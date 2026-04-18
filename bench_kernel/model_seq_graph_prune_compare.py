import argparse
import copy
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from benchmark.benchmark_bf16_msb import benchmark_cudagraph_model, benchmark_model
from bench_kernel.model_ablation_bf16_msb import configure_spatten_model
from spattn.spatten_bert_bf16_msb import (
    build_inputs_local_or_synthetic,
    load_bert_model_local_or_synthetic,
)


def parse_int_list(raw_value):
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


class BertEncoderStackWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device, dtype=torch.long)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, device=device, dtype=torch.long)

        try:
            extended_attention_mask = self.model.get_extended_attention_mask(attention_mask, input_shape, device)
        except TypeError:
            extended_attention_mask = self.model.get_extended_attention_mask(attention_mask, input_shape)

        embedding_output = self.model.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        encoder_outputs = self.model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            return_dict=True,
        )
        return encoder_outputs.last_hidden_state


def maybe_compile_model(model, enabled, dynamic=False, mode="default"):
    if not enabled:
        return model, "eager"
    try:
        compiled = torch.compile(model, dynamic=dynamic, mode=mode)
        return compiled, f"torch.compile(mode={mode}, dynamic={dynamic})"
    except Exception as exc:
        return model, f"compile_failed: {type(exc).__name__}: {exc}"


def benchmark_plain(model, inputs, warmup, iters, reset_state=False):
    return round(float(benchmark_model(model, inputs, num_iters=iters, warmup=warmup, reset_state=reset_state)), 6)


def benchmark_graph(model, inputs, warmup, iters, graph_warmup):
    return round(
        float(
            benchmark_cudagraph_model(
                model,
                inputs,
                num_iters=iters,
                warmup=warmup,
                reset_state=False,
                graph_warmup=graph_warmup,
            )
        ),
        6,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare graph-safe static SpAtten path against the real pruning-enabled dynamic path."
    )
    parser.add_argument("--seq-lens", default="1024,2048,4096,8192")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--graph-warmup", type=int, default=3)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--compile-dynamic", action="store_true")
    parser.add_argument("--spatten-mode", choices=["quant_only", "v_prune_only", "full"], default="full")
    parser.add_argument("--token-prune-num", type=int, default=1)
    parser.add_argument("--head-prune-num", type=int, default=1)
    parser.add_argument("--quant-threshold", type=float, default=0.01)
    parser.add_argument("--v-threshold", type=float, default=0.05)
    parser.add_argument("--head-prune-start-layer", type=int, default=0)
    parser.add_argument("--token-prune-start-layer", type=int, default=0)
    parser.add_argument("--head-prune-interval", type=int, default=1)
    parser.add_argument("--token-prune-interval", type=int, default=2)
    parser.add_argument("--enable-delayed-token-compaction", action="store_true")
    parser.add_argument("--token-compact-interval", type=int, default=1)
    parser.add_argument("--token-compact-min-drop-ratio", type=float, default=1.0)
    parser.add_argument("--output-dir", default="artifacts/route_b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for model_seq_graph_prune_compare.py")

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.CRITICAL)
    logging.getLogger("torch._dynamo.convert_frame").setLevel(logging.ERROR)

    seq_lens = parse_int_list(args.seq_lens)
    base_model, model_source = load_bert_model_local_or_synthetic(
        device,
        max_position_embeddings=max(512, max(seq_lens) + 2),
    )

    eager_base_model = BertEncoderStackWrapper(base_model).to(device).eval()

    spatten_static = configure_spatten_model(
        base_model,
        args.spatten_mode,
        token_prune_num=0,
        head_prune_num=0,
        quant_threshold=args.quant_threshold,
        v_threshold=args.v_threshold,
        enable_head_prune=False,
        enable_token_prune=False,
        graph_capture_mode=True,
    )
    spatten_static = BertEncoderStackWrapper(spatten_static).to(device).eval()

    all_results = []
    print("Route-B Model Seq Graph Prune Compare")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Model source: {model_source}")
    print(f"Compile mode: {args.compile_mode} (dynamic={args.compile_dynamic})")
    print(f"SpAtten mode: {args.spatten_mode}")
    print(
        f"Pruning config: head={args.head_prune_num}, token={args.token_prune_num}, "
        f"head_interval={args.head_prune_interval}, token_interval={args.token_prune_interval}"
    )
    print(
        f"{'Seq Len':<8} | {'Eager':<10} | {'Compile':<10} | {'Graph':<10} | "
        f"{'Sp-Static':<10} | {'Sp-Graph':<10} | {'Sp-Prune':<10} | {'Best':<10}"
    )
    print("-" * 98)

    for seq_len in seq_lens:
        inputs, input_source = build_inputs_local_or_synthetic(
            device,
            base_model,
            seq_len=seq_len,
            exact_seq_len=True,
        )

        compiled_base_model, compile_status = maybe_compile_model(
            BertEncoderStackWrapper(copy.deepcopy(base_model).to(device).float().eval()),
            enabled=True,
            dynamic=args.compile_dynamic,
            mode=args.compile_mode,
        )

        spatten_prune = configure_spatten_model(
            base_model,
            args.spatten_mode,
            token_prune_num=args.token_prune_num,
            head_prune_num=args.head_prune_num,
            quant_threshold=args.quant_threshold,
            v_threshold=args.v_threshold,
            head_prune_start_layer=args.head_prune_start_layer,
            token_prune_start_layer=args.token_prune_start_layer,
            head_prune_interval=args.head_prune_interval,
            token_prune_interval=args.token_prune_interval,
            enable_head_prune=True,
            enable_token_prune=True,
            enable_delayed_token_compaction=args.enable_delayed_token_compaction,
            token_compact_interval=args.token_compact_interval,
            token_compact_min_drop_ratio=args.token_compact_min_drop_ratio,
            graph_capture_mode=False,
        )

        eager_ms = benchmark_plain(eager_base_model, inputs, args.warmup, args.iters, reset_state=False)
        graph_ms = benchmark_graph(eager_base_model, inputs, args.warmup, args.iters, args.graph_warmup)
        try:
            compile_ms = benchmark_plain(compiled_base_model, inputs, args.warmup, args.iters, reset_state=False)
            compile_runtime_status = compile_status
        except Exception as exc:
            compile_ms = eager_ms
            compile_runtime_status = f"runtime_failed: {type(exc).__name__}: {exc}"

        spatten_static_ms = benchmark_plain(spatten_static, inputs, args.warmup, args.iters, reset_state=False)
        spatten_graph_ms = benchmark_graph(spatten_static, inputs, args.warmup, args.iters, args.graph_warmup)
        spatten_prune_ms = benchmark_plain(spatten_prune, inputs, args.warmup, args.iters, reset_state=True)

        bucket = {
            "seq_len": seq_len,
            "input_source": input_source,
            "eager_ms": eager_ms,
            "compile_ms": compile_ms,
            "graph_ms": graph_ms,
            "spatten_static_ms": spatten_static_ms,
            "spatten_graph_ms": spatten_graph_ms,
            "spatten_prune_ms": spatten_prune_ms,
            "compile_runtime_status": compile_runtime_status,
        }
        bucket["best"] = min(
            ("eager", "compile", "graph", "spatten_static", "spatten_graph", "spatten_prune"),
            key=lambda name: bucket[f"{name}_ms"],
        )
        all_results.append(bucket)

        print(
            f"{seq_len:<8} | {eager_ms:<10.4f} | {compile_ms:<10.4f} | {graph_ms:<10.4f} | "
            f"{spatten_static_ms:<10.4f} | {spatten_graph_ms:<10.4f} | {spatten_prune_ms:<10.4f} | "
            f"{bucket['best']:<10}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"model_seq_graph_prune_compare_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0),
        "model_source": model_source,
        "seq_lens": seq_lens,
        "warmup": args.warmup,
        "iters": args.iters,
        "graph_warmup": args.graph_warmup,
        "compile_mode": args.compile_mode,
        "compile_dynamic": args.compile_dynamic,
        "spatten_mode": args.spatten_mode,
        "token_prune_num": args.token_prune_num,
        "head_prune_num": args.head_prune_num,
        "quant_threshold": args.quant_threshold,
        "v_threshold": args.v_threshold,
        "head_prune_start_layer": args.head_prune_start_layer,
        "token_prune_start_layer": args.token_prune_start_layer,
        "head_prune_interval": args.head_prune_interval,
        "token_prune_interval": args.token_prune_interval,
        "enable_delayed_token_compaction": args.enable_delayed_token_compaction,
        "token_compact_interval": args.token_compact_interval,
        "token_compact_min_drop_ratio": args.token_compact_min_drop_ratio,
        "results": all_results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
