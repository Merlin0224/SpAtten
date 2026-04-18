import argparse
import copy
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

try:
    from benchmark_paper_bf16_msb import benchmark_model, reset_spatten_states
except ImportError:
    from benchmark.benchmark_bf16_msb import benchmark_model, reset_spatten_states

try:
    from spatten_bert_ultimate_paper_bf16_msb import (
        SpattenBertSelfAttention,
        build_inputs_local_or_synthetic,
        load_bert_model_local_or_synthetic,
        spatten_encoder_forward,
    )
except ImportError:
    from spattn.spatten_bert_bf16_msb import (
        SpattenBertSelfAttention,
        build_inputs_local_or_synthetic,
        load_bert_model_local_or_synthetic,
        spatten_encoder_forward,
    )
from transformers.models.bert.modeling_bert import BertEncoder


def parse_int_list(raw_value):
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


class BertEncoderStackWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def encoder(self):
        return self.model.encoder

    def forward(self, **inputs):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        token_type_ids = inputs.get("token_type_ids")
        inputs_embeds = inputs.get("inputs_embeds")

        if input_ids is None and inputs_embeds is None:
            raise ValueError("BertEncoderStackWrapper requires input_ids or inputs_embeds")

        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

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
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            return_dict=True,
        )
        return encoder_outputs.last_hidden_state


def configure_spatten_model(
    base_model,
    mode,
    token_prune_num,
    head_prune_num,
    quant_threshold=0.01,
    v_threshold=0.05,
    head_prune_start_layer=0,
    token_prune_start_layer=0,
    head_prune_interval=1,
    token_prune_interval=2,
    enable_head_prune=False,
    enable_token_prune=False,
    enable_delayed_token_compaction=False,
    token_compact_interval=1,
    token_compact_min_drop_ratio=1.0,
    graph_capture_mode=False,
):
    spatten_model = copy.deepcopy(base_model)
    for layer_idx, layer in enumerate(spatten_model.encoder.layer):
        orig_state = layer.attention.self.state_dict()
        new_attn = SpattenBertSelfAttention(spatten_model.config)
        new_attn.load_state_dict(orig_state)

        new_attn.enable_head_prune = enable_head_prune
        new_attn.head_prune_num = head_prune_num
        new_attn.enable_token_prune = enable_token_prune
        new_attn.token_prune_num = token_prune_num
        new_attn.enable_prog_quant = mode in {"quant_only", "full"}
        new_attn.enable_v_prune = mode in {"v_prune_only", "full"}
        new_attn.quant_threshold = quant_threshold
        new_attn.v_threshold = v_threshold
        new_attn.v_prune_num = 2
        new_attn.layer_idx = layer_idx
        new_attn.head_prune_start_layer = head_prune_start_layer
        new_attn.token_prune_start_layer = token_prune_start_layer
        new_attn.head_prune_interval = head_prune_interval
        new_attn.token_prune_interval = token_prune_interval
        new_attn.enable_delayed_token_compaction = enable_delayed_token_compaction
        new_attn.token_compact_interval = token_compact_interval
        new_attn.token_compact_min_drop_ratio = token_compact_min_drop_ratio
        new_attn.graph_capture_mode = graph_capture_mode
        layer.attention.self = new_attn

    spatten_model.encoder.forward = spatten_encoder_forward.__get__(spatten_model.encoder, BertEncoder)
    spatten_model.to(next(base_model.parameters()).device).float().eval()
    reset_spatten_states(spatten_model)
    return spatten_model


def maybe_compile_model(model, enabled, dynamic=False, mode="default"):
    if not enabled:
        return model, "eager"

    try:
        compiled = torch.compile(model, dynamic=dynamic, mode=mode)
        return compiled, f"torch.compile(mode={mode}, dynamic={dynamic})"
    except Exception as exc:
        return model, f"compile_failed: {type(exc).__name__}: {exc}"


def benchmark_variant(model, inputs, warmup, iters, reset_state=False):
    ms = benchmark_model(model, inputs, num_iters=iters, warmup=warmup, reset_state=reset_state)
    return round(float(ms), 6)


def main():
    parser = argparse.ArgumentParser(description="Compare eager baseline, torch.compile baseline, and SpAtten BF16-MSB across sequence lengths.")
    parser.add_argument("--seq-lens", default="128,256,512,1024,2048,4096")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--compile-dynamic", action="store_true")
    parser.add_argument("--enable-head-prune", action="store_true")
    parser.add_argument("--enable-token-prune", action="store_true")
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
    parser.add_argument("--spatten-mode", choices=["quant_only", "v_prune_only", "full"], default="full")
    parser.add_argument("--output-dir", default="artifacts/route_b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for route_b_model_seq_tech_compare.py")

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
    spatten_base = configure_spatten_model(
        base_model,
        args.spatten_mode,
        args.token_prune_num,
        args.head_prune_num,
        quant_threshold=args.quant_threshold,
        v_threshold=args.v_threshold,
        head_prune_start_layer=args.head_prune_start_layer,
        token_prune_start_layer=args.token_prune_start_layer,
        head_prune_interval=args.head_prune_interval,
        token_prune_interval=args.token_prune_interval,
        enable_head_prune=args.enable_head_prune,
        enable_token_prune=args.enable_token_prune,
        enable_delayed_token_compaction=args.enable_delayed_token_compaction,
        token_compact_interval=args.token_compact_interval,
        token_compact_min_drop_ratio=args.token_compact_min_drop_ratio,
    )
    spatten_base = BertEncoderStackWrapper(spatten_base).to(device).eval()

    all_results = []
    print("Route-B Model Seq Tech Compare")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Model source: {model_source}")
    print(f"Compile mode: {args.compile_mode} (dynamic={args.compile_dynamic})")
    print(f"SpAtten mode: {args.spatten_mode}")
    print(f"Head prune enabled: {args.enable_head_prune} (num={args.head_prune_num})")
    print(f"Token prune enabled: {args.enable_token_prune} (num={args.token_prune_num})")
    print(
        f"Delayed token compaction: {args.enable_delayed_token_compaction} "
        f"(interval={args.token_compact_interval}, min_drop_ratio={args.token_compact_min_drop_ratio})"
    )
    print(f"{'Seq Len':<8} | {'Eager':<10} | {'Compile':<10} | {'SpAtten':<10} | {'Best':<10}")
    print("-" * 68)

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

        eager_ms = benchmark_variant(eager_base_model, inputs, args.warmup, args.iters, reset_state=False)
        try:
            compile_ms = benchmark_variant(compiled_base_model, inputs, args.warmup, args.iters, reset_state=False)
            compile_runtime_status = compile_status
        except Exception as exc:
            compile_ms = eager_ms
            compile_runtime_status = f"runtime_failed: {type(exc).__name__}: {exc}"
        spatten_ms = benchmark_variant(spatten_base, inputs, args.warmup, args.iters, reset_state=True)

        bucket = {
            "seq_len": seq_len,
            "input_source": input_source,
            "eager_ms": eager_ms,
            "compile_ms": compile_ms,
            "spatten_ms": spatten_ms,
            "compile_runtime_status": compile_runtime_status,
            "compile_speedup_vs_eager": round(eager_ms / compile_ms, 6),
            "spatten_speedup_vs_eager": round(eager_ms / spatten_ms, 6),
        }
        bucket["best"] = min(
            ("eager", "compile", "spatten"),
            key=lambda name: bucket[f"{name}_ms"],
        )
        all_results.append(bucket)

        print(
            f"{seq_len:<8} | "
            f"{eager_ms:<10.4f} | "
            f"{compile_ms:<10.4f} | "
            f"{spatten_ms:<10.4f} | "
            f"{bucket['best']:<10}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"model_seq_tech_compare_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0),
        "model_source": model_source,
        "compile_status": f"per-seq compile with mode={args.compile_mode}, dynamic={args.compile_dynamic}",
        "compile_mode": args.compile_mode,
        "compile_dynamic": args.compile_dynamic,
        "spatten_mode": args.spatten_mode,
        "enable_head_prune": args.enable_head_prune,
        "enable_token_prune": args.enable_token_prune,
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
        "seq_lens": seq_lens,
        "results": all_results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
