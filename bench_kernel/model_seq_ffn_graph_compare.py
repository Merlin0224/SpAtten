import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from benchmark.benchmark_bf16_msb import benchmark_model, reset_spatten_states
from bench_kernel.model_ablation_bf16_msb import configure_spatten_model
from spattn.spatten_bert_bf16_msb import (
    build_inputs_local_or_synthetic,
    load_bert_model_local_or_synthetic,
)


def parse_int_list(raw_value):
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


class BertFFNBlock(nn.Module):
    def __init__(self, intermediate, output):
        super().__init__()
        self.intermediate = intermediate
        self.output = output

    def forward(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)


class FeedForwardChunkWrapper(nn.Module):
    def __init__(self, runtime_module):
        super().__init__()
        self.runtime_module = runtime_module

    def forward(self, attention_output):
        return self.runtime_module(attention_output)


class TensorCUDAGraphRunner:
    def __init__(self, model, sample_input, warmup=3):
        if not sample_input.is_cuda:
            raise RuntimeError("TensorCUDAGraphRunner requires CUDA tensors")

        self.model = model.eval()
        self.static_input = sample_input.clone()
        self.output = None
        self.graph = torch.cuda.CUDAGraph()

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            with torch.no_grad():
                for _ in range(warmup):
                    self.output = self.model(self.static_input)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        with torch.no_grad():
            with torch.cuda.graph(self.graph):
                self.output = self.model(self.static_input)

    def replay(self, new_input):
        self.static_input.copy_(new_input)
        with torch.no_grad():
            self.graph.replay()
        return self.output


class FFNCompileModule(nn.Module):
    def __init__(self, ffn_block, compile_mode="default"):
        super().__init__()
        self.fallback = ffn_block
        self.runtime = ffn_block
        self.status = "eager"
        try:
            self.runtime = torch.compile(ffn_block, dynamic=False, mode=compile_mode)
            self.status = f"torch.compile(mode={compile_mode}, dynamic=False)"
        except Exception as exc:
            self.status = f"compile_failed: {type(exc).__name__}: {exc}"

    def forward(self, attention_output):
        return self.runtime(attention_output)


class FFNGraphModule(nn.Module):
    def __init__(self, ffn_block, graph_warmup=3):
        super().__init__()
        self.ffn_block = ffn_block.eval()
        self.graph_warmup = graph_warmup
        self.status = f"cuda_graph(warmup={graph_warmup})"
        self._runner_cache = {}

    def _cache_key(self, attention_output):
        return (
            tuple(attention_output.shape),
            str(attention_output.dtype),
            tuple(attention_output.stride()),
        )

    def forward(self, attention_output):
        if not attention_output.is_cuda:
            return self.ffn_block(attention_output)

        key = self._cache_key(attention_output)
        runner = self._runner_cache.get(key)
        if runner is None:
            runner = TensorCUDAGraphRunner(self.ffn_block, attention_output, warmup=self.graph_warmup)
            self._runner_cache[key] = runner
        return runner.replay(attention_output)


def patch_ffn_branch(model, branch, compile_mode="default", graph_warmup=3):
    statuses = []
    for layer_idx, layer in enumerate(model.encoder.layer):
        ffn_block = BertFFNBlock(layer.intermediate, layer.output).to(next(model.parameters()).device).eval()
        if branch == "compile":
            patched_module = FFNCompileModule(ffn_block, compile_mode=compile_mode)
        elif branch == "graph":
            patched_module = FFNGraphModule(ffn_block, graph_warmup=graph_warmup)
        else:
            raise ValueError(f"Unknown FFN branch: {branch}")

        layer.ffn_branch_module = patched_module
        layer.feed_forward_chunk = FeedForwardChunkWrapper(patched_module)
        statuses.append(
            {
                "layer_idx": layer_idx,
                "status": getattr(patched_module, "status", branch),
            }
        )
    return model, statuses


def build_spatten_variant(base_model, args, ffn_branch=None):
    model = configure_spatten_model(
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
        enable_token_stage_pruning=args.enable_token_stage_pruning,
        token_stage_size=args.token_stage_size,
        token_stage_weighting=args.token_stage_weighting,
    )

    branch_status = []
    if ffn_branch is not None:
        model, branch_status = patch_ffn_branch(
            model,
            ffn_branch,
            compile_mode=args.ffn_compile_mode,
            graph_warmup=args.graph_warmup,
        )
    return model, branch_status


def benchmark_variant(model, inputs, warmup, iters, reset_state=False):
    ms = benchmark_model(model, inputs, num_iters=iters, warmup=warmup, reset_state=reset_state)
    return round(float(ms), 6)


def main():
    parser = argparse.ArgumentParser(description="Compare FFN-only torch.compile and CUDA Graph branches on top of the current SpAtten mainline.")
    parser.add_argument("--seq-lens", default="1024,2048,4096,8192")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--graph-warmup", type=int, default=3)
    parser.add_argument("--spatten-mode", choices=["quant_only", "v_prune_only", "full"], default="full")
    parser.add_argument("--token-prune-num", type=int, default=1)
    parser.add_argument("--head-prune-num", type=int, default=1)
    parser.add_argument("--quant-threshold", type=float, default=0.01)
    parser.add_argument("--v-threshold", type=float, default=0.05)
    parser.add_argument("--head-prune-start-layer", type=int, default=0)
    parser.add_argument("--token-prune-start-layer", type=int, default=0)
    parser.add_argument("--head-prune-interval", type=int, default=1)
    parser.add_argument("--token-prune-interval", type=int, default=2)
    parser.add_argument("--enable-head-prune", action="store_true")
    parser.add_argument("--enable-token-prune", action="store_true")
    parser.add_argument("--enable-delayed-token-compaction", action="store_true")
    parser.add_argument("--token-compact-interval", type=int, default=1)
    parser.add_argument("--token-compact-min-drop-ratio", type=float, default=1.0)
    parser.add_argument("--enable-token-stage-pruning", action="store_true")
    parser.add_argument("--token-stage-size", type=int, default=1)
    parser.add_argument("--token-stage-weighting", choices=["uniform", "linear"], default="uniform")
    parser.add_argument("--ffn-compile-mode", default="default")
    parser.add_argument("--output-dir", default="artifacts/route_b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for model_seq_ffn_graph_compare.py")

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    seq_lens = parse_int_list(args.seq_lens)
    base_model, model_source = load_bert_model_local_or_synthetic(
        device,
        max_position_embeddings=max(512, max(seq_lens) + 2),
    )

    eager_spatten, _ = build_spatten_variant(base_model, args, ffn_branch=None)
    compile_spatten, compile_status = build_spatten_variant(base_model, args, ffn_branch="compile")
    graph_spatten, graph_status = build_spatten_variant(base_model, args, ffn_branch="graph")

    all_results = []
    print("Route-B Model Seq FFN Graph Compare")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Model source: {model_source}")
    print(f"SpAtten mode: {args.spatten_mode}")
    print(f"Head prune enabled: {args.enable_head_prune} (num={args.head_prune_num})")
    print(f"Token prune enabled: {args.enable_token_prune} (num={args.token_prune_num})")
    compile_summary = compile_status[0]["status"] if compile_status else "not_patched"
    graph_summary = graph_status[0]["status"] if graph_status else "not_patched"
    print(f"FFN compile status: {compile_summary}")
    print(f"FFN graph status: {graph_summary}")
    print(
        f"{'Seq Len':<8} | {'Baseline':<10} | {'SpAtten':<10} | "
        f"{'FFN-Compile':<12} | {'FFN-Graph':<10} | {'Best':<12}"
    )
    print("-" * 80)

    for seq_len in seq_lens:
        inputs, input_source = build_inputs_local_or_synthetic(
            device,
            base_model,
            seq_len=seq_len,
            exact_seq_len=True,
        )

        baseline_ms = benchmark_variant(base_model, inputs, args.warmup, args.iters, reset_state=False)
        spatten_ms = benchmark_variant(eager_spatten, inputs, args.warmup, args.iters, reset_state=True)
        ffn_compile_ms = benchmark_variant(compile_spatten, inputs, args.warmup, args.iters, reset_state=True)
        ffn_graph_ms = benchmark_variant(graph_spatten, inputs, args.warmup, args.iters, reset_state=True)

        bucket = {
            "seq_len": seq_len,
            "input_source": input_source,
            "baseline_ms": baseline_ms,
            "spatten_ms": spatten_ms,
            "ffn_compile_ms": ffn_compile_ms,
            "ffn_graph_ms": ffn_graph_ms,
        }
        bucket["best"] = min(
            ("baseline", "spatten", "ffn_compile", "ffn_graph"),
            key=lambda name: bucket[f"{name}_ms"],
        )
        all_results.append(bucket)

        print(
            f"{seq_len:<8} | {baseline_ms:<10.4f} | {spatten_ms:<10.4f} | "
            f"{ffn_compile_ms:<12.4f} | {ffn_graph_ms:<10.4f} | {bucket['best']:<12}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"model_seq_ffn_graph_compare_{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0),
        "model_source": model_source,
        "seq_lens": seq_lens,
        "warmup": args.warmup,
        "iters": args.iters,
        "graph_warmup": args.graph_warmup,
        "spatten_mode": args.spatten_mode,
        "token_prune_num": args.token_prune_num,
        "head_prune_num": args.head_prune_num,
        "quant_threshold": args.quant_threshold,
        "v_threshold": args.v_threshold,
        "head_prune_start_layer": args.head_prune_start_layer,
        "token_prune_start_layer": args.token_prune_start_layer,
        "head_prune_interval": args.head_prune_interval,
        "token_prune_interval": args.token_prune_interval,
        "enable_head_prune": args.enable_head_prune,
        "enable_token_prune": args.enable_token_prune,
        "enable_delayed_token_compaction": args.enable_delayed_token_compaction,
        "token_compact_interval": args.token_compact_interval,
        "token_compact_min_drop_ratio": args.token_compact_min_drop_ratio,
        "enable_token_stage_pruning": args.enable_token_stage_pruning,
        "token_stage_size": args.token_stage_size,
        "token_stage_weighting": args.token_stage_weighting,
        "ffn_compile_mode": args.ffn_compile_mode,
        "ffn_compile_status": compile_status,
        "ffn_graph_status": graph_status,
        "results": all_results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
