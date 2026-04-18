import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import torch

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
    enable_token_stage_pruning=False,
    token_stage_size=1,
    token_stage_weighting="uniform",
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
        new_attn.enable_token_stage_pruning = enable_token_stage_pruning
        new_attn.token_stage_size = token_stage_size
        new_attn.token_stage_weighting = token_stage_weighting

        layer.attention.self = new_attn

    spatten_model.encoder.forward = spatten_encoder_forward.__get__(spatten_model.encoder, BertEncoder)
    spatten_model.to(next(base_model.parameters()).device).float().eval()
    reset_spatten_states(spatten_model)
    return spatten_model


def run_variant(
    base_model,
    inputs,
    mode,
    token_prune_num,
    head_prune_num,
    warmup,
    iters,
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
    enable_token_stage_pruning=False,
    token_stage_size=1,
    token_stage_weighting="uniform",
):
    if mode == "baseline":
        ms = benchmark_model(base_model, inputs, num_iters=iters, warmup=warmup, reset_state=False)
        return {
            "ms": round(float(ms), 6),
            "mode": mode,
        }

    spatten_model = configure_spatten_model(
        base_model,
        mode,
        token_prune_num,
        head_prune_num,
        quant_threshold=quant_threshold,
        v_threshold=v_threshold,
        head_prune_start_layer=head_prune_start_layer,
        token_prune_start_layer=token_prune_start_layer,
        head_prune_interval=head_prune_interval,
        token_prune_interval=token_prune_interval,
        enable_head_prune=enable_head_prune,
        enable_token_prune=enable_token_prune,
        enable_delayed_token_compaction=enable_delayed_token_compaction,
        token_compact_interval=token_compact_interval,
        token_compact_min_drop_ratio=token_compact_min_drop_ratio,
        enable_token_stage_pruning=enable_token_stage_pruning,
        token_stage_size=token_stage_size,
        token_stage_weighting=token_stage_weighting,
    )
    ms = benchmark_model(spatten_model, inputs, num_iters=iters, warmup=warmup, reset_state=True)
    return {
        "ms": round(float(ms), 6),
        "mode": mode,
    }


def recommend(results):
    baseline_ms = results["baseline"]["ms"]
    best_mode = min((name for name in results if name != "baseline"), key=lambda name: results[name]["ms"])
    best_ms = results[best_mode]["ms"]

    if best_mode == "full":
        return "Full end-to-end path is the fastest SpAtten variant; next step should target the fused kernel plus Python scheduling overhead together."
    if best_mode == "quant_only":
        return "Quant-only is the fastest end-to-end SpAtten variant; next step should keep focusing on the BF16-MSB progressive quantization path."
    if best_mode == "v_prune_only":
        return "V-prune-only is the fastest end-to-end SpAtten variant; next step should simplify the fused path so it inherits the standalone V-prune gains."
    if best_ms > baseline_ms:
        return "All SpAtten variants are still slower than baseline end-to-end; next step should focus on model-level overhead, not only the kernel."
    return "Results are mixed; next step should compare per-layer scheduling and tensor slicing overhead."


def main():
    parser = argparse.ArgumentParser(description="Route-B paper BF16-MSB model-level ablation benchmark.")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
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
    parser.add_argument("--output-dir", default="artifacts/route_b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for route_b_model_ablation_paper_bf16_msb.py")

    base_model, model_source = load_bert_model_local_or_synthetic(
        device,
        max_position_embeddings=max(512, args.seq_len + 2),
    )
    inputs, input_source = build_inputs_local_or_synthetic(
        device,
        base_model,
        seq_len=args.seq_len,
        exact_seq_len=True,
    )

    results = {}
    for mode in ["baseline", "quant_only", "v_prune_only", "full"]:
        results[mode] = run_variant(
            base_model,
            inputs,
            mode,
            args.token_prune_num,
            args.head_prune_num,
            args.warmup,
            args.iters,
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

    baseline_ms = results["baseline"]["ms"]
    for mode, item in results.items():
        item["slowdown_vs_baseline"] = round(item["ms"] / baseline_ms, 6)

    recommendation = recommend(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"model_ablation_{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0),
        "model_source": model_source,
        "input_source": input_source,
        "seq_len": args.seq_len,
        "warmup": args.warmup,
        "iters": args.iters,
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
        "results": results,
        "recommendation": recommendation,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Route-B Model Ablation (Paper BF16-MSB)")
    print(f"Device: {payload['device']}")
    print(f"Model source: {model_source}")
    print(f"Input source: {input_source}")
    print(f"Seq Len: {args.seq_len}")
    print(f"Head prune enabled: {args.enable_head_prune} (num={args.head_prune_num})")
    print(f"Token prune enabled: {args.enable_token_prune} (num={args.token_prune_num})")
    print(
        f"Delayed token compaction: {args.enable_delayed_token_compaction} "
        f"(interval={args.token_compact_interval}, min_drop_ratio={args.token_compact_min_drop_ratio})"
    )
    print(
        f"Token-only stage pruning: {args.enable_token_stage_pruning} "
        f"(token_stage_size={args.token_stage_size}, weighting={args.token_stage_weighting})"
    )
    for mode in ["baseline", "quant_only", "v_prune_only", "full"]:
        item = results[mode]
        print(
            f"{mode}: ms={item['ms']:.4f} | slowdown_vs_baseline={item['slowdown_vs_baseline']:.3f}x"
        )
    print(f"Recommendation: {recommendation}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
