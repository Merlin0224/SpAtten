import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from benchmark.benchmark_bf16_msb import benchmark_model
from spattn.spatten_qwen3_bf16_msb import (
    build_inputs_local_or_synthetic,
    configure_spatten_qwen3_model,
    load_qwen3_model_local_or_synthetic,
)


def run_variant(
    base_model,
    inputs,
    mode,
    warmup,
    iters,
    quant_threshold=0.01,
    v_threshold=0.05,
    enable_head_prune=False,
    head_prune_num=1,
    head_prune_start_layer=0,
    head_prune_interval=1,
    enable_token_prune=False,
    token_prune_num=1,
    token_prune_start_layer=0,
    token_prune_interval=1,
    token_block_size=0,
    token_recent_keep=128,
    token_prefix_keep=1,
    enable_triton_autotune=False,
):
    if mode == "baseline":
        ms = benchmark_model(base_model, inputs, num_iters=iters, warmup=warmup, reset_state=False)
        return {"ms": round(float(ms), 6), "mode": mode}

    model = configure_spatten_qwen3_model(
        base_model,
        mode,
        quant_threshold=quant_threshold,
        v_threshold=v_threshold,
        enable_head_prune=enable_head_prune,
        head_prune_num=head_prune_num,
        head_prune_start_layer=head_prune_start_layer,
        head_prune_interval=head_prune_interval,
        enable_token_prune=enable_token_prune,
        token_prune_num=token_prune_num,
        token_prune_start_layer=token_prune_start_layer,
        token_prune_interval=token_prune_interval,
        token_block_size=token_block_size,
        token_recent_keep=token_recent_keep,
        token_prefix_keep=token_prefix_keep,
        enable_triton_autotune=enable_triton_autotune,
    )
    ms = benchmark_model(
        model,
        inputs,
        num_iters=iters,
        warmup=warmup,
        reset_state=(enable_head_prune or enable_token_prune),
    )
    return {"ms": round(float(ms), 6), "mode": mode}


def recommend(results):
    baseline_ms = results["baseline"]["ms"]
    best_mode = min((name for name in results if name != "baseline"), key=lambda name: results[name]["ms"])
    best_ms = results[best_mode]["ms"]

    if best_ms > baseline_ms:
        return "All Qwen3 SpAtten variants are still slower than baseline; next step should focus on causal/GQA path tuning."
    if best_mode == "full":
        return "Full path is the fastest Qwen3 SpAtten variant; next step should continue tuning the fused causal kernel."
    if best_mode == "quant_only":
        return "Quant-only is the fastest Qwen3 SpAtten variant; next step should focus on progressive quantization under causal attention."
    return "V-prune-only is the fastest Qwen3 SpAtten variant; next step should simplify the fused causal path to inherit this gain."


def main():
    parser = argparse.ArgumentParser(description="Qwen3 BF16-MSB model-level ablation benchmark.")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--quant-threshold", type=float, default=0.01)
    parser.add_argument("--v-threshold", type=float, default=0.05)
    parser.add_argument("--enable-head-prune", action="store_true")
    parser.add_argument("--head-prune-num", type=int, default=1)
    parser.add_argument("--head-prune-start-layer", type=int, default=0)
    parser.add_argument("--head-prune-interval", type=int, default=1)
    parser.add_argument("--enable-token-prune", action="store_true")
    parser.add_argument("--token-prune-num", type=int, default=1)
    parser.add_argument("--token-prune-start-layer", type=int, default=0)
    parser.add_argument("--token-prune-interval", type=int, default=1)
    parser.add_argument("--token-block-size", type=int, default=0)
    parser.add_argument("--token-recent-keep", type=int, default=128)
    parser.add_argument("--token-prefix-keep", type=int, default=1)
    parser.add_argument("--enable-triton-autotune", action="store_true")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--allow-local-pretrained", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/route_b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for model_ablation_qwen3_bf16_msb.py")

    base_model, model_source = load_qwen3_model_local_or_synthetic(
        device,
        model_name=args.model_name,
        max_position_embeddings=max(4096, args.seq_len + 2),
        prefer_synthetic=not args.allow_local_pretrained,
    )
    inputs, input_source = build_inputs_local_or_synthetic(
        device,
        base_model,
        model_name=args.model_name,
        seq_len=args.seq_len,
        exact_seq_len=True,
    )

    results = {}
    for mode in ["baseline", "quant_only", "v_prune_only", "full"]:
        results[mode] = run_variant(
            base_model,
            inputs,
            mode,
            args.warmup,
            args.iters,
            quant_threshold=args.quant_threshold,
            v_threshold=args.v_threshold,
            enable_head_prune=args.enable_head_prune,
            head_prune_num=args.head_prune_num,
            head_prune_start_layer=args.head_prune_start_layer,
            head_prune_interval=args.head_prune_interval,
            enable_token_prune=args.enable_token_prune,
            token_prune_num=args.token_prune_num,
            token_prune_start_layer=args.token_prune_start_layer,
            token_prune_interval=args.token_prune_interval,
            token_block_size=args.token_block_size,
            token_recent_keep=args.token_recent_keep,
            token_prefix_keep=args.token_prefix_keep,
            enable_triton_autotune=args.enable_triton_autotune,
        )

    baseline_ms = results["baseline"]["ms"]
    for mode, item in results.items():
        item["slowdown_vs_baseline"] = round(item["ms"] / baseline_ms, 6)

    recommendation = recommend(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"model_ablation_qwen3_{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0),
        "model_source": model_source,
        "input_source": input_source,
        "seq_len": args.seq_len,
        "warmup": args.warmup,
        "iters": args.iters,
        "quant_threshold": args.quant_threshold,
        "v_threshold": args.v_threshold,
        "enable_head_prune": args.enable_head_prune,
        "head_prune_num": args.head_prune_num,
        "head_prune_start_layer": args.head_prune_start_layer,
        "head_prune_interval": args.head_prune_interval,
        "enable_token_prune": args.enable_token_prune,
        "token_prune_num": args.token_prune_num,
        "token_prune_start_layer": args.token_prune_start_layer,
        "token_prune_interval": args.token_prune_interval,
        "token_block_size": args.token_block_size,
        "token_recent_keep": args.token_recent_keep,
        "token_prefix_keep": args.token_prefix_keep,
        "enable_triton_autotune": args.enable_triton_autotune,
        "results": results,
        "recommendation": recommendation,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Qwen3 Model Ablation (BF16-MSB)")
    print(f"Device: {payload['device']}")
    print(f"Model source: {model_source}")
    print(f"Input source: {input_source}")
    print(f"Seq Len: {args.seq_len}")
    print(f"Head prune enabled: {args.enable_head_prune} (num={args.head_prune_num})")
    print(f"Token prune enabled: {args.enable_token_prune} (num={args.token_prune_num})")
    print(f"Token block size: {args.token_block_size}")
    print(f"Token recent keep: {args.token_recent_keep}")
    print(f"Token prefix keep: {args.token_prefix_keep}")
    print(f"Triton autotune enabled: {args.enable_triton_autotune}")
    for mode in ["baseline", "quant_only", "v_prune_only", "full"]:
        item = results[mode]
        print(f"{mode}: ms={item['ms']:.4f} | slowdown_vs_baseline={item['slowdown_vs_baseline']:.3f}x")
    print(f"Recommendation: {recommendation}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
