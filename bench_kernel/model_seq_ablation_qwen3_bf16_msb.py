import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from bench_kernel.model_ablation_qwen3_bf16_msb import recommend, run_variant
from spattn.spatten_qwen3_bf16_msb import (
    build_inputs_local_or_synthetic,
    load_qwen3_model_local_or_synthetic,
)


def parse_int_list(raw_value):
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Qwen3 BF16-MSB model-level sequence-length ablation benchmark.")
    parser.add_argument("--seq-lens", default="1024,2048,4096,8192")
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
        raise RuntimeError("CUDA is required for model_seq_ablation_qwen3_bf16_msb.py")

    seq_lens = parse_int_list(args.seq_lens)
    base_model, model_source = load_qwen3_model_local_or_synthetic(
        device,
        model_name=args.model_name,
        max_position_embeddings=max(4096, max(seq_lens) + 2),
        prefer_synthetic=not args.allow_local_pretrained,
    )

    all_results = []
    print("Qwen3 Model Seq Ablation (BF16-MSB)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Model source: {model_source}")
    print(f"Head prune enabled: {args.enable_head_prune} (num={args.head_prune_num})")
    print(f"Token prune enabled: {args.enable_token_prune} (num={args.token_prune_num})")
    print(f"Token block size: {args.token_block_size}")
    print(f"Token recent keep: {args.token_recent_keep}")
    print(f"Token prefix keep: {args.token_prefix_keep}")
    print(f"Triton autotune enabled: {args.enable_triton_autotune}")
    print(f"{'Seq Len':<8} | {'Baseline':<10} | {'Quant':<10} | {'V-Prune':<10} | {'Full':<10} | {'Best Variant':<12}")
    print("-" * 78)

    for seq_len in seq_lens:
        inputs, input_source = build_inputs_local_or_synthetic(
            device,
            base_model,
            model_name=args.model_name,
            seq_len=seq_len,
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

        best_variant = min(
            ("quant_only", "v_prune_only", "full"),
            key=lambda name: results[name]["ms"],
        )
        recommendation = recommend(results)

        record = {
            "seq_len": seq_len,
            "input_source": input_source,
            "results": results,
            "best_variant": best_variant,
            "recommendation": recommendation,
        }
        all_results.append(record)

        print(
            f"{seq_len:<8} | "
            f"{results['baseline']['ms']:<10.4f} | "
            f"{results['quant_only']['ms']:<10.4f} | "
            f"{results['v_prune_only']['ms']:<10.4f} | "
            f"{results['full']['ms']:<10.4f} | "
            f"{best_variant:<12}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"model_seq_ablation_qwen3_{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0),
        "model_source": model_source,
        "seq_lens": seq_lens,
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
        "results": all_results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
