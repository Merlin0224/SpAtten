import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from route_b_model_ablation_paper_bf16_msb import recommend, run_variant
from spatten_bert_ultimate_paper_bf16_msb import build_inputs_local_or_synthetic, load_bert_model_local_or_synthetic


def parse_int_list(raw_value):
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Route-B paper BF16-MSB model-level sequence-length ablation benchmark.")
    parser.add_argument("--seq-lens", default="128,256,512,1024,2048,4096")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--token-prune-num", type=int, default=1)
    parser.add_argument("--head-prune-num", type=int, default=1)
    parser.add_argument("--quant-threshold", type=float, default=0.05)
    parser.add_argument("--v-threshold", type=float, default=0.05)
    parser.add_argument("--enable-head-prune", action="store_true")
    parser.add_argument("--enable-token-prune", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/route_b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for route_b_model_seq_ablation_paper_bf16_msb.py")

    seq_lens = parse_int_list(args.seq_lens)
    base_model, model_source = load_bert_model_local_or_synthetic(
        device,
        max_position_embeddings=max(512, max(seq_lens) + 2),
    )

    all_results = []
    print("Route-B Model Seq Ablation (Paper BF16-MSB)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Model source: {model_source}")
    print(f"Head prune enabled: {args.enable_head_prune} (num={args.head_prune_num})")
    print(f"Token prune enabled: {args.enable_token_prune} (num={args.token_prune_num})")
    print(f"{'Seq Len':<8} | {'Baseline':<10} | {'Quant':<10} | {'V-Prune':<10} | {'Full':<10} | {'Best Variant':<12}")
    print("-" * 78)

    for seq_len in seq_lens:
        inputs, input_source = build_inputs_local_or_synthetic(
            device,
            base_model,
            seq_len=seq_len,
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
                enable_head_prune=args.enable_head_prune,
                enable_token_prune=args.enable_token_prune,
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
    output_path = output_dir / f"model_seq_ablation_{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0),
        "model_source": model_source,
        "seq_lens": seq_lens,
        "warmup": args.warmup,
        "iters": args.iters,
        "token_prune_num": args.token_prune_num,
        "head_prune_num": args.head_prune_num,
        "quant_threshold": args.quant_threshold,
        "v_threshold": args.v_threshold,
        "enable_head_prune": args.enable_head_prune,
        "enable_token_prune": args.enable_token_prune,
        "results": all_results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
