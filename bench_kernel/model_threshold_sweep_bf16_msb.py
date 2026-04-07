import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from bench_kernel.model_ablation_bf16_msb import run_variant
from spattn.spatten_bert_bf16_msb import (
    build_inputs_local_or_synthetic,
    load_bert_model_local_or_synthetic,
)


def parse_float_list(raw_value):
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Route-B BF16-MSB model-level threshold sweep.")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--mode", choices=["quant_only", "v_prune_only", "full"], default="full")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--token-prune-num", type=int, default=1)
    parser.add_argument("--head-prune-num", type=int, default=1)
    parser.add_argument("--enable-head-prune", action="store_true")
    parser.add_argument("--enable-token-prune", action="store_true")
    parser.add_argument("--quant-threshold-list", default="0.0,0.01,0.03,0.05,0.08,0.1,0.2")
    parser.add_argument("--v-threshold-list", default="0.0,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--output-dir", default="artifacts/route_b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for route_b_model_threshold_sweep_paper_bf16_msb.py")

    quant_thresholds = parse_float_list(args.quant_threshold_list)
    v_thresholds = parse_float_list(args.v_threshold_list)

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

    baseline = run_variant(
        base_model,
        inputs,
        "baseline",
        args.token_prune_num,
        args.head_prune_num,
        args.warmup,
        args.iters,
        enable_head_prune=args.enable_head_prune,
        enable_token_prune=args.enable_token_prune,
    )
    baseline_ms = baseline["ms"]

    results = []
    for quant_threshold in quant_thresholds:
        for v_threshold in v_thresholds:
            if args.mode == "quant_only" and v_threshold != v_thresholds[0]:
                continue
            if args.mode == "v_prune_only" and quant_threshold != quant_thresholds[0]:
                continue

            item = run_variant(
                base_model,
                inputs,
                args.mode,
                args.token_prune_num,
                args.head_prune_num,
                args.warmup,
                args.iters,
                quant_threshold=quant_threshold,
                v_threshold=v_threshold,
                enable_head_prune=args.enable_head_prune,
                enable_token_prune=args.enable_token_prune,
            )
            item["quant_threshold"] = quant_threshold
            item["v_threshold"] = v_threshold
            item["slowdown_vs_baseline"] = round(item["ms"] / baseline_ms, 6)
            item["speedup_vs_baseline"] = round(baseline_ms / item["ms"], 6)
            results.append(item)

    results.sort(key=lambda item: item["ms"])
    best = results[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"model_threshold_sweep_{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0),
        "model_source": model_source,
        "input_source": input_source,
        "seq_len": args.seq_len,
        "mode": args.mode,
        "warmup": args.warmup,
        "iters": args.iters,
        "token_prune_num": args.token_prune_num,
        "head_prune_num": args.head_prune_num,
        "enable_head_prune": args.enable_head_prune,
        "enable_token_prune": args.enable_token_prune,
        "baseline": baseline,
        "results": results,
        "best": best,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Route-B Model Threshold Sweep (Paper BF16-MSB)")
    print(f"Device: {payload['device']}")
    print(f"Model source: {model_source}")
    print(f"Input source: {input_source}")
    print(f"Mode: {args.mode}")
    print(f"Seq Len: {args.seq_len}")
    print(f"Head prune enabled: {args.enable_head_prune} (num={args.head_prune_num})")
    print(f"Token prune enabled: {args.enable_token_prune} (num={args.token_prune_num})")
    print(f"Baseline: {baseline_ms:.4f} ms")
    print("Top candidates:")
    for idx, item in enumerate(results[:10], start=1):
        print(
            f"{idx:02d}. quant={item['quant_threshold']:.4f} | "
            f"v={item['v_threshold']:.4f} | "
            f"ms={item['ms']:.4f} | "
            f"speedup_vs_baseline={item['speedup_vs_baseline']:.3f}x"
        )
    print(
        f"Recommendation: Promote quant_threshold={best['quant_threshold']} and "
        f"v_threshold={best['v_threshold']} into the next sequence-length ablation pass."
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
