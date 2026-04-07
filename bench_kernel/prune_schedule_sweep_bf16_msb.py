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


def parse_int_list(raw_value):
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="BF16-MSB pruning schedule sweep.")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--mode", choices=["quant_only", "v_prune_only", "full"], default="full")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--token-prune-num", type=int, default=1)
    parser.add_argument("--head-prune-num", type=int, default=1)
    parser.add_argument("--quant-threshold", type=float, default=0.0)
    parser.add_argument("--v-threshold", type=float, default=0.05)
    parser.add_argument("--head-prune-start-layers", default="0,2,4,6")
    parser.add_argument("--token-prune-start-layers", default="0,2,4,6")
    parser.add_argument("--head-prune-intervals", default="1,2")
    parser.add_argument("--token-prune-intervals", default="1,2")
    parser.add_argument("--enable-head-prune", action="store_true")
    parser.add_argument("--enable-token-prune", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/route_b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for route_b_prune_schedule_sweep_paper_bf16_msb.py")

    head_start_layers = parse_int_list(args.head_prune_start_layers)
    token_start_layers = parse_int_list(args.token_prune_start_layers)
    head_intervals = parse_int_list(args.head_prune_intervals)
    token_intervals = parse_int_list(args.token_prune_intervals)

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
        quant_threshold=args.quant_threshold,
        v_threshold=args.v_threshold,
        enable_head_prune=False,
        enable_token_prune=False,
    )
    baseline_ms = baseline["ms"]

    results = []
    for head_start in head_start_layers:
        for token_start in token_start_layers:
            for head_interval in head_intervals:
                for token_interval in token_intervals:
                    item = run_variant(
                        base_model,
                        inputs,
                        args.mode,
                        args.token_prune_num,
                        args.head_prune_num,
                        args.warmup,
                        args.iters,
                        quant_threshold=args.quant_threshold,
                        v_threshold=args.v_threshold,
                        head_prune_start_layer=head_start,
                        token_prune_start_layer=token_start,
                        head_prune_interval=head_interval,
                        token_prune_interval=token_interval,
                        enable_head_prune=args.enable_head_prune,
                        enable_token_prune=args.enable_token_prune,
                    )
                    item["head_prune_start_layer"] = head_start
                    item["token_prune_start_layer"] = token_start
                    item["head_prune_interval"] = head_interval
                    item["token_prune_interval"] = token_interval
                    item["speedup_vs_baseline"] = round(baseline_ms / item["ms"], 6)
                    results.append(item)

    results.sort(key=lambda item: item["ms"])
    best = results[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"prune_schedule_sweep_{timestamp}.json"
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
        "quant_threshold": args.quant_threshold,
        "v_threshold": args.v_threshold,
        "enable_head_prune": args.enable_head_prune,
        "enable_token_prune": args.enable_token_prune,
        "baseline": baseline,
        "results": results,
        "best": best,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Route-B Prune Schedule Sweep (Paper BF16-MSB)")
    print(f"Device: {payload['device']}")
    print(f"Mode: {args.mode}")
    print(f"Seq Len: {args.seq_len}")
    print(f"Baseline: {baseline_ms:.4f} ms")
    print("Top candidates:")
    for idx, item in enumerate(results[:10], start=1):
        print(
            f"{idx:02d}. head_start={item['head_prune_start_layer']} | "
            f"token_start={item['token_prune_start_layer']} | "
            f"head_interval={item['head_prune_interval']} | "
            f"token_interval={item['token_prune_interval']} | "
            f"ms={item['ms']:.4f} | speedup_vs_baseline={item['speedup_vs_baseline']:.3f}x"
        )
    print(
        f"Recommendation: Promote head_start={best['head_prune_start_layer']}, "
        f"token_start={best['token_prune_start_layer']}, "
        f"head_interval={best['head_prune_interval']}, "
        f"token_interval={best['token_prune_interval']}."
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
