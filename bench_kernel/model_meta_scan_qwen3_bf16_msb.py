import argparse
import json
from datetime import datetime
from itertools import product
from pathlib import Path

import torch

from benchmark.benchmark_bf16_msb import benchmark_model
from spattn.spatten_qwen3_bf16_msb import (
    TRITON_META_DEFAULTS,
    build_inputs_local_or_synthetic,
    configure_spatten_qwen3_model,
    load_qwen3_model_local_or_synthetic,
)


def parse_int_list(raw_value):
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def build_meta_candidates(block_ms, block_ns, num_warps, num_stages):
    return [
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "num_warps": warps,
            "num_stages": stages,
        }
        for block_m, block_n, warps, stages in product(block_ms, block_ns, num_warps, num_stages)
    ]


def build_meta_overrides(mode, meta):
    if mode == "quant_only":
        return {"progressive_qk": meta}
    if mode == "v_prune_only":
        return {"v_prune": meta}
    if mode == "full":
        return {"ultimate": meta}
    raise ValueError(f"Unsupported mode: {mode}")


def summarize_recommendation(results, default_meta, mode):
    best = results[0]
    if best["meta"] == default_meta:
        return f"Default {mode} launch meta remains strongest; next step should validate this config on a longer Qwen3 context."

    spread = ((results[-1]["ms"] - best["ms"]) / results[-1]["ms"]) * 100.0 if len(results) > 1 else 0.0
    return (
        f"Best {mode} candidate differs from current default. Promote {best['meta']} "
        f"into the next Qwen3 ablation pass; worst-to-best spread is {spread:.2f}%."
    )


def main():
    parser = argparse.ArgumentParser(description="Qwen3 BF16-MSB model-level Triton meta scan.")
    parser.add_argument("--mode", choices=["quant_only", "v_prune_only", "full"], default="full")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--quant-threshold", type=float, default=0.01)
    parser.add_argument("--v-threshold", type=float, default=0.05)
    parser.add_argument("--block-m-list", default="32,64,128")
    parser.add_argument("--block-n-list", default="32,64,128")
    parser.add_argument("--num-warps-list", default="2,4,8")
    parser.add_argument("--num-stages-list", default="1,2,3")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--allow-local-pretrained", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/route_b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for model_meta_scan_qwen3_bf16_msb.py")

    seq_len = args.seq_len
    base_model, model_source = load_qwen3_model_local_or_synthetic(
        device,
        model_name=args.model_name,
        max_position_embeddings=max(4096, seq_len + 2),
        prefer_synthetic=not args.allow_local_pretrained,
    )
    inputs, input_source = build_inputs_local_or_synthetic(
        device,
        base_model,
        model_name=args.model_name,
        seq_len=seq_len,
        exact_seq_len=True,
    )

    default_meta = dict(
        TRITON_META_DEFAULTS[
            "progressive_qk" if args.mode == "quant_only" else "v_prune" if args.mode == "v_prune_only" else "ultimate"
        ]
    )
    candidates = build_meta_candidates(
        parse_int_list(args.block_m_list),
        parse_int_list(args.block_n_list),
        parse_int_list(args.num_warps_list),
        parse_int_list(args.num_stages_list),
    )

    baseline_ms = benchmark_model(base_model, inputs, num_iters=args.iters, warmup=args.warmup, reset_state=False)
    valid_results = []
    skipped_results = []
    for meta in candidates:
        try:
            model = configure_spatten_qwen3_model(
                base_model,
                args.mode,
                quant_threshold=args.quant_threshold,
                v_threshold=args.v_threshold,
                meta_overrides=build_meta_overrides(args.mode, meta),
            )
            ms = benchmark_model(model, inputs, num_iters=args.iters, warmup=args.warmup, reset_state=False)
            valid_results.append(
                {
                    "meta": meta,
                    "ms": round(float(ms), 6),
                    "baseline_ms": round(float(baseline_ms), 6),
                    "slowdown_vs_baseline": round(float(ms / baseline_ms), 6),
                    "delta_vs_default_meta_ms": None,
                }
            )
        except Exception as exc:
            skipped_results.append({"meta": meta, "reason": f"{type(exc).__name__}: {exc}"})

    if not valid_results:
        raise RuntimeError("No valid Qwen3 Triton meta combinations completed successfully.")

    valid_results.sort(key=lambda item: item["ms"])
    default_result = next((item for item in valid_results if item["meta"] == default_meta), None)
    default_ms = default_result["ms"] if default_result else None
    for item in valid_results:
        if default_ms is not None:
            item["delta_vs_default_meta_ms"] = round(item["ms"] - default_ms, 6)

    recommendation = summarize_recommendation(valid_results, default_meta, args.mode)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"model_meta_scan_qwen3_{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0),
        "model_source": model_source,
        "input_source": input_source,
        "mode": args.mode,
        "seq_len": seq_len,
        "warmup": args.warmup,
        "iters": args.iters,
        "quant_threshold": args.quant_threshold,
        "v_threshold": args.v_threshold,
        "default_meta": default_meta,
        "recommendation": recommendation,
        "valid_results": valid_results,
        "skipped_results": skipped_results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Qwen3 Model Meta Scan (BF16-MSB)")
    print(f"Device: {payload['device']}")
    print(f"Model source: {model_source}")
    print(f"Input source: {input_source}")
    print(f"Mode: {args.mode}")
    print(f"Seq Len: {seq_len}")
    print(f"Baseline: {baseline_ms:.4f} ms")
    print("Top candidates:")
    for index, item in enumerate(valid_results[:10], start=1):
        print(
            f"{index:02d}. meta={item['meta']} | ms={item['ms']:.4f} "
            f"| slowdown_vs_baseline={item['slowdown_vs_baseline']:.3f}x "
            f"| delta_vs_default={item['delta_vs_default_meta_ms']}"
        )
    if skipped_results:
        print(f"Skipped configs: {len(skipped_results)}")
        for item in skipped_results[:10]:
            print(f"skip meta={item['meta']} | reason={item['reason']}")
    print(f"Recommendation: {recommendation}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
