import argparse
import json
import math
from datetime import datetime
from itertools import product
from pathlib import Path

import torch
import triton

from spattn.spatten_bert_ultimate import TRITON_META_DEFAULTS, triton_fused_spatten_ultimate


def parse_float_list(raw_value):
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def allocate_inputs(batch, heads, seq_q, seq_k, head_dim, dtype, device):
    q = torch.randn((batch, heads, seq_q, head_dim), device=device, dtype=dtype)
    k_msb = torch.randn((batch, heads, seq_k, head_dim), device=device, dtype=dtype)
    k_lsb = torch.randn((batch, heads, seq_k, head_dim), device=device, dtype=dtype)
    v = torch.randn((batch, heads, seq_k, head_dim), device=device, dtype=dtype)
    out_sum = torch.zeros((batch, heads, seq_q), device=device, dtype=torch.float32)
    return q, k_msb, k_lsb, v, out_sum


def bench_sdpa(q, k_msb, v):
    return triton.testing.do_bench(
        lambda: torch.nn.functional.scaled_dot_product_attention(q, k_msb, v)
    )


def bench_full(q, k_msb, k_lsb, v, out_sum, quant_threshold, v_threshold, sm_scale, meta):
    return triton.testing.do_bench(
        lambda: triton_fused_spatten_ultimate(
            q,
            k_msb,
            k_lsb,
            v,
            out_sum,
            quant_threshold,
            v_threshold,
            sm_scale,
            meta=meta,
            collect_stats=False,
        )
    )


def summarize_recommendation(results, default_quant, default_v):
    best = results[0]
    if best["quant_threshold"] == default_quant and best["v_threshold"] == default_v:
        return "Current thresholds remain the strongest candidate; next step should be kernel-structure simplification."

    return (
        "Best thresholds differ from the current defaults, which means branch selectivity still matters. "
        f"Promote quant_threshold={best['quant_threshold']} and v_threshold={best['v_threshold']} into the next benchmark pass."
    )


def best_by_key(results, key_name):
    best = {}
    for item in results:
        key = item[key_name]
        current = best.get(key)
        if current is None or item["triton_ms"] < current["triton_ms"]:
            best[key] = item
    return [
        {
            key_name: key,
            "best_partner_threshold": value["v_threshold"] if key_name == "quant_threshold" else value["quant_threshold"],
            "triton_ms": value["triton_ms"],
            "slowdown_vs_attention_only": value["slowdown_vs_attention_only"],
        }
        for key, value in sorted(best.items(), key=lambda pair: pair[0])
    ]


def main():
    parser = argparse.ArgumentParser(description="Route-B threshold scan for the fused SpAtten Triton kernel.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--seq-q", type=int, default=4096)
    parser.add_argument("--seq-k", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--quant-threshold-list", default="0.0,0.01,0.03,0.05,0.08,0.1")
    parser.add_argument("--v-threshold-list", default="0.0,0.005,0.01,0.02,0.05")
    parser.add_argument("--default-quant-threshold", type=float, default=0.05)
    parser.add_argument("--default-v-threshold", type=float, default=0.01)
    parser.add_argument("--output-dir", default="artifacts/route_b")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Route-B threshold scan.")

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    torch.manual_seed(0)

    q, k_msb, k_lsb, v, out_sum = allocate_inputs(
        batch=args.batch,
        heads=args.heads,
        seq_q=args.seq_q,
        seq_k=args.seq_k,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
    )
    sm_scale = 1.0 / math.sqrt(args.head_dim)
    meta = dict(TRITON_META_DEFAULTS["ultimate"])

    attention_only_ms = float(bench_sdpa(q, k_msb, v))
    results = []
    for quant_threshold, v_threshold in product(
        parse_float_list(args.quant_threshold_list),
        parse_float_list(args.v_threshold_list),
    ):
        torch.cuda.empty_cache()
        triton_ms = float(
            bench_full(
                q,
                k_msb,
                k_lsb,
                v,
                out_sum,
                quant_threshold,
                v_threshold,
                sm_scale,
                meta,
            )
        )
        results.append(
            {
                "quant_threshold": quant_threshold,
                "v_threshold": v_threshold,
                "triton_ms": round(triton_ms, 6),
                "attention_only_ms": round(attention_only_ms, 6),
                "slowdown_vs_attention_only": round(triton_ms / attention_only_ms, 6),
                "delta_vs_default_threshold_ms": None,
            }
        )

    results.sort(key=lambda item: item["triton_ms"])
    default_result = next(
        (
            item
            for item in results
            if item["quant_threshold"] == args.default_quant_threshold
            and item["v_threshold"] == args.default_v_threshold
        ),
        None,
    )
    default_ms = default_result["triton_ms"] if default_result else None
    for item in results:
        if default_ms is not None:
            item["delta_vs_default_threshold_ms"] = round(item["triton_ms"] - default_ms, 6)

    recommendation = summarize_recommendation(
        results,
        args.default_quant_threshold,
        args.default_v_threshold,
    )
    quant_sweep_summary = best_by_key(results, "quant_threshold")
    v_sweep_summary = best_by_key(results, "v_threshold")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"threshold_scan_{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0),
        "shape": {
            "batch": args.batch,
            "heads": args.heads,
            "seq_q": args.seq_q,
            "seq_k": args.seq_k,
            "head_dim": args.head_dim,
        },
        "dtype": args.dtype,
        "meta": meta,
        "default_quant_threshold": args.default_quant_threshold,
        "default_v_threshold": args.default_v_threshold,
        "recommendation": recommendation,
        "quant_sweep_summary": quant_sweep_summary,
        "v_sweep_summary": v_sweep_summary,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Threshold Scan")
    print(f"Device: {payload['device']}")
    print(f"Shape: B={args.batch}, H={args.heads}, M={args.seq_q}, N={args.seq_k}, D={args.head_dim}")
    print(f"Attention-only baseline: {attention_only_ms:.4f} ms")
    print(f"Meta: {meta}")
    print("Top candidates:")
    for index, item in enumerate(results[:10], start=1):
        print(
            f"{index:02d}. quant={item['quant_threshold']:.4f} | v={item['v_threshold']:.4f} "
            f"| triton={item['triton_ms']:.4f} ms | slowdown={item['slowdown_vs_attention_only']:.3f}x "
            f"| delta_vs_default={item['delta_vs_default_threshold_ms']}"
        )
    print(f"Recommendation: {recommendation}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
