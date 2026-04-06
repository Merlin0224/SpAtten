import argparse
import json
import math
from datetime import datetime
from itertools import product
from pathlib import Path

import torch
import triton

from spatten_bert_ultimate import TRITON_META_DEFAULTS, triton_fused_spatten_ultimate

try:
    from triton.runtime.errors import OutOfResources
except ImportError:  # pragma: no cover
    OutOfResources = RuntimeError

def parse_int_list(raw_value):
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]

def build_meta_candidates(block_ms, block_ns, num_warps, num_stages):
    return [
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "num_warps": warps,
            "num_stages": stages
        }
        for block_m, block_n, warps, stages in product(block_ms, block_ns, num_warps, num_stages)
    ]


def allocate_inputs(batch, heads, seq_q, seq_k, head_dim, dtype, device):
    q = torch.randn((batch, heads, seq_q, head_dim), device=device, dtype=dtype)
    k_msb = torch.randn((batch, heads, seq_k, head_dim), device=device, dtype=dtype)
    k_lsb = torch.randn((batch, heads, seq_k, head_dim), device=device, dtype=dtype)
    v = torch.randn((batch, heads, seq_k, head_dim), device=device, dtype=dtype)
    out_sum = torch.zeros((batch, heads, seq_q), device=device, dtype=torch.float32)
    return q, k_msb, k_lsb, v, out_sum

def benchmark_sdpa(q, k, v):
    return triton.testing.do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))

def benchmark_ultimate(q, k_msb, k_lsb, v, out_sum, quant_threshold, v_threshold, sm_scale, meta):
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
            meta
        )
    )

def summarize_recommendation(results, default_meta):
    best = results[0]
    if best["meta"] == default_meta:
        return "Default launch meta remains the strongest candidate; next step should be ablation under this config."

    spread = ((results[-1]["triton_ms"] - best["triton_ms"]) / results[-1]["triton_ms"]) * 100.0
    return (
        "Best candidate differs from the current default, which points to launch-meta mismatch. "
        f"Promote {best['meta']} into the next ablation pass; worst-to-best spread is {spread:.2f}%."
    )

def main():
    parser = argparse.ArgumentParser(description="Route-B meta scan for the SpAtten fused Triton kernel.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--seq-q", type=int, default=4096)
    parser.add_argument("--seq-k", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--quant-threshold", type=float, default=0.05)
    parser.add_argument("--v-threshold", type=float, default=0.01)
    parser.add_argument("--block-m-list", default="32,64,128")
    parser.add_argument("--block-n-list", default="32,64,128")
    parser.add_argument("--num-warps-list", default="4,8")
    parser.add_argument("--num-stages-list", default="1,2,3,4")
    parser.add_argument("--output-dir", default="artifacts/route_b")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Route-B meta scan.")

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
        device=device
    )

    sm_scale = 1.0 / math.sqrt(args.head_dim)

    default_meta = dict(TRITON_META_DEFAULTS["ultimate"])
    candidates = build_meta_candidates(
        parse_int_list(args.block_m_list),
        parse_int_list(args.block_n_list),
        parse_int_list(args.num_warps_list),
        parse_int_list(args.num_stages_list)
    )

    sdpa_ms = benchmark_sdpa(q, k_msb, v)
    valid_results = []
    skipped_results = []
    for meta in candidates:
        try:
            torch.cuda.empty_cache()
            triton_ms = benchmark_ultimate(
                q,
                k_msb,
                k_lsb,
                v,
                out_sum,
                args.quant_threshold,
                args.v_threshold,
                sm_scale,
                meta,
            )
            valid_results.append(
                {
                    "meta": meta,
                    "triton_ms": round(float(triton_ms), 6),
                    "sdpa_ms": round(float(sdpa_ms), 6),
                    "slowdown_vs_sdpa": round(float(triton_ms / sdpa_ms), 6),
                    "delta_vs_default_meta_ms": None,
                }
            )
        except (OutOfResources, RuntimeError) as exc:
            skipped_results.append(
                {
                    "meta": meta,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    if not valid_results:
        raise RuntimeError("No valid Triton meta combinations completed successfully.")

    valid_results.sort(key=lambda item: item["triton_ms"])
    default_result = next((item for item in valid_results if item["meta"] == default_meta), None)
    default_ms = default_result["triton_ms"] if default_result else None
    for item in valid_results:
        if default_ms is not None:
            item["delta_vs_default_meta_ms"] = round(item["triton_ms"] - default_ms, 6)

    recommendation = summarize_recommendation(valid_results, default_meta)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"meta_scan_{timestamp}.json"

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
        "quant_threshold": args.quant_threshold,
        "v_threshold": args.v_threshold,
        "default_meta": default_meta,
        "recommendation": recommendation,
        "valid_results": valid_results,
        "skipped_results": skipped_results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Route-B Meta Scan")
    print(f"Device: {payload['device']}")
    print(f"Shape: B={args.batch}, H={args.heads}, M={args.seq_q}, N={args.seq_k}, D={args.head_dim}")
    print(f"SDPA baseline: {sdpa_ms:.4f} ms")
    print("Top candidates:")
    for index, item in enumerate(valid_results[:10], start=1):
        print(
            f"{index:02d}. meta={item['meta']} | triton={item['triton_ms']:.4f} ms "
            f"| slowdown={item['slowdown_vs_sdpa']:.3f}x | delta_vs_default={item['delta_vs_default_meta_ms']}"
        )
    if skipped_results:
        print(f"Skipped configs: {len(skipped_results)}")
        for item in skipped_results[:10]:
            print(f"skip meta={item['meta']} | reason={item['reason']}")
    print(f"Recommendation: {recommendation}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
