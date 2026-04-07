import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import triton

from spatten_bert_ultimate import (
    TRITON_META_DEFAULTS,
    triton_fused_spatten_ultimate,
    triton_fused_spatten_ultimate_block_skip,
    triton_fused_spatten_v_prune,
    triton_fused_spatten_v_prune_block_skip,
    triton_progressive_qk,
)


def allocate_inputs(batch, heads, seq_q, seq_k, head_dim, dtype, device):
    q = torch.randn((batch, heads, seq_q, head_dim), device=device, dtype=dtype)
    k_full = torch.randn((batch, heads, seq_k, head_dim), device=device, dtype=dtype)
    k_msb = k_full * 0.8
    k_lsb = k_full * 0.2
    v = torch.randn((batch, heads, seq_k, head_dim), device=device, dtype=dtype)
    out_sum = torch.zeros((batch, heads, seq_q), device=device, dtype=torch.float32)
    return q, k_full, k_msb, k_lsb, v, out_sum


def bench_sdpa(q, k_full, v):
    return triton.testing.do_bench(
        lambda: torch.nn.functional.scaled_dot_product_attention(q, k_full, v)
    )


def bench_quant_only(q, k_msb, k_lsb, v, quant_threshold, sm_scale, meta):
    return triton.testing.do_bench(
        lambda: triton_progressive_qk(
            q, k_msb, k_lsb, v, quant_threshold, sm_scale, meta=meta
        )
    )


def bench_v_prune_only(q, k_full, v, v_threshold, sm_scale, meta):
    return triton.testing.do_bench(
        lambda: triton_fused_spatten_v_prune(
            q, k_full, v, v_threshold, sm_scale, meta=meta
        )
    )


def bench_v_prune_block_only(q, k_full, v, v_threshold, sm_scale, meta):
    return triton.testing.do_bench(
        lambda: triton_fused_spatten_v_prune_block_skip(
            q, k_full, v, v_threshold, sm_scale, meta=meta
        )
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


def bench_full_block_only(q, k_msb, k_lsb, v, out_sum, quant_threshold, v_threshold, sm_scale, meta):
    return triton.testing.do_bench(
        lambda: triton_fused_spatten_ultimate_block_skip(
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


def make_recommendation(results):
    sdpa_ms = results["attention_only"]["ms"]
    quant_ms = results["quant_only"]["ms"]
    v_prune_ms = results["v_prune_only"]["ms"]
    v_prune_block_ms = results["v_prune_block_only"]["ms"]
    full_ms = results["full"]["ms"]
    full_block_ms = results["full_block_only"]["ms"]

    triton_variants = {
        "quant_only": quant_ms,
        "v_prune_only": v_prune_ms,
        "v_prune_block_only": v_prune_block_ms,
        "full": full_ms,
        "full_block_only": full_block_ms,
    }
    best_variant = min(triton_variants, key=triton_variants.get)

    if full_block_ms < full_ms:
        return (
            "Block-skip-only fused kernel beats the current fine-grained fused path. "
            "Next step should be simplifying or removing the fine-grained V mask path from the hot kernel."
        )
    if v_prune_block_ms < v_prune_ms:
        return (
            "Block-skip-only V pruning beats the fine-grained V-prune kernel. "
            "This points directly to masked V loads and per-element pruning as the current drag."
        )
    if full_ms <= min(quant_ms, v_prune_ms, v_prune_block_ms, full_block_ms):
        return (
            "Full fused path is the best Triton variant under the chosen meta. "
            "Next step should be threshold sensitivity scanning around quant_threshold and v_threshold."
        )
    if best_variant == "quant_only":
        return (
            "Quant-only beats the fused kernel, which points to local V pruning as the current drag. "
            "Next step should isolate the V-prune branch structure and test a simpler block-level skip policy."
        )
    if best_variant == "v_prune_only":
        return (
            "V-prune-only beats the fused kernel, which points to progressive quantization or fusion overhead as the current drag. "
            "Next step should inspect the quant branch and LSB load path separately."
        )
    if full_ms > sdpa_ms * 1.5:
        return (
            "Even after tuning launch meta, the fused kernel is still far from SDPA. "
            "Next step should focus on structural simplification, not further launch tuning."
        )
    return "Results are mixed; next step should be threshold scanning with the current best meta."


def main():
    parser = argparse.ArgumentParser(description="Route-B ablation benchmark for SpAtten kernels.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--seq-q", type=int, default=4096)
    parser.add_argument("--seq-k", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--quant-threshold", type=float, default=0.05)
    parser.add_argument("--v-threshold", type=float, default=0.01)
    parser.add_argument("--output-dir", default="artifacts/route_b")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Route-B ablation benchmarking.")

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    torch.manual_seed(0)

    q, k_full, k_msb, k_lsb, v, out_sum = allocate_inputs(
        batch=args.batch,
        heads=args.heads,
        seq_q=args.seq_q,
        seq_k=args.seq_k,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
    )
    sm_scale = 1.0 / math.sqrt(args.head_dim)

    results = {
        "attention_only": {
            "ms": round(float(bench_sdpa(q, k_full, v)), 6),
            "meta": None,
        },
        "quant_only": {
            "ms": round(
                float(
                    bench_quant_only(
                        q,
                        k_msb,
                        k_lsb,
                        v,
                        args.quant_threshold,
                        sm_scale,
                        meta=dict(TRITON_META_DEFAULTS["progressive_qk"]),
                    )
                ),
                6,
            ),
            "meta": dict(TRITON_META_DEFAULTS["progressive_qk"]),
        },
        "v_prune_only": {
            "ms": round(
                float(
                    bench_v_prune_only(
                        q,
                        k_full,
                        v,
                        args.v_threshold,
                        sm_scale,
                        meta=dict(TRITON_META_DEFAULTS["v_prune"]),
                    )
                ),
                6,
            ),
            "meta": dict(TRITON_META_DEFAULTS["v_prune"]),
        },
        "v_prune_block_only": {
            "ms": round(
                float(
                    bench_v_prune_block_only(
                        q,
                        k_full,
                        v,
                        args.v_threshold,
                        sm_scale,
                        meta=dict(TRITON_META_DEFAULTS["v_prune"]),
                    )
                ),
                6,
            ),
            "meta": dict(TRITON_META_DEFAULTS["v_prune"]),
        },
        "full": {
            "ms": round(
                float(
                    bench_full(
                        q,
                        k_msb,
                        k_lsb,
                        v,
                        out_sum,
                        args.quant_threshold,
                        args.v_threshold,
                        sm_scale,
                        meta=dict(TRITON_META_DEFAULTS["ultimate"]),
                    )
                ),
                6,
            ),
            "meta": dict(TRITON_META_DEFAULTS["ultimate"]),
        },
        "full_block_only": {
            "ms": round(
                float(
                    bench_full_block_only(
                        q,
                        k_msb,
                        k_lsb,
                        v,
                        out_sum,
                        args.quant_threshold,
                        args.v_threshold,
                        sm_scale,
                        meta=dict(TRITON_META_DEFAULTS["ultimate"]),
                    )
                ),
                6,
            ),
            "meta": dict(TRITON_META_DEFAULTS["ultimate"]),
        },
    }

    baseline_ms = results["attention_only"]["ms"]
    for name, item in results.items():
        item["slowdown_vs_attention_only"] = round(item["ms"] / baseline_ms, 6)

    recommendation = make_recommendation(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ablation_{timestamp}.json"

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
        "results": results,
        "recommendation": recommendation,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Ablation")
    print(f"Device: {payload['device']}")
    print(f"Shape: B={args.batch}, H={args.heads}, M={args.seq_q}, N={args.seq_k}, D={args.head_dim}")
    for name in ["attention_only", "quant_only", "v_prune_only", "v_prune_block_only", "full", "full_block_only"]:
        item = results[name]
        print(
            f"{name}: ms={item['ms']:.4f} | slowdown_vs_attention_only={item['slowdown_vs_attention_only']:.3f}x "
            f"| meta={item['meta']}"
        )
    print(f"Recommendation: {recommendation}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
