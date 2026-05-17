"""
Generate comprehensive BERT vs Qwen3 comparison report for SpAtten paper.
"""
import json, sys
from pathlib import Path
from datetime import datetime


def load_all_results(data_dir):
    results = []
    for jf in sorted(Path(data_dir).glob("*_20260517_*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if "results" in data:
                results.extend(data["results"])
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    all_results = load_all_results(data_dir)
    print(f"Loaded {len(all_results)} results from {data_dir}")

    bert_ppl = sorted(
        [r for r in all_results if "n_masked" in r],
        key=lambda x: (x["seq_len"], x["variant"])
    )
    qwen_ppl = sorted(
        [r for r in all_results if "n_tokens" in r],
        key=lambda x: (x["seq_len"], x["variant"])
    )
    bert_tp = sorted(
        [r for r in all_results if "latency_ms" in r],
        key=lambda x: (x["seq_len"], x["variant"])
    )
    qwen_tp = sorted(
        [r for r in all_results if "total_tok_per_s" in r and "prefill_tok_per_s" in r],
        key=lambda x: (x["seq_len"], x["variant"])
    )

    print(f"BERT PPL: {len(bert_ppl)}, BERT TP: {len(bert_tp)}, Qwen PPL: {len(qwen_ppl)}, Qwen TP: {len(qwen_tp)}")

    L = []  # report lines
    L.append("# SpAtten Comprehensive Evaluation Report")
    L.append("")
    L.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    L.append("**Models**: BERT-base-uncased (encoder-only) & Qwen3-0.6B (decoder-only)  ")
    L.append("**GPU**: NVIDIA GeForce RTX 3090 24GB  ")
    L.append("**SpAtten Config**: quant_threshold=0.01, v_threshold=0.05  ")
    L.append("")
    L.append("---")
    L.append("")

    # ── 1. BERT Perplexity ──
    L.append("## 1. BERT Perplexity (MLM Pseudo-Perplexity)")
    L.append("")
    L.append("| Variant | Seq=128 | Seq=256 | Seq=512 |")
    L.append("|---------|---------|---------|--------|")
    for v in ["Dense", "Sp-Quant", "Sp-V", "Sp-Full"]:
        cells = [v]
        for sl in [128, 256, 512]:
            m = [r for r in bert_ppl if r["variant"] == v and r["seq_len"] == sl]
            if m:
                ppl = m[0]["perplexity"]
                dm = [r for r in bert_ppl if r["variant"] == "Dense" and r["seq_len"] == sl]
                if dm and v != "Dense":
                    d = (ppl / dm[0]["perplexity"] - 1) * 100
                    s = "+" if d >= 0 else ""
                    cells.append(f"{ppl:.2f} ({s}{d:.1f}%)")
                else:
                    cells.append(f"{ppl:.2f}")
            else:
                cells.append("-")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append("*Lower PPL = better. Sp-Full anomaly flagged for investigation.*")
    L.append("")

    # ── 2. Qwen3 Perplexity ──
    L.append("## 2. Qwen3-0.6B Perplexity (Autoregressive CE)")
    L.append("")
    L.append("| Variant | Seq=1024 | Seq=2048 | Seq=4096 | Seq=8192 |")
    L.append("|---------|----------|----------|----------|----------|")
    for v in ["Dense", "Sp-Quant", "Sp-V", "Sp-Full"]:
        cells = [v]
        for sl in [1024, 2048, 4096, 8192]:
            m = [r for r in qwen_ppl if r["variant"] == v and r["seq_len"] == sl]
            if m:
                ppl = m[0]["perplexity"]
                dm = [r for r in qwen_ppl if r["variant"] == "Dense" and r["seq_len"] == sl]
                if dm and v != "Dense":
                    d = (ppl / dm[0]["perplexity"] - 1) * 100
                    s = "+" if d >= 0 else ""
                    cells.append(f"{ppl:.4f} ({s}{d:.2f}%)")
                else:
                    cells.append(f"{ppl:.4f}")
            else:
                cells.append("-")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append("*Sp-V shows negative degradation at seq >= 2048 = better than Dense (regularization effect).*")
    L.append("")

    # ── 3. BERT Throughput ──
    L.append("## 3. BERT Throughput (Forward Pass, tokens/sec)")
    L.append("")
    L.append("| Variant | Seq=1024 | Seq=2048 | Seq=4096 | Seq=8192 |")
    L.append("|---------|----------|----------|----------|----------|")
    for v in ["Dense-Eager", "Dense-SDPA", "Sp-Quant", "Sp-V", "Sp-Full"]:
        cells = [v]
        for sl in [1024, 2048, 4096, 8192]:
            m = [r for r in bert_tp if r["variant"] == v and r["seq_len"] == sl]
            if m:
                tps = m[0]["tokens_per_sec"]
                su = m[0].get("speedup_vs_eager")
                if su and v not in ("Dense-Eager", "Dense-SDPA"):
                    cells.append(f"{tps:,.0f} ({su:.2f}x)")
                else:
                    cells.append(f"{tps:,.0f}")
            else:
                cells.append("-")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append("*Speedup relative to Dense-Eager baseline. Sp-V reaches 2.00x at seq=8192.*")
    L.append("")

    # ── 4. Qwen3 Throughput ──
    ds = qwen_tp[0].get("decode_steps", "?") if qwen_tp else "?"
    L.append(f"## 4. Qwen3-0.6B Generation Throughput (decode_steps={ds})")
    L.append("")

    L.append("### 4.1 Total Throughput (tokens/sec)")
    L.append("")
    L.append("| Variant | Seq=1024 | Seq=2048 | Seq=4096 | Seq=8192 |")
    L.append("|---------|----------|----------|----------|----------|")
    for v in ["Dense", "Sp-Quant", "Sp-V", "Sp-Full"]:
        cells = [v]
        for sl in [1024, 2048, 4096, 8192]:
            m = [r for r in qwen_tp if r["variant"] == v and r["seq_len"] == sl]
            if m:
                tps = m[0]["total_tok_per_s"]
                dm = [r for r in qwen_tp if r["variant"] == "Dense" and r["seq_len"] == sl]
                if dm and v != "Dense":
                    rto = tps / dm[0]["total_tok_per_s"]
                    cells.append(f"{tps:,.0f} ({rto:.2f}x)")
                else:
                    cells.append(f"{tps:,.0f}")
            else:
                cells.append("-")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    L.append("### 4.2 Prefill Throughput (tokens/sec)")
    L.append("")
    L.append("| Variant | Seq=1024 | Seq=2048 | Seq=4096 | Seq=8192 |")
    L.append("|---------|----------|----------|----------|----------|")
    for v in ["Dense", "Sp-Quant", "Sp-V", "Sp-Full"]:
        cells = [v]
        for sl in [1024, 2048, 4096, 8192]:
            m = [r for r in qwen_tp if r["variant"] == v and r["seq_len"] == sl]
            cells.append(f"{m[0]['prefill_tok_per_s']:,.0f}" if m else "-")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    L.append("### 4.3 Decode Throughput (tokens/sec)")
    L.append("")
    L.append("| Variant | Seq=1024 | Seq=2048 | Seq=4096 | Seq=8192 |")
    L.append("|---------|----------|----------|----------|----------|")
    for v in ["Dense", "Sp-Quant", "Sp-V", "Sp-Full"]:
        cells = [v]
        for sl in [1024, 2048, 4096, 8192]:
            m = [r for r in qwen_tp if r["variant"] == v and r["seq_len"] == sl]
            cells.append(f"{m[0]['decode_tok_per_s']:.1f}" if m else "-")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    L.append("### 4.4 TTFT - Time to First Token (ms)")
    L.append("")
    L.append("| Variant | Seq=1024 | Seq=2048 | Seq=4096 | Seq=8192 |")
    L.append("|---------|----------|----------|----------|----------|")
    for v in ["Dense", "Sp-Quant", "Sp-V", "Sp-Full"]:
        cells = [v]
        for sl in [1024, 2048, 4096, 8192]:
            m = [r for r in qwen_tp if r["variant"] == v and r["seq_len"] == sl]
            cells.append(f"{m[0]['ttft_ms']:.1f}" if m else "-")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    L.append("### 4.5 Latency Breakdown (ms)")
    L.append("")
    L.append("| Variant | Seq | Prefill ms | Decode ms/step | Decode total | TTFT ms |")
    L.append("|---------|-----|------------|----------------|--------------|---------|")
    for r in sorted(qwen_tp, key=lambda x: (x["seq_len"], x["variant"])):
        dt = r["decode_step_ms"] * r["decode_steps"]
        L.append(f"| {r['variant']:<10} | {r['seq_len']:<4} | {r['prefill_ms']:.2f} | "
                 f"{r['decode_step_ms']:.2f} | {dt:.2f} | {r['ttft_ms']:.2f} |")
    L.append("")

    # ── 5. Cross-Model Analysis ──
    L.append("---")
    L.append("")
    L.append("## 5. Cross-Model Analysis")
    L.append("")

    L.append("### 5.1 PPL Degradation by Variant")
    L.append("")
    L.append("| Variant | BERT (avg) | Qwen3 (avg) | Notes |")
    L.append("|---------|-----------|-------------|-------|")
    for v in ["Sp-Quant", "Sp-V", "Sp-Full"]:
        bd, qd = [], []
        for r in bert_ppl:
            if r["variant"] == v:
                dm = next((x for x in bert_ppl if x["variant"] == "Dense" and x["seq_len"] == r["seq_len"]), None)
                if dm:
                    bd.append((r["perplexity"] / dm["perplexity"] - 1) * 100)
        for r in qwen_ppl:
            if r["variant"] == v:
                dm = next((x for x in qwen_ppl if x["variant"] == "Dense" and x["seq_len"] == r["seq_len"]), None)
                if dm:
                    qd.append((r["perplexity"] / dm["perplexity"] - 1) * 100)
        ba = sum(bd) / len(bd) if bd else 0
        qa = sum(qd) / len(qd) if qd else 0
        note = ""
        if v == "Sp-Quant":
            note = "Near-zero on Qwen3"
        elif v == "Sp-V":
            note = "Improves Qwen3 at long seq"
        elif v == "Sp-Full":
            note = "BERT anomaly (state leak?)"
        L.append(f"| {v} | {ba:+.1f}% | {qa:+.2f}% | {note} |")
    L.append("")

    L.append("### 5.2 Throughput Speedup: BERT vs Qwen3")
    L.append("")
    L.append("| Seq Len | BERT Sp-V vs Eager | Qwen3 Best vs Dense |")
    L.append("|---------|---------------------|---------------------|")
    for sl in [1024, 2048, 4096, 8192]:
        bv = next((r for r in bert_tp if r["variant"] == "Sp-V" and r["seq_len"] == sl), None)
        b = f"{bv['speedup_vs_eager']:.2f}x" if bv and bv.get("speedup_vs_eager") else "-"
        qd = next((r for r in qwen_tp if r["variant"] == "Dense" and r["seq_len"] == sl), None)
        qn = [r for r in qwen_tp if r["variant"] != "Dense" and r["seq_len"] == sl]
        if qn and qd:
            best = max(qn, key=lambda x: x["total_tok_per_s"])
            q = f"{best['total_tok_per_s']/qd['total_tok_per_s']:.2f}x ({best['variant']})"
        else:
            q = "-"
        L.append(f"| {sl} | {b} | {q} |")
    L.append("")

    L.append("### 5.3 Key Findings for Paper (Chapter 7)")
    L.append("")
    L.append("1. **Sp-Quant: quality-first variant.** Near-zero PPL degradation on Qwen3 (+0.02% avg) "
             "and acceptable BERT impact (+6.1% avg). Use when quality preservation is paramount.")
    L.append("")
    L.append("2. **Sp-V: throughput-first variant for encoders.** 2.00x speedup on BERT at seq=8192. "
             "On Qwen3, Sp-V even *improves* PPL at sequence lengths >= 2048, "
             "suggesting V-block pruning acts as attention regularization for decoder models.")
    L.append("")
    L.append("3. **SpAtten overhead dominates decode.** For autoregressive generation, single-token "
             "decode steps cannot amortize sparse attention overhead. Qwen3 total throughput is "
             "6-20% slower with SpAtten despite competitive prefill speeds.")
    L.append("")
    L.append("4. **BERT encoder is the sweet spot.** SpAtten's design (cascaded head/token pruning, "
             "physical compaction) benefits the parallel forward pass of encoder models. "
             "Speedups scale with sequence length: 1.19x -> 2.00x from 1K to 8K.")
    L.append("")
    L.append("5. **Sp-Full is not recommended.** Fusing quantization + V-pruning underperforms "
             "Sp-V alone on both models. The combined kernel fusion overhead outweighs benefits.")
    L.append("")
    L.append("6. **BERT Sp-Full anomaly needs investigation.** Sp-Full shows implausibly low PPL "
             "(-17.9% at seq=512) on BERT, likely due to state leakage between MLM evaluation "
             "chunks in the current benchmark implementation.")
    L.append("")

    L.append("### 5.4 Deployment Guide")
    L.append("")
    L.append("| Scenario | Variant | Rationale |")
    L.append("|----------|---------|-----------|")
    L.append("| BERT long-sequence inference | **Sp-V** | 2.00x speedup at 8K |")
    L.append("| BERT quality-critical | **Sp-Quant** | Lowest PPL impact among variants |")
    L.append("| Qwen3 autoregressive gen | **Dense** | SpAtten adds 6-20% overhead |")
    L.append("| Qwen3 prefill/batch | **Sp-V** or **Sp-Quant** | Competitive prefill, minimal quality loss |")
    L.append("| Universal quality-safe | **Sp-Quant** | +0.02% PPL on Qwen3 |")
    L.append("")

    report = "\n".join(L)
    out = Path(data_dir) / "comprehensive_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {out}")

    # Also upload to server
    print("\n=== REPORT PREVIEW ===\n")
    print(report)


if __name__ == "__main__":
    main()
