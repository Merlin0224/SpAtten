"""
Throughput benchmark for SpAtten BERT variants.

Measures forward pass latency (ms) and throughput (tokens/sec) across:
  Dense-Eager, Dense-SDPA, Sp-Quant, Sp-V, Sp-Full

For encoder-only BERT, throughput = seq_len / latency * 1000 (batch=1).
"""
import argparse
import copy
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from spattn.spatten_bert_bf16_msb import (
    SpattenBertSelfAttention,
    build_inputs_local_or_synthetic,
    load_bert_model_local_or_synthetic,
    spatten_encoder_forward,
)
from transformers.models.bert.modeling_bert import BertEncoder
from benchmark.benchmark_bf16_msb import benchmark_model


def configure_spatten_bert(
    base_model,
    mode,
    quant_threshold=0.01,
    v_threshold=0.05,
    enable_head_prune=False,
    head_prune_num=1,
    head_prune_start_layer=0,
    head_prune_interval=1,
    enable_token_prune=False,
    token_prune_num=1,
    token_prune_interval=2,
):
    """Configure BERT with SpAtten attention variants."""
    spatten_model = copy.deepcopy(base_model)

    for layer_idx, layer in enumerate(spatten_model.encoder.layer):
        orig_state = layer.attention.self.state_dict()
        new_attn = SpattenBertSelfAttention(spatten_model.config)
        new_attn.load_state_dict(orig_state)
        new_attn.enable_prog_quant = mode in {"quant_only", "full"}
        new_attn.enable_v_prune = mode in {"v_prune_only", "full"}
        new_attn.quant_threshold = quant_threshold
        new_attn.v_threshold = v_threshold
        new_attn.enable_head_prune = enable_head_prune
        new_attn.head_prune_num = head_prune_num
        new_attn.head_prune_start_layer = head_prune_start_layer
        new_attn.head_prune_interval = head_prune_interval
        new_attn.enable_token_prune = enable_token_prune
        new_attn.token_prune_num = token_prune_num
        new_attn.token_prune_interval = token_prune_interval
        new_attn.layer_idx = layer_idx
        layer.attention.self = new_attn

    spatten_model.encoder.forward = spatten_encoder_forward.__get__(
        spatten_model.encoder, BertEncoder
    )
    return spatten_model


def parse_int_list(raw):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="BERT Throughput Benchmark")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--seq-lens", default="128,256,512,1024,2048,4096,8192")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--modes", default="dense_eager,dense_sdpa,quant_only,v_prune_only,full")
    parser.add_argument("--quant-threshold", type=float, default=0.01)
    parser.add_argument("--v-threshold", type=float, default=0.05)
    parser.add_argument("--enable-head-prune", action="store_true")
    parser.add_argument("--head-prune-num", type=int, default=1)
    parser.add_argument("--head-prune-interval", type=int, default=1)
    parser.add_argument("--enable-token-prune", action="store_true")
    parser.add_argument("--token-prune-num", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/bert_throughput")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_lens = parse_int_list(args.seq_lens)
    modes = [m.strip() for m in args.modes.split(",")]

    print("=" * 70)
    print("SpAtten BERT Throughput Benchmark")
    print(f"Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print(f"Seq lens: {seq_lens}")
    print(f"Modes: {modes}")
    print(f"Warmup: {args.warmup}, Iters: {args.iters}")
    print("=" * 70)

    all_results = []

    max_seq = max(seq_lens)

    # Dense-Eager baseline
    if "dense_eager" in modes:
        print("\n--- Dense-Eager baseline ---")
        base_model, model_source = load_bert_model_local_or_synthetic(
            device, max_position_embeddings=max_seq + 2
        )
        base_model = base_model.to(device).float().eval()

        for seq_len in seq_lens:
            inputs, input_source = build_inputs_local_or_synthetic(
                device, base_model, seq_len=seq_len, exact_seq_len=True
            )
            # Wrap inputs for benchmark_model which expects kwargs
            wrapped_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "token_type_ids": torch.zeros_like(inputs["input_ids"]),
            }
            ms = benchmark_model(base_model, wrapped_inputs, num_iters=args.iters,
                                warmup=args.warmup)
            tok_per_s = seq_len / ms * 1000
            result = {
                "variant": "Dense-Eager", "mode": "dense_eager", "seq_len": seq_len,
                "latency_ms": round(float(ms), 4),
                "tokens_per_sec": round(float(tok_per_s), 1),
            }
            all_results.append(result)
            print(f"  seq={seq_len} | {ms:.4f} ms | {tok_per_s:.1f} tok/s")

        del base_model
        gc.collect()
        torch.cuda.empty_cache()

    # Dense-SDPA baseline
    if "dense_sdpa" in modes:
        print("\n--- Dense-SDPA baseline ---")
        from transformers import BertConfig, BertModel

        config = BertConfig.from_pretrained(args.model_name)
        config.max_position_embeddings = max_seq + 2
        config._attn_implementation = "sdpa"
        base_sdpa = BertModel(config).to(device).float().eval()

        for seq_len in seq_lens:
            inputs, _ = build_inputs_local_or_synthetic(
                device, base_sdpa, seq_len=seq_len, exact_seq_len=True
            )
            wrapped_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "token_type_ids": torch.zeros_like(inputs["input_ids"]),
            }
            ms = benchmark_model(base_sdpa, wrapped_inputs, num_iters=args.iters,
                                warmup=args.warmup)
            tok_per_s = seq_len / ms * 1000
            result = {
                "variant": "Dense-SDPA", "mode": "dense_sdpa", "seq_len": seq_len,
                "latency_ms": round(float(ms), 4),
                "tokens_per_sec": round(float(tok_per_s), 1),
            }
            all_results.append(result)
            print(f"  seq={seq_len} | {ms:.4f} ms | {tok_per_s:.1f} tok/s")

        del base_sdpa
        gc.collect()
        torch.cuda.empty_cache()

    # SpAtten variants
    mode_to_label = {
        "quant_only": "Sp-Quant",
        "v_prune_only": "Sp-V",
        "full": "Sp-Full",
    }
    spatten_modes = [m for m in modes if m not in {"dense_eager", "dense_sdpa"}]

    for mode in spatten_modes:
        label = mode_to_label[mode]
        print(f"\n--- {label} ---")
        try:
            base_model, _ = load_bert_model_local_or_synthetic(
                device, max_position_embeddings=max_seq + 2
            )
            base_model = base_model.to(device).float().eval()
            spatten_model = configure_spatten_bert(
                base_model, mode,
                quant_threshold=args.quant_threshold,
                v_threshold=args.v_threshold,
                enable_head_prune=args.enable_head_prune,
                head_prune_num=args.head_prune_num,
                head_prune_interval=args.head_prune_interval,
                enable_token_prune=args.enable_token_prune,
                token_prune_num=args.token_prune_num,
            )
            spatten_model = spatten_model.to(device).float().eval()
        except Exception as e:
            print(f"  {label} | SKIP (config error: {e})")
            continue

        for seq_len in seq_lens:
            inputs, _ = build_inputs_local_or_synthetic(
                device, spatten_model, seq_len=seq_len, exact_seq_len=True
            )
            wrapped_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "token_type_ids": torch.zeros_like(inputs["input_ids"]),
            }
            ms = benchmark_model(spatten_model, wrapped_inputs, num_iters=args.iters,
                                warmup=args.warmup)
            tok_per_s = seq_len / ms * 1000

            dense_ms = next((r["latency_ms"] for r in all_results
                           if r["variant"] == "Dense-Eager" and r["seq_len"] == seq_len), None)
            vs_eager = f"{dense_ms/ms:.2f}x" if dense_ms and ms > 0 else ""

            result = {
                "variant": label, "mode": mode, "seq_len": seq_len,
                "latency_ms": round(float(ms), 4),
                "tokens_per_sec": round(float(tok_per_s), 1),
                "speedup_vs_eager": round(float(dense_ms/ms), 4) if dense_ms else None,
            }
            all_results.append(result)
            print(f"  seq={seq_len} | {ms:.4f} ms | {tok_per_s:.1f} tok/s | {vs_eager} vs Eager")

        del base_model, spatten_model
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("BERT Throughput Summary")
    print("=" * 80)
    header = f"{'Variant':<14} {'Seq':<7} {'Latency ms':<12} {'Tokens/sec':<12} {'vs Eager':<10}"
    print(header)
    print("-" * 65)
    for seq_len in seq_lens:
        for r in all_results:
            if r["seq_len"] != seq_len:
                continue
            vs = f"{r.get('speedup_vs_eager', ''):.2f}x" if r.get("speedup_vs_eager") else "(baseline)"
            print(f"{r['variant']:<14} {r['seq_len']:<7} {r['latency_ms']:<12.4f} "
                  f"{r['tokens_per_sec']:<12.1f} {vs:<10}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"bert_throughput_{timestamp}.json"
    output_path.write_text(json.dumps({
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "model_name": args.model_name,
        "warmup": args.warmup, "iters": args.iters,
        "results": all_results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
