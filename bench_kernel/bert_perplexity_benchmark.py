"""
Perplexity benchmark for SpAtten BERT variants (Masked Language Modeling).

Evaluates MLM perplexity across:
  Dense (HF baseline), Sp-Quant, Sp-V, Sp-Full
with configurable head/token pruning, quantization, and V-threshold.

Uses pseudo-perplexity approach: random mask 15% tokens, compute loss on masked positions.
"""
import argparse
import copy
import gc
import json
import math
import random
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


def load_eval_corpus_bert(tokenizer, seq_len, stride=None, max_samples=None):
    """Load text corpus for BERT MLM perplexity."""
    if stride is None:
        stride = seq_len // 2

    # Try local GSM8K
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test",
                          download_mode="reuse_dataset_if_exists")
        texts = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in ds]
        combined = " ".join(texts)
        encodings = tokenizer(combined, return_tensors="pt", add_special_tokens=True,
                              truncation=True, max_length=seq_len * (max_samples or 200))
        input_ids = encodings["input_ids"][0]
        source = f"GSM8K ({len(ds)} examples)"
    except Exception:
        # Synthetic fallback
        generator = torch.Generator()
        generator.manual_seed(42)
        total_tokens = max(seq_len * (max_samples or 20), seq_len * 20)
        input_ids = torch.randint(100, 30000, (total_tokens,),
                                  generator=generator, dtype=torch.long)
        source = "synthetic random tokens"

    chunks = []
    for i in range(0, len(input_ids) - seq_len, stride):
        chunks.append(input_ids[i:i + seq_len])
    if max_samples and len(chunks) > max_samples:
        step = max(len(chunks) // max_samples, 1)
        chunks = chunks[::step][:max_samples]
    return chunks, source


@torch.no_grad()
def compute_mlm_perplexity(model, tokenizer, input_chunks, device, mlm_prob=0.15):
    """Compute BERT MLM perplexity over chunks.

    Uses random masking: 15% tokens masked, loss only on masked positions.
    """
    model.eval()
    total_loss = 0.0
    total_masked = 0
    mask_token_id = tokenizer.mask_token_id

    for chunk in input_chunks:
        input_ids = chunk.unsqueeze(0).to(device)

        # Create random mask (15% of tokens)
        rand = torch.rand(input_ids.shape, device=device)
        mask_arr = (rand < mlm_prob) & (input_ids != tokenizer.cls_token_id) \
                   & (input_ids != tokenizer.sep_token_id) & (input_ids != tokenizer.pad_token_id)

        # Prepare labels: -100 for non-masked, original token for masked
        labels = input_ids.clone()
        labels[~mask_arr] = -100

        # Replace masked tokens with [MASK]
        masked_input = input_ids.clone()
        masked_input[mask_arr] = mask_token_id

        outputs = model(input_ids=masked_input, labels=labels)
        loss = outputs.loss

        n_masked = mask_arr.sum().item()
        if loss is not None and not torch.isnan(loss) and n_masked > 0:
            total_loss += loss.item() * n_masked
            total_masked += n_masked

    avg_loss = total_loss / total_masked if total_masked > 0 else float("inf")
    ppl = math.exp(avg_loss)
    return ppl, avg_loss, total_masked


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

    for layer_idx, layer in enumerate(spatten_model.bert.encoder.layer):
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

    spatten_model.bert.encoder.forward = spatten_encoder_forward.__get__(
        spatten_model.bert.encoder, BertEncoder
    )
    return spatten_model


def parse_int_list(raw):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="BERT MLM Perplexity Benchmark")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--seq-lens", default="128,256,512")
    parser.add_argument("--max-eval-chunks", type=int, default=200)
    parser.add_argument("--modes", default="dense,quant_only,v_prune_only,full")
    parser.add_argument("--quant-threshold", type=float, default=0.01)
    parser.add_argument("--v-threshold", type=float, default=0.05)
    parser.add_argument("--enable-head-prune", action="store_true")
    parser.add_argument("--head-prune-num", type=int, default=1)
    parser.add_argument("--head-prune-interval", type=int, default=1)
    parser.add_argument("--enable-token-prune", action="store_true")
    parser.add_argument("--token-prune-num", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/bert_perplexity")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_lens = parse_int_list(args.seq_lens)
    modes = [m.strip() for m in args.modes.split(",")]

    print("=" * 70)
    print("SpAtten BERT Perplexity Benchmark")
    print(f"Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print(f"Seq lens: {seq_lens}")
    print(f"Modes: {modes}")
    print("=" * 70)

    from transformers import BertForMaskedLM, AutoTokenizer

    # Load tokenizer
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"  Tokenizer: {args.model_name}")

    # Prepare data
    print("[2/3] Preparing evaluation data...")
    eval_data = {}
    for seq_len in seq_lens:
        chunks, source = load_eval_corpus_bert(
            tokenizer, seq_len, max_samples=args.max_eval_chunks
        )
        eval_data[seq_len] = chunks
        print(f"  seq_len={seq_len}: {len(chunks)} chunks | source: {source}")

    all_results = []

    # Dense baseline
    if "dense" in modes:
        print("\n[3/3] Evaluating Dense baseline...")
        base_model = BertForMaskedLM.from_pretrained(args.model_name).to(device).eval()
        for seq_len in seq_lens:
            ppl, avg_loss, n_masked = compute_mlm_perplexity(
                base_model, tokenizer, eval_data[seq_len], device
            )
            result = {
                "variant": "Dense", "mode": "dense", "seq_len": seq_len,
                "head_prune_num": 0, "head_prune_interval": 0,
                "token_prune_num": 0, "quant_threshold": None, "v_threshold": None,
                "perplexity": round(float(ppl), 4),
                "avg_loss": round(float(avg_loss), 6), "n_masked": n_masked,
            }
            all_results.append(result)
            print(f"  Dense | seq={seq_len} | PPL={ppl:.4f} | masked={n_masked}")

    # SpAtten variants
    mode_to_label = {
        "quant_only": "Sp-Quant",
        "v_prune_only": "Sp-V",
        "full": "Sp-Full",
    }
    spatten_modes = [m for m in modes if m != "dense"]

    for mode in spatten_modes:
        label = mode_to_label[mode]
        print(f"\n  Configuring {label}...")
        try:
            base_mlm = BertForMaskedLM.from_pretrained(args.model_name).to(device).eval()
            spatten_mlm = configure_spatten_bert(
                base_mlm, mode,
                quant_threshold=args.quant_threshold,
                v_threshold=args.v_threshold,
                enable_head_prune=args.enable_head_prune,
                head_prune_num=args.head_prune_num,
                head_prune_interval=args.head_prune_interval,
                enable_token_prune=args.enable_token_prune,
                token_prune_num=args.token_prune_num,
            )
            spatten_mlm = spatten_mlm.to(device).eval()
        except Exception as e:
            print(f"  {label} | SKIP (config error: {e})")
            continue

        for seq_len in seq_lens:
            ppl, avg_loss, n_masked = compute_mlm_perplexity(
                spatten_mlm, tokenizer, eval_data[seq_len], device
            )
            result = {
                "variant": label, "mode": mode, "seq_len": seq_len,
                "head_prune_num": args.head_prune_num if args.enable_head_prune else 0,
                "head_prune_interval": args.head_prune_interval if args.enable_head_prune else 0,
                "token_prune_num": args.token_prune_num if args.enable_token_prune else 0,
                "quant_threshold": args.quant_threshold,
                "v_threshold": args.v_threshold,
                "perplexity": round(float(ppl), 4),
                "avg_loss": round(float(avg_loss), 6), "n_masked": n_masked,
            }
            all_results.append(result)

            dense_ppl = next((r["perplexity"] for r in all_results
                            if r["variant"] == "Dense" and r["seq_len"] == seq_len), None)
            degradation = ""
            if dense_ppl and dense_ppl > 0:
                degradation = f" | +{(ppl/dense_ppl - 1)*100:.2f}% vs Dense"
            print(f"  {label} | seq={seq_len} | PPL={ppl:.4f}{degradation}")

        del base_mlm, spatten_mlm
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("BERT Perplexity Summary")
    print("=" * 70)
    print(f"{'Variant':<12} {'Seq':<6} {'hpn':<4} {'tpn':<4} {'PPL':<10} {'vs Dense':<10}")
    print("-" * 55)

    for seq_len in seq_lens:
        dense_ppl = next((r["perplexity"] for r in all_results
                        if r["variant"] == "Dense" and r["seq_len"] == seq_len), None)
        for r in all_results:
            if r["seq_len"] != seq_len:
                continue
            vs_dense = ""
            if dense_ppl and r["variant"] != "Dense":
                delta = (r["perplexity"] / dense_ppl - 1) * 100
                vs_dense = f"+{delta:.2f}%"
            elif r["variant"] == "Dense":
                vs_dense = "(baseline)"
            print(f"{r['variant']:<12} {r['seq_len']:<6} "
                  f"{r['head_prune_num']:<4} {r['token_prune_num']:<4} "
                  f"{r['perplexity']:<10.4f} {vs_dense:<10}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"bert_perplexity_{timestamp}.json"
    output_path.write_text(json.dumps({
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "model_name": args.model_name,
        "results": all_results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
