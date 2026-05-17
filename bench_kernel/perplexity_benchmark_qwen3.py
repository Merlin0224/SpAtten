"""
Perplexity benchmark for SpAtten variants on Qwen3-0.6B.

Evaluates language modeling quality (perplexity) across:
  Dense (HF baseline), Sp-Quant, Sp-V, Sp-Full
with configurable head pruning, quantization, and V-threshold parameters.

Uses WikiText-2 test set with sliding-window evaluation.
"""
import argparse
import copy
import gc
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark.benchmark_bf16_msb import reset_spatten_states
from spattn.spatten_qwen3_bf16_msb import SpattenQwen3Attention, spatten_qwen3_model_forward


def load_eval_corpus(tokenizer, seq_len, stride=None, max_samples=None, prefer_local=True):
    """Load evaluation corpus and tokenize into fixed-length chunks.

    Tries in order: local GSM8K, WikiText-2 (network), synthetic fallback.
    """
    if stride is None:
        stride = seq_len

    # Attempt 1: local GSM8K dataset (no network needed if cached)
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "openai/gsm8k", "main", split="test",
            download_mode="reuse_dataset_if_exists",
        )
        texts = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in ds]
        combined = "\n\n".join(texts)
        encodings = tokenizer(combined, return_tensors="pt", add_special_tokens=False,
                              truncation=False)
        input_ids = encodings["input_ids"][0]
        if len(input_ids) >= seq_len * 3:
            source = f"GSM8K test ({len(ds)} examples, {len(input_ids)} tokens)"
            return _chunkify(input_ids, seq_len, stride, max_samples), source
    except Exception:
        pass

    # Attempt 2: WikiText-2 via HF (requires network)
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"])
        encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = encodings["input_ids"][0]
        source = f"WikiText-2 test ({len(input_ids)} tokens)"
        return _chunkify(input_ids, seq_len, stride, max_samples), source
    except Exception:
        pass

    # Attempt 3: synthetic random tokens with fixed seed
    vocab_size = getattr(tokenizer, "vocab_size", 151936)
    generator = torch.Generator()
    generator.manual_seed(42)
    total_tokens = max(seq_len * max_samples, seq_len * 20)
    input_ids = torch.randint(
        10, max(11, vocab_size - 1), (total_tokens,),
        generator=generator, dtype=torch.long,
    )
    source = f"synthetic random tokens ({total_tokens} tokens, seed=42)"
    return _chunkify(input_ids, seq_len, stride, max_samples), source


def _chunkify(input_ids, seq_len, stride, max_samples):
    chunks = []
    for i in range(0, len(input_ids) - seq_len, stride):
        chunks.append(input_ids[i : i + seq_len])
    if max_samples and len(chunks) > max_samples:
        step = max(len(chunks) // max_samples, 1)
        chunks = chunks[::step][:max_samples]
    return chunks


@torch.no_grad()
def compute_perplexity(model, input_ids_chunks, device):
    """Compute perplexity over tokenized chunks.

    Returns:
        ppl: exp(average cross-entropy loss)
        avg_loss: average negative log-likelihood
        total_tokens: total number of tokens evaluated
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for chunk in input_ids_chunks:
        input_ids = chunk.unsqueeze(0).to(device)
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        if loss is not None and not torch.isnan(loss):
            total_loss += loss.item() * chunk.size(0)
            total_tokens += chunk.size(0)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_loss)
    return ppl, avg_loss, total_tokens


def configure_spatten_causal_lm(
    base_causal_lm,
    mode,
    quant_threshold=0.01,
    v_threshold=0.05,
    enable_head_prune=False,
    head_prune_num=1,
    head_prune_start_layer=0,
    head_prune_interval=1,
):
    """Patch Qwen3ForCausalLM with SpAtten attention, keeping lm_head intact."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    model = copy.deepcopy(base_causal_lm)
    base_model = model.model
    device = next(base_causal_lm.parameters()).device
    target_dtype = next(base_causal_lm.parameters()).dtype

    for layer_idx, layer in enumerate(base_model.layers):
        orig_state = layer.self_attn.state_dict()
        new_attn = SpattenQwen3Attention(base_model.config, layer_idx=layer_idx)
        new_attn = new_attn.to(device=device, dtype=target_dtype)
        new_attn.load_state_dict(orig_state)
        new_attn.enable_prog_quant = mode in {"quant_only", "full"}
        new_attn.enable_v_prune = mode in {"v_prune_only", "full"}
        new_attn.quant_threshold = quant_threshold
        new_attn.v_threshold = v_threshold
        new_attn.enable_head_prune = enable_head_prune
        new_attn.head_prune_num = head_prune_num
        new_attn.head_prune_start_layer = head_prune_start_layer
        new_attn.head_prune_interval = head_prune_interval
        new_attn.enable_token_prune = False
        layer.self_attn = new_attn

    base_model.forward = spatten_qwen3_model_forward.__get__(base_model, Qwen3Model)
    return model.to(device=device, dtype=target_dtype).eval()


def parse_int_list(raw):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Perplexity benchmark: SpAtten variants on Qwen3-0.6B"
    )
    parser.add_argument("--model-name", default="/root/autodl-tmp/models/Qwen3-0.6B-ms")
    parser.add_argument("--seq-lens", default="1024,2048,4096,8192")
    parser.add_argument("--max-eval-chunks", type=int, default=500,
                        help="Max chunks per seq_len (limit eval time)")
    parser.add_argument("--modes", default="dense,quant_only,v_prune_only,full",
                        help="Comma-separated modes: dense,quant_only,v_prune_only,full")
    parser.add_argument("--quant-thresholds", default="0.01",
                        help="Comma-separated quant_threshold values to sweep")
    parser.add_argument("--v-thresholds", default="0.05",
                        help="Comma-separated v_threshold values to sweep")
    parser.add_argument("--enable-head-prune", action="store_true")
    parser.add_argument("--head-prune-nums", default="1",
                        help="Comma-separated head_prune_num values")
    parser.add_argument("--head-prune-intervals", default="1",
                        help="Comma-separated head_prune_interval values")
    parser.add_argument("--output-dir", default="artifacts/perplexity")
    parser.add_argument("--allow-remote-download", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = "cuda"
    seq_lens = parse_int_list(args.seq_lens)
    modes = [m.strip() for m in args.modes.split(",")]
    quant_thresholds = parse_float_list(args.quant_thresholds)
    v_thresholds = parse_float_list(args.v_thresholds)
    head_prune_nums = parse_int_list(args.head_prune_nums)
    head_prune_intervals = parse_int_list(args.head_prune_intervals)

    print("=" * 70)
    print("SpAtten Perplexity Benchmark - Qwen3-0.6B")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Seq lens: {seq_lens}")
    print(f"Modes: {modes}")
    print(f"Head prune: {'on' if args.enable_head_prune else 'off'}")
    print("=" * 70)

    # Load base model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n[1/4] Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=not args.allow_remote_download,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=not args.allow_remote_download,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    base_model = base_model.to(device).eval()
    print(f"  Model loaded: {args.model_name}")

    # Prepare evaluation chunks for each seq_len
    print("\n[2/4] Preparing evaluation corpus...")
    eval_data = {}
    corpus_source = None
    for seq_len in seq_lens:
        chunks, source = load_eval_corpus(
            tokenizer, seq_len, stride=seq_len // 2, max_samples=args.max_eval_chunks
        )
        eval_data[seq_len] = chunks
        if corpus_source is None:
            corpus_source = source
        print(f"  seq_len={seq_len}: {len(chunks)} chunks | source: {source}")

    all_results = []

    # Dense baseline
    if "dense" in modes:
        print("\n[3/4] Evaluating Dense baseline...")
        for seq_len in seq_lens:
            chunks = eval_data[seq_len]
            ppl, avg_loss, n_tokens = compute_perplexity(base_model, chunks, device)
            result = {
                "variant": "Dense",
                "mode": "dense",
                "seq_len": seq_len,
                "head_prune_num": 0,
                "head_prune_interval": 0,
                "quant_threshold": None,
                "v_threshold": None,
                "perplexity": round(float(ppl), 4),
                "avg_loss": round(float(avg_loss), 6),
                "n_tokens": n_tokens,
            }
            all_results.append(result)
            print(f"  Dense | seq={seq_len} | PPL={ppl:.4f} | loss={avg_loss:.6f}")

    # SpAtten variants
    configs = []
    for mode in [m for m in modes if m != "dense"]:
        if args.enable_head_prune:
            for hpn in head_prune_nums:
                for hpi in head_prune_intervals:
                    for qt in quant_thresholds:
                        for vt in v_thresholds:
                            configs.append((mode, hpn, hpi, qt, vt))
        else:
            for qt in quant_thresholds:
                for vt in v_thresholds:
                    configs.append((mode, 0, 0, qt, vt))

    print(f"\n[4/4] Evaluating {len(configs)} SpAtten configurations...")
    for idx, (mode, hpn, hpi, qt, vt) in enumerate(configs):
        label = {
            "quant_only": "Sp-Quant",
            "v_prune_only": "Sp-V",
            "full": "Sp-Full",
        }[mode]

        try:
            spatten_model = configure_spatten_causal_lm(
                base_model,
                mode,
                quant_threshold=qt,
                v_threshold=vt,
                enable_head_prune=args.enable_head_prune,
                head_prune_num=hpn,
                head_prune_start_layer=0,
                head_prune_interval=hpi,
            )
        except Exception as e:
            print(f"  [{idx+1}/{len(configs)}] {label} | hpn={hpn} hpi={hpi} "
                  f"qt={qt} vt={vt} | SKIP (config error: {e})")
            continue

        for seq_len in seq_lens:
            chunks = eval_data[seq_len]

            def reset_fn(model=spatten_model):
                reset_spatten_states(model.model)

            ppl, avg_loss, n_tokens = compute_perplexity(spatten_model, chunks, device)
            # reset state between seq lens to avoid cross-contamination
            reset_fn()

            result = {
                "variant": label,
                "mode": mode,
                "seq_len": seq_len,
                "head_prune_num": hpn,
                "head_prune_interval": hpi,
                "quant_threshold": qt,
                "v_threshold": vt,
                "perplexity": round(float(ppl), 4),
                "avg_loss": round(float(avg_loss), 6),
                "n_tokens": n_tokens,
            }
            all_results.append(result)

            # Get dense PPL for this seq_len for comparison
            dense_ppl = next(
                (r["perplexity"] for r in all_results
                 if r["variant"] == "Dense" and r["seq_len"] == seq_len),
                None,
            )
            degradation = ""
            if dense_ppl and dense_ppl > 0:
                degradation = f" | +{(ppl/dense_ppl - 1)*100:.2f}% vs Dense"

            print(f"  [{idx+1}/{len(configs)}] {label} | seq={seq_len} | "
                  f"hpn={hpn} hpi={hpi} qt={qt} vt={vt} | "
                  f"PPL={ppl:.4f}{degradation}")

        del spatten_model
        gc.collect()
        torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 70)
    print("Perplexity Results Summary")
    print("=" * 70)
    print(f"{'Variant':<12} {'Seq':<6} {'hpn':<4} {'hpi':<4} {'qt':<6} {'vt':<6} {'PPL':<10} {'vs Dense':<10}")
    print("-" * 70)

    for seq_len in seq_lens:
        dense_ppl = next(
            (r["perplexity"] for r in all_results
             if r["variant"] == "Dense" and r["seq_len"] == seq_len),
            None,
        )
        for r in all_results:
            if r["seq_len"] != seq_len:
                continue
            vs_dense = ""
            if dense_ppl and r["variant"] != "Dense":
                delta = (r["perplexity"] / dense_ppl - 1) * 100
                vs_dense = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
            elif r["variant"] == "Dense":
                vs_dense = "(baseline)"
            print(f"{r['variant']:<12} {r['seq_len']:<6} {r['head_prune_num']:<4} "
                  f"{r['head_prune_interval']:<4} {r['quant_threshold'] or '-':<6} "
                  f"{r['v_threshold'] or '-':<6} {r['perplexity']:<10.4f} {vs_dense:<10}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"perplexity_benchmark_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0),
        "model_name": args.model_name,
        "enable_head_prune": args.enable_head_prune,
        "results": all_results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
