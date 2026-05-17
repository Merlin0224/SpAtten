"""
Throughput benchmark for SpAtten variants on Qwen3-0.6B.

Measures generation throughput (tokens/sec) across:
  Dense (HF baseline), Sp-Quant, Sp-V, Sp-Full
with prefill + multi-step decode phase breakdown.

Reports: prefill tok/s, decode tok/s, total tok/s, TTFT, and decode latency.
"""
import argparse
import copy
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from benchmark.benchmark_bf16_msb import reset_spatten_states
from spattn.spatten_qwen3_bf16_msb import SpattenQwen3Attention, spatten_qwen3_model_forward
def parse_int_list(raw_value):
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def build_exact_prompt_token_ids(vocab_size, seq_len, seed=0):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + seq_len)
    token_ids = torch.randint(
        low=10, high=max(11, vocab_size), size=(seq_len,),
        generator=generator, dtype=torch.long,
    )
    return token_ids.tolist()


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
    """Patch Qwen3ForCausalLM with SpAtten attention."""
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


@torch.no_grad()
def benchmark_prefill_throughput(model, prompt_token_ids, device):
    """Measure prefill throughput in tokens/sec."""
    input_ids = torch.tensor([prompt_token_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    seq_len = input_ids.size(1)

    # warmup
    for _ in range(5):
        _ = model(input_ids=input_ids, attention_mask=attention_mask,
                   use_cache=True, return_dict=True)

    torch.cuda.synchronize()
    start = time.perf_counter()
    n_iters = 20
    for _ in range(n_iters):
        _ = model(input_ids=input_ids, attention_mask=attention_mask,
                   use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = seq_len * n_iters
    return total_tokens / elapsed, elapsed / n_iters * 1000


@torch.no_grad()
def benchmark_decode_throughput(model, prompt_token_ids, device, decode_steps=16):
    """Measure decode throughput for multi-step generation."""
    input_ids = torch.tensor([prompt_token_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    # prefill + get past_key_values
    prefill_out = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=True, return_dict=True)
    past_kv = prefill_out.past_key_values
    next_token = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
    decode_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)], dim=1
    )

    # warmup decode
    _past = past_kv
    _token = next_token
    for _ in range(3):
        out = model(input_ids=_token, attention_mask=decode_mask,
                    past_key_values=_past, use_cache=True, return_dict=True)
        _past = out.past_key_values
        _token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    torch.cuda.synchronize()
    start = time.perf_counter()
    n_iters = 10
    for _ in range(n_iters):
        _past = past_kv
        _token = next_token
        for _step in range(decode_steps):
            out = model(input_ids=_token, attention_mask=decode_mask,
                        past_key_values=_past, use_cache=True, return_dict=True)
            _past = out.past_key_values
            _token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = decode_steps * n_iters
    tok_per_s = total_tokens / elapsed
    avg_step_ms = elapsed / (decode_steps * n_iters) * 1000
    return tok_per_s, avg_step_ms


@torch.no_grad()
def benchmark_generate_ttft(model, prompt_token_ids, device):
    """Measure Time-To-First-Token via generate()."""
    input_ids = torch.tensor([prompt_token_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": 1,
        "min_new_tokens": 1,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": 0,
    }

    for _ in range(5):
        _ = model.generate(**gen_kwargs)

    torch.cuda.synchronize()
    start = time.perf_counter()
    n_iters = 20
    for _ in range(n_iters):
        _ = model.generate(**gen_kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / n_iters * 1000


def main():
    parser = argparse.ArgumentParser(
        description="Throughput benchmark: SpAtten variants on Qwen3-0.6B"
    )
    parser.add_argument("--model-name", default="/root/autodl-tmp/models/Qwen3-0.6B-ms")
    parser.add_argument("--seq-lens", default="1024,2048,4096,8192")
    parser.add_argument("--decode-steps", type=int, default=16,
                        help="Number of decode steps for throughput measurement")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--modes", default="dense,quant_only,v_prune_only,full")
    parser.add_argument("--quant-thresholds", default="0.01")
    parser.add_argument("--v-thresholds", default="0.05")
    parser.add_argument("--enable-head-prune", action="store_true")
    parser.add_argument("--head-prune-num", type=int, default=1)
    parser.add_argument("--head-prune-interval", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/throughput")
    parser.add_argument("--allow-remote-download", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = "cuda"
    seq_lens = parse_int_list(args.seq_lens)
    modes = [m.strip() for m in args.modes.split(",")]

    print("=" * 70)
    print("SpAtten Throughput Benchmark - Qwen3-0.6B")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Seq lens: {seq_lens}")
    print(f"Decode steps: {args.decode_steps}")
    print(f"Modes: {modes}")
    print(f"Head prune: {'on' if args.enable_head_prune else 'off'}")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n[1/3] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=not args.allow_remote_download,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=not args.allow_remote_download,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    base_model = base_model.to(device).eval()
    print(f"  Model loaded: {args.model_name}")

    prompt_map = {
        seq_len: build_exact_prompt_token_ids(vocab_size, seq_len, seed=20260517)
        for seq_len in seq_lens
    }

    all_results = []

    # Dense baseline
    if "dense" in modes:
        print("\n[2/3] Benchmarking Dense baseline...")
        for seq_len in seq_lens:
            prompt = prompt_map[seq_len]
            prefill_tps, prefill_ms = benchmark_prefill_throughput(base_model, prompt, device)
            decode_tps, decode_ms = benchmark_decode_throughput(
                base_model, prompt, device, decode_steps=args.decode_steps
            )
            ttft_ms = benchmark_generate_ttft(base_model, prompt, device)

            seq_tokens = seq_len + args.decode_steps
            total_time = prefill_ms + decode_ms * args.decode_steps
            total_tps = seq_tokens / total_time * 1000 if total_time > 0 else 0

            result = {
                "variant": "Dense",
                "mode": "dense",
                "seq_len": seq_len,
                "decode_steps": args.decode_steps,
                "prefill_tok_per_s": round(float(prefill_tps), 2),
                "prefill_ms": round(float(prefill_ms), 4),
                "decode_tok_per_s": round(float(decode_tps), 2),
                "decode_step_ms": round(float(decode_ms), 4),
                "ttft_ms": round(float(ttft_ms), 4),
                "total_tok_per_s": round(float(total_tps), 2),
            }
            all_results.append(result)
            print(f"  Dense | seq={seq_len} | prefill={prefill_tps:.1f} tok/s | "
                  f"decode={decode_tps:.1f} tok/s | TTFT={ttft_ms:.2f}ms | "
                  f"total={total_tps:.1f} tok/s")

    # SpAtten variants
    print(f"\n[3/3] Benchmarking SpAtten variants...")
    mode_to_label = {
        "quant_only": "Sp-Quant",
        "v_prune_only": "Sp-V",
        "full": "Sp-Full",
    }
    spatten_modes = [m for m in modes if m != "dense"]

    for mode in spatten_modes:
        label = mode_to_label[mode]
        try:
            spatten_model = configure_spatten_causal_lm(
                base_model,
                mode,
                quant_threshold=float(args.quant_thresholds.split(",")[0]),
                v_threshold=float(args.v_thresholds.split(",")[0]),
                enable_head_prune=args.enable_head_prune,
                head_prune_num=args.head_prune_num,
                head_prune_start_layer=0,
                head_prune_interval=args.head_prune_interval,
            )
        except Exception as e:
            print(f"  {label} | SKIP (config error: {e})")
            continue

        for seq_len in seq_lens:
            prompt = prompt_map[seq_len]

            def reset_fn(model=spatten_model):
                reset_spatten_states(model.model)

            reset_fn()
            prefill_tps, prefill_ms = benchmark_prefill_throughput(
                spatten_model, prompt, device
            )

            reset_fn()
            decode_tps, decode_ms = benchmark_decode_throughput(
                spatten_model, prompt, device, decode_steps=args.decode_steps
            )

            reset_fn()
            ttft_ms = benchmark_generate_ttft(spatten_model, prompt, device)

            seq_tokens = seq_len + args.decode_steps
            total_time = prefill_ms + decode_ms * args.decode_steps
            total_tps = seq_tokens / total_time * 1000 if total_time > 0 else 0

            result = {
                "variant": label,
                "mode": mode,
                "seq_len": seq_len,
                "decode_steps": args.decode_steps,
                "head_prune_num": args.head_prune_num if args.enable_head_prune else 0,
                "head_prune_interval": args.head_prune_interval if args.enable_head_prune else 0,
                "prefill_tok_per_s": round(float(prefill_tps), 2),
                "prefill_ms": round(float(prefill_ms), 4),
                "decode_tok_per_s": round(float(decode_tps), 2),
                "decode_step_ms": round(float(decode_ms), 4),
                "ttft_ms": round(float(ttft_ms), 4),
                "total_tok_per_s": round(float(total_tps), 2),
            }
            all_results.append(result)

            # Compare against dense
            dense_total = next(
                (r["total_tok_per_s"] for r in all_results
                 if r["variant"] == "Dense" and r["seq_len"] == seq_len),
                None,
            )
            speedup = ""
            if dense_total and dense_total > 0:
                speedup = f" | {total_tps/dense_total:.2f}x vs Dense"

            print(f"  {label} | seq={seq_len} | prefill={prefill_tps:.1f} tok/s | "
                  f"decode={decode_tps:.1f} tok/s | TTFT={ttft_ms:.2f}ms | "
                  f"total={total_tps:.1f} tok/s{speedup}")

        del spatten_model
        gc.collect()
        torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 80)
    print("Throughput Results Summary")
    print("=" * 80)
    header = (f"{'Variant':<12} {'Seq':<6} {'Prefill tok/s':<15} "
              f"{'Decode tok/s':<14} {'TTFT ms':<10} {'Total tok/s':<12} {'vs Dense':<10}")
    print(header)
    print("-" * 80)
    for seq_len in seq_lens:
        dense_total = next(
            (r["total_tok_per_s"] for r in all_results
             if r["variant"] == "Dense" and r["seq_len"] == seq_len),
            None,
        )
        for r in all_results:
            if r["seq_len"] != seq_len:
                continue
            vs_dense = ""
            if dense_total and r["variant"] != "Dense":
                ratio = r["total_tok_per_s"] / dense_total
                vs_dense = f"{ratio:.2f}x"
            elif r["variant"] == "Dense":
                vs_dense = "(baseline)"
            print(f"{r['variant']:<12} {r['seq_len']:<6} "
                  f"{r['prefill_tok_per_s']:<15.1f} {r['decode_tok_per_s']:<14.1f} "
                  f"{r['ttft_ms']:<10.2f} {r['total_tok_per_s']:<12.1f} {vs_dense:<10}")

    # Print prefill vs decode breakdown
    print("\n" + "-" * 80)
    print("Prefill vs Decode Latency Breakdown (ms)")
    print("-" * 80)
    print(f"{'Variant':<12} {'Seq':<6} {'Prefill ms':<12} {'Decode ms/step':<16} {'Decode total ms':<16} {'TTFT ms':<10}")
    print("-" * 75)
    for r in all_results:
        decode_total = r["decode_step_ms"] * r["decode_steps"]
        print(f"{r['variant']:<12} {r['seq_len']:<6} "
              f"{r['prefill_ms']:<12.4f} {r['decode_step_ms']:<16.4f} "
              f"{decode_total:<16.4f} {r['ttft_ms']:<10.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"throughput_benchmark_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0),
        "model_name": args.model_name,
        "decode_steps": args.decode_steps,
        "enable_head_prune": args.enable_head_prune,
        "head_prune_num": args.head_prune_num if args.enable_head_prune else 0,
        "head_prune_interval": args.head_prune_interval if args.enable_head_prune else 0,
        "results": all_results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
