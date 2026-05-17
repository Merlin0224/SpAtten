import argparse
import copy
import gc
import json
import os
from datetime import datetime
from pathlib import Path

import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from benchmark.generate_benchmark_exp_cachepath import benchmark_hf_generation_metrics
from bench_kernel.model_seq_vllm_compare_qwen3 import (
    build_exact_prompt_token_ids,
    load_hf_causal_lm,
    parse_int_list,
)
from spattn.spatten_qwen3_bf16_msb_exp_cachepath import (
    SpattenQwen3Attention,
    spatten_qwen3_model_forward,
)


def reset_spatten_states_qwen3(model):
    target = getattr(model, "model", model)
    layers = getattr(target, "layers", None)
    if layers is None:
        return

    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue

        if hasattr(attn, "next_active_head_indices"):
            attn.next_active_head_indices = None
        if hasattr(attn, "active_head_indices_for_this_layer"):
            attn.active_head_indices_for_this_layer = None
        if hasattr(attn, "cached_incoming_head_indices"):
            attn.cached_incoming_head_indices = None
        if hasattr(attn, "cached_next_head_indices"):
            attn.cached_next_head_indices = None
        if hasattr(attn, "cumulative_token_score"):
            attn.cumulative_token_score = None
        if hasattr(attn, "next_active_token_indices"):
            attn.next_active_token_indices = None
        if hasattr(attn, "current_active_head_indices"):
            attn.current_active_head_indices = None
        if hasattr(attn, "stage_token_score_accum"):
            attn.stage_token_score_accum = None
        if hasattr(attn, "stage_token_score_count"):
            attn.stage_token_score_count = 0
        if hasattr(attn, "stage_token_weight_total"):
            attn.stage_token_weight_total = 0.0


def configure_spatten_qwen3_causal_lm_exp(
    base_causal_lm,
    mode,
    *,
    quant_threshold=0.01,
    v_threshold=0.05,
    enable_head_prune=False,
    head_prune_num=1,
    head_prune_start_layer=0,
    head_prune_interval=1,
    enable_prefill_static_reuse=True,
    enable_prefill_sdpa_fast_path=False,
    prefill_sdpa_max_seq_len=0,
    enable_decode_fast_path=True,
    enable_cache_aware_head_reuse=True,
    enable_cache_aware_kv_prune=True,
    enable_decode_kv_window=False,
    decode_kv_window_size=0,
    decode_kv_prefix_keep=0,
):
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
    except Exception as exc:
        raise RuntimeError("transformers Qwen3 backend is required for SpAtten causal LM patching.") from exc

    model = copy.deepcopy(base_causal_lm)
    base_model = model.model
    device = next(base_causal_lm.parameters()).device
    target_dtype = next(base_causal_lm.parameters()).dtype

    for layer_idx, layer in enumerate(base_model.layers):
        orig_state = layer.self_attn.state_dict()
        new_attn = SpattenQwen3Attention(base_model.config, layer_idx=layer_idx).to(device=device, dtype=target_dtype)
        new_attn.load_state_dict(orig_state)

        new_attn.enable_prog_quant = mode in {"quant_only", "full"}
        new_attn.enable_v_prune = mode in {"v_prune_only", "full"}
        new_attn.quant_threshold = quant_threshold
        new_attn.v_threshold = v_threshold

        new_attn.enable_head_prune = enable_head_prune
        new_attn.head_prune_num = head_prune_num
        new_attn.head_prune_start_layer = head_prune_start_layer
        new_attn.head_prune_interval = head_prune_interval

        # keep token pruning off for stable use_cache experiments
        new_attn.enable_token_prune = False
        new_attn.token_prune_num = 0
        new_attn.next_active_token_indices = None
        new_attn.cumulative_token_score = None

        # prefill path options
        new_attn.enable_prefill_static_reuse = enable_prefill_static_reuse
        new_attn.enable_prefill_sdpa_fast_path = enable_prefill_sdpa_fast_path
        new_attn.prefill_sdpa_max_seq_len = prefill_sdpa_max_seq_len

        # decode path options
        new_attn.enable_decode_fast_path = enable_decode_fast_path
        new_attn.enable_cache_aware_head_reuse = enable_cache_aware_head_reuse
        new_attn.enable_cache_aware_kv_prune = enable_cache_aware_kv_prune
        new_attn.enable_decode_kv_window = enable_decode_kv_window
        new_attn.decode_kv_window_size = decode_kv_window_size
        new_attn.decode_kv_prefix_keep = decode_kv_prefix_keep

        layer.self_attn = new_attn

    base_model.forward = spatten_qwen3_model_forward.__get__(base_model, Qwen3Model)
    return model.to(device=device, dtype=target_dtype).eval()


def _run_metrics(model, prompt_token_ids, warmup, iters, decode_steps, reset_state):
    reset_fn = (lambda model=model: reset_spatten_states_qwen3(model)) if reset_state else None
    return benchmark_hf_generation_metrics(
        model,
        prompt_token_ids,
        warmup=warmup,
        iters=iters,
        decode_steps=decode_steps,
        reset_state_fn=reset_fn,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Experimental use_cache path exploration on Qwen3 (prefill/decode/KV-cache separated)."
    )
    parser.add_argument("--seq-lens", default="1024,2048,4096,8192")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--allow-remote-download", action="store_true")
    parser.add_argument("--quant-threshold", type=float, default=0.01)
    parser.add_argument("--v-threshold", type=float, default=0.05)
    parser.add_argument("--enable-head-prune", action="store_true")
    parser.add_argument("--head-prune-num", type=int, default=1)
    parser.add_argument("--head-prune-start-layer", type=int, default=0)
    parser.add_argument("--head-prune-interval", type=int, default=1)
    parser.add_argument("--enable-prefill-sdpa-fast-path", action="store_true")
    parser.add_argument("--prefill-sdpa-max-seq-len", type=int, default=0)
    parser.add_argument("--kv-window-size", type=int, default=2048)
    parser.add_argument("--kv-prefix-keep", type=int, default=0)
    parser.add_argument("--output-dir", default="artifacts/route_b")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model_seq_generate_compare_qwen3_spatten_exp_cachepath.py")

    try:
        from transformers import AutoConfig
    except Exception as exc:
        raise RuntimeError("transformers is required for this experiment.") from exc

    device = "cuda"
    seq_lens = parse_int_list(args.seq_lens)
    config = AutoConfig.from_pretrained(
        args.model_name,
        local_files_only=not args.allow_remote_download,
        trust_remote_code=True,
    )
    vocab_size = int(getattr(config, "vocab_size", 32000))
    prompt_map = {
        seq_len: build_exact_prompt_token_ids(vocab_size, seq_len, seed=20260421)
        for seq_len in seq_lens
    }

    print("Qwen3 use_cache Path Explore (Experimental)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model_name}")
    print(f"Decode steps (multi-step metric): {args.decode_steps}")
    print(f"Head prune enabled: {args.enable_head_prune} (num={args.head_prune_num})")
    print(f"Prefill SDPA fast path: {args.enable_prefill_sdpa_fast_path} (max_seq_len={args.prefill_sdpa_max_seq_len})")
    print(f"Decode KV window: size={args.kv_window_size}, prefix_keep={args.kv_prefix_keep}")

    hf_model = load_hf_causal_lm(
        args.model_name,
        device=device,
        allow_remote_download=args.allow_remote_download,
    )

    variants = {
        "hf_dense": None,
        "sp_prefill_opt": {
            "mode": "v_prune_only",
            "enable_prefill_static_reuse": True,
            "enable_prefill_sdpa_fast_path": args.enable_prefill_sdpa_fast_path,
            "prefill_sdpa_max_seq_len": args.prefill_sdpa_max_seq_len,
            "enable_decode_fast_path": False,
            "enable_cache_aware_head_reuse": False,
            "enable_cache_aware_kv_prune": False,
            "enable_decode_kv_window": False,
        },
        "sp_decode_opt": {
            "mode": "v_prune_only",
            "enable_prefill_static_reuse": True,
            "enable_prefill_sdpa_fast_path": False,
            "prefill_sdpa_max_seq_len": 0,
            "enable_decode_fast_path": True,
            "enable_cache_aware_head_reuse": True,
            "enable_cache_aware_kv_prune": True,
            "enable_decode_kv_window": False,
        },
        "sp_kv_window_opt": {
            "mode": "v_prune_only",
            "enable_prefill_static_reuse": True,
            "enable_prefill_sdpa_fast_path": False,
            "prefill_sdpa_max_seq_len": 0,
            "enable_decode_fast_path": True,
            "enable_cache_aware_head_reuse": True,
            "enable_cache_aware_kv_prune": True,
            "enable_decode_kv_window": True,
        },
    }

    results = {}
    for seq_len in seq_lens:
        row = {}
        prompt_ids = prompt_map[seq_len]

        row["hf_dense"] = benchmark_hf_generation_metrics(
            hf_model,
            prompt_ids,
            warmup=args.warmup,
            iters=args.iters,
            decode_steps=args.decode_steps,
            reset_state_fn=None,
        )

        for variant_name, opts in variants.items():
            if variant_name == "hf_dense":
                continue

            model = configure_spatten_qwen3_causal_lm_exp(
                hf_model,
                opts["mode"],
                quant_threshold=args.quant_threshold,
                v_threshold=args.v_threshold,
                enable_head_prune=args.enable_head_prune,
                head_prune_num=args.head_prune_num,
                head_prune_start_layer=args.head_prune_start_layer,
                head_prune_interval=args.head_prune_interval,
                enable_prefill_static_reuse=opts["enable_prefill_static_reuse"],
                enable_prefill_sdpa_fast_path=opts["enable_prefill_sdpa_fast_path"],
                prefill_sdpa_max_seq_len=opts["prefill_sdpa_max_seq_len"],
                enable_decode_fast_path=opts["enable_decode_fast_path"],
                enable_cache_aware_head_reuse=opts["enable_cache_aware_head_reuse"],
                enable_cache_aware_kv_prune=opts["enable_cache_aware_kv_prune"],
                enable_decode_kv_window=opts["enable_decode_kv_window"],
                decode_kv_window_size=args.kv_window_size,
                decode_kv_prefix_keep=args.kv_prefix_keep,
            )
            row[variant_name] = _run_metrics(
                model,
                prompt_ids,
                warmup=args.warmup,
                iters=args.iters,
                decode_steps=args.decode_steps,
                reset_state=args.enable_head_prune,
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        results[seq_len] = row

    print("\nPrefill Latency (ms)")
    print(f"{'Seq Len':<8} | {'HF':<10} | {'PrefillOpt':<10} | {'DecodeOpt':<10} | {'KVWinOpt':<10}")
    print("-" * 66)
    for seq_len in seq_lens:
        row = results[seq_len]
        print(
            f"{seq_len:<8} | "
            f"{row['hf_dense']['prefill_ms']:<10.4f} | "
            f"{row['sp_prefill_opt']['prefill_ms']:<10.4f} | "
            f"{row['sp_decode_opt']['prefill_ms']:<10.4f} | "
            f"{row['sp_kv_window_opt']['prefill_ms']:<10.4f}"
        )

    print("\nDecode-1tok Latency (ms)")
    print(f"{'Seq Len':<8} | {'HF':<10} | {'PrefillOpt':<10} | {'DecodeOpt':<10} | {'KVWinOpt':<10}")
    print("-" * 66)
    for seq_len in seq_lens:
        row = results[seq_len]
        print(
            f"{seq_len:<8} | "
            f"{row['hf_dense']['decode_1tok_ms']:<10.4f} | "
            f"{row['sp_prefill_opt']['decode_1tok_ms']:<10.4f} | "
            f"{row['sp_decode_opt']['decode_1tok_ms']:<10.4f} | "
            f"{row['sp_kv_window_opt']['decode_1tok_ms']:<10.4f}"
        )

    print(f"\nDecode-{args.decode_steps}tok Avg Latency (ms/token)")
    print(f"{'Seq Len':<8} | {'HF':<10} | {'PrefillOpt':<10} | {'DecodeOpt':<10} | {'KVWinOpt':<10}")
    print("-" * 66)
    for seq_len in seq_lens:
        row = results[seq_len]
        print(
            f"{seq_len:<8} | "
            f"{row['hf_dense']['decode_multi_tok_ms']:<10.4f} | "
            f"{row['sp_prefill_opt']['decode_multi_tok_ms']:<10.4f} | "
            f"{row['sp_decode_opt']['decode_multi_tok_ms']:<10.4f} | "
            f"{row['sp_kv_window_opt']['decode_multi_tok_ms']:<10.4f}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"model_seq_generate_compare_qwen3_spatten_exp_cachepath_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0),
        "model_name": args.model_name,
        "seq_lens": seq_lens,
        "warmup": args.warmup,
        "iters": args.iters,
        "decode_steps": args.decode_steps,
        "quant_threshold": args.quant_threshold,
        "v_threshold": args.v_threshold,
        "enable_head_prune": args.enable_head_prune,
        "head_prune_num": args.head_prune_num,
        "head_prune_start_layer": args.head_prune_start_layer,
        "head_prune_interval": args.head_prune_interval,
        "enable_prefill_sdpa_fast_path": args.enable_prefill_sdpa_fast_path,
        "prefill_sdpa_max_seq_len": args.prefill_sdpa_max_seq_len,
        "kv_window_size": args.kv_window_size,
        "kv_prefix_keep": args.kv_prefix_keep,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
