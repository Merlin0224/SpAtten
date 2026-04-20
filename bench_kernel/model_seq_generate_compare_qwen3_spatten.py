import os
import argparse
import copy
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from benchmark.benchmark_bf16_msb import reset_spatten_states
from benchmark.generate_benchmark import benchmark_hf_generation_metrics
from bench_kernel.model_seq_vllm_compare_qwen3 import (
    build_exact_prompt_token_ids,
    load_hf_causal_lm,
    load_vllm_engine,
    benchmark_vllm_generate,
    parse_int_list,
)
from spattn.spatten_qwen3_bf16_msb import SpattenQwen3Attention, spatten_qwen3_model_forward


def configure_spatten_qwen3_causal_lm(
    base_causal_lm,
    mode,
    quant_threshold=0.01,
    v_threshold=0.05,
    enable_head_prune=False,
    head_prune_num=1,
    head_prune_start_layer=0,
    head_prune_interval=1,
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
        new_attn.token_prune_num = 0
        new_attn.next_active_token_indices = None
        new_attn.cumulative_token_score = None
        layer.self_attn = new_attn

    base_model.forward = spatten_qwen3_model_forward.__get__(base_model, Qwen3Model)
    return model.to(device=device, dtype=target_dtype).eval()


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3 generate benchmark template with unified prefill/decode/TTFT metrics."
    )
    parser.add_argument("--seq-lens", default="1024,2048,4096,8192")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--allow-remote-download", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--quant-threshold", type=float, default=0.01)
    parser.add_argument("--v-threshold", type=float, default=0.05)
    parser.add_argument("--enable-head-prune", action="store_true")
    parser.add_argument("--head-prune-num", type=int, default=1)
    parser.add_argument("--head-prune-start-layer", type=int, default=0)
    parser.add_argument("--head-prune-interval", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/route_b")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model_seq_generate_compare_qwen3_spatten.py")

    try:
        from transformers import AutoConfig
    except Exception as exc:
        raise RuntimeError("transformers is required for this comparison.") from exc

    device = "cuda"
    seq_lens = parse_int_list(args.seq_lens)
    max_seq_len = max(seq_lens)
    config = AutoConfig.from_pretrained(
        args.model_name,
        local_files_only=not args.allow_remote_download,
        trust_remote_code=True,
    )
    vocab_size = int(getattr(config, "vocab_size", 32000))
    prompt_map = {
        seq_len: build_exact_prompt_token_ids(vocab_size, seq_len, seed=20260419)
        for seq_len in seq_lens
    }

    print("Qwen3 Generate Compare (HF / vLLM / SpAtten)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model_name}")
    print("Metrics: prefill latency, 1-token decode latency, and TTFT")
    print(f"Head prune enabled: {args.enable_head_prune} (num={args.head_prune_num})")

    results = {}

    hf_model = load_hf_causal_lm(
        args.model_name,
        device=device,
        allow_remote_download=args.allow_remote_download,
    )
    for seq_len in seq_lens:
        results.setdefault(seq_len, {})["hf"] = benchmark_hf_generation_metrics(
            hf_model,
            prompt_map[seq_len],
            args.warmup,
            args.iters,
            reset_state_fn=None,
        )

    vllm = load_vllm_engine(
        args.model_name,
        max_model_len=max_seq_len + 8,
        gpu_memory_utilization=args.gpu_memory_utilization,
        allow_remote_download=args.allow_remote_download,
    )
    for seq_len in seq_lens:
        vllm_ms = benchmark_vllm_generate(vllm, prompt_map[seq_len], args.warmup, args.iters)
        results.setdefault(seq_len, {})["vllm"] = {
            "ttft_ms": round(float(vllm_ms), 6),
            "prefill_ms": None,
            "decode_1tok_ms": None,
            "decode_tok_per_s": None,
        }

    # Keep one HF model in memory and derive SpAtten variants from it sequentially.
    mode_to_label = {
        "quant_only": "spatten_quant",
        "v_prune_only": "spatten_v",
        "full": "spatten_full",
    }
    for mode in ["quant_only", "v_prune_only", "full"]:
        spatten_model = configure_spatten_qwen3_causal_lm(
            hf_model,
            mode,
            quant_threshold=args.quant_threshold,
            v_threshold=args.v_threshold,
            enable_head_prune=args.enable_head_prune,
            head_prune_num=args.head_prune_num,
            head_prune_start_layer=args.head_prune_start_layer,
            head_prune_interval=args.head_prune_interval,
        )
        for seq_len in seq_lens:
            metrics = benchmark_hf_generation_metrics(
                spatten_model,
                prompt_map[seq_len],
                args.warmup,
                args.iters,
                reset_state_fn=(lambda model=spatten_model: reset_spatten_states(model.model)) if args.enable_head_prune else None,
            )
            results.setdefault(seq_len, {})[mode_to_label[mode]] = metrics
        del spatten_model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print(
        f"{'Seq Len':<8} | {'HF-TTFT':<10} | {'vLLM-TTFT':<10} | "
        f"{'SpQ-TTFT':<10} | {'SpV-TTFT':<10} | {'SpF-TTFT':<10} | {'Best':<10}"
    )
    print("-" * 93)
    
    for seq_len in seq_lens:
        row = results[seq_len]
        candidates = {
            "hf": row["hf"]["ttft_ms"],
            "vllm": row["vllm"]["ttft_ms"],
            "sp_quant": row["spatten_quant"]["ttft_ms"],
            "sp_v": row["spatten_v"]["ttft_ms"],
            "sp_full": row["spatten_full"]["ttft_ms"],
        }
        best = min(candidates, key=candidates.get)
        row["best_variant"] = best
        print(
            f"{seq_len:<8} | "
            f"{row['hf']['ttft_ms']:<10.4f} | "
            f"{row['vllm']['ttft_ms']:<10.4f} | "
            f"{row['spatten_quant']['ttft_ms']:<10.4f} | "
            f"{row['spatten_v']['ttft_ms']:<10.4f} | "
            f"{row['spatten_full']['ttft_ms']:<10.4f} | "
            f"{best:<10}"
        )

    print("\nDecode-1tok Latency (HF / SpAtten)")
    print(
        f"{'Seq Len':<8} | {'HF-Dec':<10} | {'SpQ-Dec':<10} | "
        f"{'SpV-Dec':<10} | {'SpF-Dec':<10}"
    )
    print("-" * 63)
    for seq_len in seq_lens:
        row = results[seq_len]
        print(
            f"{seq_len:<8} | "
            f"{row['hf']['decode_1tok_ms']:<10.4f} | "
            f"{row['spatten_quant']['decode_1tok_ms']:<10.4f} | "
            f"{row['spatten_v']['decode_1tok_ms']:<10.4f} | "
            f"{row['spatten_full']['decode_1tok_ms']:<10.4f}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"model_seq_generate_compare_qwen3_spatten_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0),
        "model_name": args.model_name,
        "metric_template": {
            "prefill_ms": "Prompt forward latency with use_cache=True",
            "decode_1tok_ms": "Single cached decode step latency after prefill",
            "ttft_ms": "Time-to-first-token measured by generate(max_new_tokens=1)",
            "decode_tok_per_s": "1000 / decode_1tok_ms",
        },
        "seq_lens": seq_lens,
        "warmup": args.warmup,
        "iters": args.iters,
        "quant_threshold": args.quant_threshold,
        "v_threshold": args.v_threshold,
        "enable_head_prune": args.enable_head_prune,
        "head_prune_num": args.head_prune_num,
        "head_prune_start_layer": args.head_prune_start_layer,
        "head_prune_interval": args.head_prune_interval,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
