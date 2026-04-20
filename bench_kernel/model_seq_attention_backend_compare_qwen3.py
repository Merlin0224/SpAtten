import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from benchmark.benchmark_bf16_msb import benchmark_model
from bench_kernel.model_seq_vllm_compare_qwen3 import parse_int_list
from spattn.spatten_qwen3_bf16_msb import (
    build_inputs_local_or_synthetic,
    configure_spatten_qwen3_model,
)


def load_qwen3_dense_model(device, model_name, seq_len, attn_implementation, allow_local_pretrained):
    try:
        from transformers import Qwen3Config, Qwen3Model
    except Exception as exc:
        raise RuntimeError("transformers Qwen3 backend is required for attention backend comparison.") from exc

    if allow_local_pretrained:
        model = Qwen3Model.from_pretrained(
            model_name,
            local_files_only=True,
            attn_implementation=attn_implementation,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        source = f"local pretrained: {model_name} ({attn_implementation})"
        return model.to(device).eval(), source

    config = Qwen3Config(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=12,
        vocab_size=32000,
        max_position_embeddings=max(4096, seq_len + 2),
    )
    config._attn_implementation = attn_implementation
    model = Qwen3Model(config)
    source = f"synthetic Qwen3Config() ({attn_implementation})"
    return model.to(device).float().eval(), source


def run_dense_variant(model_name, device, seq_len, attn_implementation, warmup, iters, allow_local_pretrained):
    model, model_source = load_qwen3_dense_model(
        device=device,
        model_name=model_name,
        seq_len=seq_len,
        attn_implementation=attn_implementation,
        allow_local_pretrained=allow_local_pretrained,
    )
    inputs, input_source = build_inputs_local_or_synthetic(
        device,
        model,
        model_name=model_name,
        seq_len=seq_len,
        exact_seq_len=True,
    )
    ms = benchmark_model(model, inputs, num_iters=iters, warmup=warmup, reset_state=False)
    return {
        "ms": round(float(ms), 6),
        "model_source": model_source,
        "input_source": input_source,
    }


def run_spatten_variant(
    base_model,
    model_name,
    device,
    seq_len,
    mode,
    warmup,
    iters,
    quant_threshold,
    v_threshold,
    enable_head_prune,
    head_prune_num,
    head_prune_start_layer,
    head_prune_interval,
    enable_spatten_sdpa_prefill,
):
    inputs, input_source = build_inputs_local_or_synthetic(
        device,
        base_model,
        model_name=model_name,
        seq_len=seq_len,
        exact_seq_len=True,
    )
    model = configure_spatten_qwen3_model(
        base_model,
        mode,
        quant_threshold=quant_threshold,
        v_threshold=v_threshold,
        enable_head_prune=enable_head_prune,
        head_prune_num=head_prune_num,
        head_prune_start_layer=head_prune_start_layer,
        head_prune_interval=head_prune_interval,
        enable_prefill_sdpa_fast_path=enable_spatten_sdpa_prefill,
        enable_token_prune=False,
        token_prune_num=0,
    )
    ms = benchmark_model(
        model,
        inputs,
        num_iters=iters,
        warmup=warmup,
        reset_state=enable_head_prune,
    )
    return {
        "ms": round(float(ms), 6),
        "input_source": input_source,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3 attention-only backend comparison: dense eager/sdpa/flash vs SpAtten."
    )
    parser.add_argument("--seq-lens", default="1024,2048,4096,8192")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--allow-local-pretrained", action="store_true")
    parser.add_argument("--quant-threshold", type=float, default=0.01)
    parser.add_argument("--v-threshold", type=float, default=0.05)
    parser.add_argument("--enable-head-prune", action="store_true")
    parser.add_argument("--head-prune-num", type=int, default=1)
    parser.add_argument("--head-prune-start-layer", type=int, default=0)
    parser.add_argument("--head-prune-interval", type=int, default=1)
    parser.add_argument("--enable-spatten-sdpa-prefill", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/route_b")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model_seq_attention_backend_compare_qwen3.py")

    device = "cuda"
    seq_lens = parse_int_list(args.seq_lens)

    base_model, base_model_source = load_qwen3_dense_model(
        device=device,
        model_name=args.model_name,
        seq_len=max(seq_lens),
        attn_implementation="eager",
        allow_local_pretrained=args.allow_local_pretrained,
    )

    dense_backends = [
        ("dense_eager", "eager"),
        ("dense_sdpa", "sdpa"),
        ("dense_flash", "flash_attention_2"),
    ]
    spatten_modes = [
        ("spatten_quant", "quant_only"),
        ("spatten_v", "v_prune_only"),
        ("spatten_full", "full"),
    ]

    print("Qwen3 Attention Backend Compare")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Base model source: {base_model_source}")
    print(f"Head prune enabled: {args.enable_head_prune} (num={args.head_prune_num})")
    print(f"SpAtten SDPA prefill fast path: {args.enable_spatten_sdpa_prefill}")
    print(
        f"{'Seq Len':<8} | {'Dense-Eager':<12} | {'Dense-SDPA':<12} | {'Dense-Flash':<12} | "
        f"{'Sp-Quant':<12} | {'Sp-V':<12} | {'Sp-Full':<12} | {'Best':<12} | {'SpBest/SDPA':<12}"
    )
    print("-" * 128)

    all_results = []
    for seq_len in seq_lens:
        record = {
            "seq_len": seq_len,
            "dense": {},
            "spatten": {},
        }

        dense_candidates = {}
        for label, impl in dense_backends:
            try:
                result = run_dense_variant(
                    model_name=args.model_name,
                    device=device,
                    seq_len=seq_len,
                    attn_implementation=impl,
                    warmup=args.warmup,
                    iters=args.iters,
                    allow_local_pretrained=args.allow_local_pretrained,
                )
                record["dense"][label] = result
                dense_candidates[label] = result["ms"]
            except Exception as exc:
                record["dense"][label] = {"ms": None, "error": str(exc)}

        spatten_candidates = {}
        for label, mode in spatten_modes:
            result = run_spatten_variant(
                base_model=base_model,
                model_name=args.model_name,
                device=device,
                seq_len=seq_len,
                mode=mode,
                warmup=args.warmup,
                iters=args.iters,
                quant_threshold=args.quant_threshold,
                v_threshold=args.v_threshold,
                enable_head_prune=args.enable_head_prune,
                head_prune_num=args.head_prune_num,
                head_prune_start_layer=args.head_prune_start_layer,
                head_prune_interval=args.head_prune_interval,
                enable_spatten_sdpa_prefill=args.enable_spatten_sdpa_prefill,
            )
            record["spatten"][label] = result
            spatten_candidates[label] = result["ms"]

        candidates = {**dense_candidates, **spatten_candidates}
        best_variant = min(candidates, key=candidates.get)
        spatten_best_variant = min(spatten_candidates, key=spatten_candidates.get)
        dense_sdpa_ms = dense_candidates.get("dense_sdpa")
        spatten_vs_sdpa = (
            (spatten_candidates[spatten_best_variant] / dense_sdpa_ms)
            if dense_sdpa_ms is not None
            else None
        )
        record["best_variant"] = best_variant
        record["spatten_best_variant"] = spatten_best_variant
        record["spatten_vs_sdpa"] = None if spatten_vs_sdpa is None else round(float(spatten_vs_sdpa), 6)
        all_results.append(record)

        def _fmt(section, key):
            value = record[section][key]["ms"]
            return "n/a" if value is None else f"{value:.4f}"

        print(
            f"{seq_len:<8} | "
            f"{_fmt('dense', 'dense_eager'):<12} | "
            f"{_fmt('dense', 'dense_sdpa'):<12} | "
            f"{_fmt('dense', 'dense_flash'):<12} | "
            f"{_fmt('spatten', 'spatten_quant'):<12} | "
            f"{_fmt('spatten', 'spatten_v'):<12} | "
            f"{_fmt('spatten', 'spatten_full'):<12} | "
            f"{best_variant:<12} | "
            f"{('n/a' if spatten_vs_sdpa is None else f'{spatten_vs_sdpa * 100:.1f}%'):<12}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"model_seq_attention_backend_compare_qwen3_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0),
        "model_name": args.model_name,
        "base_model_source": base_model_source,
        "seq_lens": seq_lens,
        "warmup": args.warmup,
        "iters": args.iters,
        "allow_local_pretrained": args.allow_local_pretrained,
        "quant_threshold": args.quant_threshold,
        "v_threshold": args.v_threshold,
        "enable_head_prune": args.enable_head_prune,
        "head_prune_num": args.head_prune_num,
        "head_prune_start_layer": args.head_prune_start_layer,
        "head_prune_interval": args.head_prune_interval,
        "enable_spatten_sdpa_prefill": args.enable_spatten_sdpa_prefill,
        "results": all_results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
