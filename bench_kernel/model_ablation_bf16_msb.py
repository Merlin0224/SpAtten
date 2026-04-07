import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import torch

from benchmark.benchmark_bf16_msb import benchmark_model, reset_spatten_states
from spattn.spatten_bert_bf16_msb import (
    SpattenBertSelfAttention,
    build_inputs_local_or_synthetic,
    load_bert_model_local_or_synthetic,
    spatten_encoder_forward,
)
from transformers.models.bert.modeling_bert import BertEncoder


def configure_spatten_model(
    base_model,
    mode,
    token_prune_num,
    head_prune_num,
    enable_head_prune=False,
    enable_token_prune=False,
):
    spatten_model = copy.deepcopy(base_model)
    for layer in spatten_model.encoder.layer:
        orig_state = layer.attention.self.state_dict()
        new_attn = SpattenBertSelfAttention(spatten_model.config)
        new_attn.load_state_dict(orig_state)

        new_attn.enable_head_prune = enable_head_prune
        new_attn.head_prune_num = head_prune_num

        new_attn.enable_token_prune = enable_token_prune
        new_attn.token_prune_num = token_prune_num

        new_attn.enable_prog_quant = mode in {"quant_only", "full"}
        new_attn.enable_v_prune = mode in {"v_prune_only", "full"}
        new_attn.quant_threshold = 0.05
        new_attn.v_prune_num = 2

        layer.attention.self = new_attn

    spatten_model.encoder.forward = spatten_encoder_forward.__get__(spatten_model.encoder, BertEncoder)
    spatten_model.to(next(base_model.parameters()).device).float().eval()
    reset_spatten_states(spatten_model)
    return spatten_model


def run_variant(
    base_model,
    inputs,
    mode,
    token_prune_num,
    head_prune_num,
    warmup,
    iters,
    enable_head_prune=False,
    enable_token_prune=False,
):
    if mode == "baseline":
        ms = benchmark_model(base_model, inputs, num_iters=iters, warmup=warmup, reset_state=False)
        return {
            "ms": round(float(ms), 6),
            "mode": mode,
        }

    spatten_model = configure_spatten_model(
        base_model,
        mode,
        token_prune_num,
        head_prune_num,
        enable_head_prune=enable_head_prune,
        enable_token_prune=enable_token_prune,
    )
    ms = benchmark_model(spatten_model, inputs, num_iters=iters, warmup=warmup, reset_state=True)
    return {
        "ms": round(float(ms), 6),
        "mode": mode,
    }


def recommend(results):
    baseline_ms = results["baseline"]["ms"]
    best_mode = min((name for name in results if name != "baseline"), key=lambda name: results[name]["ms"])
    best_ms = results[best_mode]["ms"]

    if best_mode == "full":
        return "Full end-to-end path is the fastest SpAtten variant; next step should target the fused kernel plus Python scheduling overhead together."
    if best_mode == "quant_only":
        return "Quant-only is the fastest end-to-end SpAtten variant; next step should keep focusing on the BF16-MSB progressive quantization path."
    if best_mode == "v_prune_only":
        return "V-prune-only is the fastest end-to-end SpAtten variant; next step should simplify the fused path so it inherits the standalone V-prune gains."
    if best_ms > baseline_ms:
        return "All SpAtten variants are still slower than baseline end-to-end; next step should focus on model-level overhead, not only the kernel."
    return "Results are mixed; next step should compare per-layer scheduling and tensor slicing overhead."


def main():
    parser = argparse.ArgumentParser(description="Route-B paper BF16-MSB model-level ablation benchmark.")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--token-prune-num", type=int, default=1)
    parser.add_argument("--head-prune-num", type=int, default=1)
    parser.add_argument("--enable-head-prune", action="store_true")
    parser.add_argument("--enable-token-prune", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/route_b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for route_b_model_ablation_paper_bf16_msb.py")

    base_model, model_source = load_bert_model_local_or_synthetic(
        device,
        max_position_embeddings=max(512, args.seq_len + 2),
    )
    inputs, input_source = build_inputs_local_or_synthetic(
        device,
        base_model,
        seq_len=args.seq_len,
        exact_seq_len=True,
    )

    results = {}
    for mode in ["baseline", "quant_only", "v_prune_only", "full"]:
        results[mode] = run_variant(
            base_model,
            inputs,
            mode,
            args.token_prune_num,
            args.head_prune_num,
            args.warmup,
            args.iters,
            enable_head_prune=args.enable_head_prune,
            enable_token_prune=args.enable_token_prune,
        )

    baseline_ms = results["baseline"]["ms"]
    for mode, item in results.items():
        item["slowdown_vs_baseline"] = round(item["ms"] / baseline_ms, 6)

    recommendation = recommend(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"model_ablation_{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0),
        "model_source": model_source,
        "input_source": input_source,
        "seq_len": args.seq_len,
        "warmup": args.warmup,
        "iters": args.iters,
        "token_prune_num": args.token_prune_num,
        "head_prune_num": args.head_prune_num,
        "enable_head_prune": args.enable_head_prune,
        "enable_token_prune": args.enable_token_prune,
        "results": results,
        "recommendation": recommendation,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Route-B Model Ablation (Paper BF16-MSB)")
    print(f"Device: {payload['device']}")
    print(f"Model source: {model_source}")
    print(f"Input source: {input_source}")
    print(f"Seq Len: {args.seq_len}")
    print(f"Head prune enabled: {args.enable_head_prune} (num={args.head_prune_num})")
    print(f"Token prune enabled: {args.enable_token_prune} (num={args.token_prune_num})")
    for mode in ["baseline", "quant_only", "v_prune_only", "full"]:
        item = results[mode]
        print(
            f"{mode}: ms={item['ms']:.4f} | slowdown_vs_baseline={item['slowdown_vs_baseline']:.3f}x"
        )
    print(f"Recommendation: {recommendation}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
