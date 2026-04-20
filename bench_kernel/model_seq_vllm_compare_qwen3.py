import os
import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def parse_int_list(raw_value):
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def build_exact_prompt_token_ids(vocab_size, seq_len, seed=0):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + seq_len)
    token_ids = torch.randint(
        low=10,
        high=max(11, vocab_size),
        size=(seq_len,),
        generator=generator,
        dtype=torch.long,
    )
    return token_ids.tolist()


def load_hf_causal_lm(model_name, device, allow_remote_download=False):
    try:
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        raise RuntimeError(
            "transformers is required for the Hugging Face baseline. "
            "Please run this script inside the project pixi environment."
        ) from exc

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=not allow_remote_download,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    except Exception as exc:
        if not allow_remote_download:
            raise RuntimeError(
                "The model is not present in the local Hugging Face cache. "
                "Re-run with --allow-remote-download after enabling network access."
            ) from exc
        raise
    return model.to(device).eval()


def benchmark_hf_generate(model, prompt_token_ids, warmup, iters):
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_token_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                min_new_tokens=1,
                do_sample=False,
                use_cache=True,
                pad_token_id=0,
            )
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    with torch.no_grad():
        for _ in range(iters):
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                min_new_tokens=1,
                do_sample=False,
                use_cache=True,
                pad_token_id=0,
            )
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iters


def load_vllm_engine(model_name, max_model_len, gpu_memory_utilization, allow_remote_download=False):
    try:
        from vllm import LLM
    except Exception as exc:
        raise RuntimeError(
            "vLLM is required for this comparison. Install it in the server environment "
            "before running this script."
        ) from exc

    # vLLM handles its own tokenizer / model loading. We use prompt_token_ids directly,
    # so skip_tokenizer_init avoids unnecessary tokenizer work.
    return LLM(
        model=model_name,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        download_dir=None if allow_remote_download else None,
    )


def benchmark_vllm_generate(llm, prompt_token_ids, warmup, iters):
    try:
        from vllm import SamplingParams
    except Exception as exc:
        raise RuntimeError("vLLM is required for this comparison.") from exc

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        min_tokens=1,
        ignore_eos=True,
    )

    prompts = [{"prompt_token_ids": prompt_token_ids}]

    for _ in range(warmup):
        _ = llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

    start_time = time.perf_counter()
    for _ in range(iters):
        _ = llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
    end_time = time.perf_counter()
    return (end_time - start_time) * 1000.0 / iters


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3 Hugging Face vs vLLM generation latency comparison (prefill + 1 token)."
    )
    parser.add_argument("--seq-lens", default="1024,2048,4096,8192")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--allow-remote-download", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--output-dir", default="artifacts/route_b")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model_seq_vllm_compare_qwen3.py")

    device = "cuda"
    seq_lens = parse_int_list(args.seq_lens)
    max_seq_len = max(seq_lens)

    try:
        from transformers import AutoConfig
    except Exception as exc:
        raise RuntimeError(
            "transformers is required for this comparison. "
            "Please run this script inside the project pixi environment."
        ) from exc

    try:
        config = AutoConfig.from_pretrained(
            args.model_name,
            local_files_only=not args.allow_remote_download,
            trust_remote_code=True,
        )
    except Exception as exc:
        if not args.allow_remote_download:
            raise RuntimeError(
                "The model config is not available in the local cache. "
                "Re-run with --allow-remote-download after enabling network access."
            ) from exc
        raise
    vocab_size = int(getattr(config, "vocab_size", 32000))
    prompt_map = {
        seq_len: build_exact_prompt_token_ids(vocab_size, seq_len, seed=20260419)
        for seq_len in seq_lens
    }

    print("Qwen3 HF vs vLLM Compare")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model_name}")
    print("Metric: prefill + 1 token generation latency")
    print(f"vLLM gpu_memory_utilization: {args.gpu_memory_utilization}")
    print(f"{'Seq Len':<8} | {'HF-Generate':<12} | {'vLLM':<12} | {'Speedup(vLLM)':<15}")
    print("-" * 66)

    results = {}

    hf_model = load_hf_causal_lm(
        args.model_name,
        device=device,
        allow_remote_download=args.allow_remote_download,
    )
    for seq_len in seq_lens:
        hf_ms = benchmark_hf_generate(
            hf_model,
            prompt_map[seq_len],
            warmup=args.warmup,
            iters=args.iters,
        )
        results.setdefault(seq_len, {})["hf_generate_ms"] = round(float(hf_ms), 6)

    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    llm = load_vllm_engine(
        args.model_name,
        max_model_len=max_seq_len + 8,
        gpu_memory_utilization=args.gpu_memory_utilization,
        allow_remote_download=args.allow_remote_download,
    )
    for seq_len in seq_lens:
        vllm_ms = benchmark_vllm_generate(
            llm,
            prompt_map[seq_len],
            warmup=args.warmup,
            iters=args.iters,
        )
        results.setdefault(seq_len, {})["vllm_ms"] = round(float(vllm_ms), 6)

    for seq_len in seq_lens:
        hf_ms = results[seq_len]["hf_generate_ms"]
        vllm_ms = results[seq_len]["vllm_ms"]
        speedup = hf_ms / vllm_ms if vllm_ms > 0 else float("inf")
        results[seq_len]["vllm_speedup_vs_hf"] = round(float(speedup), 6)
        print(f"{seq_len:<8} | {hf_ms:<12.4f} | {vllm_ms:<12.4f} | {speedup:<15.3f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"model_seq_vllm_compare_qwen3_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "device": torch.cuda.get_device_name(0),
        "model_name": args.model_name,
        "metric": "prefill_plus_1_token_generation_latency_ms",
        "seq_lens": seq_lens,
        "warmup": args.warmup,
        "iters": args.iters,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
