import time
from typing import Callable

import torch


def build_generation_inputs(prompt_token_ids, device):
    input_ids = torch.tensor([prompt_token_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _clone_attention_mask(attention_mask):
    return attention_mask.clone()


def benchmark_hf_prefill_with_cache(model, prompt_token_ids, warmup, iters, reset_state_fn: Callable | None = None):
    device = next(model.parameters()).device
    input_ids, attention_mask = build_generation_inputs(prompt_token_ids, device)
    model_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
        "return_dict": True,
    }

    with torch.no_grad():
        for _ in range(warmup):
            if reset_state_fn is not None:
                reset_state_fn()
            _ = model(**model_kwargs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    with torch.no_grad():
        for _ in range(iters):
            if reset_state_fn is not None:
                reset_state_fn()
            _ = model(**model_kwargs)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iters


def benchmark_hf_decode_step_with_cache(model, prompt_token_ids, warmup, iters, reset_state_fn: Callable | None = None):
    device = next(model.parameters()).device
    input_ids, attention_mask = build_generation_inputs(prompt_token_ids, device)

    def run_decode_once():
        if reset_state_fn is not None:
            reset_state_fn()
        prefill_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        next_token = torch.argmax(prefill_outputs.logits[:, -1, :], dim=-1, keepdim=True)
        decode_attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.size(0), 1), device=device, dtype=attention_mask.dtype)],
            dim=1,
        )
        return model(
            input_ids=next_token,
            attention_mask=decode_attention_mask,
            past_key_values=prefill_outputs.past_key_values,
            use_cache=True,
            return_dict=True,
        )

    with torch.no_grad():
        for _ in range(warmup):
            _ = run_decode_once()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    with torch.no_grad():
        for _ in range(iters):
            _ = run_decode_once()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iters


def benchmark_hf_generate_ttft(model, prompt_token_ids, warmup, iters, reset_state_fn: Callable | None = None):
    device = next(model.parameters()).device
    input_ids, attention_mask = build_generation_inputs(prompt_token_ids, device)
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": 1,
        "min_new_tokens": 1,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": 0,
    }

    with torch.no_grad():
        for _ in range(warmup):
            if reset_state_fn is not None:
                reset_state_fn()
            _ = model.generate(**generation_kwargs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    with torch.no_grad():
        for _ in range(iters):
            if reset_state_fn is not None:
                reset_state_fn()
            _ = model.generate(**generation_kwargs)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iters


def benchmark_hf_generation_metrics(model, prompt_token_ids, warmup, iters, reset_state_fn: Callable | None = None):
    prefill_ms = benchmark_hf_prefill_with_cache(
        model,
        prompt_token_ids,
        warmup=warmup,
        iters=iters,
        reset_state_fn=reset_state_fn,
    )
    decode_ms = benchmark_hf_decode_step_with_cache(
        model,
        prompt_token_ids,
        warmup=warmup,
        iters=iters,
        reset_state_fn=reset_state_fn,
    )
    ttft_ms = benchmark_hf_generate_ttft(
        model,
        prompt_token_ids,
        warmup=warmup,
        iters=iters,
        reset_state_fn=reset_state_fn,
    )
    return {
        "prefill_ms": round(float(prefill_ms), 6),
        "decode_1tok_ms": round(float(decode_ms), 6),
        "ttft_ms": round(float(ttft_ms), 6),
        "decode_tok_per_s": round(float(1000.0 / decode_ms), 6) if decode_ms > 0 else float("inf"),
    }
