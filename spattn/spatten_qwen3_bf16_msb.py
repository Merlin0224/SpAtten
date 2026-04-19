import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import copy

import torch
import triton
import triton.language as tl
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    create_causal_mask,
    create_sliding_window_causal_mask,
    apply_rotary_pos_emb,
    repeat_kv,
)

from spattn.spatten_bert_bf16_msb import prepare_progressive_k_bf16_msb
from module import compact_token_state


TRITON_META_DEFAULTS = {
    "progressive_qk": {
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "num_warps": 4,
        "num_stages": 1,
    },
    "v_prune": {
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "num_warps": 4,
        "num_stages": 2,
    },
    "ultimate": {
        "BLOCK_M": 32,
        "BLOCK_N": 64,
        "num_warps": 4,
        "num_stages": 2,
    },
}


def _make_triton_configs_qk(config_specs):
    return [
        triton.Config(
            {
                "BLOCK_M": spec["BLOCK_M"],
                "BLOCK_N": spec["BLOCK_N"],
            },
            num_warps=spec["num_warps"],
            num_stages=spec["num_stages"],
        )
        for spec in config_specs
    ]


def _make_triton_configs_v(config_specs):
    return [
        triton.Config(
            {
                "BLOCK_M": spec["BLOCK_M"],
                "BLOCK_N": spec["BLOCK_N"],
                "BLOCK_D": spec.get("BLOCK_D", 64),
            },
            num_warps=spec["num_warps"],
            num_stages=spec["num_stages"],
        )
        for spec in config_specs
    ]


QWEN3_AUTOTUNE_CONFIGS = {
    "progressive_qk": _make_triton_configs_qk(
        [
            {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 4, "num_stages": 1},
            {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1},
            {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4, "num_stages": 1},
            {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
            {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 8, "num_stages": 2},
        ]
    ),
    "v_prune": _make_triton_configs_v(
        [
            {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2},
            {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
            {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2},
            {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
            {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 8, "num_stages": 2},
        ]
    ),
    "ultimate": _make_triton_configs_qk(
        [
            {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2},
            {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
            {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2},
            {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
            {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 8, "num_stages": 2},
            {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1},
        ]
    ),
}

def _resolve_triton_meta(path_name, meta=None):
    config = dict(TRITON_META_DEFAULTS[path_name])
    if meta is None:
        return config

    unknown_keys = sorted(set(meta) - set(config))
    if unknown_keys:
        raise ValueError(f"Unknown Triton meta keys for {path_name}: {unknown_keys}")

    config.update(meta)
    return config


def load_qwen3_model_local_or_synthetic(
    device,
    model_name="Qwen/Qwen3-0.6B",
    max_position_embeddings=None,
    prefer_synthetic=True,
):
    if not prefer_synthetic:
        try:
            model = Qwen3Model.from_pretrained(model_name, local_files_only=True)
            source = f"local pretrained cache: {model_name}"
            return model.to(device).float().eval(), source
        except Exception:
            pass

    config_kwargs = {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "vocab_size": 32000,
    }
    if max_position_embeddings is not None:
        config_kwargs["max_position_embeddings"] = max_position_embeddings
    model = Qwen3Model(Qwen3Config(**config_kwargs))
    source = (
        f"synthetic Qwen3Config() weights "
        f"(max_position_embeddings={model.config.max_position_embeddings})"
    )
    return model.to(device).float().eval(), source


def build_inputs_local_or_synthetic(
    device,
    model,
    model_name="Qwen/Qwen3-0.6B",
    seq_len=32,
    exact_seq_len=False,
):
    text = "SpAtten validates sparse attention acceleration in a decoder-only long-context model."
    if exact_seq_len:
        input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
        attention_mask = torch.ones((1, seq_len), device=device, dtype=torch.long)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        source = f"synthetic exact-length token ids (seq_len={seq_len})"
        return inputs, source

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        source = f"local tokenizer cache: {model_name}"
    except Exception:
        input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
        attention_mask = torch.ones((1, seq_len), device=device, dtype=torch.long)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        source = f"synthetic token ids (seq_len={seq_len})"
    return inputs, source


@triton.jit
def _progressive_qk_causal_kernel(
    Q, K_MSB, K_LSB, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX_Q, N_CTX_K,
    sm_scale, threshold,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    d_model: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_b = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_b * stride_qz + off_h * stride_qh
    k_msb_base = K_MSB + off_b * stride_kz + off_h * stride_kh
    k_lsb_base = K_LSB + off_b * stride_kz + off_h * stride_kh
    v_base = V + off_b * stride_vz + off_h * stride_vh
    out_base = Out + off_b * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d_model)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    qi = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)
    neg_inf = -1.0e6

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, d_model], dtype=tl.float32)

    max_q = start_m * BLOCK_M + BLOCK_M - 1

    for j in range(0, tl.cdiv(N_CTX_K, BLOCK_N)):
        cur_n = j * BLOCK_N
        if cur_n <= max_q:
            offs_n_curr = cur_n + offs_n
            valid = (offs_n_curr[None, :] <= offs_m[:, None]) & (offs_n_curr[None, :] < N_CTX_K) & (
                offs_m[:, None] < N_CTX_Q
            )
            row_has_any = tl.sum(valid.to(tl.int32), axis=1) > 0

            k_msb_ptrs = k_msb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k_msb = tl.load(k_msb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0).to(tl.float32)
            qk = tl.dot(qi, tl.trans(k_msb)) * sm_scale
            qk = tl.where(valid, qk, neg_inf)

            max_score = tl.max(qk)
            if max_score < threshold:
                k_lsb_ptrs = k_lsb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
                k_lsb = tl.load(k_lsb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)
                qk_lsb = tl.dot(qi, tl.trans(k_lsb)) * sm_scale
                qk = tl.where(valid, qk + qk_lsb, neg_inf)

            m_ij = tl.max(qk, 1)
            safe_m_ij = tl.where(row_has_any, m_ij, 0.0)
            p = tl.where(valid, tl.exp(qk - safe_m_ij[:, None]), 0.0)
            l_ij = tl.sum(p, 1)

            m_next = tl.where(row_has_any, tl.maximum(m_i, m_ij), m_i)
            alpha = tl.where(row_has_any, tl.exp(m_i - m_next), 1.0)
            beta = tl.where(row_has_any, tl.exp(m_ij - m_next), 0.0)

            l_i = l_i * alpha + l_ij * beta
            acc = acc * alpha[:, None]

            v_ptrs = v_base + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)

            p_scaled = p * beta[:, None]
            acc = tl.dot(p_scaled.to(tl.float16), v.to(tl.float16), acc)
            m_i = m_next

    acc = acc / l_i[:, None]
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX_Q)


@triton.autotune(
    configs=QWEN3_AUTOTUNE_CONFIGS["progressive_qk"],
    key=["N_CTX_Q", "N_CTX_K"],
)
@triton.jit
def _progressive_qk_causal_autotuned_kernel(
    Q, K_MSB, K_LSB, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX_Q, N_CTX_K,
    sm_scale, threshold,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    d_model: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_b = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_b * stride_qz + off_h * stride_qh
    k_msb_base = K_MSB + off_b * stride_kz + off_h * stride_kh
    k_lsb_base = K_LSB + off_b * stride_kz + off_h * stride_kh
    v_base = V + off_b * stride_vz + off_h * stride_vh
    out_base = Out + off_b * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d_model)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    qi = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)
    neg_inf = -1.0e6

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, d_model], dtype=tl.float32)

    max_q = start_m * BLOCK_M + BLOCK_M - 1

    for j in range(0, tl.cdiv(N_CTX_K, BLOCK_N)):
        cur_n = j * BLOCK_N
        if cur_n <= max_q:
            offs_n_curr = cur_n + offs_n
            valid = (offs_n_curr[None, :] <= offs_m[:, None]) & (offs_n_curr[None, :] < N_CTX_K) & (
                offs_m[:, None] < N_CTX_Q
            )
            row_has_any = tl.sum(valid.to(tl.int32), axis=1) > 0

            k_msb_ptrs = k_msb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k_msb = tl.load(k_msb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0).to(tl.float32)
            qk = tl.dot(qi, tl.trans(k_msb)) * sm_scale
            qk = tl.where(valid, qk, neg_inf)

            max_score = tl.max(qk)
            if max_score < threshold:
                k_lsb_ptrs = k_lsb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
                k_lsb = tl.load(k_lsb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)
                qk_lsb = tl.dot(qi, tl.trans(k_lsb)) * sm_scale
                qk = tl.where(valid, qk + qk_lsb, neg_inf)

            m_ij = tl.max(qk, 1)
            safe_m_ij = tl.where(row_has_any, m_ij, 0.0)
            p = tl.where(valid, tl.exp(qk - safe_m_ij[:, None]), 0.0)
            l_ij = tl.sum(p, 1)

            m_next = tl.where(row_has_any, tl.maximum(m_i, m_ij), m_i)
            alpha = tl.where(row_has_any, tl.exp(m_i - m_next), 1.0)
            beta = tl.where(row_has_any, tl.exp(m_ij - m_next), 0.0)

            l_i = l_i * alpha + l_ij * beta
            acc = acc * alpha[:, None]

            v_ptrs = v_base + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)

            p_scaled = p * beta[:, None]
            acc = tl.dot(p_scaled.to(tl.float16), v.to(tl.float16), acc)
            m_i = m_next

    acc = acc / l_i[:, None]
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX_Q)


@triton.jit
def _spatten_v_prune_block_skip_causal_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX_Q, N_CTX_K,
    sm_scale, v_threshold,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_b = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_b * stride_qz + off_h * stride_qh
    k_base = K + off_b * stride_kz + off_h * stride_kh
    v_base = V + off_b * stride_vz + off_h * stride_vh
    out_base = Out + off_b * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    qi = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)
    neg_inf = -1.0e6

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    max_q = start_m * BLOCK_M + BLOCK_M - 1

    for j in range(0, tl.cdiv(N_CTX_K, BLOCK_N)):
        cur_n = j * BLOCK_N
        if cur_n <= max_q:
            offs_n_curr = cur_n + offs_n
            valid = (offs_n_curr[None, :] <= offs_m[:, None]) & (offs_n_curr[None, :] < N_CTX_K) & (
                offs_m[:, None] < N_CTX_Q
            )
            row_has_any = tl.sum(valid.to(tl.int32), axis=1) > 0

            k_ptrs = k_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k = tl.load(k_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)
            qk = tl.dot(qi, tl.trans(k)) * sm_scale
            qk = tl.where(valid, qk, neg_inf)

            m_ij = tl.max(qk, 1)
            safe_m_ij = tl.where(row_has_any, m_ij, 0.0)
            p = tl.where(valid, tl.exp(qk - safe_m_ij[:, None]), 0.0)

            m_next = tl.where(row_has_any, tl.maximum(m_i, m_ij), m_i)
            alpha = tl.where(row_has_any, tl.exp(m_i - m_next), 1.0)
            beta = tl.where(row_has_any, tl.exp(m_ij - m_next), 0.0)

            max_p_for_v = tl.max(p, axis=0)
            l_ij = tl.sum(p, 1)
            l_i_new = l_i * alpha + l_ij * beta
            acc_new = acc * alpha[:, None]

            skip_v = tl.max(max_p_for_v) <= v_threshold
            if skip_v:
                acc = acc_new
            else:
                v_load_mask = max_p_for_v > v_threshold
                v_mask_2d = v_load_mask[:, None] & (offs_n_curr[:, None] < N_CTX_K)
                v_ptrs = v_base + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk
                v = tl.load(v_ptrs, mask=v_mask_2d, other=0.0)
                p_pruned = tl.where(p > v_threshold, p, 0.0)
                p_scaled = p_pruned * beta[:, None]
                acc = tl.dot(p_scaled.to(tl.float16), v.to(tl.float16), acc_new)

            l_i = l_i_new
            m_i = m_next

    acc = acc / l_i[:, None]
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX_Q)


@triton.autotune(
    configs=QWEN3_AUTOTUNE_CONFIGS["v_prune"],
    key=["N_CTX_Q", "N_CTX_K"],
)
@triton.jit
def _spatten_v_prune_block_skip_causal_autotuned_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX_Q, N_CTX_K,
    sm_scale, v_threshold,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_b = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_b * stride_qz + off_h * stride_qh
    k_base = K + off_b * stride_kz + off_h * stride_kh
    v_base = V + off_b * stride_vz + off_h * stride_vh
    out_base = Out + off_b * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    qi = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)
    neg_inf = -1.0e6

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    max_q = start_m * BLOCK_M + BLOCK_M - 1

    for j in range(0, tl.cdiv(N_CTX_K, BLOCK_N)):
        cur_n = j * BLOCK_N
        if cur_n <= max_q:
            offs_n_curr = cur_n + offs_n
            valid = (offs_n_curr[None, :] <= offs_m[:, None]) & (offs_n_curr[None, :] < N_CTX_K) & (
                offs_m[:, None] < N_CTX_Q
            )
            row_has_any = tl.sum(valid.to(tl.int32), axis=1) > 0

            k_ptrs = k_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k = tl.load(k_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)
            qk = tl.dot(qi, tl.trans(k)) * sm_scale
            qk = tl.where(valid, qk, neg_inf)

            m_ij = tl.max(qk, 1)
            safe_m_ij = tl.where(row_has_any, m_ij, 0.0)
            p = tl.where(valid, tl.exp(qk - safe_m_ij[:, None]), 0.0)

            m_next = tl.where(row_has_any, tl.maximum(m_i, m_ij), m_i)
            alpha = tl.where(row_has_any, tl.exp(m_i - m_next), 1.0)
            beta = tl.where(row_has_any, tl.exp(m_ij - m_next), 0.0)

            max_p_for_v = tl.max(p, axis=0)
            l_ij = tl.sum(p, 1)
            l_i_new = l_i * alpha + l_ij * beta
            acc_new = acc * alpha[:, None]

            skip_v = tl.max(max_p_for_v) <= v_threshold
            if skip_v:
                acc = acc_new
            else:
                v_load_mask = max_p_for_v > v_threshold
                v_mask_2d = v_load_mask[:, None] & (offs_n_curr[:, None] < N_CTX_K)
                v_ptrs = v_base + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk
                v = tl.load(v_ptrs, mask=v_mask_2d, other=0.0)
                p_pruned = tl.where(p > v_threshold, p, 0.0)
                p_scaled = p_pruned * beta[:, None]
                acc = tl.dot(p_scaled.to(tl.float16), v.to(tl.float16), acc_new)

            l_i = l_i_new
            m_i = m_next

    acc = acc / l_i[:, None]
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX_Q)


@triton.jit
def _spatten_fused_ultimate_causal_kernel(
    Q, K_MSB, K_LSB, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX_Q, N_CTX_K,
    sm_scale, quant_threshold, v_threshold,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    d_model: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_b = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_b * stride_qz + off_h * stride_qh
    k_msb_base = K_MSB + off_b * stride_kz + off_h * stride_kh
    k_lsb_base = K_LSB + off_b * stride_kz + off_h * stride_kh
    v_base = V + off_b * stride_vz + off_h * stride_vh
    out_base = Out + off_b * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d_model)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    qi = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)
    neg_inf = -1.0e6

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, d_model], dtype=tl.float32)

    max_q = start_m * BLOCK_M + BLOCK_M - 1

    for j in range(0, tl.cdiv(N_CTX_K, BLOCK_N)):
        cur_n = j * BLOCK_N
        if cur_n <= max_q:
            offs_n_curr = cur_n + offs_n
            valid = (offs_n_curr[None, :] <= offs_m[:, None]) & (offs_n_curr[None, :] < N_CTX_K) & (
                offs_m[:, None] < N_CTX_Q
            )
            row_has_any = tl.sum(valid.to(tl.int32), axis=1) > 0

            k_msb_ptrs = k_msb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k_msb = tl.load(k_msb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0).to(tl.float32)
            qk = tl.dot(qi, tl.trans(k_msb)) * sm_scale
            qk = tl.where(valid, qk, neg_inf)

            max_score = tl.max(qk)
            if max_score < quant_threshold:
                k_lsb_ptrs = k_lsb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
                k_lsb = tl.load(k_lsb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)
                qk_lsb = tl.dot(qi, tl.trans(k_lsb)) * sm_scale
                qk = tl.where(valid, qk + qk_lsb, neg_inf)

            m_ij = tl.max(qk, 1)
            safe_m_ij = tl.where(row_has_any, m_ij, 0.0)
            p = tl.where(valid, tl.exp(qk - safe_m_ij[:, None]), 0.0)

            m_next = tl.where(row_has_any, tl.maximum(m_i, m_ij), m_i)
            alpha = tl.where(row_has_any, tl.exp(m_i - m_next), 1.0)
            beta = tl.where(row_has_any, tl.exp(m_ij - m_next), 0.0)

            max_p_for_v = tl.max(p, axis=0)
            l_ij = tl.sum(p, 1)
            l_i_new = l_i * alpha + l_ij * beta
            acc_new = acc * alpha[:, None]

            skip_v = tl.max(max_p_for_v) <= v_threshold
            if skip_v:
                acc = acc_new
            else:
                v_load_mask = max_p_for_v > v_threshold
                v_mask_2d = v_load_mask[:, None] & (offs_n_curr[:, None] < N_CTX_K)
                v_ptrs = v_base + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk
                v = tl.load(v_ptrs, mask=v_mask_2d, other=0.0)
                p_pruned = tl.where(p > v_threshold, p, 0.0)
                p_scaled = p_pruned * beta[:, None]
                acc = tl.dot(p_scaled.to(tl.float16), v.to(tl.float16), acc_new)

            l_i = l_i_new
            m_i = m_next

    acc = acc / l_i[:, None]
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX_Q)


@triton.autotune(
    configs=QWEN3_AUTOTUNE_CONFIGS["ultimate"],
    key=["N_CTX_Q", "N_CTX_K"],
)
@triton.jit
def _spatten_fused_ultimate_causal_autotuned_kernel(
    Q, K_MSB, K_LSB, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX_Q, N_CTX_K,
    sm_scale, quant_threshold, v_threshold,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    d_model: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_b = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_b * stride_qz + off_h * stride_qh
    k_msb_base = K_MSB + off_b * stride_kz + off_h * stride_kh
    k_lsb_base = K_LSB + off_b * stride_kz + off_h * stride_kh
    v_base = V + off_b * stride_vz + off_h * stride_vh
    out_base = Out + off_b * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d_model)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    qi = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)
    neg_inf = -1.0e6

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, d_model], dtype=tl.float32)

    max_q = start_m * BLOCK_M + BLOCK_M - 1

    for j in range(0, tl.cdiv(N_CTX_K, BLOCK_N)):
        cur_n = j * BLOCK_N
        if cur_n <= max_q:
            offs_n_curr = cur_n + offs_n
            valid = (offs_n_curr[None, :] <= offs_m[:, None]) & (offs_n_curr[None, :] < N_CTX_K) & (
                offs_m[:, None] < N_CTX_Q
            )
            row_has_any = tl.sum(valid.to(tl.int32), axis=1) > 0

            k_msb_ptrs = k_msb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k_msb = tl.load(k_msb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0).to(tl.float32)
            qk = tl.dot(qi, tl.trans(k_msb)) * sm_scale
            qk = tl.where(valid, qk, neg_inf)

            max_score = tl.max(qk)
            if max_score < quant_threshold:
                k_lsb_ptrs = k_lsb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
                k_lsb = tl.load(k_lsb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)
                qk_lsb = tl.dot(qi, tl.trans(k_lsb)) * sm_scale
                qk = tl.where(valid, qk + qk_lsb, neg_inf)

            m_ij = tl.max(qk, 1)
            safe_m_ij = tl.where(row_has_any, m_ij, 0.0)
            p = tl.where(valid, tl.exp(qk - safe_m_ij[:, None]), 0.0)

            m_next = tl.where(row_has_any, tl.maximum(m_i, m_ij), m_i)
            alpha = tl.where(row_has_any, tl.exp(m_i - m_next), 1.0)
            beta = tl.where(row_has_any, tl.exp(m_ij - m_next), 0.0)

            max_p_for_v = tl.max(p, axis=0)
            l_ij = tl.sum(p, 1)
            l_i_new = l_i * alpha + l_ij * beta
            acc_new = acc * alpha[:, None]

            skip_v = tl.max(max_p_for_v) <= v_threshold
            if skip_v:
                acc = acc_new
            else:
                v_load_mask = max_p_for_v > v_threshold
                v_mask_2d = v_load_mask[:, None] & (offs_n_curr[:, None] < N_CTX_K)
                v_ptrs = v_base + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk
                v = tl.load(v_ptrs, mask=v_mask_2d, other=0.0)
                p_pruned = tl.where(p > v_threshold, p, 0.0)
                p_scaled = p_pruned * beta[:, None]
                acc = tl.dot(p_scaled.to(tl.float16), v.to(tl.float16), acc_new)

            l_i = l_i_new
            m_i = m_next

    acc = acc / l_i[:, None]
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX_Q)


def triton_progressive_qk_causal(q, k_msb, k_lsb, v, threshold, sm_scale, meta=None, use_autotune=False):
    z, h, m, d = q.shape
    _, _, n, _ = k_msb.shape
    out = torch.empty_like(q)

    if use_autotune and meta is None:
        grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]), z * h)
        _progressive_qk_causal_autotuned_kernel[grid](
            q, k_msb, k_lsb, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k_msb.stride(0), k_msb.stride(1), k_msb.stride(2), k_msb.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            z, h, m, n,
            sm_scale, threshold,
            d_model=d,
        )
        return out

    config = _resolve_triton_meta("progressive_qk", meta)
    block_m = config["BLOCK_M"]
    block_n = config["BLOCK_N"]
    grid = (triton.cdiv(m, block_m), z * h)

    _progressive_qk_causal_kernel[grid](
        q, k_msb, k_lsb, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_msb.stride(0), k_msb.stride(1), k_msb.stride(2), k_msb.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        z, h, m, n,
        sm_scale, threshold,
        BLOCK_M=block_m, BLOCK_N=block_n, d_model=d,
        num_warps=config["num_warps"], num_stages=config["num_stages"],
    )
    return out


def triton_fused_spatten_v_prune_block_skip_causal(q, k, v, v_threshold, sm_scale, meta=None, use_autotune=False):
    z, h, m, d = q.shape
    _, _, n, _ = k.shape
    out = torch.empty_like(q)

    if use_autotune and meta is None:
        grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]), z * h)
        _spatten_v_prune_block_skip_causal_autotuned_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            z, h, m, n,
            sm_scale, v_threshold,
        )
        return out

    config = _resolve_triton_meta("v_prune", meta)
    block_m = config["BLOCK_M"]
    block_n = config["BLOCK_N"]
    grid = (triton.cdiv(m, block_m), z * h)

    _spatten_v_prune_block_skip_causal_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        z, h, m, n,
        sm_scale, v_threshold,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=d,
        num_warps=config["num_warps"], num_stages=config["num_stages"],
    )
    return out


def triton_fused_spatten_ultimate_causal(
    q,
    k_msb,
    k_lsb,
    v,
    quant_threshold,
    v_threshold,
    sm_scale,
    meta=None,
    use_autotune=False,
):
    z, h, m, d = q.shape
    _, _, n, _ = k_msb.shape
    out = torch.empty_like(q)

    if use_autotune and meta is None:
        grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]), z * h)
        _spatten_fused_ultimate_causal_autotuned_kernel[grid](
            q, k_msb, k_lsb, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k_msb.stride(0), k_msb.stride(1), k_msb.stride(2), k_msb.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            z, h, m, n,
            sm_scale, quant_threshold, v_threshold,
            d_model=d,
        )
        return out

    config = _resolve_triton_meta("ultimate", meta)
    block_m = config["BLOCK_M"]
    block_n = config["BLOCK_N"]
    grid = (triton.cdiv(m, block_m), z * h)

    _spatten_fused_ultimate_causal_kernel[grid](
        q, k_msb, k_lsb, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_msb.stride(0), k_msb.stride(1), k_msb.stride(2), k_msb.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        z, h, m, n,
        sm_scale, quant_threshold, v_threshold,
        BLOCK_M=block_m, BLOCK_N=block_n, d_model=d,
        num_warps=config["num_warps"], num_stages=config["num_stages"],
    )
    return out


class SpattenQwen3Attention(Qwen3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.num_heads = config.num_attention_heads
        self.enable_prog_quant = False
        self.enable_v_prune = False
        self.quant_threshold = 0.01
        self.v_threshold = 0.05
        self.enable_head_prune = False
        self.head_prune_num = 0
        self.head_prune_start_layer = 0
        self.head_prune_interval = 1
        self.enable_token_prune = False
        self.token_prune_num = 0
        self.token_prune_start_layer = 0
        self.token_prune_interval = 1
        self.cumulative_token_score = None
        self.next_active_token_indices = None
        self.active_head_indices_for_this_layer = None
        self.next_active_head_indices = None
        self.progressive_qk_meta = None
        self.v_prune_meta = None
        self.ultimate_meta = None
        self.enable_triton_autotune = False
        self.token_block_size = 0
        self.token_recent_keep = 128
        self.token_prefix_keep = 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.active_head_indices_for_this_layer is not None:
            active_head_indices = self.active_head_indices_for_this_layer
        else:
            active_head_indices = torch.arange(self.num_heads, device=hidden_states.device)
        cur_heads = active_head_indices.numel()

        if cur_heads != self.num_heads:
            query_states = query_states.index_select(1, active_head_indices)
            key_states = key_states.index_select(1, active_head_indices)
            value_states = value_states.index_select(1, active_head_indices)

        if self.enable_prog_quant and self.enable_v_prune:
            k_msb, k_lsb = prepare_progressive_k_bf16_msb(key_states)
            context_layer = triton_fused_spatten_ultimate_causal(
                query_states,
                k_msb,
                k_lsb,
                value_states,
                quant_threshold=self.quant_threshold,
                v_threshold=self.v_threshold,
                sm_scale=self.scaling,
                meta=self.ultimate_meta,
                use_autotune=self.enable_triton_autotune,
            )
        elif self.enable_prog_quant:
            k_msb, k_lsb = prepare_progressive_k_bf16_msb(key_states)
            context_layer = triton_progressive_qk_causal(
                query_states,
                k_msb,
                k_lsb,
                value_states,
                threshold=self.quant_threshold,
                sm_scale=self.scaling,
                meta=self.progressive_qk_meta,
                use_autotune=self.enable_triton_autotune,
            )
        elif self.enable_v_prune:
            context_layer = triton_fused_spatten_v_prune_block_skip_causal(
                query_states,
                key_states,
                value_states,
                v_threshold=self.v_threshold,
                sm_scale=self.scaling,
                meta=self.v_prune_meta,
                use_autotune=self.enable_triton_autotune,
            )
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            context_layer = attn_output

        current_token_importance = context_layer.abs().mean(dim=(1, 3))
        head_importance = context_layer.abs().mean(dim=(0, 2, 3))
        next_head_indices = active_head_indices
        head_prune_active = (
            self.enable_head_prune
            and self.layer_idx >= self.head_prune_start_layer
            and ((self.layer_idx - self.head_prune_start_layer) % max(1, self.head_prune_interval) == 0)
        )
        if head_prune_active and cur_heads > self.head_prune_num:
            if self.head_prune_num == 1:
                drop_idx = torch.argmin(head_importance)
                keep_mask = torch.ones(cur_heads, device=hidden_states.device, dtype=torch.bool)
                keep_mask[drop_idx] = False
                next_head_indices = active_head_indices[keep_mask]
            else:
                keep_k = max(1, cur_heads - self.head_prune_num)
                _, topk_indices = torch.topk(head_importance, k=keep_k)
                next_head_indices = torch.sort(active_head_indices[topk_indices]).values
        self.next_active_head_indices = next_head_indices

        if not self.enable_token_prune or self.token_prune_num <= 0:
            self.next_active_token_indices = None
            self.cumulative_token_score = None
        else:
            if self.cumulative_token_score is None:
                self.cumulative_token_score = current_token_importance
            else:
                self.cumulative_token_score = self.cumulative_token_score + current_token_importance

            token_prune_active = (
                self.layer_idx >= self.token_prune_start_layer
                and ((self.layer_idx - self.token_prune_start_layer) % max(1, self.token_prune_interval) == 0)
            )
            next_token_indices = None
            seq_len = current_token_importance.size(1)
            if token_prune_active and seq_len > self.token_prune_num:
                prefix_keep = min(self.token_prefix_keep, seq_len)
                remaining_after_prefix = max(0, seq_len - prefix_keep)
                recent_keep = min(self.token_recent_keep, remaining_after_prefix)
                candidate_start = prefix_keep
                candidate_end = seq_len - recent_keep

                prefix_indices = None
                if prefix_keep > 0:
                    prefix_indices = torch.arange(prefix_keep, device=hidden_states.device).unsqueeze(0)

                recent_indices = None
                if recent_keep > 0:
                    recent_indices = torch.arange(
                        seq_len - recent_keep,
                        seq_len,
                        device=hidden_states.device,
                    ).unsqueeze(0)

                if candidate_end <= candidate_start:
                    parts = [part for part in (prefix_indices, recent_indices) if part is not None]
                    next_token_indices = torch.cat(parts, dim=1) if parts else None
                elif self.token_block_size and self.token_block_size > 1:
                    block_size = self.token_block_size
                    candidate_scores = self.cumulative_token_score[:, candidate_start:candidate_end]
                    candidate_len = candidate_scores.size(1)
                    padded = ((candidate_len + block_size - 1) // block_size) * block_size
                    if padded != candidate_len:
                        pad_width = padded - candidate_len
                        padded_scores = torch.nn.functional.pad(
                            candidate_scores,
                            (0, pad_width),
                            value=torch.finfo(self.cumulative_token_score.dtype).max,
                        )
                    else:
                        padded_scores = candidate_scores
                    num_blocks = padded // block_size
                    block_scores = padded_scores.view(padded_scores.size(0), num_blocks, block_size).mean(dim=-1)
                    prune_blocks = min(max(1, self.token_prune_num), max(0, num_blocks - 1))
                    keep_blocks = max(1, num_blocks - prune_blocks)
                    _, topk_blocks = torch.topk(block_scores, k=keep_blocks, dim=1, largest=True)
                    sorted_blocks = torch.sort(topk_blocks, dim=1).values
                    block_offsets = torch.arange(block_size, device=hidden_states.device).view(1, 1, block_size)
                    expanded = sorted_blocks.unsqueeze(-1) * block_size + block_offsets + candidate_start
                    candidate_indices = expanded.view(expanded.size(0), -1)
                    candidate_indices = candidate_indices[:, candidate_indices[0] < candidate_end]
                    parts = [part for part in (prefix_indices, candidate_indices, recent_indices) if part is not None]
                    next_token_indices = torch.cat(parts, dim=1) if parts else None
                else:
                    candidate_scores = self.cumulative_token_score[:, candidate_start:candidate_end]
                    candidate_len = candidate_scores.size(1)
                    keep_k = max(1, candidate_len - self.token_prune_num)
                    _, topk_indices = torch.topk(candidate_scores, k=keep_k, dim=1)
                    candidate_indices = torch.sort(topk_indices + candidate_start, dim=1).values
                    parts = [part for part in (prefix_indices, candidate_indices, recent_indices) if part is not None]
                    next_token_indices = torch.cat(parts, dim=1) if parts else None
                self.cumulative_token_score = None
            self.next_active_token_indices = next_token_indices

        full_context = torch.zeros(
            input_shape[0],
            input_shape[1],
            self.num_heads,
            self.head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        full_context[:, :, active_head_indices, :] = context_layer.transpose(1, 2)
        attn_output = full_context.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


def configure_spatten_qwen3_model(
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
    token_prune_start_layer=0,
    token_prune_interval=1,
    token_block_size=0,
    token_recent_keep=128,
    token_prefix_keep=1,
    enable_triton_autotune=False,
    meta_overrides=None,
):
    model = copy.deepcopy(base_model)
    device = next(base_model.parameters()).device
    for layer_idx, layer in enumerate(model.layers):
        orig_state = layer.self_attn.state_dict()
        new_attn = SpattenQwen3Attention(model.config, layer_idx=layer_idx)
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
        new_attn.token_prune_start_layer = token_prune_start_layer
        new_attn.token_prune_interval = token_prune_interval
        new_attn.token_block_size = token_block_size
        new_attn.token_recent_keep = token_recent_keep
        new_attn.token_prefix_keep = token_prefix_keep
        new_attn.enable_triton_autotune = enable_triton_autotune
        if meta_overrides:
            new_attn.progressive_qk_meta = meta_overrides.get("progressive_qk")
            new_attn.v_prune_meta = meta_overrides.get("v_prune")
            new_attn.ultimate_meta = meta_overrides.get("ultimate")
        layer.self_attn = new_attn
    model.forward = spatten_qwen3_model_forward.__get__(model, Qwen3Model)
    return model.to(device).float().eval()


def spatten_qwen3_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    **kwargs,
):
    if use_cache and any(getattr(layer.self_attn, "enable_token_prune", False) for layer in self.layers):
        raise ValueError("Qwen3 minimal token pruning currently supports prefill-only execution (use_cache=False).")

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

    hidden_states = inputs_embeds
    active_head_indices = None
    active_token_indices = None
    cumulative_token_score = None

    if isinstance(attention_mask, dict):
        if any(getattr(layer.self_attn, "enable_token_prune", False) for layer in self.layers):
            raise ValueError("Qwen3 token pruning does not support dict attention_mask inputs in the current minimal path.")
        base_attention_mask = attention_mask
    else:
        base_attention_mask = attention_mask
        if base_attention_mask is None:
            base_attention_mask = torch.ones(
                (hidden_states.size(0), hidden_states.size(1)),
                device=hidden_states.device,
                dtype=torch.long,
            )

    def _gather_2d(source, keep_indices):
        expanded = keep_indices.unsqueeze(-1).expand(-1, -1, source.size(-1))
        return torch.gather(source, 1, expanded)

    for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        if active_token_indices is not None:
            hidden_states, _, cumulative_token_score = compact_token_state(
                hidden_states,
                active_token_indices,
                attention_mask=None,
                cumulative_token_score=cumulative_token_score,
            )
            position_ids = _gather_2d(position_ids.unsqueeze(-1), active_token_indices).squeeze(-1)
            if not isinstance(base_attention_mask, dict) and base_attention_mask is not None:
                base_attention_mask = _gather_2d(base_attention_mask.unsqueeze(-1), active_token_indices).squeeze(-1)
            active_token_indices = None

        if isinstance(base_attention_mask, dict):
            causal_mask_mapping = base_attention_mask
        else:
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": hidden_states,
                "attention_mask": base_attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        decoder_layer.self_attn.active_head_indices_for_this_layer = active_head_indices
        decoder_layer.self_attn.cumulative_token_score = cumulative_token_score
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[self.config.layer_types[i]],
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        active_head_indices = decoder_layer.self_attn.next_active_head_indices
        active_token_indices = decoder_layer.self_attn.next_active_token_indices
        cumulative_token_score = decoder_layer.self_attn.cumulative_token_score

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SpAtten Qwen3 BF16-MSB Running on: {device}")

    original_model, model_source = load_qwen3_model_local_or_synthetic(device)
    spatten_model = configure_spatten_qwen3_model(original_model, "full")
    inputs, input_source = build_inputs_local_or_synthetic(device, original_model)

    print(f"Model source: {model_source}")
    print(f"Input source: {input_source}")
    print(f"Original Sequence Length: {inputs['input_ids'].shape[1]}")

    print("\n[Step 1] Baseline Validation (Sparse Paths OFF)...")
    off_model = configure_spatten_qwen3_model(original_model, "baseline")
    with torch.no_grad():
        orig_out = original_model(**inputs).last_hidden_state
        off_out = off_model(**inputs).last_hidden_state
    print(f"Max Difference: {(orig_out - off_out).abs().max().item():.6f} (Should be very small)")

    print("\n[Step 2] Enabling Qwen3 SpAtten Full Path...")
    with torch.no_grad():
        final_out = spatten_model(**inputs).last_hidden_state
    print(f"Final Hidden States Shape: {list(final_out.shape)}")


if __name__ == "__main__":
    main()
