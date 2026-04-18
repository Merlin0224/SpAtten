import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from transformers.models.bert.modeling_bert import BertSelfAttention, BertModel, BertEncoder
from transformers import BertConfig, BertTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

import triton
import triton.language as tl

from module import spatten_encoder_forward

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
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "num_warps": 4,
        "num_stages": 1,
    },
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


def load_bert_model_local_or_synthetic(device, model_name="bert-base-uncased", max_position_embeddings=None):
    try:
        if max_position_embeddings is not None and max_position_embeddings > 512:
            raise ValueError("synthetic model requested for extended max_position_embeddings")
        model = BertModel.from_pretrained(model_name, local_files_only=True)
        source = f"local pretrained cache: {model_name}"
    except Exception:
        config_kwargs = {}
        if max_position_embeddings is not None:
            config_kwargs["max_position_embeddings"] = max_position_embeddings
        model = BertModel(BertConfig(**config_kwargs))
        source = f"synthetic BertConfig() weights (max_position_embeddings={model.config.max_position_embeddings})"
    return model.to(device).float().eval(), source


def split_fp32_to_bf16_msb_residual(x):
    if x.dtype != torch.float32:
        x = x.float()

    x_msb = x.to(torch.bfloat16).contiguous()
    x_lsb = (x - x_msb.float()).contiguous()
    return x_msb, x_lsb


def prepare_progressive_k_bf16_msb(key_layer):
    return split_fp32_to_bf16_msb_residual(key_layer)


def build_inputs_local_or_synthetic(
    device,
    model,
    model_name="bert-base-uncased",
    seq_len=32,
    exact_seq_len=False,
):
    text = "SpAtten is an algorithm-architecture co-design that leverages token, head, and quantization sparsity."
    if exact_seq_len:
        input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
        attention_mask = torch.ones((1, seq_len), device=device, dtype=torch.long)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        source = f"synthetic exact-length token ids (seq_len={seq_len})"
        return inputs, source

    try:
        tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        source = f"local tokenizer cache: {model_name}"
    except Exception:
        input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
        attention_mask = torch.ones((1, seq_len), device=device, dtype=torch.long)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        source = f"synthetic token ids (seq_len={seq_len})"
    return inputs, source


@triton.jit
def _block_argmin_kernel(
    in_ptr,
    out_val_ptr,
    out_idx_ptr,
    length,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < length
    vals = tl.load(in_ptr + offs, mask=mask, other=float("inf"))
    min_val = tl.min(vals, axis=0)
    is_min = vals == min_val
    candidate = tl.where(is_min, offs, length)
    min_idx = tl.min(candidate, axis=0)
    tl.store(out_val_ptr + pid, min_val)
    tl.store(out_idx_ptr + pid, min_idx)


def triton_argmin_1d(x, block_size=256):
    if x.dim() != 1:
        raise ValueError(f"triton_argmin_1d expects a 1D tensor, got shape {list(x.shape)}")
    if not x.is_cuda:
        return torch.argmin(x)

    length = x.numel()
    num_blocks = triton.cdiv(length, block_size)
    block_min_vals = torch.empty((num_blocks,), device=x.device, dtype=x.dtype)
    block_min_idxs = torch.empty((num_blocks,), device=x.device, dtype=torch.int32)

    _block_argmin_kernel[(num_blocks,)](
        x,
        block_min_vals,
        block_min_idxs,
        length,
        BLOCK=block_size,
        num_warps=4,
        num_stages=1,
    )
    best_block = torch.argmin(block_min_vals)
    return block_min_idxs[best_block].to(torch.long)

# =====================================================================
# Triton 渐进式量化 Kernel (Progressive Quantization) 和 FlashAttention
# =====================================================================
@triton.jit
def _progressive_qk_kernel(
    Q, K_MSB, K_LSB, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX_Q, N_CTX_K,
    sm_scale, threshold,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    d_model: tl.constexpr
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

    # 加载 Q (形状:[BLOCK_M, d_model])
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    qi = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, d_model], dtype=tl.float32)

    for j in range(0, tl.cdiv(N_CTX_K, BLOCK_N)):
        cur_n = j * BLOCK_N
        offs_n_curr = cur_n + offs_n

        # 激进加载 K_MSB (形状为[BLOCK_N, d_model])
        k_msb_ptrs = k_msb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_msb = tl.load(k_msb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0).to(tl.float32)

        # qk:[BLOCK_M, d_model] @ [d_model, BLOCK_N] = [BLOCK_M, BLOCK_N]
        qk = tl.dot(qi, tl.trans(k_msb)) * sm_scale

        #  渐进式分支预测
        max_score = tl.max(qk)
        if max_score < threshold:
            k_lsb_ptrs = k_lsb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k_lsb = tl.load(k_lsb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)
            qk += tl.dot(qi, tl.trans(k_lsb)) * sm_scale

        # Online Softmax 逻辑
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        m_next = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(m_ij - m_next)

        l_i = l_i * alpha + l_ij * beta
        acc = acc * alpha[:, None]

        # 加载 V 并累加输出 (形状为 [BLOCK_N, d_model])
        v_ptrs = v_base + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)

        p_scaled = p * beta[:, None]
        acc = tl.dot(p_scaled.to(tl.float16), v.to(tl.float16), acc)

        m_i = m_next

    # 归一化并写回显存
    acc = acc / l_i[:, None]
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX_Q)


@triton.jit
def _progressive_qk_stats_kernel(
    Q, K_MSB, K_LSB, V, Out, Out_Sum,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX_Q, N_CTX_K,
    sm_scale, threshold,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    d_model: tl.constexpr
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

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, d_model], dtype=tl.float32)
    row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)

    for j in range(0, tl.cdiv(N_CTX_K, BLOCK_N)):
        cur_n = j * BLOCK_N
        offs_n_curr = cur_n + offs_n

        k_msb_ptrs = k_msb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_msb = tl.load(k_msb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0).to(tl.float32)

        qk = tl.dot(qi, tl.trans(k_msb)) * sm_scale

        max_score = tl.max(qk)
        if max_score < threshold:
            k_lsb_ptrs = k_lsb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k_lsb = tl.load(k_lsb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)
            qk += tl.dot(qi, tl.trans(k_lsb)) * sm_scale

        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        m_next = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(m_ij - m_next)

        row_sum += l_ij * beta
        l_i = l_i * alpha + l_ij * beta
        acc = acc * alpha[:, None]

        v_ptrs = v_base + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)

        p_scaled = p * beta[:, None]
        acc = tl.dot(p_scaled.to(tl.float16), v.to(tl.float16), acc)

        m_i = m_next

    acc = acc / l_i[:, None]
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    sum_ptrs = Out_Sum + off_b * H * N_CTX_Q + off_h * N_CTX_Q + offs_m
    tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX_Q)
    tl.store(sum_ptrs, row_sum, mask=offs_m < N_CTX_Q)


@triton.jit
def _spatten_v_prune_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX_Q, N_CTX_K,
    sm_scale, v_threshold, # 传入 V 剪枝的动态阈值
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

    # 1. 从 HBM 加载到 Q 块到 SRAM
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    qi = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # 在 K 和 V 的序列长度上循环
    for j in range(0, tl.cdiv(N_CTX_K, BLOCK_N)):
        cur_n = j * BLOCK_N
        offs_n_curr = cur_n + offs_n

        # 2. 加载 K 块
        k_ptrs = k_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)

        # 3. 计算局部 Attention Score (QK^T)
        qk = tl.dot(qi, tl.trans(k)) * sm_scale

        m_ij = tl.max(qk, 1)
        m_next = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(m_ij - m_next)

        p = tl.exp(qk - m_next[:, None])

         # -------------------------------------------------------------
        # SRAM 内闭环的局部 V 物理剪枝 (按需访存)
        # -------------------------------------------------------------
        # 探测当前 Block 内每个 V 对应的最大概率
        max_p_for_v = tl.max(p, axis=0) #[BLOCK_N]
        
        l_ij = tl.sum(p, 1)
        l_i_new = l_i * alpha + l_ij * beta
        acc_new = acc * alpha[:, None]

        # 计算是否要跳过整个 V Block 
        skip_v = tl.max(max_p_for_v) <= v_threshold

        if skip_v:
            # 优化A：整个 Block 没用，拒绝读 V！直接继承衰减后的 acc
            acc = acc_new
        else:
            # 优化B：细粒度掩码生成
            v_load_mask = max_p_for_v > v_threshold
            v_mask_2d = v_load_mask[:, None] & (offs_n_curr[:, None] < N_CTX_K)
            
            # 物理拒绝读取！DRAM 只回传有用的 V 向量数据
            v_ptrs = v_base + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=v_mask_2d, other=0.0)

            # 概率也施加剪枝阈值，将废弃 V 对应的概率彻底抹零
            p_pruned = tl.where(p > v_threshold, p, 0.0)
            p_scaled = p_pruned * beta[:, None]
            
            acc = tl.dot(p_scaled.to(tl.float16), v.to(tl.float16), acc_new)

        l_i = l_i_new
        m_i = m_next
    # 5. 归一化并写回 HBM
    acc = acc / l_i[:, None]
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX_Q)

@triton.jit
def _spatten_fused_ultimate_kernel(
    Q, K_MSB, K_LSB, V, Out, Out_Sum,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX_Q, N_CTX_K, 
    sm_scale, quant_threshold, v_threshold,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    d_model: tl.constexpr
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

    # 1. 加载 Q (形状:[BLOCK_M, d_model])
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    qi = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, d_model], dtype=tl.float32)
    row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)

    for j in range(0, tl.cdiv(N_CTX_K, BLOCK_N)):
        cur_n = j * BLOCK_N
        offs_n_curr = cur_n + offs_n

        # 2. 激进加载 K_MSB 
        k_msb_ptrs = k_msb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_msb = tl.load(k_msb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0).to(tl.float32)

        # qk = Q @ K_NSB^T
        qk = tl.dot(qi, tl.trans(k_msb)) * sm_scale

        # ---------------------------------------------------------
        # 特性 1：渐进式量化 (Progressive Quantization on K)
        # ---------------------------------------------------------
        max_score = tl.max(qk)
        if max_score < quant_threshold:
            k_lsb_ptrs = k_lsb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k_lsb = tl.load(k_lsb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)
            qk += tl.dot(qi, tl.trans(k_lsb)) * sm_scale
        
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None]) # 此时 p 的区间是 (0, 1]
        
        m_next = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(m_ij - m_next)
        row_sum += tl.sum(p, 1) * beta
        # ---------------------------------------------------------
        # 特性 2：局部 V 向量按需物理剪枝 (Local V Pruning)
        # ---------------------------------------------------------
        # 探测当前 Block 内每个 V 对应的最大概率
        max_p_for_v = tl.max(p, axis=0) #[BLOCK_N]
        
        l_ij = tl.sum(p, 1)
        l_i_new = l_i * alpha + l_ij * beta
        acc_new = acc * alpha[:, None]

        # 计算是否要跳过整个 V Block
        skip_v = tl.max(max_p_for_v) <= v_threshold

        if skip_v:
            # 优化A：整个 Block 没用，拒绝读 V！直接继承衰减后的 acc
            acc = acc_new
        else:
            # 优化B：细粒度掩码生成
            v_load_mask = max_p_for_v > v_threshold
            v_mask_2d = v_load_mask[:, None] & (offs_n_curr[:, None] < N_CTX_K)
            
            # 物理拒绝读取！DRAM 只回传有用的 V 向量数据
            v_ptrs = v_base + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=v_mask_2d, other=0.0)

            # 概率也施加剪枝阈值，将废弃 V 对应的概率彻底抹零
            p_pruned = tl.where(p > v_threshold, p, 0.0)
            p_scaled = p_pruned * beta[:, None]

            acc = tl.dot(p_scaled.to(tl.float16), v.to(tl.float16), acc_new)

        l_i = l_i_new
        m_i = m_next
    
    # 归一化并写回显存
    acc = acc / l_i[:, None]
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    sum_ptrs = Out_Sum + off_b * H * N_CTX_Q + off_h * N_CTX_Q + offs_m
    tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX_Q)
    tl.store(sum_ptrs, row_sum, mask=offs_m < N_CTX_Q)


@triton.jit
def _spatten_fused_ultimate_nostats_kernel(
    Q, K_MSB, K_LSB, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX_Q, N_CTX_K,
    sm_scale, quant_threshold, v_threshold,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    d_model: tl.constexpr
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

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, d_model], dtype=tl.float32)

    for j in range(0, tl.cdiv(N_CTX_K, BLOCK_N)):
        cur_n = j * BLOCK_N
        offs_n_curr = cur_n + offs_n

        k_msb_ptrs = k_msb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_msb = tl.load(k_msb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0).to(tl.float32)

        qk = tl.dot(qi, tl.trans(k_msb)) * sm_scale

        max_score = tl.max(qk)
        if max_score < quant_threshold:
            k_lsb_ptrs = k_lsb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k_lsb = tl.load(k_lsb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)
            qk += tl.dot(qi, tl.trans(k_lsb)) * sm_scale

        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])

        m_next = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(m_ij - m_next)

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
def _spatten_v_prune_block_skip_kernel(
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

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for j in range(0, tl.cdiv(N_CTX_K, BLOCK_N)):
        cur_n = j * BLOCK_N
        offs_n_curr = cur_n + offs_n

        k_ptrs = k_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)

        qk = tl.dot(qi, tl.trans(k)) * sm_scale

        m_ij = tl.max(qk, 1)
        m_next = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(m_ij - m_next)

        p = tl.exp(qk - m_next[:, None])
        max_p_for_v = tl.max(p, axis=0)

        l_ij = tl.sum(p, 1)
        l_i_new = l_i * alpha + l_ij * beta
        acc_new = acc * alpha[:, None]

        skip_v = tl.max(max_p_for_v) <= v_threshold
        if skip_v:
            acc = acc_new
        else:
            v_ptrs = v_base + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)
            p_scaled = p * beta[:, None]
            acc = tl.dot(p_scaled.to(tl.float16), v.to(tl.float16), acc_new)

        l_i = l_i_new
        m_i = m_next

    acc = acc / l_i[:, None]
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX_Q)


@triton.jit
def _spatten_v_prune_block_skip_stats_kernel(
    Q, K, V, Out, Out_Sum,
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

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)

    for j in range(0, tl.cdiv(N_CTX_K, BLOCK_N)):
        cur_n = j * BLOCK_N
        offs_n_curr = cur_n + offs_n

        k_ptrs = k_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)

        qk = tl.dot(qi, tl.trans(k)) * sm_scale

        m_ij = tl.max(qk, 1)
        m_next = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(m_ij - m_next)

        p = tl.exp(qk - m_next[:, None])
        max_p_for_v = tl.max(p, axis=0)

        l_ij = tl.sum(p, 1)
        row_sum += l_ij
        l_i_new = l_i * alpha + l_ij * beta
        acc_new = acc * alpha[:, None]

        skip_v = tl.max(max_p_for_v) <= v_threshold
        if skip_v:
            acc = acc_new
        else:
            v_ptrs = v_base + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)
            p_scaled = p * beta[:, None]
            acc = tl.dot(p_scaled.to(tl.float16), v.to(tl.float16), acc_new)

        l_i = l_i_new
        m_i = m_next

    acc = acc / l_i[:, None]
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    sum_ptrs = Out_Sum + off_b * H * N_CTX_Q + off_h * N_CTX_Q + offs_m
    tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX_Q)
    tl.store(sum_ptrs, row_sum, mask=offs_m < N_CTX_Q)


@triton.jit
def _spatten_fused_ultimate_block_skip_nostats_kernel(
    Q, K_MSB, K_LSB, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX_Q, N_CTX_K,
    sm_scale, quant_threshold, v_threshold,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    d_model: tl.constexpr
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

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, d_model], dtype=tl.float32)

    for j in range(0, tl.cdiv(N_CTX_K, BLOCK_N)):
        cur_n = j * BLOCK_N
        offs_n_curr = cur_n + offs_n

        k_msb_ptrs = k_msb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_msb = tl.load(k_msb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0).to(tl.float32)

        qk = tl.dot(qi, tl.trans(k_msb)) * sm_scale

        max_score = tl.max(qk)
        if max_score < quant_threshold:
            k_lsb_ptrs = k_lsb_base + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k_lsb = tl.load(k_lsb_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)
            qk += tl.dot(qi, tl.trans(k_lsb)) * sm_scale

        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])

        m_next = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(m_ij - m_next)

        max_p_for_v = tl.max(p, axis=0)
        l_ij = tl.sum(p, 1)
        l_i_new = l_i * alpha + l_ij * beta
        acc_new = acc * alpha[:, None]

        skip_v = tl.max(max_p_for_v) <= v_threshold
        if skip_v:
            acc = acc_new
        else:
            v_ptrs = v_base + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)
            p_scaled = p * beta[:, None]
            acc = tl.dot(p_scaled.to(tl.float16), v.to(tl.float16), acc_new)

        l_i = l_i_new
        m_i = m_next

    acc = acc / l_i[:, None]
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX_Q)



def triton_progressive_qk(q, k_msb, k_lsb, v, threshold, sm_scale, meta=None, out_sum=None, collect_stats=False):
    Z, H, M, D = q.shape
    _, _, N, _ = k_msb.shape
    out = torch.empty_like(q)

    config = _resolve_triton_meta("progressive_qk", meta)
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    grid = (triton.cdiv(M, BLOCK_M), Z * H)

    if collect_stats:
        if out_sum is None:
            raise ValueError("out_sum must be provided when collect_stats=True for triton_progressive_qk")
        _progressive_qk_stats_kernel[grid](
            q, k_msb, k_lsb, v, out, out_sum,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k_msb.stride(0), k_msb.stride(1), k_msb.stride(2), k_msb.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            Z, H, M, N,
            sm_scale, threshold,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, d_model=D,
            num_warps=config["num_warps"], num_stages=config["num_stages"]
        )
    else:
        _progressive_qk_kernel[grid](
            q, k_msb, k_lsb, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k_msb.stride(0), k_msb.stride(1), k_msb.stride(2), k_msb.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            Z, H, M, N,
            sm_scale, threshold,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, d_model=D,
            num_warps=config["num_warps"], num_stages=config["num_stages"]
        )
    return out


def triton_fused_spatten_v_prune(q, k, v, v_threshold, sm_scale, meta=None):
    Z, H, M, D = q.shape
    _, _, N, _ = k.shape
    out = torch.empty_like(q)

    config = _resolve_triton_meta("v_prune", meta)
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    grid = (triton.cdiv(M, BLOCK_M), Z * H)

    _spatten_v_prune_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, M, N,
        sm_scale, v_threshold,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D,
        num_warps=config["num_warps"], num_stages=config["num_stages"]
    )
    return out

def triton_fused_spatten_v_prune_block_skip(q, k, v, v_threshold, sm_scale, meta=None, out_sum=None, collect_stats=False):
    Z, H, M, D = q.shape
    _, _, N, _ = k.shape
    out = torch.empty_like(q)

    config = _resolve_triton_meta("v_prune", meta)
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    grid = (triton.cdiv(M, BLOCK_M), Z * H)

    if collect_stats:
        if out_sum is None:
            raise ValueError("out_sum must be provided when collect_stats=True for triton_fused_spatten_v_prune_block_skip")
        _spatten_v_prune_block_skip_stats_kernel[grid](
            q, k, v, out, out_sum,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            Z, H, M, N,
            sm_scale, v_threshold,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D,
            num_warps=config["num_warps"], num_stages=config["num_stages"]
        )
    else:
        _spatten_v_prune_block_skip_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            Z, H, M, N,
            sm_scale, v_threshold,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D,
            num_warps=config["num_warps"], num_stages=config["num_stages"]
        )
    return out

def triton_fused_spatten_ultimate(
    q,
    k_msb,
    k_lsb,
    v,
    out_sum,
    quant_threshold,
    v_threshold,
    sm_scale,
    meta=None,
    collect_stats=False,
):
    Z, H, M, D = q.shape
    _, _, N, _ = k_msb.shape
    out = torch.empty_like(q)

    config = _resolve_triton_meta("ultimate", meta)
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    grid = (triton.cdiv(M, BLOCK_M), Z * H)

    if collect_stats:
        _spatten_fused_ultimate_kernel[grid](
            q, k_msb, k_lsb, v, out, out_sum,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k_msb.stride(0), k_msb.stride(1), k_msb.stride(2), k_msb.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            Z, H, M, N,
            sm_scale, quant_threshold, v_threshold,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, d_model=D,
            num_warps=config["num_warps"], num_stages=config["num_stages"]
        )
    else:
        _spatten_fused_ultimate_nostats_kernel[grid](
            q, k_msb, k_lsb, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k_msb.stride(0), k_msb.stride(1), k_msb.stride(2), k_msb.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            Z, H, M, N,
            sm_scale, quant_threshold, v_threshold,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, d_model=D,
            num_warps=config["num_warps"], num_stages=config["num_stages"]
        )
    return out


def triton_fused_spatten_ultimate_block_skip(
    q,
    k_msb,
    k_lsb,
    v,
    out_sum,
    quant_threshold,
    v_threshold,
    sm_scale,
    meta=None,
    collect_stats=False,
):
    if collect_stats:
        raise NotImplementedError("block-skip fused kernel is currently benchmark-only (collect_stats=False).")

    Z, H, M, D = q.shape
    _, _, N, _ = k_msb.shape
    out = torch.empty_like(q)

    config = _resolve_triton_meta("ultimate", meta)
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    grid = (triton.cdiv(M, BLOCK_M), Z * H)

    _spatten_fused_ultimate_block_skip_nostats_kernel[grid](
        q, k_msb, k_lsb, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_msb.stride(0), k_msb.stride(1), k_msb.stride(2), k_msb.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, M, N,
        sm_scale, quant_threshold, v_threshold,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, d_model=D,
        num_warps=config["num_warps"], num_stages=config["num_stages"]
    )
    return out

# ========================================================
# Spatten Attention Module
# ========================================================

class SpattenBertSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size // config.num_attention_heads)

        # Head 剪枝控制
        self.head_prune_num = 0 # 本层要剪掉的 Head 数量
        self.enable_head_prune = False
        self.next_active_head_indices = None # 用于级联传递的 Buffer
        self.active_head_indices_for_this_layer = None

        # Token 剪枝控制
        self.enable_token_prune = False
        self.token_prune_start_layer = 0
        self.token_prune_interval = 1
        self.token_prune_num = 0 # 本层要剪掉的Token数量
        self.cumulative_token_score = None # 累积的Token重要性分数，用于级联传递 [Batch, SeqLen]
        self.next_active_token_indices = None # 下一层要保留的Token索引
        self.enable_delayed_token_compaction = False
        self.token_compact_interval = 1
        self.token_compact_min_drop_ratio = 1.0
        self.graph_capture_mode = False
        self.enable_token_stage_pruning = False
        self.token_stage_size = 1
        self.token_stage_weighting = "uniform"
        self.stage_token_score_accum = None
        self.stage_token_score_count = 0
        self.stage_token_weight_total = 0.0

        # 局部 V 剪枝控制
        self.enable_v_prune = False # 是否启用局部 Value 剪枝
        self.v_prune_num = 2 # 每个Head内部要剪掉的Value维度数量

        # 渐近式量化控制
        self.enable_prog_quant = False
        self.quant_threshold = 1.0
        self.v_threshold = 0.05
        self.head_prune_start_layer = 0
        self.head_prune_interval = 1
        self.layer_idx = -1
    
    def transpose_for_out(self, x, n_heads):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        bs, seq_len, _ = x.shape
        x = x.view(bs, seq_len, n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3) # [B, N, S, D]
  
    
    def forward(
        self, hidden_states, attention_mask=None, **kwargs
    ):
        device = hidden_states.device
        bs, seq_len, _ = hidden_states.shape

        # 1. 级联掩码生效：提取当前活跃的 Heads
        if self.active_head_indices_for_this_layer is not None:
            active_head_indices = self.active_head_indices_for_this_layer
        else:
            active_head_indices = torch.arange(self.num_heads, device=device)
        cur_heads = active_head_indices.size(0)

        # 切片获取活跃的 Q, K, V 权重
        q_proj = self.query(hidden_states)
        k_proj = self.key(hidden_states)
        v_proj = self.value(hidden_states)

        query_layer = self.transpose_for_out(q_proj, self.num_heads)
        key_layer = self.transpose_for_out(k_proj, self.num_heads)
        value_layer = self.transpose_for_out(v_proj, self.num_heads)

        if cur_heads != self.num_heads:
            query_layer = query_layer.index_select(1, active_head_indices)
            key_layer = key_layer.index_select(1, active_head_indices)
            value_layer = value_layer.index_select(1, active_head_indices)
        
        sm_scale = 1.0 / math.sqrt(self.head_dim)
        attention_probs = None
        context_layer = None
        op = True

        # ====================================================================
        # 核心算子路由：决定走哪条加速路径
        # ====================================================================
        if self.enable_prog_quant and self.enable_v_prune:
            # 【终极融合路径】同时开启渐进式量化和局部V剪枝
            k_msb, k_lsb = prepare_progressive_k_bf16_msb(key_layer)
            
            # 准备一个专门放概率和的 buffer
            out_sum = torch.zeros((bs, cur_heads, seq_len), device=device, dtype=torch.float32)
            
            context_layer = triton_fused_spatten_ultimate(
                query_layer, k_msb, k_lsb, value_layer, out_sum, # 传入 buffer
                quant_threshold=self.quant_threshold,
                v_threshold=self.v_threshold,
                sm_scale=sm_scale,
                collect_stats=True,
            )
            current_token_importance = out_sum.sum(dim=1) # [B, S_Q]
            op = False

        elif self.enable_prog_quant:
            # 【仅开启渐进式量化】
            k_msb, k_lsb = prepare_progressive_k_bf16_msb(key_layer)
            out_sum = torch.zeros((bs, cur_heads, seq_len), device=device, dtype=torch.float32)
            context_layer = triton_progressive_qk(
                query_layer, k_msb, k_lsb, value_layer,
                self.quant_threshold, sm_scale,
                out_sum=out_sum,
                collect_stats=True,
            )
            current_token_importance = out_sum.sum(dim=1)
            op = False

        elif self.enable_v_prune and self.v_prune_num > 0:
            # 【仅开启局部 V 剪枝】
            out_sum = torch.zeros((bs, cur_heads, seq_len), device=device, dtype=torch.float32)
            context_layer = triton_fused_spatten_v_prune_block_skip(
                query_layer, key_layer, value_layer,
                v_threshold=self.v_threshold, sm_scale=sm_scale,
                out_sum=out_sum,
                collect_stats=True,
            )
            current_token_importance = out_sum.sum(dim=1)
            op = False

        else:
            # 【原生 PyTorch 路径】没有任何硬件级优化
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2) * sm_scale)
            if attention_mask is not None:
                if attention_mask.dim() > attention_scores.dim():
                    attention_mask = attention_mask.squeeze(1)
                attention_scores = attention_scores + attention_mask

            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, value_layer)

        # ====================================================================
        # 级联剪枝决策逻辑 (Head & Token Pruning)
        # ====================================================================
        # 此时的 attention_probs (不论是原生还是近似) 都会用于 Token Pruning
        # context_layer 将用于 Head Pruning
        if self.graph_capture_mode:
            self.next_active_head_indices = None
            self.next_active_token_indices = None
            self.cumulative_token_score = None
            self.stage_token_score_accum = None
            self.stage_token_score_count = 0
            self.stage_token_weight_total = 0.0
        else:
            if op:
                current_token_importance = attention_probs.sum(dim=(1, 2))

            # Head Pruning 决策
            importance_dims = (0, 2, 3) if context_layer.dim() == 4 else (1, 2)
            head_importance = context_layer.abs().mean(dim=importance_dims)
            next_head_indices = active_head_indices

            head_prune_active = (
                self.enable_head_prune
                and self.layer_idx >= self.head_prune_start_layer
                and ((self.layer_idx - self.head_prune_start_layer) % max(1, self.head_prune_interval) == 0)
            )
            if head_prune_active and cur_heads > self.head_prune_num:
                if self.head_prune_num == 1:
                    drop_idx = torch.argmin(head_importance)
                    keep_mask = torch.ones(cur_heads, device=device, dtype=torch.bool)
                    keep_mask[drop_idx] = False
                    next_head_indices = active_head_indices[keep_mask]
                else:
                    keep_k = max(1, cur_heads - self.head_prune_num) # Ensure at least 1 head remains
                    _, topk_indices = torch.topk(head_importance, k=keep_k)
                    next_head_indices = active_head_indices[topk_indices]
                    next_head_indices = torch.sort(next_head_indices).values
            
            self.next_active_head_indices = next_head_indices

            # Token Pruning 决策
            if self.enable_token_stage_pruning:
                stage_pos = ((self.layer_idx - self.token_prune_start_layer) % max(1, self.token_stage_size)) + 1
                if self.token_stage_weighting == "linear":
                    stage_weight = float(stage_pos)
                else:
                    stage_weight = 1.0
                if self.stage_token_score_accum is None:
                    self.stage_token_score_accum = current_token_importance * stage_weight
                    self.stage_token_score_count = 1
                    self.stage_token_weight_total = stage_weight
                else:
                    self.stage_token_score_accum = self.stage_token_score_accum + current_token_importance * stage_weight
                    self.stage_token_score_count += 1
                    self.stage_token_weight_total += stage_weight
                token_score_for_prune = self.stage_token_score_accum / max(1.0, self.stage_token_weight_total)
                token_stage_boundary = (
                    self.layer_idx >= self.token_prune_start_layer
                    and ((self.layer_idx - self.token_prune_start_layer + 1) % max(1, self.token_stage_size) == 0)
                )
                self.cumulative_token_score = token_score_for_prune
            else:
                if self.cumulative_token_score is None:
                    self.cumulative_token_score = current_token_importance
                else:
                    self.cumulative_token_score += current_token_importance
                token_score_for_prune = self.cumulative_token_score
                token_stage_boundary = True

            next_token_indices = None
            token_prune_active = (
                self.enable_token_prune
                and self.layer_idx >= self.token_prune_start_layer
                and ((self.layer_idx - self.token_prune_start_layer) % max(1, self.token_prune_interval) == 0)
                and token_stage_boundary
            )
            if token_prune_active and seq_len > self.token_prune_num and self.token_prune_num == 1 and token_score_for_prune.size(0) == 1:
                drop_idx = triton_argmin_1d(token_score_for_prune[0])
                keep_mask = torch.ones(seq_len, device=device, dtype=torch.bool)
                keep_mask[drop_idx] = False
                next_token_indices = torch.arange(seq_len, device=device).unsqueeze(0)[:, keep_mask]
            elif token_prune_active and seq_len > self.token_prune_num:
                keep_k_tokens = max(1, seq_len - self.token_prune_num)
                _, topk_token_indices = torch.topk(token_score_for_prune, k=keep_k_tokens, dim=1)
                # 保持 Token 的原有句子的顺序
                next_token_indices = torch.sort(topk_token_indices, dim=1).values
            if token_prune_active and self.enable_token_stage_pruning:
                self.stage_token_score_accum = None
                self.stage_token_score_count = 0
                self.stage_token_weight_total = 0.0
                self.cumulative_token_score = None
            
            self.next_active_token_indices = next_token_indices # Store for next layer to use

        # ====================================================================
        # 统一形状处理与输出返回
        # ====================================================================
        if context_layer.dim() != 4:
            raise RuntimeError(f"Unexpected context_layer shape: {list(context_layer.shape)}")

        # [B, H_active, S, D] -> [B, S, H_active, D]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        bs = context_layer.shape[0]
        seq_len_actual = context_layer.shape[1]

        # Scatter active heads back into the original BERT hidden layout.
        full_context = torch.zeros(
            bs,
            seq_len_actual,
            self.num_heads,
            self.head_dim,
            device=device,
            dtype=hidden_states.dtype,
        )
        full_context[:, :, active_head_indices, :] = context_layer
        full_context = full_context.view(bs, seq_len_actual, -1)

        return (full_context, attention_probs)

# ====================================================================
# 测试
# ====================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" SpAtten Paper BF16-MSB Integration Running on: {device} \n")

    original_model, model_source = load_bert_model_local_or_synthetic(device)
    # 为了让 Triton Kernel 发挥最佳性能，将模型转换为 FP16 精度
    spatten_model = copy.deepcopy(original_model)

    for i, layer in enumerate(spatten_model.encoder.layer):
        orig_state_dict = layer.attention.self.state_dict()
        new_attn = SpattenBertSelfAttention(spatten_model.config)
        new_attn.load_state_dict(orig_state_dict)
        new_attn.layer_idx = i
        layer.attention.self = new_attn
    
    spatten_model.encoder.forward = spatten_encoder_forward.__get__(spatten_model.encoder, BertEncoder)
    spatten_model.to(device).float().eval()

    inputs, input_source = build_inputs_local_or_synthetic(device, original_model)
    print(f"Model source: {model_source}")
    print(f"Input source: {input_source}")
    print(f"Original Sequence Length: {inputs['input_ids'].shape[1]}")

    print("\n[Step 1] Baseline Validation (All Pruning OFF)...")
    with torch.no_grad():
        orig_out = original_model(**inputs).last_hidden_state
        sp_out = spatten_model(**inputs).last_hidden_state
    print(f"Max Difference: {(orig_out - sp_out).abs().max().item():.6f} (Should be very small)")

    print("\n[Step 2] Enabling ALL SpAtten Features (Head + Token + V Pruning + Triton Progressive Quantization)...")
    for i in range(12): 
        attn = spatten_model.encoder.layer[i].attention.self
        attn.enable_head_prune = True
        attn.head_prune_num = 1      # 每层剪去 1 个不重要的 Head
        
        attn.enable_token_prune = True
        if i >= 1: 
            attn.token_prune_num = 1 # 从第 1 层起，每层淘汰 1 个废话 Token
            
        attn.enable_v_prune = True
        attn.v_prune_num = 2         # 在软Softmax阶段，屏蔽 2 个最无关的 V 向量
        
        attn.enable_prog_quant = True # 开启 Triton 硬件级量化加速！
        attn.quant_threshold = 0.01  # 论文忠实默认值：保留 k_lsb 参与机会，避免退化成近似 K_MSB-only
        attn.v_threshold = 0.05
        attn.token_prune_interval = 2

    with torch.no_grad():
        final_out = spatten_model(**inputs).last_hidden_state
        
    print(f"\n✅ All modules executed successfully without crashing!")
    print(f"📉 Final Hidden States Shape (Length compressed!): {list(final_out.shape)}")
    
    print("\n📊 Hardware & Pruning Cascade Report:")
    for i in range(4):
        attn = spatten_model.encoder.layer[i].attention.self
        kept_heads = attn.next_active_head_indices.size(0)
        kept_tokens = attn.next_active_token_indices.shape[1] if attn.next_active_token_indices is not None else inputs['input_ids'].shape[1]
        print(f"  -> Layer {i:02d} Output | Active Heads: {kept_heads}/12 | Sent Tokens down to: {kept_tokens}")

if __name__ == "__main__":
    main()
