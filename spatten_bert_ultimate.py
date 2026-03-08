import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from transformers.models.bert.modeling_bert import BertSelfAttention, BertModel, BertEncoder
from transformers import BertTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

import triton
import triton.language as tl

from module import slice_linear_weights, spatten_encoder_forward

# ========================================================
# Triton 渐进式量化 Kernel (Progressive Quantization)
# ========================================================
@triton.jit
def _progressive_qk_kernel(
    Q, K_MSB, K_LSB, Scores,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_sz, stride_sh, stride_sm, stride_sn,
    Z, H, N_CTX_Q, N_CTX_K,
    Head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCORE_THRESHOLD: tl.constexpr,
    SQRT_D: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    batch_id = off_hz // H
    head_id = off_hz % H
    
    # 指针基础
    q_ptr_base = Q + batch_id * stride_qz + head_id * stride_qh
    k_msb_ptr_base = K_MSB + batch_id * stride_kz + head_id * stride_kh
    k_lsb_ptr_base = K_LSB + batch_id * stride_kz + head_id * stride_kh
    s_ptr_base = Scores + batch_id * stride_sz + head_id * stride_sh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, Head_dim)

    # 读入 Q 的 Block
    q_ptrs = q_ptr_base + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)

    for start_n in range(0, N_CTX_K, BLOCK_N):
        start_n_offs = start_n + offs_n

        # [激进读取]：只读 K_MSB
        k_msb_ptrs = k_msb_ptr_base + start_n_offs[None, :] * stride_kn + offs_k[:, None] * stride_kk
        k_msb = tl.load(k_msb_ptrs, mask=start_n_offs[None, :] < N_CTX_K, other=0.0)

        qk = tl.dot(q, k_msb) / SQRT_D

        # [分支预测]：如果 MSB 算出的最高分不够大，说明分布平坦，需要回退读取
        if tl.max(qk) < SCORE_THRESHOLD:
            k_lsb_ptrs = k_lsb_ptr_base + start_n_offs[None, :] * stride_kn + offs_k[:, None] * stride_kk
            k_lsb = tl.load(k_lsb_ptrs, mask=start_n_offs[None, :] < N_CTX_K, other=0.0)
            qk += tl.dot(q, k_lsb) / SQRT_D

        s_ptrs = s_ptr_base + offs_m[:, None] * stride_sm + start_n_offs[None, :] * stride_sn
        tl.store(s_ptrs, qk, mask=(offs_m[:, None] < N_CTX_Q) & (start_n_offs[None, :] < N_CTX_K))

def triton_progressive_qk(q, k_msb, k_lsb, threshold):
    Z, H, M, D = q.shape
    _, _, N, _ = k_msb.shape
    scores = torch.empty((Z, H, M, D), device=q.device, dtype=q.dtype)

    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), Z * H)

    _progressive_qk_kernel[grid](
        q, k_msb, k_lsb, scores,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_msb.stride(0), k_msb.stride(1), k_msb.stride(2), k_msb.stride(3),
        scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
        Z, H, M, N,
        Head_dim=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        SCORE_THRESHOLD=threshold, SQRT_D=math.sqrt(D)
    )
    return scores

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
        self.token_prune_num = 0 # 本层要剪掉的Token数量
        self.cumulative_token_score = None # 累积的Token重要性分数，用于级联传递 [Batch, SeqLen]
        self.next_active_token_indices = None # 下一层要保留的Token索引

        # 局部 V 剪枝控制
        self.enable_v_prune = False # 是否启用局部 Value 剪枝
        self.v_prune_num = 2 # 每个Head内部要剪掉的Value维度数量

        # 将近式量化控制
        self.enable_prog_quant = False
        self.quant_threshold = 1.0 
    
    def transpose_for_scores(self, x, n_heads):
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

        if self.active_head_indices_for_this_layer is not None:
            active_head_indices = active_head_indices_for_this_layer
        else:
            active_head_indices = torch.arange(self.num_heads, device=device)
        cur_heads = active_head_indices.size(0)

        wq, bq = slice_linear_weights(self.query, active_head_indices, self.num_heads, self.head_dim)
        wk, bk = slice_linear_weights(self.key, active_head_indices, self.num_heads, self.head_dim)
        wv, bv = slice_linear_weights(self.value, active_head_indices, self.num_heads, self.head_dim)
                
        query_layer = self.transpose_for_scores(F.linear(hidden_states, wq, bq), cur_heads)
        key_layer = self.transpose_for_scores(F.linear(hidden_states, wk, bk), cur_heads)
        value_layer = self.transpose_for_scores(F.linear(hidden_states, wv, bv), cur_heads)
        
        if self.enable_prog_quant:
            k_msb = key_layer * 0.8 # 主要特征
            k_lsb = key_layer * 0.2 # 残差补偿

            attention_scores = triton_progressive_qk(
                query_layer.contiguous(),
                k_msb.contiguous(),
                k_lsb.contiguous(),
                self.quant_threshold
            )
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2) / math.sqrt(self.head_dim))
        
        if attention_mask is not None:
            if attention_mask.size(-1) != attention_scores.size(-1):
                attention_mask = attention_mask[:, :, :, :attention_scores.size(-1)]
            attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        print(f"DEBUG: Q:{query_layer.shape}, K:{key_layer.shape}, V:{value_layer.shape}, Probs:{attention_probs.shape}")
        if self.enable_v_prune and self.v_prune_num > 0 and attention_probs.size(-1) > self.v_prune_num:
            keep_v = attention_probs.size(-1) - self.v_prune_num

            # 使用更安全的掩码生成方式
            probs_flat = attention_probs.view(-1, attention_probs.size(-1))
            _, topk_idx = torch.topk(probs_flat, k=keep_v, dim=-1)
            
            mask = torch.zeros_like(probs_flat)
            # 使用 scatter 的 safe 版本
            mask.scatter_(dim=-1, index=topk_idx, value=1.0)
            attention_probs = (attention_probs.view(-1, attention_probs.size(-1)) * mask).view(attention_probs.shape)
            print(f"Mask shape: {mask.shape}, Topk_idx shape: {topk_idx.shape}")

        # query: [B, H, Seq_q, D]
        # prob:  [B, H, Seq_q, Seq_k]
        # value: [B, H, Seq_k, D]
        seq_k_len = value_layer.size(-2)
        if attention_probs.size(-1) != seq_k_len:
            attention_probs = attention_probs[..., :seq_k_len]

        context_layer = torch.matmul(attention_probs, value_layer)

        importance_dims = (0, 2, 3) if context_layer.dim() == 4 else (1, 2)
        head_importance = context_layer.abs().mean(dim=importance_dims)  # [cur_heads]

        next_head_indices = active_head_indices
        if self.enable_head_prune and cur_heads > self.head_prune_num:
            keep_k = max(1, cur_heads - self.head_prune_num) # Ensure at least 1 head remains
            _, topk_indices = torch.topk(head_importance, k=keep_k)
            next_head_indices = active_head_indices[topk_indices]
            next_head_indices = torch.sort(next_head_indices).values
        
        self.next_active_head_indices = next_head_indices  # Store for next layer to use

        curren_token_importance = attention_probs.sum(dim=(1, 2))  # [Batch, Seq_k]
        
        # 累积重要性分数：级联传递给下一层
        if self.cumulative_token_score is None:
            self.cumulative_token_score = curren_token_importance
        else:
            self.cumulative_token_score += curren_token_importance
        next_token_indices = None
        if self.enable_token_prune and seq_len > self.token_prune_num:
            keep_k_tokens = max(1, seq_len - self.token_prune_num)
            _, topk_token_indices = torch.topk(self.cumulative_token_score, k=keep_k_tokens, dim=1)
            # 保持 Token 的原有句子的顺序
            next_token_indices = torch.sort(topk_token_indices, dim=1).values
        
        self.next_active_token_indices = next_token_indices # Store for next layer to use

        if context_layer.dim() == 4:
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            full_context = torch.zeros(context_layer.shape[0], context_layer.shape[1], self.num_heads, self.head_dim, device=device, dtype=hidden_states.dtype)
            full_context[:, :, active_head_indices, :] = context_layer 
            full_context = full_context.view(context_layer.shape[0], context_layer.shape[1], -1)
        else:
            context_layer = context_layer.permute(1, 0, 2).contiguous()
            full_context = torch.zeros(context_layer.shape[0], self.num_heads, self.head_dim, device=device, dtype=hidden_states.dtype)
            full_context[:, active_head_indices, :] = context_layer
            full_context = full_context.view(context_layer.shape[0], -1)
        # 始终只返回 2 个值：(Output, Weights)
        return (full_context, attention_probs)


# ====================================================================
# 测试
# ====================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" SpAtten Ultimate Integration Running on: {device} \n")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # 为了让 Triton Kernel 发挥最佳性能，将模型转换为 FP16 精度
    original_model = BertModel.from_pretrained("bert-base-uncased").to(device).half().eval()
    spatten_model = copy.deepcopy(original_model)

    for i, layer in enumerate(spatten_model.encoder.layer):
        orig_state_dict = layer.attention.self.state_dict()
        new_attn = SpattenBertSelfAttention(spatten_model.config)
        new_attn.load_state_dict(orig_state_dict)
        layer.attention.self = new_attn
    
    spatten_model.encoder.forward = spatten_encoder_forward.__get__(spatten_model.encoder, BertEncoder)
    spatten_model.to(device).half().eval()

    inputs = tokenizer("SpAtten is an algorithm-architecture co-design that leverages token, head, and quantization sparsity.", return_tensors="pt").to(device)
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
        attn.quant_threshold = 0.05   # 设定动态提取高精度的容忍阈值

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