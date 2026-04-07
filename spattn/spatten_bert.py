import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from transformers.models.bert.modeling_bert import BertSelfAttention, BertConfig, BertModel, BertEncoder
from transformers import BertTokenizer
from module import slice_linear_weights, spatten_encoder_forward

# ==========================================

# 定义 SpattenBertSelfAttention 类, Spatten 级联头剪枝模块

# ==========================================
class SpattenBertSelfAttention(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config)

        # [Spatten] Additional parameters for Spatten
        self.layer_id = 0
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

        self.enable_v_prune = False # 是否启用局部 Value 剪枝
        self.v_prune_num = 2 # 每个Head内部要剪掉的Value维度数量

    def transpose_for_scores(self, x, n_heads):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        bs, seq_len, _ = x.shape
        x = x.view(bs, seq_len, n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3) # [B, N, S, D]


    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            **kwargs
        ):
        device = hidden_states.device
        bs, seq_len, _ = hidden_states.size()

        # === [Spatten] 级联头剪枝逻辑 ===
        # 确定当前层存活的索引
        if self.active_head_indices_for_this_layer is not None:
            active_head_indices = self.active_head_indices_for_this_layer
        else:
            active_head_indices = torch.arange(self.num_heads, device=device)
        
        cur_heads = active_head_indices.size(0)

        # 1. 物理投影
        wq, bq = slice_linear_weights(self.query, active_head_indices, self.num_heads, self.head_dim)
        wk, bk = slice_linear_weights(self.key, active_head_indices, self.num_heads, self.head_dim)
        wv, bv = slice_linear_weights(self.value, active_head_indices, self.num_heads, self.head_dim)
                
        query_layer = self.transpose_for_scores(F.linear(hidden_states, wq, bq), cur_heads)
        key_layer = self.transpose_for_scores(F.linear(hidden_states, wk, bk), cur_heads)
        value_layer = self.transpose_for_scores(F.linear(hidden_states, wv, bv), cur_heads)

        # 2. 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)


        # 3. 应用注意力掩码
        if attention_mask is not None:
            if attention_mask.dim() > attention_scores.dim():
                attention_mask = attention_mask.squeeze(1)
            attention_scores = attention_scores + attention_mask
        
       
        # 4. 计算注意力权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # Dropout
        attention_probs = self.dropout(attention_probs)

        # === Local Value Pruning (局部 V 向量剪枝) ===
        # Spatten 论文中提到的局部 Value 剪枝：在每个 Head 内部剪掉某些维度
        # 注意力概率分布形状： [Batch, Heads, Seq_q, Seq_k]
        if getattr(self, "enable_v_prune", False) and getattr(self, "v_prune_num", 0) > 0:
            seq_k_len = attention_probs.size(-1)
            if seq_k_len > self.v_prune_num:
                keep_v = seq_k_len - self.v_prune_num

                # 精确获取需要的保留的 Top-K 索引
                _, topk_idx = torch.topk(attention_probs, k=keep_v, dim=-1)

                # 创建全 0 掩码，并把保留位置 1
                mask = torch.zeros_like(attention_probs)
                mask.scatter_(-1, topk_idx, 1)

                # 应用掩码
                attention_probs = attention_probs * mask
        # === [Spatten] 级联头剪枝逻辑 ===
        # 5. Context & 重要性评估（算法二）
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # 计算当前Heads的重要性分数
        # 计算重要性 (适应 3D/4D)
        importance_dims = (0, 2, 3) if context_layer.dim() == 4 else (1, 2)
        head_importance = context_layer.abs().mean(dim=importance_dims)  # [cur_heads]

        next_head_indices = active_head_indices
        if self.enable_head_prune and cur_heads > self.head_prune_num:
            keep_k = max(1, cur_heads - self.head_prune_num) # Ensure at least 1 head remains
            _, topk_indices = torch.topk(head_importance, k=keep_k)
            next_head_indices = active_head_indices[topk_indices]
            next_head_indices = torch.sort(next_head_indices).values
        
        self.next_active_head_indices = next_head_indices  # Store for next layer to use

        # === [Spatten] 级联 Token 剪枝逻辑 ===
        # 根据Spatten论文，Token 重要性是 Attention Probs
        # probs shape: [Batch, Heads, Seq_q, Seq_k]
        # 求和：某个Token(k)被所有 Head 和所有 Query 位置关注的总强度
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

        # 6. 还原形状 （Pad 0 以匹配 Bert 剩余框架）
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




