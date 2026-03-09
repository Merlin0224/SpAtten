import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from transformers.models.bert.modeling_bert import BertSelfAttention, BertConfig, BertModel, BertEncoder
from transformers import BertTokenizer
from transformers.modeling_outputs import  BaseModelOutputWithPastAndCrossAttentions
# ==========================================
# 辅助函数：权重物理切片 
# ==========================================
def slice_linear_weights(linear_layer, active_indices, num_heads, head_dim):
    """
    从 Linear 层中提取 active_indices 对应的 Head 权重和偏置。
    核心逻辑：物理减少矩阵维度，压榨 GPU 推理性能。
    """
    weight = linear_layer.weight
    bias = linear_layer.bias
    device = weight.device

    # Weight: [all_heads, head_dim, hidden_size]
    w_view = weight.view(num_heads, head_dim, -1)
    # Bias: [all_heads, head_dim]
    b_view = bias.view(num_heads, head_dim)

    # 物理提取需要的 Head
    w_subset = torch.index_select(w_view, 0, active_indices)
    b_subset = torch.index_select(b_view, 0, active_indices)

    # 展平回 2D 供 F.linear 使用
    w_out = w_subset.reshape(-1, w_subset.size(-1))
    b_out = b_subset.reshape(-1)
    
    return w_out, b_out


# ==========================================
# 级联注入：修改 BertEncoder 循环逻辑
# ==========================================
def spatten_encoder_forward(self, hidden_states, attention_mask=None, **kwargs):
    active_head_indices = None # Start with all heads active
    active_token_indices = None # Start with all tokens active
    cumulative_token_score = None # For token importance accumulation

    for i, layer_module in enumerate(self.layer):
        # ----------------------------------------------------
        # [Token 剪枝执行阶段]: 物理切片 hidden_states 和 attention_mask
        # ----------------------------------------------------
        if active_token_indices is not None:

            # 1. 切片 hidden_states [Batch, Seq, Hidden] -> [Batch, Keep_Seq, Hidden]
            # 扩展索引的维度以匹配 hidden_states
            expanded_indices = active_token_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
            hidden_states = torch.gather(hidden_states, dim=1, index=expanded_indices)

            # 2. 切片 attention_mask [Batch, 1, 1, Seq] -> [Batch, 1, 1, Keep_Seq]
            if attention_mask is not None:
                mask_indices = active_token_indices.unsqueeze(1).unsqueeze(2)
                attention_mask = torch.gather(attention_mask, 3, index=mask_indices)
            
            # 必须同步切片历史累积的 Token Score，否则层数增加会产生形状冲突！
            if cumulative_token_score is not None:
                cumulative_token_score = torch.gather(cumulative_token_score, 1, active_token_indices)
            
        # ----------------------------------------------------
        # 状态同步
        # ----------------------------------------------------

        layer_module.attention.self.active_head_indices_for_this_layer = active_head_indices
        layer_module.attention.self.cumulative_token_score = cumulative_token_score

        # ----------------------------------------------------
        # 执行前向传播 (此处的 hidden_states 已经变短了，FFN 计算量也减小了)
        # ----------------------------------------------------
        layer_outputs = layer_module(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )

        if isinstance(layer_outputs, (tuple,list)):
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs
        
        # ----------------------------------------------------
        # 提取当前层做出的决策
        # ----------------------------------------------------
        active_head_indices = layer_module.attention.self.next_active_head_indices
        active_token_indices = layer_module.attention.self.next_active_token_indices
        cumulative_token_score = layer_module.attention.self.cumulative_token_score

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
        cross_attentions=None
    )
