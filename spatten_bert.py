import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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

# 定义 SpattenBertSelfAttention 类, Spatten 级联头剪枝模块

# ==========================================
class SpattenBertSelfAttention(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config)

        # [Spatten] Additional parameters for Spatten
        self.layer_id = 0
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size // config.num_attention_heads)
        # 剪枝控制
        self.prune_num = 0
        self.enable_prune = False

        # 用于级联传递的 Buffer
        self.next_active_indices = None
        self.active_indices_for_this_layer = None

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

        # 确定当前层存活的索引
        if self.active_indices_for_this_layer is not None:
            active_indices = self.active_indices_for_this_layer
        else:
            active_indices = torch.arange(self.num_heads, device=device)
        
        cur_heads = active_indices.size(0)

        # 1. 物理投影
        wq, bq = slice_linear_weights(self.query, active_indices, self.num_heads, self.head_dim)
        wk, bk = slice_linear_weights(self.key, active_indices, self.num_heads, self.head_dim)
        wv, bv = slice_linear_weights(self.value, active_indices, self.num_heads, self.head_dim)
                
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

        # 5. Context & 重要性评估（算法二）
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # 计算当前Heads的重要性分数
        # 计算重要性 (适应 3D/4D)
        importance_dims = (0, 2, 3) if context_layer.dim() == 4 else (1, 2)
        importance = context_layer.abs().mean(dim=importance_dims)  # [cur_heads]

        # 6. [Spatten] 级联头剪枝逻辑
        next_indices = active_indices
        if self.enable_prune and cur_heads > self.prune_num:
            keep_k = max(1, cur_heads - self.prune_num)# Ensure at least 1 head remains
            _, topk_indices = torch.topk(importance, k=keep_k)
            next_indices = active_indices[topk_indices]
            next_indices, _ = torch.sort(next_indices)
        
        self.next_active_indices = next_indices  # Store for next layer to use
        # 7. 还原形状 （Pad 0 以匹配 Bert 剩余框架）
        if context_layer.dim() == 4:
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            full_context = torch.zeros(context_layer.shape[0], context_layer.shape[1], self.num_heads, self.head_dim, device=device, dtype=hidden_states.dtype)
            full_context[:, :, active_indices, :] = context_layer # [Fix]: 使用 active_indices
            full_context = full_context.view(context_layer.shape[0], context_layer.shape[1], -1)
        else:
            context_layer = context_layer.permute(1, 0, 2).contiguous()
            full_context = torch.zeros(context_layer.shape[0], self.num_heads, self.head_dim, device=device, dtype=hidden_states.dtype)
            full_context[:, active_indices, :] = context_layer
            full_context = full_context.view(context_layer.shape[0], -1)
        # 始终只返回 2 个值：(Output, Weights)
        return (full_context, attention_probs)

# ==========================================
# 级联注入：修改 BertEncoder 循环逻辑
# ==========================================
def spatten_encoder_forward(self, hidden_states, attention_mask=None, **kwargs):
    active_indices = None # Start with all heads active

    for i, layer_module in enumerate(self.layer):
         # 将上一层的决策手动同步到当前层的类属性中
        layer_module.attention.self.active_indices_for_this_layer = active_indices
        
        layer_outputs = layer_module(hidden_states, attention_mask, **kwargs)
        if isinstance(layer_outputs, (tuple, list)):
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs
        
        # 提取当前层做出的“下一层剪枝决策”
        active_indices = layer_module.attention.self.next_active_indices

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
        cross_attentions=None
    )



# ==========================================

# 测试函数：替换模型并验证

# ==========================================
def main():
    device ="cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 加载预训练的 BERT 模型和分词器
    print("loading original BERT model...")
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    original_model = BertModel.from_pretrained(model_name).to(device)
    original_model.eval()

    # 2. 替换 BERT 模型中的 Self-Attention 模块为 SpattenBertSelfAttention
    print("replacing self-attention modules with SpattenBertSelfAttention...")
    spatten_model = copy.deepcopy(original_model)

    print("Modifying self-attention modules...")
    for i, layer in enumerate(spatten_model.encoder.layer):
        orig_atten_module = layer.attention.self
        orig_state_dict = orig_atten_module.state_dict()

        config = spatten_model.config
        new_atten_module = SpattenBertSelfAttention(config)
        new_atten_module.layer_id = i  # Set layer ID for pruning logic

        # Load original weights into the new attention module
        new_atten_module.load_state_dict(orig_state_dict)

        # Replace the original attention module with the new one
        layer.attention.self = new_atten_module
    
    # 注入级联逻辑：替换原来的 encoder.forward
    # 使用类型绑定的方式进行替换
    spatten_model.encoder.forward = spatten_encoder_forward.__get__(spatten_model.encoder, BertEncoder)
    spatten_model.to(device)
    spatten_model.eval()

    print("Replacement complete.")

    # 3. 验证替换后的模型在输入上的输出与原模型一致
    print("validating the modified model...")
    inputs_sentence = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(inputs_sentence, return_tensors="pt").to(device)

    with torch.no_grad():
        orig_outputs = original_model(**inputs)
        spatten_outputs = spatten_model(**inputs)


    # 验证逻辑：第一阶段 (精度对齐)
    # Compare the outputs
    print("\n--- Step 1: Baseline Validation (Pruning OFF) ---")
    with torch.no_grad():
        orig_out = original_model(**inputs).last_hidden_state
        sp_out = spatten_model(**inputs).last_hidden_state

    diff = (orig_out - sp_out).abs().max().item()
    print(f"Max Difference: {diff:.9f}")

    if diff < 1e-5: # 修正阈值
        print("✅ Logic Success: 模型对齐成功！")
    else:
        print("❌ Validation Failed.")
        return
    # 验证逻辑：第二阶段 (剪枝逻辑)
    print("\n--- Step 2: Enabling Pruning and Validating ---")
    # 开启前三层的剪枝，测试级联效果
    for i in range(3):
        attn = spatten_model.encoder.layer[i].attention.self
        attn.enable_prune = True
        attn.prune_num = 2 # 每层多剪掉 2 个 Head

    with torch.no_grad():
        _ = spatten_model(**inputs)
        
    print(f"Cascade Pruning Results:")
    for i in range(4): # 打印前 4 层情况
        attn = spatten_model.encoder.layer[i].attention.self
        kept = attn.active_indices_for_this_layer if attn.active_indices_for_this_layer is not None else torch.arange(12)
        print(f"Layer {i} active heads: {kept.tolist()} (Count: {len(kept)})")

    print("\n✅ Success: 级联物理剪枝逻辑跑通！")

if __name__ == "__main__":
    main()