import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from transformers.models.bert.modeling_bert import BertSelfAttention, BertConfig
from transformers import BertModel, BertTokenizer


#===============================================================================

# 1. 定义 SpattenBertSelfAttention 类，继承自 BertSelfAttention

#===============================================================================
class SpattenBertSelfAttention(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)

        # [Spatten] Additional parameters for Spatten
        self.layer_id = 0
        self.token_prune_ratio = 0.0
        self.head_prune_ratio = 0.0

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
        ):
        #--------------------
        # TODO : Implement the forward pass for SpattenBertSelfAttention, including token and head pruning logic
        #--------------------
        # For now, we will just call the original forward method from BertSelfAttention
        # 1. 线性变换得到 Q, K, V
        mixed_query_layer = self.query(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.key(encoder_hidden_states)
            value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
    
        # 2. 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 3. 应用注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
       
        # 4. 计算注意力权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # TODO : Implement Progressive Quantization for attention_probs if head_prune_ratio > 0

        # Dropout
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        # 5. 计算上下文向量(probs @ V)
        context_layer = torch.matmul(attention__probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

#===============================================================================

# 2. 测试函数：替换模型并验证

#===============================================================================
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
    
