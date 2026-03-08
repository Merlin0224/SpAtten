import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from transformers.models.bert.modeling_bert import BertSelfAttention, BertConfig, BertModel, BertEncoder
from transformers import BertTokenizer
from spattrn_bert import SpattenBertSelfAttention
from module import slice_linear_weights, spatten_encoder_forward
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
    
    # 注入 Monkey Patch 级联逻辑：替换原来的 encoder.forward
    # 使用类型绑定的方式进行替换
    spatten_model.encoder.forward = spatten_encoder_forward.__get__(spatten_model.encoder, BertEncoder)
    spatten_model.to(device)
    spatten_model.eval()

    print("Replacement complete.")

    # 3. 验证替换后的模型在输入上的输出与原模型一致
    print("validating the modified model...")
    inputs_sentence = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(inputs_sentence, return_tensors="pt").to(device)
    seq_len = inputs['input_ids'].shape[1]

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
    for i in range(12):
        attn = spatten_model.encoder.layer[i].attention.self
        attn.enable_head_prune = True
        attn.enable_token_prune = True
        
        if i >= 1:
            attn.token_prune_num = 1

    with torch.no_grad():
        spatten_outputs = spatten_model(**inputs)
    
    final_hidden_states = spatten_outputs.last_hidden_state
    print(f"Final hidden states shape after pruning: {final_hidden_states.shape}")

    print(f"Cascade Pruning Results:")
    for i in range(4): # 打印前 4 层情况
        attn = spatten_model.encoder.layer[i].attention.self
        kept = attn.active_indices_for_this_layer if attn.active_indices_for_this_layer is not None else torch.arange(12)
        print(f"Layer {i} active heads: {kept.tolist()} (Count: {len(kept)})")
    
    for i in range(5):
        attn = spatten_model.encoder.layer[i].attention.self
        kept_tokens = attn.next_active_token_indices.shape[1] if attn.next_active_token_indices is not None else seq_len
        print(f"Layer {i} Passed tokens down: {kept_tokens}")

    print("\n✅ Success: 级联物理剪枝逻辑跑通！")

if __name__ == "__main__":
    main()