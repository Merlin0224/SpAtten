import torch
import time
import copy
from transformers import BertConfig, BertModel

from spatten_bert_ultimate import SpattenBertSelfAttention
from module import spatten_encoder_forward
from transformers.models.bert.modeling_bert import BertEncoder

def reset_spatten_states(model):
    """
    每次 forward 前，必须清理上一轮残留的累积状态
    否则 cumulative_token_score 会在不同 iteration 之间错误累加
    """
    for layer in model.encoder.layer:
        attn = layer.attention.self
        if hasattr(attn, 'cumulative_token_score'):
            attn.cumulative_token_score = None
            attn.next_active_head_indices = None
            attn.next_active_token_indices = None

def benchmark_latency(model, input_data, is_spatten=False, warmup=5, iters=20):
    # 预热
    for _ in range(warmup):
        if is_spatten:
            reset_spatten_states(model)
        _ = model(**input_data)
    
    torch.cuda.synchronize()
    
    # 测速
    start_time = time.perf_counter()
    for _ in range(iters):
        if is_spatten:
            reset_spatten_states(model)
        _ = model(**input_data)
    torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    return ((end_time - start_time) / iters) * 1000  # 返回毫秒

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Sequence Length Scalability Benchmark on {device}...\n")

    # 突破长度限制：初始化一个支持最大 8192 长度的骨架模型
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.max_position_embeddings = 8192  # 突破默认的 512 限制
    
    print("Building models (using random weights for performance profiling)...")
    original_model = BertModel(config).to(device).half().eval()
    spatten_model = copy.deepcopy(original_model)

    # 替换 SpAtten 模块
    for i, layer in enumerate(spatten_model.encoder.layer):
        new_attn = SpattenBertSelfAttention(config)
        layer.attention.self = new_attn
    
    spatten_model.encoder.forward = spatten_encoder_forward.__get__(spatten_model.encoder, BertEncoder)
    spatten_model.to(device).half().eval()

    # 测试的序列长度列表
    seq_lengths =[128, 256, 512, 1024, 2048, 4096]
    batch_size = 1

    print(f"{'Seq Len':<10} | {'Orig Time(ms)':<15} | {'SpAtten Time(ms)':<18} | {'Speedup':<10}")
    print("-" * 60)

    for seq_len in seq_lengths:
        # 动态配置 SpAtten 的剪枝强度
        # 长度越长，剪得越狠，这样才能体现算法设计的价值
        for i in range(12): 
            attn = spatten_model.encoder.layer[i].attention.self
            attn.enable_head_prune = True
            attn.head_prune_num = 2  # 每层剪 2 个 Head (最终保留一部分)
            
            attn.enable_token_prune = True
            if i >= 1: 
                # 每层剪掉当前长度的 5%，实现级联缩短
                attn.token_prune_num = max(1, int(seq_len * 0.05))
                
            attn.enable_v_prune = True
            attn.v_prune_num = 16  # 剪掉 16 个 V 通道 (64 -> 48)
            
            attn.enable_prog_quant = True
            attn.quant_threshold = 0.05 

        # 生成随机输入数据
        inputs = {
            "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device),
            "attention_mask": torch.ones(batch_size, seq_len).to(device)
        }

        # 运行基准测试
        try:
            # 清理显存以防 OOM 互相影响
            torch.cuda.empty_cache()
            
            orig_ms = benchmark_latency(original_model, inputs, is_spatten=False)
            
            torch.cuda.empty_cache()
            
            sp_ms = benchmark_latency(spatten_model, inputs, is_spatten=True)
            
            speedup = orig_ms / sp_ms
            print(f"{seq_len:<10} | {orig_ms:<15.2f} | {sp_ms:<18.2f} | {speedup:<10.2f}x")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{seq_len:<10} | {'OOM':<15} | {'---':<18} | {'---':<10}")
            else:
                raise e

if __name__ == "__main__":
    with torch.no_grad():
        main()