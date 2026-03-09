import torch
import time
import logging
from transformers import BertModel, BertTokenizer
import copy
from spatten_bert_ultimate import SpattenBertSelfAttention, spatten_encoder_forward
from transformers.models.bert.modeling_bert import BertEncoder

# 配置日志保存
logging.basicConfig(
    filename='benchmark_result.log',
    filemode='w',
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)

def benchmark_model(model, inputs, num_iters=100, warmup=10):
    model.eval()
    device = next(model.parameters()).device

    # 预热 GPU
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(**inputs)
    torch.cuda.synchronize()

    # 开始计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event =torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(**inputs)
    end_event.record()
    torch.cuda.synchronize()

    avg_time_ms = start_event.elapsed_time(end_event) / num_iters
    return avg_time_ms

def main():
    device = "cuda"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    input_text = " ".join(["algorithm"] * 128)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    orig_model = BertModel.from_pretrained("bert-base-uncased").to(device).half().eval()

    spatten_model = copy.deepcopy(orig_model)
    for layer in spatten_model.encoder.layer:
        orig_state = layer.attention.self.state_dict()
        new_atten = SpattenBertSelfAttention(spatten_model.config)
        new_atten.load_state_dict(orig_state)

        new_atten.enable_head_prune = True
        new_atten.head_prune_num = 1
        new_atten.enable_token_prune = True
        new_atten.token_prune_num = 1
        new_atten.enable_prog_quant = True
        layer.attention.self = new_atten

    spatten_model.encoder.forward = spatten_encoder_forward.__get__(spatten_model.encoder, BertEncoder)
    spatten_model.to(device).half().eval()

    # 执行性能测试
    print("Running Benchmark...")
    orig_time = benchmark_model(orig_model, inputs)
    spatten_time = benchmark_model(spatten_model, inputs)
    
    throughput_orig = 1000 / orig_time
    throughput_spatten = 1000 / spatten_time
    speedup = spatten_time / orig_time 

    msg = (f"Results:\n"
           f"Original Avg Time: {orig_time:.2f} ms | Throughput: {throughput_orig:.2f} sent/sec\n"
           f"SpAtten Avg Time:  {spatten_time:.2f} ms | Throughput: {throughput_spatten:.2f} sent/sec\n"
           f"Speedup: {orig_time/spatten_time:.2f}x\n")
    
    print(msg)
    logging.info(msg)
    print("Result saved to benchmark_results.log")

if __name__ == "__main__":
    main()