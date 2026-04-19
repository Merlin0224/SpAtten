import torch
import time
import logging
import copy
import torch.nn as nn

try:
    from spatten_bert_ultimate_paper_bf16_msb import (
        SpattenBertSelfAttention,
        build_inputs_local_or_synthetic,
        load_bert_model_local_or_synthetic,
        spatten_encoder_forward,
    )
except ImportError:
    from spattn.spatten_bert_bf16_msb import (
        SpattenBertSelfAttention,
        build_inputs_local_or_synthetic,
        load_bert_model_local_or_synthetic,
        spatten_encoder_forward,
    )
from transformers.models.bert.modeling_bert import BertEncoder

# 配置日志保存
logging.basicConfig(
    filename='benchmark_result_paper_bf16_msb.log',
    filemode='w',
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)

def reset_spatten_states(model):
    if hasattr(model, "encoder"):
        for layer in model.encoder.layer:
            attn = layer.attention.self
            if hasattr(attn, "cumulative_token_score"):
                attn.cumulative_token_score = None
                attn.next_active_head_indices = None
                attn.next_active_token_indices = None
                attn.active_head_indices_for_this_layer = None
                if hasattr(attn, "current_active_head_indices"):
                    attn.current_active_head_indices = None
                if hasattr(attn, "stage_token_score_accum"):
                    attn.stage_token_score_accum = None
                if hasattr(attn, "stage_token_score_count"):
                    attn.stage_token_score_count = 0
                if hasattr(attn, "stage_token_weight_total"):
                    attn.stage_token_weight_total = 0.0
        return

    if hasattr(model, "layers"):
        for layer in model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            if hasattr(attn, "next_active_head_indices"):
                attn.next_active_head_indices = None
            if hasattr(attn, "active_head_indices_for_this_layer"):
                attn.active_head_indices_for_this_layer = None
            if hasattr(attn, "cumulative_token_score"):
                attn.cumulative_token_score = None
            if hasattr(attn, "next_active_token_indices"):
                attn.next_active_token_indices = None


def clone_inputs(inputs):
    return {name: tensor.clone() for name, tensor in inputs.items()}


class KwargsModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.model(**kwargs)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs


class CUDAGraphRunner:
    def __init__(self, model, sample_inputs, warmup=3, reset_state_fn=None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAGraphRunner requires CUDA")

        self.model = model.eval()
        self.reset_state_fn = reset_state_fn
        self.static_inputs = {name: tensor.clone() for name, tensor in sample_inputs.items()}
        self.output = None
        self.graph = torch.cuda.CUDAGraph()

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(warmup):
                if self.reset_state_fn is not None:
                    self.reset_state_fn(self.model)
                self.output = self.model(**self.static_inputs)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        if self.reset_state_fn is not None:
            self.reset_state_fn(self.model)
        with torch.cuda.graph(self.graph):
            self.output = self.model(**self.static_inputs)

    def replay(self, new_inputs=None):
        if new_inputs is not None:
            for name, tensor in new_inputs.items():
                self.static_inputs[name].copy_(tensor)
        self.graph.replay()
        return self.output


def benchmark_cudagraph_model(model, inputs, num_iters=100, warmup=10, reset_state=False, graph_warmup=3):
    model.eval()
    wrapped = KwargsModelWrapper(model).to(next(model.parameters()).device).eval()
    runner = CUDAGraphRunner(
        wrapped,
        clone_inputs(inputs),
        warmup=graph_warmup,
        reset_state_fn=(lambda _: reset_spatten_states(model)) if reset_state else None,
    )

    for _ in range(warmup):
        runner.replay(inputs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_iters):
        runner.replay(inputs)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_iters


def benchmark_model(model, inputs, num_iters=100, warmup=10, reset_state=False):
    model.eval()
    device = next(model.parameters()).device

    # 预热 GPU
    for _ in range(warmup):
        if reset_state:
            reset_spatten_states(model)
        with torch.no_grad():
            _ = model(**inputs)
    torch.cuda.synchronize()

    # 开始计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event =torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        if reset_state:
            reset_spatten_states(model)
        with torch.no_grad():
            _ = model(**inputs)
    end_event.record()
    torch.cuda.synchronize()

    avg_time_ms = start_event.elapsed_time(end_event) / num_iters
    return avg_time_ms

def main():
    device = "cuda"
    orig_model, model_source = load_bert_model_local_or_synthetic(device)
    inputs, input_source = build_inputs_local_or_synthetic(device, orig_model, seq_len=128, exact_seq_len=True)
    print(f"Model source: {model_source}")
    print(f"Input source: {input_source}")

    spatten_model = copy.deepcopy(orig_model)
    for layer_idx, layer in enumerate(spatten_model.encoder.layer):
        orig_state = layer.attention.self.state_dict()
        new_atten = SpattenBertSelfAttention(spatten_model.config)
        new_atten.load_state_dict(orig_state)

        new_atten.enable_head_prune = True
        new_atten.head_prune_num = 1
        new_atten.enable_token_prune = True
        new_atten.token_prune_num = 1
        new_atten.enable_prog_quant = True
        new_atten.quant_threshold = 0.01
        new_atten.v_threshold = 0.05
        new_atten.token_prune_interval = 2
        new_atten.layer_idx = layer_idx
        layer.attention.self = new_atten

    spatten_model.encoder.forward = spatten_encoder_forward.__get__(spatten_model.encoder, BertEncoder)
    spatten_model.to(device).float().eval()

    # 执行性能测试
    print("Running Paper BF16-MSB Benchmark...")
    orig_time = benchmark_model(orig_model, inputs)
    spatten_time = benchmark_model(spatten_model, inputs, reset_state=True)
    
    throughput_orig = 1000 / orig_time
    throughput_spatten = 1000 / spatten_time
    speedup = spatten_time / orig_time 

    msg = (f"Results:\n"
           f"Original Avg Time: {orig_time:.2f} ms | Throughput: {throughput_orig:.2f} sent/sec\n"
           f"SpAtten Avg Time:  {spatten_time:.2f} ms | Throughput: {throughput_spatten:.2f} sent/sec\n"
           f"Speedup: {orig_time/spatten_time:.2f}x\n")
    
    print(msg)
    logging.info(msg)
    print("Result saved to benchmark_result_paper_bf16_msb.log")

if __name__ == "__main__":
    main()
