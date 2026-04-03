from spatten_bert_ultimate import _spatten_fused_ultimate_kernel, triton_fused_spatten_ultimate
import torch
import time
import triton
import triton.language as tl
import math

def benchmark_spatten():
    Z, H, M, N = 1, 12, 4096, 4096
    D = 64

    q = torch.randn((Z, H, M, D), device='cuda', dtype=torch.float16)
    k_msb = torch.randn((Z, H, M, D), device='cuda', dtype=torch.float16)
    k_lsb = torch.randn((Z, H, M, D), device='cuda', dtype=torch.float16)
    v = torch.randn((Z, H, M, D), device='cuda', dtype=torch.float16)
    Out_sum = torch.zeros((Z, H, M), device='cuda', dtype=torch.float32)

    sm_scale = 1.0 / math.sqrt(D)
    quant_threshold = 0.05
    v_threshold = 0.01

    # print("Warming up...")
    # for _ in range(5):
    #     triton_fused_spatten_ultimate(
    #         q, k_msb, k_lsb, v, Out_sum,
    #         sm_scale, quant_threshold, v_threshold
    #     )
    
    # print("Running kernel for NCU...")
    # triton_fused_spatten_ultimate(q, k_msb, k_lsb, v, Out_sum, quant_threshold, v_threshold, sm_scale)
    # torch.cuda.synchronize()
    # print("Done!")

    # 1. 测 Triton 的耗时
    triton_ms = triton.testing.do_bench(
        lambda: triton_fused_spatten_ultimate(q, k_msb, k_lsb, v, Out_sum, quant_threshold, v_threshold, sm_scale)
    )
    
    # 2. 测 PyTorch 原生 FlashAttention 的耗时（作为极限参考线）
    # 注意：SDPA 内部自动调用 FlashAttention2
    pytorch_ms = triton.testing.do_bench(
        lambda: torch.nn.functional.scaled_dot_product_attention(q, k_msb, v)
    )
    
    print(f"PyTorch SDPA Baseline: {pytorch_ms:.4f} ms")
    print(f"Triton SpAtten Kernel: {triton_ms:.4f} ms")
    print(f"Slowdown Factor: {triton_ms / pytorch_ms:.2f}x")

if __name__ == "__main__":
    benchmark_spatten()
