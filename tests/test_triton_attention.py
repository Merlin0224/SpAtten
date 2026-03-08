import torch
import triton
import triton.language as tl
from triton_attention import _fwd_kernel, triton_attention

# ==========================================
# 测试函数：验证 Triton Attention 的正确性
# ==========================================
def main():
    print("Testing Triton Attention...")
    torch.manual_seed(0)
    Z, H, N_CTX, D = 2, 4, 128, 64  # Batch, Heads, Seq_len, Head_dim

    q = torch.rand((Z, H, N_CTX, D), device='cuda', dtype=torch.float16)
    k = torch.rand((Z, H, N_CTX, D), device='cuda', dtype=torch.float16)
    v = torch.rand((Z, H, N_CTX, D), device='cuda', dtype=torch.float16)

    Out_triton = triton_attention(q, k, v)

    scores = torch.matmul(q, k.transpose(-1, -2)) / (D ** 0.5)
    probs = torch.softmax(scores, dim=-1)
    Out_torch = torch.matmul(probs, v)

    diff = torch.abs(Out_triton - Out_torch).max().item()
    print(f"Max difference between Triton and PyTorch attention outputs: {diff:.6f}")
    assert diff < 1e-2, "Triton Attention output does not match PyTorch output within tolerance!"
    print("Triton Attention test passed!")

if __name__ == "__main__":
    main()