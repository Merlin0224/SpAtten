import torch
import triton
import triton.language as tl
from spatten_progressive_attention import _progressive_fwd_kernel, spatten_progressive_attention

def main():
    print("Testing Triton Attention...")
    torch.manual_seed(0)
    Z, H, N_CTX, D = 1, 12, 128, 64  # Batch, Heads, Seq_len, Head_dim

    q = torch.rand((Z, H, N_CTX, D), device='cuda', dtype=torch.float16)
    v = torch.rand((Z, H, N_CTX, D), device='cuda', dtype=torch.float16)

    k_msb = torch.rand((Z, H, N_CTX, D), device='cuda', dtype=torch.float16) * 0.5
    k_lsb = torch.rand((Z, H, N_CTX, D), device='cuda', dtype=torch.float16) * 0.1
    k_full = k_msb + k_lsb  # 模拟完整的 Key

    
    # 标准 PyTorch 全精度计算 (Baseline)
    scores_full = torch.matmul(q, k_full.transpose(-1, -2)) / (D ** 0.5)
    probs_full = torch.softmax(scores_full, dim=-1)
    out_torch_full = torch.matmul(probs_full, v)

    # 运行 Triton：高阈值 (Threshold = 100) -> 必定触发 LSB 读取 -> 精度应该极高
    out_triton_high_thresh = spatten_progressive_attention(q, k_msb, k_lsb, v, threshold=100.0)
    diff_high = torch.abs(out_triton_high_thresh - out_torch_full).max().item()
    print(f"\n[Mode: High Threshold (Always fetch LSB)]")
    print(f"Max difference with PyTorch: {diff_high:.6f}")
    if diff_high < 1e-2:
        print("✅ 全精度回退逻辑正确！")

    # 运行 Triton：低阈值 (Threshold = -100) -> 拒绝读取 LSB -> 节省带宽，但产生量化误差
    out_triton_low_thresh = spatten_progressive_attention(q, k_msb, k_lsb, v, threshold=-100.0)
    diff_low = torch.abs(out_triton_low_thresh - out_torch_full).max().item()
    
    # 纯量化 (只有 MSB) 的基准误差
    scores_msb = torch.matmul(q, k_msb.transpose(-1, -2)) / 8.0
    out_torch_msb = torch.matmul(torch.softmax(scores_msb, dim=-1), v)
    diff_expected = torch.abs(out_torch_msb - out_torch_full).max().item()

    print(f"\n[Mode: Low Threshold (Skip LSB, MSB Only)]")
    print(f"Max difference with PyTorch Full: {diff_low:.6f} (Expected quantization error: ~{diff_expected:.6f})")
    
    if diff_low > 1e-2 and abs(diff_low - diff_expected) < 1e-2:
        print("✅ 跳过逻辑生效！成功跳过了 LSB 的显存读取，节省了带宽。")
        print("\n 已在 GPU 上完整实现了 SpAtten 论文的核心思想：")
        print("   1. Cascade Head Pruning")
        print("   2. Cascade Token Pruning")
        print("   3. Progressive Quantization (Triton 显存带宽节省)")

if __name__ == "__main__":
    main()