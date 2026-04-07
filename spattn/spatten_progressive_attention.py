import torch
import triton
import triton.language as tl

# ==============================================================
# Triton Kernel: 渐进式量化 Attention (Progressive Quantization)
# ==============================================================
@triton.jit
def _progressive_fwd_kernel(
    Q, K_MSB, K_LSB, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX, Head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SCORES_THRESHOLD: tl.constexpr # [Spatten] 判断分布平坦的阈值
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offest = off_hz * stride_qh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, Head_dim)

    q_ptrs = Q + qvk_offest + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_msb_ptrs = K_MSB + qvk_offest + offs_k[:, None] * stride_kk + tl.arange(0, BLOCK_N)[None, :] * stride_kn
    k_lsb_ptrs = K_LSB + qvk_offest + offs_k[:, None] * stride_kk + tl.arange(0, BLOCK_N)[None, :] * stride_kn
    v_ptrs = V + qvk_offest + tl.arange(0, BLOCK_N)[:, None] * stride_vn + offs_k[None, :] * stride_vk
    o_ptrs = Out + qvk_offest + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on

    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, Head_dim], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # ------------------------------------------
        # 激进读取：只读取高位置信度特征（MSB)
        # ------------------------------------------
        k_msb = tl.load(k_msb_ptrs + start_n * stride_kk, mask=offs_n[None, :] < N_CTX, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k_msb, allow_tf32=True)
        qk = qk / (Head_dim ** 0.5)

        # ------------------------------------------
        # Spatten 核心： Progressive Quantization 判断
        # ------------------------------------------
        block_max_score = tl.max(qk)

        # 如果最大分数低于阈值，说明分布较平坦，继续读取低位特征（LSB）进行补充
        if block_max_score < SCORES_THRESHOLD:
            # 读取 LSB 并重构完整的 Key
            k_lsb = tl.load(k_lsb_ptrs + start_n * stride_kk)
            qk += tl.dot(q, k_lsb, allow_tf32=True) / (Head_dim ** 0.5)
        else:
            # 否则，直接使用 MSB 计算结果，跳过 LSB 以节省计算
            pass
        

        # Softmax 计算
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_n[None, :] < N_CTX, other=0.0)
        p = p.to(v.dtype)
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=True)

        m_i = m_i_new
        l_i = l_i * alpha + tl.sum(p, 1)
    
    acc = acc / l_i[:, None]
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < N_CTX)


def spatten_progressive_attention(q, k_msb, k_lsb, v, threshold=0.1):
    Z, H, N_CTX, D = q.shape
    Out = torch.empty_like(q)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    _progressive_fwd_kernel[grid](
        q, k_msb, k_lsb, v, Out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_msb.stride(0), k_msb.stride(1), k_msb.stride(2), k_msb.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        Z, H, N_CTX,
        Head_dim=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        SCORES_THRESHOLD=threshold
    )

    return Out

