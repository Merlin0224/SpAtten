import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX, Head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Batch 和 Head 的全局偏移
    qvk_offest = off_hz * stride_qh

    # Query 块的指针初始化
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, Head_dim)

    q_ptrs = Q + qvk_offest + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K + qvk_offest + offs_k[:, None] * stride_kk + tl.arange(0, BLOCK_N)[None, :] * stride_kn
    v_ptrs = V + qvk_offest + tl.arange(0, BLOCK_N)[:, None] * stride_vn + offs_k[None, :] * stride_vk
    o_ptrs = Out + qvk_offest + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on

    # 加载 Query 块到SRAM
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # 模拟 Softmax 累加器
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')  # max accumulator
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # logsum accumulator
    acc = tl.zeros([BLOCK_M, Head_dim], dtype=tl.float32)  # 输出累加器

    # 循环遍历 Key 和 Value 块
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n =tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # 加载 Key 和 Value 块到SRAM
        k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_n[None, :] < N_CTX, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_n[None, :] < N_CTX, other=0.0)

        # 计算注意力分数 Q * K^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, allow_tf32=True)
        qk = qk / (Head_dim ** 0.5)

        # Softmax 计算
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        # 累加Attention
        v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_n[None, :] < N_CTX, other=0.0)
        p = p.to(v.dtype)
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=True)

        # 更新累加器
        m_i = m_i_new
        l_i = l_i * alpha + tl.sum(p, 1)

    # 归一化输出并写回内存
    acc = acc / l_i[:, None]
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < N_CTX)

def triton_attention(q, k, v):
    # q, k, v: [Batch, Heads, Seq_len, Head_dim]
    Z, H, N_CTX, D = q.shape
    Out = torch.empty_like(q)
    
    BLOCK_M = 64  # Query 块大小
    BLOCK_N = 64  # Key/Value 块大小
    grid = (triton.cdiv(N_CTX, BLOCK_N), Z * H)  # 网格大小

    
    _fwd_kernel[grid](
        q, k, v, Out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        Z, H, N_CTX, D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return Out

