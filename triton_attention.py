import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qd,
    stride_kz, stride_kh, stride_kd,
    stride_vz, stride_vh, stride_vd,
    stride_oz, stride_oh, stride_od,
    Z, H, N_CTX, Head_dim: tl.constexpr
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    