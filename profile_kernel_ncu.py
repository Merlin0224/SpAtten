import argparse
import math
import time

import torch

from spattn.spatten_bert_ultimate import triton_fused_spatten_ultimate


def build_inputs(z: int, h: int, m: int, d: int, seed: int, dtype: torch.dtype, device: str):
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    q = torch.randn((z, h, m, d), device=device, dtype=dtype)
    k_msb = torch.randn((z, h, m, d), device=device, dtype=dtype)
    k_lsb = torch.randn((z, h, m, d), device=device, dtype=dtype)
    v = torch.randn((z, h, m, d), device=device, dtype=dtype)
    out_sum = torch.zeros((z, h, m), device=device, dtype=torch.float32)
    return q, k_msb, k_lsb, v, out_sum


def launch_kernel(
    q: torch.Tensor,
    k_msb: torch.Tensor,
    k_lsb: torch.Tensor,
    v: torch.Tensor,
    out_sum: torch.Tensor,
    quant_threshold: float,
    v_threshold: float,
    sm_scale: float,
):
    triton_fused_spatten_ultimate(
        q,
        k_msb,
        k_lsb,
        v,
        out_sum,
        quant_threshold,
        v_threshold,
        sm_scale,
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="NCU-focused single-kernel profiler for SpAtten Triton kernel")
    parser.add_argument("--Z", type=int, default=1)
    parser.add_argument("--H", type=int, default=12)
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5, help="Warmup launches before the profiled run")
    parser.add_argument("--iters", type=int, default=1, help="Number of timed/profiled launches after warmup")
    parser.add_argument("--quant-threshold", type=float, default=0.05)
    parser.add_argument("--v-threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Input tensor dtype",
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="Also report average latency with CUDA events after warmup",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please use a CUDA-enabled PyTorch build.")

    device = "cuda"
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    q, k_msb, k_lsb, v, out_sum = build_inputs(
        z=args.Z,
        h=args.H,
        m=args.M,
        d=args.D,
        seed=args.seed,
        dtype=dtype,
        device=device,
    )

    sm_scale = 1.0 / math.sqrt(args.D)

    # 先初始化 CUDA context，避免首次建图/初始化污染 NCU 数据。
    torch.cuda.synchronize()

    print("[SpAtten NCU Profile Config]")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"shape=(Z={args.Z}, H={args.H}, M={args.M}, D={args.D})")
    print(f"dtype={dtype}")
    print(f"warmup={args.warmup}, iters={args.iters}")
    print(f"quant_threshold={args.quant_threshold}, v_threshold={args.v_threshold}, sm_scale={sm_scale:.6f}")

    # Warmup：让 Triton JIT、CUDA context、缓存分配都在 profiling 前稳定下来。
    for _ in range(args.warmup):
        launch_kernel(q, k_msb, k_lsb, v, out_sum, args.quant_threshold, args.v_threshold, sm_scale)
    torch.cuda.synchronize()

    if args.time:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            launch_kernel(q, k_msb, k_lsb, v, out_sum, args.quant_threshold, args.v_threshold, sm_scale)
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
        print(f"timed_avg_ms={total_ms / args.iters:.6f}")
    else:
        # 只执行正式 profile run，尽量减少 NCU report 中的噪音。
        for _ in range(args.iters):
            launch_kernel(q, k_msb, k_lsb, v, out_sum, args.quant_threshold, args.v_threshold, sm_scale)
        torch.cuda.synchronize()
        print("profile_run_done=1")


if __name__ == "__main__":
    main()
