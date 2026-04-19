# SpAtten on General GPU

面向长上下文的稀疏注意力加速机制研究项目。

本项目围绕毕业设计课题展开，目标是在通用 GPU 上实现并优化稀疏注意力机制，使其在长上下文场景下获得真实、可测量的端到端加速收益。项目以 `PyTorch + Triton` 为主要技术栈，围绕 `SpAtten` 思想构建了一套从模型改造、级联剪枝、底层算子优化到实验验证的完整实现。

## Overview

本项目关注的问题是：Transformer 在长上下文场景下面临注意力计算复杂度高、显存访问压力大、很多稀疏方法难以在通用 GPU 上转化为真实收益等问题。因此，本项目的重点不是单纯复现稀疏注意力算法，而是研究如何把逻辑稀疏转化为真实的计算与访存下降，并验证该方法是否能够从 BERT 这类 encoder-only 模型泛化到 Qwen3 这类 decoder-only 模型。

## Technical Route

项目整体采用“上层动态控制 + 下层算子优化”的协同路线。在模型层，通过对 BERT 与 Qwen3 的注意力模块进行 Monkey Patching，引入级联 `Head pruning`、级联 `Token pruning` 以及跨层状态传递机制；在底层实现上，通过 Triton 编写并优化 `progressive quantization`、`local V pruning` 与 `full fused SpAtten` 等路径，使模型层的稀疏性能真正转化为物理收益。整个项目通过 benchmark、模型级消融、序列长度 sweep、参数扫描和图化对比等实验方式不断收敛主线。

## Highlights

本项目的技术亮点与创新点主要体现在以下几个方面：

- 强调“逻辑稀疏到物理收益”的转化，而不是停留在 mask 层面。
- 将动态剪枝、状态传递与 Triton kernel 优化结合成统一的系统设计。
- 采用 `BF16-MSB` 路线实现高位/残差量化思路，兼顾论文语义与实际位宽压缩。
- 在 BERT 上验证完整主线后，进一步扩展到 Qwen3，验证方法的跨架构泛化能力。
- 针对 decoder-only 模型，继续探索更适合的 token pruning 形式，如 block token pruning 与 recent-window 保留策略。

## Core Implementations

### BERT Mainline

当前 BERT 主线基于 `BF16-MSB` 路线实现，核心文件包括：

- `spattn/spatten_bert_bf16_msb.py`
- `module.py`
- `benchmark/benchmark_bf16_msb.py`
- `bench_kernel/model_ablation_bf16_msb.py`
- `bench_kernel/model_seq_ablation_bf16_msb.py`

主要实现内容包括：

- `progressive quantization`
- `local V pruning`
- `cascade head pruning`
- `cascade token pruning`
- Triton kernel 优化

### Qwen3 Generalization Branch

为了验证方法的泛化性，项目额外实现了 `Qwen3` 分支，适配了：

- decoder-only
- causal attention
- GQA / repeat_kv
- RoPE

核心文件包括：

- `spattn/spatten_qwen3_bf16_msb.py`
- `bench_kernel/model_ablation_qwen3_bf16_msb.py`
- `bench_kernel/model_seq_ablation_qwen3_bf16_msb.py`

## Results

### BERT Mainline Results

在 RTX 3090 上，BERT 主线长上下文结果如下：

| Seq Len | Baseline (ms) | Quant (ms) | V-Prune (ms) | Full (ms) | Best Variant |
| --- | ---: | ---: | ---: | ---: | --- |
| 1024 | 18.03 | 17.89 | 17.31 | 17.83 | `v_prune_only` |
| 2048 | 45.49 | 31.82 | 31.25 | 31.63 | `v_prune_only` |
| 4096 | 125.91 | 64.52 | 62.46 | 62.06 | `full` |
| 8192 | 432.92 | 154.73 | 146.26 | 139.53 | `full` |

对应加速比：

| Seq Len | Full Speedup vs Baseline |
| --- | ---: |
| 2048 | `1.44x` |
| 4096 | `2.03x` |
| 8192 | `3.10x` |

结论：

- `2048` 时已经出现明显收益。
- `4096/8192` 时 `full` 路径成为最优主线。
- 项目已在通用 GPU 上验证了长上下文稀疏注意力的工程可行性。

### Compile / Graph Comparison

项目还额外比较了 `torch.compile`、`CUDA Graph` 与 graph-safe SpAtten 路径：

| Seq Len | Eager | Compile | Graph | SpAtten | SpAtten+Graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | 14.48 | 14.95 | 14.11 | 10.18 | 9.96 |
| 2048 | 38.76 | 38.84 | 38.65 | 24.19 | 24.05 |
| 4096 | 111.95 | 111.86 | 112.04 | 53.56 | 53.38 |
| 8192 | 403.16 | 403.31 | 403.93 | 140.42 | 140.11 |

结论：

- `torch.compile` 与 `CUDA Graph` 对稳定子路径有帮助。
- 但当前主收益仍然来自动态稀疏注意力与级联剪枝主线。

### FFN Graph Experiments

在当前 SpAtten 主线基础上，项目进一步对 FFN 单独做了 `torch.compile` 与 `CUDA Graph` 测试：

| Seq Len | Baseline | SpAtten | FFN-Compile | FFN-Graph | Best |
| --- | ---: | ---: | ---: | ---: | --- |
| 1024 | 14.47 | 15.64 | 15.68 | 15.39 | `baseline` |
| 2048 | 37.86 | 24.69 | 24.68 | 24.79 | `FFN-Compile` |
| 4096 | 111.48 | 46.85 | 46.87 | 46.92 | `SpAtten` |
| 8192 | 399.98 | 106.10 | 105.76 | 105.97 | `FFN-Compile` |

结论：

- FFN 图化只带来极小边际收益。
- 当前端到端瓶颈并不主要位于 FFN。

## Qwen3 Generalization

为了验证方法的泛化能力，项目将主线方法迁移到 `Qwen3`。实验过程中依次验证了 `quant_only`、`v_prune_only`、`full`，随后加入了最小版 `head pruning`、最小版 `token pruning`，并进一步尝试了 Triton meta 调整、`Triton autotune` 以及更适合 decoder-only 的 block token pruning。

当前收敛出的主要结论如下：

- 方法在 `Qwen3` 上是可行的，说明它具有跨模型架构的泛化能力。
- 在 `Qwen3` 上，`quant_only`、`v_prune_only`、`full` 都能稳定优于 baseline。
- `head pruning` 已验证有效。
- `token pruning` 在 decoder-only 上更敏感，目前还没有形成稳定超越。
- 当前最稳的主线是：
  - `progressive quantization`
  - `minimal head pruning`

近期 `Qwen3` 结果示例如下：

| Seq Len | Baseline (ms) | Quant (ms) | V-Prune (ms) | Full (ms) | Best Variant |
| --- | ---: | ---: | ---: | ---: | --- |
| 2048 | 61.74 | 56.59 | 57.40 | 56.72 | `quant_only` |
| 4096 | 150.77 | 113.41 | 117.90 | 113.57 | `quant_only` |
| 8192 | 416.51 | 256.13 | 272.72 | 261.21 | `quant_only` |

同时，在更适合 decoder-only 的 block token pruning 试验中，也观察到：

| Seq Len | Baseline (ms) | Quant (ms) | V-Prune (ms) | Full (ms) | Best Variant |
| --- | ---: | ---: | ---: | ---: | --- |
| 2048 | 61.70 | 58.37 | 59.53 | 58.60 | `quant_only` |
| 4096 | 150.43 | 118.69 | 122.24 | 117.96 | `full` |
| 8192 | 419.92 | 259.95 | 276.95 | 264.48 | `quant_only` |

结论：

- `Qwen3` 已经证明方法具有泛用性。
- 但其最优路径与 BERT 并不相同。
- decoder-only 上的 token pruning 仍需更有针对性的设计。

## Current Conclusions

截至目前，本项目可以得到以下结论：

- SpAtten 风格的稀疏注意力方法可以在通用 GPU 上获得真实收益。
- 对 BERT 这类 encoder-only 模型，项目主线已经收敛到 `BF16-MSB + Full SpAtten + Cascade Pruning`。
- 对 Qwen3 这类 decoder-only 模型，方法同样可行，但最优路径更偏向 `progressive quantization + minimal head pruning`。
- `torch.compile`、`CUDA Graph` 和 FFN 图化都已验证，但并不是当前主收益来源。
- 不同模型架构下的最优稀疏策略并不相同，这也是本项目的重要研究发现之一。

## Project Structure

- `spattn/`
  - 主模型与注意力实现
- `bench_kernel/`
  - 模型级消融、对比实验、参数扫描脚本
- `benchmark/`
  - 基础 benchmark 与图化测试工具
- `module.py`
  - 层间状态传递与 token compaction
- `tests/`
  - 测试脚本

## Quick Start

### BERT Mainline

```bash
pixi run python -m bench_kernel.model_seq_ablation_bf16_msb \
  --seq-lens 1024,2048,4096,8192 \
  --warmup 10 \
  --iters 30 \
  --enable-head-prune \
  --enable-token-prune \
  --head-prune-num 1 \
  --token-prune-num 1
```

### Qwen3 Generalization

```bash
pixi run python -m bench_kernel.model_seq_ablation_qwen3_bf16_msb \
  --seq-lens 2048,4096,8192 \
  --enable-head-prune \
  --head-prune-num 1
```

### Qwen3 Decoder-Oriented Token Pruning

```bash
pixi run python -m bench_kernel.model_seq_ablation_qwen3_bf16_msb \
  --seq-lens 2048,4096,8192 \
  --enable-head-prune \
  --enable-token-prune \
  --head-prune-num 1 \
  --token-prune-num 1 \
  --token-block-size 16 \
  --token-recent-keep 128 \
  --token-prefix-keep 1
```

## Future Directions

后续更值得继续投入的方向包括：

- 更适合 GPU 的动态 `Head/Token` 决策策略
- 更适合 decoder-only 的 token pruning 设计
- 更稳定的 `Qwen3 full` 路径优化
- 稀疏注意力与物理 compaction 的进一步协同

一句话总结：

**本项目已经在通用 GPU 上验证了面向长上下文的稀疏注意力加速机制的可行性，并进一步证明该方法具有跨模型架构的泛化潜力。**
