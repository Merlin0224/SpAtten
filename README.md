# SpAtten on General GPU

## Qwen3 泛化验证

为了验证方法是否只对 BERT 这类 encoder-only 模型有效，我们额外构建了一个 `Qwen3` 分支，用来测试方法在以下场景下的可行性：

- decoder-only
- causal attention
- GQA / repeat_kv

当前 `Qwen3` 分支的核心文件包括：

- `spattn/spatten_qwen3_bf16_msb.py`
- `bench_kernel/model_ablation_qwen3_bf16_msb.py`
- `bench_kernel/model_seq_ablation_qwen3_bf16_msb.py`

在 `Qwen3` 上，我们按顺序做了这些尝试：

- 先迁移 attention 主线，验证：
  - `quant_only`
  - `v_prune_only`
  - `full`
- 再加入最小版 `head pruning`
- 再加入最小版 `token pruning`
- 尝试为 `Qwen3 causal + GQA` 单独调整 Triton meta
- 尝试引入 `Triton autotune`
- 将逐 token pruning 改成更适合 decoder-only 的形式：
  - 保留前缀锚点
  - 保留最近窗口
  - 只在中间旧上下文上做裁剪
  - 可选 block-based token pruning

当前可以收敛出的结论是：

- 方法在 `Qwen3` 上是可行的，说明它具有跨模型架构的泛化性。
- 在 `Qwen3` 上，`quant_only`、`v_prune_only`、`full` 都可以稳定优于 baseline。
- `head pruning` 在 `Qwen3` 上已经验证有效，可以带来明确收益。
- `token pruning` 在 decoder-only 上更敏感，目前还没有形成稳定超越。
- `Triton autotune` 已完成接入，但当前没有证明自己能稳定优于人工调过的主线配置。
- 更适合 decoder-only 的 block/prefix/recent token pruning 方向已经完成框架验证，但目前仍属于探索性分支。

因此，当前 `Qwen3` 分支最稳的主线可以概括为：

- `progressive quantization`
- `minimal head pruning`

也就是：

**在 Qwen3 上，最稳的收益来源于量化路径与最小版 head pruning；token pruning 与更复杂的 fused 路径仍需更有针对性的 decoder-only 设计。**

面向长上下文的稀疏注意力加速机制研究项目。当前主线围绕 `BF16-MSB + Full SpAtten` 展开，目标是在通用 GPU 上把逻辑稀疏尽量转化为真实的物理加速。

## 当前主线

- 模型：BERT
- 平台：PyTorch + Triton
- 主线实现：
  - `spattn/spatten_bert_bf16_msb.py`
  - `module.py`
- 主线策略：
  - `progressive quantization`
  - `local V pruning`
  - `cascade head/token pruning`
- 当前推荐配置：
  - `quant_threshold = 0.01`
  - `v_threshold = 0.05`
  - `head_prune_num = 1`
  - `token_prune_num = 1`
  - `head_prune_interval = 1`
  - `token_prune_interval = 2`

## 目录说明

- `spattn/`
  - 主模型与注意力实现
  - 包含 `BERT` 主线与 `Qwen3` 最小泛化验证分支
- `bench_kernel/`
  - 端到端消融、技术对比、阈值和调度实验脚本
- `benchmark/`
  - 基础 benchmark 与 CUDA Graph benchmark 工具
- `module.py`
  - encoder 层间状态传递与 token compaction
- `archive/`
  - 历史尝试与早期实验入口，保留实验脉络，不作为当前主线

## 关键文件

- 主线实现：
  - `spattn/spatten_bert_bf16_msb.py`
- `Qwen3` 最小验证分支：
  - `spattn/spatten_qwen3_bf16_msb.py`
  - `bench_kernel/model_ablation_qwen3_bf16_msb.py`
  - `bench_kernel/model_seq_ablation_qwen3_bf16_msb.py`
- 层间级联逻辑：
  - `module.py`
- 主线模型级消融：
  - `bench_kernel/model_seq_ablation_bf16_msb.py`
- `torch.compile` 对比：
  - `bench_kernel/model_seq_tech_compare.py`
- Graph-safe 对比：
  - `bench_kernel/model_seq_graph_compare.py`
- Graph-safe 与真实剪枝对比：
  - `bench_kernel/model_seq_graph_prune_compare.py`
- FFN 单独图化对比：
  - `bench_kernel/model_seq_ffn_graph_compare.py`

## 当前结论

### 1. BF16-MSB 是当前最值得保留的论文主线

- `k_msb = key.to(torch.bfloat16)`
- `k_lsb = key - k_msb.float()`

这条路线同时满足两点：

- 语义上更接近论文中的高位/残差补偿思路
- `k_msb` 真正降到 16 bit，具备实际带宽收益基础

### 2. 长上下文下，`full` 路径已经具备稳定收益

在 RTX 3090 上，当前主线结果如下：

| Seq Len | Baseline (ms) | Quant (ms) | V-Prune (ms) | Full (ms) | Best |
| --- | ---: | ---: | ---: | ---: | --- |
| 1024 | 18.03 | 17.89 | 17.31 | 17.83 | `v_prune_only` |
| 2048 | 45.49 | 31.82 | 31.25 | 31.63 | `v_prune_only` |
| 4096 | 125.91 | 64.52 | 62.46 | 62.06 | `full` |
| 8192 | 432.92 | 154.73 | 146.26 | 139.53 | `full` |

对应加速比：

- `2048`: `1.44x`
- `4096`: `2.03x`
- `8192`: `3.10x`

### 3. `torch.compile` 和 CUDA Graph 不是当前主突破口

在稳定子路径上：

- `torch.compile` 对 baseline 帮助有限
- CUDA Graph 对 graph-safe SpAtten 子路径只有小幅收益

graph 对比结果显示：

| Seq Len | Eager | Compile | Graph | SpAtten | SpAtten+Graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | 14.48 | 14.95 | 14.11 | 10.18 | 9.96 |
| 2048 | 38.76 | 38.84 | 38.65 | 24.19 | 24.05 |
| 4096 | 111.95 | 111.86 | 112.04 | 53.56 | 53.38 |
| 8192 | 403.16 | 403.31 | 403.93 | 140.42 | 140.11 |

结论：

- 图化可以作为稳定子路径的辅助优化
- 但它不能替代动态级联剪枝

### 4. 真正长上下文收益仍然主要来自动态剪枝

graph-safe 静态路径与真实剪枝路径对比如下：

| Seq Len | Sp-Static | Sp-Graph | Sp-Prune | Best |
| --- | ---: | ---: | ---: | --- |
| 1024 | 10.23 | 9.96 | 15.57 | `Sp-Graph` |
| 2048 | 24.35 | 24.07 | 24.87 | `Sp-Graph` |
| 4096 | 53.85 | 53.19 | 46.86 | `Sp-Prune` |
| 8192 | 140.73 | 140.49 | 106.45 | `Sp-Prune` |

结论：

- 为了引入 CUDA Graph 而放弃动态剪枝，不值得
- 真正决定长上下文性能上限的，仍然是动态剪枝算法本身

### 5. 在 SpAtten 主线上单独对 FFN 做 `torch.compile` / CUDA Graph，只有极小边际收益

在 SpAtten `full` 主线基础上，我们只对每层 FFN（`BertIntermediate + BertOutput`）做图化对比，得到结果如下：

| Seq Len | Baseline | SpAtten | FFN-Compile | FFN-Graph | Best |
| --- | ---: | ---: | ---: | ---: | --- |
| 1024 | 14.47 | 15.64 | 15.68 | 15.39 | `baseline` |
| 2048 | 37.86 | 24.69 | 24.68 | 24.79 | `FFN-Compile` |
| 4096 | 111.48 | 46.85 | 46.87 | 46.92 | `SpAtten` |
| 8192 | 399.98 | 106.10 | 105.76 | 105.97 | `FFN-Compile` |

结论：

- `FFN-Compile` 和 `FFN-Graph` 都已经包含当前 SpAtten 稀疏注意力主线，并不是“只测 FFN”
- 在 SpAtten 主线之上，FFN 单独图化只带来极小边际收益，大约是 `0.01 ~ 0.34 ms`
- 这说明当前主瓶颈仍然不在 FFN，而在动态稀疏注意力与级联剪枝路径

## 当前不保留为主线的尝试

以下尝试做过实验，但当前不作为默认方向：

- Triton token compaction 替代层间 `gather/index_select`
- Triton `argmin` 替代 `token_prune_num == 1` 的选择逻辑
- delayed token compaction
- stage pruning 及其简单平均、线性加权等变体
- FFN 单独 `torch.compile` / CUDA Graph 图化
- 早期 `FP32` / 伪量化验证分支

这些尝试要么收益不稳定，要么被当前 `BF16-MSB` 主线覆盖。

## 推荐运行命令

### 主线模型级消融

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

### 与 `torch.compile` 对比

```bash
pixi run python -m bench_kernel.model_seq_tech_compare \
  --seq-lens 1024,2048,4096,8192 \
  --warmup 10 \
  --iters 30 \
  --spatten-mode full
```

### Graph-safe 路径对比

```bash
pixi run python -m bench_kernel.model_seq_graph_compare \
  --seq-lens 1024,2048,4096,8192 \
  --warmup 10 \
  --iters 30 \
  --spatten-mode full
```

### Graph-safe 与真实剪枝路径对比

```bash
pixi run python -m bench_kernel.model_seq_graph_prune_compare \
  --seq-lens 1024,2048,4096,8192 \
  --warmup 10 \
  --iters 30 \
  --spatten-mode full \
  --token-prune-num 1 \
  --head-prune-num 1 \
  --quant-threshold 0.01 \
  --v-threshold 0.05 \
  --head-prune-interval 1 \
  --token-prune-interval 2
```

### FFN 单独图化对比

```bash
pixi run python -m bench_kernel.model_seq_ffn_graph_compare \
  --seq-lens 1024,2048,4096,8192 \
  --warmup 10 \
  --iters 30 \
  --spatten-mode full \
  --enable-head-prune \
  --enable-token-prune \
  --head-prune-num 1 \
  --token-prune-num 1 \
  --quant-threshold 0.01 \
  --v-threshold 0.05 \
  --head-prune-interval 1 \
  --token-prune-interval 2
```

## 下一步方向

当前最值得继续做的，不是继续抠 kernel，也不是继续在 FFN 图化上花时间，而是优化动态级联剪枝算法本身，重点包括：

- 更 GPU-friendly 的 token/head 决策策略
- 降低动态决策频率
- 优化 `cumulative_token_score` 的维护方式
- 只在收益足够大时再触发物理 compaction

一句话总结：

**当前主线已经证明，SpAtten 在长上下文下能在通用 GPU 上取得稳定收益；下一步真正值得投入的，是把动态级联剪枝做得更“GPU 友好”。**
