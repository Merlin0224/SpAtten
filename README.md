# SpAtten on General GPU

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
- `bench_kernel/`
  - 端到端消融、技术对比、阈值/调度实验脚本
- `benchmark/`
  - 基础 benchmark 与 CUDA Graph benchmark 工具
- `module.py`
  - encoder 层间状态传递与 token compaction

## 关键文件

- 主线实现：
  - `spattn/spatten_bert_bf16_msb.py`
- 层间级联逻辑：
  - `module.py`
- 主线模型级消融：
  - `bench_kernel/model_seq_ablation_bf16_msb.py`
- compile 对比：
  - `bench_kernel/model_seq_tech_compare.py`
- graph 对比：
  - `bench_kernel/model_seq_graph_compare.py`
- graph 与真实剪枝对比：
  - `bench_kernel/model_seq_graph_prune_compare.py`

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
- `CUDA Graph` 对 graph-safe SpAtten 子路径只有小幅收益

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

## 当前不保留为主线的尝试

以下尝试做过实验，但当前不作为默认方向：

- Triton token compaction 替代层间 `gather/index_select`
- Triton `argmin` 替代 `token_prune_num == 1` 的选择逻辑
- delayed token compaction
- 早期 `FP32`/伪量化验证分支

这些尝试要么收益不稳定，要么被当前 BF16-MSB 主线覆盖。

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

## 下一步方向

当前最值得继续做的不是继续抠 kernel，而是优化动态级联剪枝算法本身，重点包括：

- 阶段式 token/head pruning
- 降低动态决策频率
- 优化 `cumulative_token_score` 的维护方式
- 只在收益足够大时再触发物理 compact

一句话总结：

**当前主线已经证明：SpAtten 在长上下文下能在通用 GPU 上取得稳定收益；下一步真正值得投入的，是把动态级联剪枝做得更“GPU 友好”。**
