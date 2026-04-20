# SpAtten on General GPU

面向长序列场景的稀疏注意力加速项目。  
本项目以毕业设计为背景，目标是在通用 GPU 上把“逻辑稀疏”转化为“可测量的真实加速”，并验证该方法在 BERT（encoder-only）与 Qwen3（decoder-only）上的泛化能力。当前主线聚焦注意力计算本身，不把调度系统与 KV cache 管理作为主要贡献点。

## 项目定位

我们关注的核心问题不是“能不能剪”，而是“剪完之后能不能真快”。围绕这一点，项目构建了从模型改造、级联剪枝、Triton 内核实现到序列级 benchmark 的完整闭环，并持续将对比口径收敛到长序列注意力路径，以保证结果可解释、可复现、可用于论文主结论。

## 技术亮点（创新点合并）

本项目的技术亮点在于将模型层动态稀疏策略与算子层执行优化联合设计：在模型层引入 progressive quantization、V-prune、head pruning 并做层间状态传递，在内核层使用 Triton 实现因果注意力相关路径，把“理论稀疏”变成“计算与访存都减少”的物理收益；同时通过 BERT 与 Qwen3 两条主线交叉验证，证明该方法并非只对单一架构有效，而是可迁移到 decoder-only 的长序列场景并保持稳定收益。

## 主线进展

- 完成 BERT 主线：BF16-MSB 路线、量化与剪枝协同、模型级与序列级消融。
- 完成 Qwen3 泛化：causal attention、GQA、RoPE、use_cache 路径适配。
- 收敛主实验口径：从 generate 全链路回到 attention backend 对比，专注长序列注意力加速。

## 核心结果

### 1) BERT 主线（长序列）

| Seq Len | Baseline (ms) | Quant (ms) | V-Prune (ms) | Full (ms) | Best |
| --- | ---: | ---: | ---: | ---: | --- |
| 1024 | 18.03 | 17.89 | 17.31 | 17.83 | V-Prune |
| 2048 | 45.49 | 31.82 | 31.25 | 31.63 | V-Prune |
| 4096 | 125.91 | 64.52 | 62.46 | 62.06 | Full |
| 8192 | 432.92 | 154.73 | 146.26 | 139.53 | Full |

### 2) Qwen3 上的 vLLM baseline 尝试（已纳入对照）

测试脚本：`bench_kernel/model_seq_vllm_compare_qwen3.py`  
评测指标：`prefill + 1 token generation latency (TTFT)`

| 配置项 | 值 |
| --- | --- |
| model | `/root/autodl-tmp/models/Qwen3-0.6B-ms` |
| vLLM gpu_memory_utilization | `0.6` |
| max_model_len | `8200` |
| chunked prefill | 启用（日志显示 `max_num_batched_tokens=8192`） |
| prefix caching | 启用 |

| Seq Len | HF-Generate (ms) | vLLM (ms) | Speedup (vLLM) |
| --- | ---: | ---: | ---: |
| 1024 | 43.4665 | 7.7041 | 5.64x |
| 2048 | 65.5810 | 10.5171 | 6.24x |
| 4096 | 138.3004 | 17.7483 | 7.79x |
| 8192 | 326.5602 | 26.9872 | 12.10x |

> 说明：vLLM 结果反映的是系统级优化（调度、缓存、图捕获等）叠加收益，不能直接等价为“注意力内核本身”的对比结论，因此后续主线改为 attention backend 的同口径对照。

### 3) Qwen3 注意力后端对比（当前主线）

测试脚本：`bench_kernel/model_seq_attention_backend_compare_qwen3.py`  
基线重点：`Dense-SDPA`（主流稠密注意力后端）

| Seq Len | Dense-SDPA (ms) | SpAtten Best (ms) | Best Variant | SpBest / SDPA |
| --- | ---: | ---: | --- | ---: |
| 1024 | 39.2716 | 43.6746 | Sp-V | 111.2% |
| 2048 | 62.9569 | 61.5272 | Sp-V | 97.7% |
| 4096 | 136.7730 | 105.7964 | Sp-V | 77.4% |
| 8192 | 329.4735 | 230.1167 | Sp-V | 69.8% |

> 结论：在长序列（4096/8192）下，SpAtten 已显著超过 SDPA。

### 4) 激进开关对比（SDPA prefill fast path）

| Seq Len | Dense-SDPA (ms) | SpAtten Best (ms) | SpBest / SDPA |
| --- | ---: | ---: | ---: |
| 1024 | 37.7418 | 42.0040 | 111.3% |
| 2048 | 63.3164 | 61.8311 | 97.7% |
| 4096 | 137.7169 | 121.2103 | 88.0% |
| 8192 | 329.7879 | 302.8387 | 91.8% |

> 结论：该开关对短序列有帮助，但会削弱长序列优势，因此不作为默认主线。

## 当前阶段结论

项目已完成从“可运行”到“可在长序列注意力路径稳定领先主流稠密后端”的关键跨越。  
vLLM baseline 对照已经做过并明确记录；在论文主结论中，我们将其作为“系统级参考上界”，而将 `Dense-SDPA vs SpAtten` 作为“注意力加速能力本身”的核心证据。

## 快速复现（主线命令）

```bash
unset OMP_NUM_THREADS
pixi run python -m bench_kernel.model_seq_attention_backend_compare_qwen3 \
  --seq-lens 4096,8192 \
  --warmup 10 --iters 30 \
  --model-name /root/autodl-tmp/models/Qwen3-0.6B-ms \
  --allow-local-pretrained \
  --enable-head-prune --head-prune-num 1 --head-prune-interval 1
```

## 项目结构

- `spattn/`: SpAtten 核心实现（BERT / Qwen3、Triton 内核、forward patch）
- `bench_kernel/`: 模型级与序列级对比实验脚本
- `benchmark/`: 通用 benchmark 工具
- `module.py`: 权重切片、token compaction 与通用辅助模块
- `tests/`: 测试脚本

