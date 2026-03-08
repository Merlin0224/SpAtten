# SpAtten: GPU-Accelerated Sparse Attention Architecture

![Static Badge](https://img.shields.io/badge/Status-Project_Completed-brightgreen)
![Static Badge](https://img.shields.io/badge/CUDA-Compatible-blue)
![Static Badge](https://img.shields.io/badge/Triton-Kernel_Optimized-red)

## 📖 项目简介 (Project Overview)
本项目复现并适配了 **SpAtten: Efficient Sparse Attention Architecture** 的核心算法逻辑。我们通过软硬件协同设计思想，在通用 GPU (NVIDIA RTX 20/30 Series) 上实现了高效的稀疏注意力推理。

该项目打破了传统 Transformer 推理时静态计算图的限制，实现了**动态序列长度裁剪**与**动态通道裁剪**，并结合 **Triton 算子开发**实现了硬件级的渐进式量化加速。

---

## 🚀 核心特性 (Key Features)
*   **Cascade Token Pruning (级联 Token 剪枝)**：在推理过程中实时根据注意力分数计算重要性，在 Encoder 层间动态缩短序列长度，实现 FFN 和 Attention 层的联合加速。
*   **Cascade Head Pruning (级联头剪枝)**：通过权重物理切片 (Slicing) 技术，在每一层动态剔除冗余的注意力头，直接减少矩阵乘法维度。
*   **Local Value Pruning (局部 V 剪枝)**：在 Softmax 后，根据注意力概率分布屏蔽无关的 Value 向量。
*   **Triton Progressive Quantization (渐进式量化)**：使用 OpenAI Triton 编写底层 Kernel，实现“按需读取显存”。模型根据注意力分布的置信度，动态决定是否加载 LSB（低位残差）数据，从而节省显存带宽。

---

## 🛠 技术实现路径
1.  **Monkey Patching**: 巧妙利用动态注入技术 (`spatten_encoder_forward`)，在不破坏 Hugging Face BERT 结构的前提下，实现跨层的状态传递与剪枝指令同步。
2.  **物理切片引擎**: 使用 `torch.index_select` 和 `F.linear` 替换原始线性层，将逻辑上的稀疏化转化为真实的 GPU 物理计算维度下降。
3.  **Triton Kernel 算子优化**: 使用 Triton JIT 编写自定义算子，在 GPU 的 SRAM 缓存中完成 Attention 逻辑，并通过分支预测减少 DRAM 访存。

---

## 📊 实验效果 (Experimental Results)
在 BERT-Base 模型上，该实现能够：
*   **动态缩短序列**：将初始 Token 序列逐层从 27 缩减至最终的 17（或更少）。
*   **动态剪除注意力头**：每一层自动识别并剔除冗余 Head，减少权重搬运量。
*   **精度对齐**：在 Pruning 关闭时，与原生 BERT 的 Max Difference 低于 `1e-5`，保证了算法的正确性。

---

## 📂 项目结构
```text
├── spatten_bert_ultimate.py    # 项目核心，整合了所有剪枝逻辑与 Triton Kernel
├── spatten_progressive_attention.py # 独立的 Triton 量化 Kernel 测试
├── spatten_bert.py              # 早期阶段的架构探索记录
└── ...
```

---

## ⚙️ 如何运行
1. **环境准备**：
   ```bash
   pip install torch transformers triton scipy accelerate
   ```
2. **运行最终整合版本**：
   ```bash
   python spatten_bert_ultimate.py
   ```

---