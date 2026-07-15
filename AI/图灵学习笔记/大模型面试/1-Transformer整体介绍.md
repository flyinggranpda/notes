# Transformer 整体介绍

## 1. 课程元信息

- **主题**：Transformer 源码整体概览 — 架构总览与模块对应关系
- **主讲老师**：未知（图灵面试课程）
- **适用阶段**：初级 → 进阶（需了解基本的神经网络概念）
- **核心目标**：建立 Transformer 经典架构图与 PyTorch 源码之间的对应关系，知道各个模块在代码中处于什么位置

---

## 2. 核心概念图谱

| 术语 | 英文 | 通俗解释 |
|------|------|---------|
| 张量 | Tensor | PyTorch 中的统一数据格式。0阶=标量（25度），1阶=向量，2阶=矩阵，3阶=向量组成的矩阵，4阶=矩阵组成的矩阵 |
| 掩码 | Mask | 防止模型在预测时"偷看"未来信息的技术手段 |
| 上三角掩码矩阵 | Upper Triangular Mask | 对角线以上为0（或负无穷），以下为有效值的矩阵，用于遮盖未来信息 |
| 缩放点积注意力 | Scaled Dot-Product Attention | Transformer 的核心机制：`Attention(Q,K,V) = softmax(QK^T/√d_k)V` |
| 多头注意力 | Multi-Head Attention | 将 Q/K/V 投影到多个子空间分别做注意力，再拼接回原维度 |
| 前馈神经网络 | FFN (Feed-Forward Network) | 每个位置的向量独立通过的两层全连接网络（Position-wise） |
| 位置编码 | Positional Encoding | 给序列中每个位置添加位置信息，弥补自注意力无位置感知的缺陷 |
| 填充掩码 | Padding Mask | 标记序列中 padding 的位置，让注意力机制忽略这些无效位置 |
| 注意力加权矩阵 | Attention Weight Matrix | Q 和 K 计算出的注意力分数矩阵 |
| 注意力加权特征 | Attention Weighted Feature | 注意力矩阵与 V 加权求和后的特征向量 |
| 编码器 | Encoder | 由多头注意力和 FFN 组成的模块，将输入序列编码为特征表示 |
| 解码器 | Decoder | 由掩码自注意力、交叉注意力和 FFN 组成的模块，生成输出序列 |

---

## 3. 技术原理 / 流程拆解

### 3.1 Transformer 三大组成部分

```
┌─────────────────────────────────────────────────────┐
│                   Transformer                         │
│                                                       │
│  ┌──────────────┐    ┌──────────────┐                 │
│  │   Encoder     │    │   Decoder     │                │
│  │  (编码层)     │    │  (解码层)     │                │
│  │              │    │              │                 │
│  │  Multi-Head  │    │ Masked MH-Attn│                │
│  │  Attention   │    │ Cross-Attn   │                 │
│  │  + FFN       │    │ + FFN        │                 │
│  └──────┬───────┘    └──────┬───────┘                 │
│         │                   │                         │
│         └────────┬──────────┘                         │
│                  │                                    │
│          ┌───────▼────────┐                           │
│          │   输出层       │                           │
│          │  (Output Layer)│                           │
│          └────────────────┘                           │
└─────────────────────────────────────────────────────┘
```

### 3.2 Transformer 内部单 Block 运算流程

```
[输入特征] (feature from previous block)
      ↓
[注意力加权矩阵] (Attention Weight Matrix)
  → Q × K^T / √d_k → softmax
      ↓
[注意力加权特征] (Attention Weighted Feature)
  → Weight_Matrix × V
      ↓
[W + b] (权重 + 偏置)
      ↓
[ReLU 激活函数]
      ↓
[Position-wise FFN] (前馈神经网络)
      ↓
[输出到下一个 Block]
```

### 3.3 完整源码模块对应关系

```
1. 数据处理部分
   └─ 文本 → input_batch → torch.LongTensor（张量化）

2. Subsequence Mask（掩码生成）
   └─ 生成上三角掩码矩阵，防止偷看未来信息

3. Scaled Dot-Product Attention（缩放点积注意力） ← 重中之重
   └─ Attention(Q,K,V) = softmax(Q·K^T / √d_k)·V

4. Multi-Head Attention（多头注意力） ← Transformer 的灵魂
   └─ 将 Q/K/V 拆成多个头分别计算注意力后拼接

5. Position-wise FFN（前馈神经网络）
   └─ 用卷积层模拟全连接层，每个位置独立计算

6. Padding Mask（填充掩码）
   └─ 标记 padding 位置，与注意力机制协同

7. Positional Encoding（位置编码）
   └─ 写死的传统位置编码方案

8. Encoder（编码器层）
   └─ Multi-Head Attention + FFN（多个 block 堆叠）

9. Decoder（解码器层）
   └─ Masked Multi-Head Attention + Cross-Attention + FFN

10. 参数词表与句子输入规范
```

---

## 4. 案例 / 代码实战复盘（理念层）

本节是**源码整体概览**（第一遍浏览），不涉及逐行代码实战，而是建立**架构图 ↔ 代码**的映射关系。

> **参考资源**：老师引用了 tomer 在 "AI by hand" 项目中的注意力运算示意图（[GitHub](https://github.com/tomere)），该图清晰展示了单 Block 内的数据流：特征输入 → 注意力加权矩阵 → 注意力加权特征 → W+b → ReLU → FFN → 下一 Block。

| 模块 | 图中位置 | 代码位置 | 功能简述 |
|------|---------|---------|---------|
| Input Embedding | 底部输入 | 数据处理部分 | 文本 → 张量 |
| Positional Encoding | 嵌入后叠加 | positional encoding 部分 | 加入位置信息 |
| Multi-Head Attention | Encoder 核心块 | attention / multi-head attention | 捕捉序列关系 |
| Add & Norm | 残差连接 | 各模块内嵌 | 防止梯度消失 |
| FFN | Encoder 核心块 | position-wise FFN | 非线性变换 |
| Masked Multi-Head Attention | Decoder 第一个子层 | subsequence mask + attention | 防止未来信息泄露 |
| Cross-Attention | Decoder 第二个子层 | decoder attention | 连接编码器和解码器 |
| Output Layer | 顶部输出 | 输出层 | 映射为词汇分布 |

---

## 5. 避坑指南

| 注意点 | 说明 |
|--------|------|
| 张量的"阶"容易混淆 | 0阶=标量，1阶=向量，2阶=矩阵，3阶=向量组成的矩阵，4阶=矩阵组成的矩阵 |
| 上三角掩码的方向 | Subsequence mask 生成的是**上三角**矩阵，遮蔽**未来**位置（下半部分可见） |
| 注意力公式中的 √d_k 不能省略 | 缩放因子防止 softmax 落在梯度饱和区 |
| Padding mask 和 Subsequence mask 是两回事 | 前者遮蔽无效填充位，后者遮蔽未来信息，两者需要**共同作用** |
| Position-wise 的含义 | 不是对位置做变换，而是**每个位置独立计算**相同结构的 FFN |
| 源码版本差异 | 不同版本的 PyTorch 或 Transformer 实现细节可能不同 |

---

## 6. 对比与思考

### 6.1 Transformer 整体设计哲学

| 设计选择 | 解决什么问题 | 关键机制 |
|---------|------------|---------|
| 自注意力 | 长距离依赖 + 并行计算 | QKV + 多头 |
| 位置编码 | 自注意力无位置感知 | 三角函数的固定编码 |
| 残差连接 | 深层网络梯度消失 | Add & Norm |
| 掩码机制 | 生成任务中的信息泄露 | 上三角矩阵 |
| 多头注意力 | 捕捉不同子空间的特征 | 拆分 → 计算 → 拼接 |

### 6.2 编码器 vs 解码器

| 对比维度 | Encoder | Decoder |
|---------|---------|---------|
| 输入 | 完整的源序列 | 目标序列（逐步生成） |
| 注意力类型 | Self-Attention | Masked Self-Attention + Cross-Attention |
| 是否能看到未来 | 能（完整序列） | 不能（必须掩码） |
| 输出 | 序列的特征表示 | 目标词汇的概率分布 |
| 堆叠数量 | N 层 | N 层 |

---

## 7. 本节课思维导图

- Transformer 整体介绍
  - 三大组成部分
    - Encoder（编码层）
    - Decoder（解码层）
    - Output Layer（输出层）
  - 源码模块映射
    - 数据处理 → Tensor
    - Subsequence Mask（上三角掩码）
    - Scaled Dot-Product Attention（灵魂机制）
    - Multi-Head Attention（多头拆分）
    - Position-wise FFN（卷积模拟全连接）
    - Padding Mask（填充标记）
    - Positional Encoding（固定编码）
    - Encoder Layer（Multi-Head + FFN）
    - Decoder Layer（掩码自注意力 + 交叉注意力 + FFN）
    - 参数词表 + 句子规范
  - 注意力运算流程（一个 Block）
    - 特征输入
    - Attention Weight Matrix（QK^T）
    - Attention Weighted Feature（× V）
    - W + b → ReLU
    - FFN（Position-wise）
    - 输出到下一个 Block
  - 张量概念
    - 0阶（标量）→ 1阶（向量）→ 2阶（矩阵）→ 3阶... 4阶...
    - PyTorch 中统一用 Tensor 表示
