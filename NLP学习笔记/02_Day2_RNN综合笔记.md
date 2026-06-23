# Day2 — RNN（循环神经网络）综合笔记

> 📅 日期：2026-06-22
> 📚 来源：尚硅谷 AI 大模型 NLP 课程 Day2
> 📝 涵盖：字幕 9~16，共 8 个视频

---

# 第一部分 · RNN 基本概念

---

## 1.1 🧠 什么是 RNN

**RNN（Recurrent Neural Network，循环神经网络）** 是一种**带记忆**的神经网络，专门用来处理**序列数据**（如文本、时间序列、音频）。

> 🍳 **费曼理解**：
> 普通神经网络像看一眼照片——只看当前输入，不记得之前看过什么。
> RNN 像读书——每读一个字，脑子还记得前面读过的内容，阅读理解更好。

### 为什么需要 RNN？

在 RNN 出现之前，处理序列数据很困难：

| 问题 | 例子 | 为什么难 |
|------|------|---------|
| 变长输入 | 句子有长有短 | 普通网络需要固定输入大小 |
| 时序依赖 | "我**打**开门" vs "我**打**篮球" | 前文决定后文的意思 |
| 上下文联系 | "小明来自北京，他..." | "他"指谁？需要记住前面的信息 |

### RNN 的核心思想

```
普通神经网络：
输入 → 网络 → 输出（一次性，无记忆）

RNN：
输入₁ → 网络 → 输出₁
             ↓
          记忆传递
             ↓
输入₂ → 网络 → 输出₂  ← 带着输入₁的记忆
             ↓
          记忆传递
             ↓
输入₃ → 网络 → 输出₃  ← 带着输入₁+₂的记忆
```

> 关键：**隐藏状态（Hidden State）** 在时间步之间传递，这就是 RNN 的"记忆"。

---

## 1.2 🔄 RNN 的展开结构

RNN 在时间轴上"展开"，就像把循环复制成一条链：

```
时间步：   t=1       t=2       t=3       t=4
          ┌───┐     ┌───┐     ┌───┐     ┌───┐
输出：    │ o₁│     │ o₂│     │ o₃│     │ o₄│
          └─┬─┘     └─┬─┘     └─┬─┘     └─┬─┘
            ↑         ↑         ↑         ↑
隐藏状态：  h₁ ──→   h₂ ──→   h₃ ──→   h₄
            ↑         ↑         ↑         ↑
输入：     x₁        x₂        x₃        x₄
         ("我")    ("爱")    ("自")    ("然")
```

关键点：
- **每个时间步**都有一个输入 xₜ 和输出 oₜ
- **隐藏状态 hₜ** 承载了过去所有信息的"总结"
- hₜ 由 **当前输入 xₜ** 和 **上一时刻的隐藏状态 hₜ₋₁** 共同决定

---

## 1.3 🏷️ RNN 中的关键术语

| 术语 | 英文 | 解释 |
|------|------|------|
| **时间步** | Time Step | 序列中的每个位置（如句子中的每个词） |
| **Token** | Token | 每个时间步输入的单元（一个词或一个字） |
| **隐藏状态** | Hidden State (h) | RNN 的"记忆"，在时间步之间传递 |
| **Embedding** | Embedding | 把词转换成向量（数值表示） |
| **序列长度** | Sequence Length | 句子中包含多少个 token |

### 向量维度先导概念

> ⚠️ 接下来的内容会频繁遇到**形状（Shape）**的概念，先记住这个：
> - `input_size`：每个 token 用多少维向量表示（比如词向量是 300 维）
> - `hidden_size`：隐藏状态的维度（自己设定的超参数）
> - `batch_size`：一次同时处理多少个样本
> - `seq_len`：序列长度（句子有多少个 token）

---

# 第二部分 · RNN 数学原理

---

## 2.1 🔢 RNN 的核心公式

RNN 的工作原理可以用一个公式概括：

```
hₜ = tanh( Wₓₕ · xₜ + Wₕₕ · hₜ₋₁ + b )
```

### 公式拆解

| 符号 | 含义 | 维度 | 类比 |
|------|------|------|------|
| **hₜ** | 当前时刻的隐藏状态 | (hidden_size,) | 当前的"记忆" |
| **xₜ** | 当前时刻的输入向量 | (input_size,) | 刚读到的新词 |
| **hₜ₋₁** | 上一时刻的隐藏状态 | (hidden_size,) | 之前的"记忆" |
| **Wₓₕ** | 输入→隐藏的权重矩阵 | (hidden_size, input_size) | 如何理解新词 |
| **Wₕₕ** | 隐藏→隐藏的权重矩阵 | (hidden_size, hidden_size) | 如何更新记忆 |
| **b** | 偏置项 | (hidden_size,) | 调节阈值 |
| **tanh** | 激活函数 | — | 把值压缩到 [-1, 1] |

### 用语言解释

> 新记忆 = 激活函数（ 新词 × 权重 + 旧记忆 × 权重 + 偏置 ）

**类比**：
- **xₜ** = 你刚听到的一句话
- **hₜ₋₁** = 你之前记住的背景信息
- **hₜ** = 结合新信息和旧记忆后，你现在的理解
- **Wₓₕ** = 你"听进去"的能力（新信息对你影响多大）
- **Wₕₕ** = 你"记忆力"的强弱（旧记忆保留多少）

---

## 2.2 📐 维度变化示例

假设：
- `input_size = 3`（每个词用 3 维向量表示）
- `hidden_size = 4`（隐藏状态用 4 维向量表示）

```
xₜ 的形状:  (3,)       ← 每个词是 3 维向量
hₜ₋₁ 的形状: (4,)      ← 隐藏状态是 4 维向量
Wₓₕ 的形状:  (4, 3)    ← 把 3 维输入映射到 4 维空间
Wₕₕ 的形状:  (4, 4)    ← 4 维隐藏状态到 4 维隐藏状态
b 的形状:    (4,)

计算过程:
Wₓₕ @ xₜ      → (4,3)×(3,) → (4,)    ✦ 矩阵×向量（内部是逐行点积）
Wₕₕ @ hₜ₋₁    → (4,4)×(4,) → (4,)    ✦ 矩阵×向量
相加 + b       → (4,)                  ✦ 向量加法
tanh           → (4,) = hₜ             ✦ 逐元素激活

注：这里的运算不是哈达玛积（逐元素乘），每次要同时算 hidden_size 个不同的"映射"。
如果用哈达玛积，形状必须相同，且每个维度只跟同维度交互，那学不到跨维度的特征。
```

---

## 2.3 🧮 整体计算流程

```
输入序列: [x₁, x₂, x₃, x₄]  每个 x 形状 (3,)

第一步：   h₁ = tanh(Wₓₕ·x₁ + Wₕₕ·h₀ + b)    (h₀ 通常初始化为全 0)
第二步：   h₂ = tanh(Wₓₕ·x₂ + Wₕₕ·h₁ + b)    
第三步：   h₃ = tanh(Wₓₕ·x₃ + Wₕₕ·h₂ + b)    
第四步：   h₄ = tanh(Wₓₕ·x₄ + Wₕₕ·h₃ + b)    

注意：Wₓₕ、Wₕₕ、b 在所有时间步是共享的（参数共享！）
```

### 🔑 参数共享（重要）

在 RNN 中，**每个时间步用同一套参数**（Wₓₕ, Wₕₕ, b）。

> 🍳 **类比**：不管你看第 1 个字还是第 100 个字，你的"理解能力"（参数）是一样的，不会因为看到后面就换了个大脑。

**优点**：
- 参数数量固定，不随序列长度变化
- 可以处理任意长度的序列
- 学到的是"通用的序列处理能力"

---

# 第三部分 · RNN 的变体结构

---

## 3.1 📚 多层 RNN（Stacked RNN）

### 为什么需要多层？

单层 RNN 只能捕捉较简单的模式，**堆叠多层**可以学习更复杂的特征层次。

```
                   输出
                    ↑
          ┌──────────────────┐
第 2 层：  │  RNN Layer 2     │
          │  h¹₁→h¹₂→h¹₃→h¹₄│
          └────────┬─────────┘
                   ↑（上一层的隐藏状态作为下一层的输入）
          ┌──────────────────┐
第 1 层：  │  RNN Layer 1     │
          │  h¹₁→h¹₂→h¹₃→h¹₄│
          └────────┬─────────┘
                   ↑
               x₁  x₂  x₃  x₄
```

### 类比 CNN

CNN 用多层卷积提取从低级到高级的特征：
```
CNN：    边缘 → 纹理 → 形状 → 物体
RNN：    词义 → 短语 → 句子 → 语义
```

### 数学表示

对于 **L 层 RNN**：

```
第 1 层：  h₁¹ = tanh(Wₓₕ·x₁ + Wₕₕ·h₀¹ + b¹)     ← 输入是原始 x
           h₂¹ = tanh(Wₓₕ·x₂ + Wₕₕ·h₁¹ + b¹)
           ...

第 2 层：  h₁² = tanh(Wₓₕ·h₁¹ + Wₕₕ·h₀² + b²)     ← 输入是第 1 层的输出
           h₂² = tanh(Wₓₕ·h₂¹ + Wₕₕ·h₁² + b²)
           ...

第 L 层：  ...                                     ← 最后一层输出给任务
```

> 每层有自己的参数 Wₓₕ、Wₕₕ、b，不共享。

### 多层 vs 单层

| 对比 | 单层 RNN | 多层 RNN |
|------|---------|---------|
| 表达能力 | 有限，只能学简单模式 | 更强，能学层次化特征 |
| 训练难度 | 容易 | 更难（梯度问题更严重） |
| 参数量 | 少 | 线性增加 |
| 适用场景 | 简单序列任务 | 复杂语言理解 |

---

## 3.2 🔄 双向 RNN（Bidirectional RNN）

### 为什么需要双向？

**单向 RNN** 只能看到**过去**的信息，但在很多 NLP 任务中，**未来**的信息也很重要。

```
单向 RNN：
"我 从 北京 来"
           ↑
        看到"北京"时，已经忘了"我"和"从"

双向 RNN：
"我 从 北京 来"
       ↑
   同时看左边（我 从）和右边（来），
   更好地理解"北京"在这个上下文中的角色
```

### 哪些任务需要双向？

| 任务 | 为什么需要双向 | 例子 |
|------|--------------|------|
| **NER**（命名实体识别） | "北京"是城市还是机构？看左右才知道 | "去**北京**" vs "**北京**大学" |
| **词性标注** | 词性由前后文决定 | "打**报告**"（名词）vs "**报告**情况"（动词） |
| **情感分析** | 转折词改变前半句意思 | "电影不错，**但**..." |

### 双向 RNN 的结构

```
        输出：    o₁      o₂      o₃      o₄
                ┌───┐   ┌───┐   ┌───┐   ┌───┐
                │   │   │   │   │   │   │   │
                └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
                  │       │       │       │
正向 RNN：  h₁ᶠ → h₂ᶠ → h₃ᶠ → h₄ᶠ
          (x₁)   (x₂)   (x₃)   (x₄)
          
反向 RNN：  h₁ᵇ ← h₂ᵇ ← h₃ᵇ ← h₄ᵇ
          (x₁)   (x₂)   (x₃)   (x₄)

最终输出：  h₁ = [h₁ᶠ, h₁ᵇ]     ← 拼接正向和反向
           h₂ = [h₂ᶠ, h₂ᵇ]
           h₃ = [h₃ᶠ, h₃ᵇ]
           h₄ = [h₄ᶠ, h₄ᵇ]
```

### 核心要点

- **两个独立的 RNN**：一个从左往右读，一个从右往左读
- **拼接**每个时间步两个方向的隐藏状态 → 每个 token 的表示既包含过去信息又包含未来信息
- **参数不共享**：正向和反向 RNN 各自有独立的参数
- **输出维度翻倍**：hidden_size × 2

### 数学表示

```
正向 RNN:   hₜᶠ = tanh(Wₓₕᶠ·xₜ + Wₕₕᶠ·hₜ₋₁ᶠ + bᶠ)
反向 RNN:   hₜᵇ = tanh(Wₓₕᵇ·xₜ + Wₕₕᵇ·hₜ₊₁ᵇ + bᵇ)
最终状态:   hₜ = [hₜᶠ ; hₜᵇ]      ← 拼接
```

---

## 3.3 🏗️ 多层双向 RNN

把多层和双向组合起来：

```
第 2 层（双向）：
  h₁²ᶠ→h₂²ᶠ→h₃²ᶠ→h₄²ᶠ    +    h₁²ᵇ←h₂²ᵇ←h₃²ᵇ←h₄²ᵇ

第 1 层（双向）：
  h₁¹ᶠ→h₂¹ᶠ→h₃¹ᶠ→h₄¹ᶠ    +    h₁¹ᵇ←h₂¹ᵇ←h₃¹ᵇ←h₄¹ᵇ

输入：  x₁      x₂      x₃      x₄
```

这里的维度变化会更复杂：
```
第 1 层：每个时间步输出 hidden_size × 2（因为双向）
第 1 层的输出 → 作为第 2 层的输入
第 2 层：输入维度 = hidden_size × 2
第 2 层输出维度 = hidden_size × 2（又是双向）
```

### 参数量计算公式

```
单层单向 RNN 参数量：
  Wₓₕ:  hidden_size × input_size
  Wₕₕ:  hidden_size × hidden_size
  b:    hidden_size
  总计：input_size × hidden_size + hidden_size² + hidden_size

多层双向 RNN 参数量：
  每层有正向和反向两组参数，再乘以层数
  总计 ≈ num_layers × 2 × (input_size × hidden_size + hidden_size² + hidden_size)
```

---

# 第四部分 · PyTorch RNN API 详解

---

## 4.1 🏗️ `nn.RNN` 构造参数

```python
import torch.nn as nn

rnn = nn.RNN(
    input_size=3,       # 输入向量的维度（每个 token 的维度）
    hidden_size=4,      # 隐藏状态的维度
    num_layers=2,       # RNN 的层数（默认 1）
    nonlinearity='tanh', # 激活函数：'tanh' 或 'relu'（默认 'tanh'）
    bias=True,          # 是否使用偏置（默认 True）
    batch_first=True,   # 输入输出是否把 batch 放在第一维（默认 False）
    dropout=0.0,        # 层与层之间的 dropout 概率（默认 0）
    bidirectional=False  # 是否双向（默认 False）
)
```

### 参数详解

| 参数 | 含义 | 默认值 | 注意 |
|------|------|:-----:|------|
| `input_size` | ⭐ 每个 token 的向量维度 | 必填 | 词向量维度（如 300） |
| `hidden_size` | ⭐ 隐藏状态维度 | 必填 | 越大模型容量越大，但计算量也大 |
| `num_layers` | RNN 层数 | 1 | 多层可增强表达能力 |
| `nonlinearity` | 激活函数 | 'tanh' | tanh 最常用，也可用 relu |
| `bias` | 是否加偏置 | True | 通常保持默认 |
| `batch_first` | ⭐ batch 维度的位置 | False | **建议设为 True**，更直观 |
| `dropout` | 层间的 dropout 比例 | 0 | 只在 num_layers>1 时有效 |
| `bidirectional` | ⭐ 是否双向 | False | True 时输出维度翻倍 |

---

## 4.2 📐 输入输出形状

> ⚠️ **`batch_first=True` vs `False` 的差异**

### 推荐方式：`batch_first=True`

```
输入 input:  (batch_size, seq_len, input_size)
隐藏状态 h₀: (num_layers * num_directions, batch_size, hidden_size)

输出 output: (batch_size, seq_len, hidden_size * num_directions)
隐藏状态 hₙ: (num_layers * num_directions, batch_size, hidden_size)
```

#### 🤔 直观理解：与线性回归对比

线性回归 `Y = WX + B` 中，`X` 的形状是 `(m, n)`，其中：
- **m** = 样本数（案例数）
- **n** = 特征数（每个样本用几个数值描述）

RNN 的输入 `(batch_size, seq_len, input_size)` 可以类比理解：

| RNN 参数 | 类比线性回归 | 含义 |
|----------|-------------|------|
| **batch_size** | ↔ **m**（样本数） | 一次处理多少条独立序列 |
| **seq_len** | ✨ **新增维度**（时间步） | 每条序列展开成多长（几句话，一句话有几个词、几天的股价） |
| **input_size** | ↔ **n**（特征数） | 每个时间步用几个数值描述 |

**关键区别**：线性回归是"一个样本 → 一个输出"的扁平映射；RNN 多了一个**时间轴**，每个样本本身是一条序列 `[x₁, x₂, ..., x_T]`，模型沿着时间步逐个读取。相当于把线性回归的每个"案例"从单个点展开成一条时间线。

> 例：情感分析，一条评论 padding 到 10 个词，每个词用 300 维词向量表示
> → `(batch_size=32, seq_len=10, input_size=300)`
> → 32 条评论，每条 10 个时间步，每步 300 个特征

### 默认方式：`batch_first=False`

```
输入 input:  (seq_len, batch_size, input_size)
隐藏状态 h₀: (num_layers * num_directions, batch_size, hidden_size)

输出 output: (seq_len, batch_size, hidden_size * num_directions)
隐藏状态 hₙ: (num_layers * num_directions, batch_size, hidden_size)
```

> 💡 **建议**：**总是设置 `batch_first=True`**，这样输入形状是 (batch, seq_len, features)，更符合直觉。

### 形状示例

假设：`batch_size=2, seq_len=4, input_size=3, hidden_size=4, num_layers=2, bidirectional=True`

```
输入 input:    (2, 4, 3)       ← (batch, seq_len, input_size)
隐藏态 h₀:     (4, 2, 4)       ← (2层×2方向, batch, hidden_size)

输出 output:   (2, 4, 8)       ← (batch, seq_len, hidden_size×2)
隐藏态 hₙ:     (4, 2, 4)       ← (2层×2方向, batch, hidden_size)
```

### 输出说明

**output**：
- 每个时间步的**最后一层**隐藏状态
- 形状中的 `hidden_size * num_directions`：单向 = hidden_size，双向 = hidden_size×2
- 通常用 output[:, -1, :] 取最后一个时间步的输出作为整个序列的表示

**hₙ**：
- 所有层的**最后一个时间步**隐藏状态
- `num_layers * num_directions`行：第 1 层正向、第 1 层反向（如双向）、第 2 层正向……
- 常用 hₙ[-1, :, :] 取最后一层的最后一个时间步

### 常见疑问

#### ❓ 同一个 batch 里的句子必须等长吗？

**批量训练时：必须等长**。PyTorch 张量要求矩形结构，`(batch_size, seq_len, input_size)` 中所有序列的 `seq_len` 必须一致。

但自然语言天然是变长的：

| 场景 | 需要等长？ | 如何处理 |
|------|-----------|---------|
| **单条推断** | ❌ 不需要 | 任意长度直接输入 |
| **批量训练** | ✅ 张量要求矩形 | Padding（填充）+ pack_padded_sequence |

**Padding 策略**：将短句子补上 `<PAD>` 标记到 batch 内最长句子的长度。

```
原始：
  "I love NLP"                          → [I, love, NLP]
  "I love natural language processing"  → [I, love, natural, language, processing]

Padding 后（补 <PAD> 到最长长度）：
  [I, love, NLP, <PAD>, <PAD>]            ← seq_len=5
  [I, love, natural, language, processing] ← seq_len=5
```

但 padding 会引入噪声，所以需要配合 **`pack_padded_sequence`** 告诉 RNN 忽略 <PAD> 位置：

```python
# 1. 按真实长度排序，padding
# 2. pack 压缩成变长格式
packed_input = nn.utils.rnn.pack_padded_sequence(padded_input, lengths, batch_first=True)
# 3. RNN 处理时自动跳过 padding 位置
packed_output, h_n = rnn(packed_input)
# 4. 解压回普通张量
output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
```

#### ❓ `seq_len` 在初始化时该如何确定？取最长的吗？

**关键理解：`seq_len` 不是模型参数，是输入数据的形状决定的。**

```python
rnn = nn.RNN(input_size=300, hidden_size=128, num_layers=2)  # ✅ 没有 seq_len！
```

`nn.RNN` 的构造函数里**不需要**传 `seq_len`，它是每次前向传播时由输入 tensor 的形状动态决定的。

实际项目中通常有三种策略：

| 方案 | 做法 | 适用场景 |
|-----|------|---------|
| **① 逐 batch padding** | 每个 batch 各自 pad 到该 batch 的最长值 | 数据长短差异大，追求召回 |
| **② 全局固定长度** | 统计长度分布，取 95% 分位数作为 `MAX_SEQ_LEN`，超出截断、不足 padding | 大多数工业项目 |
| **③ Bucket batching** | 按长度分桶，同桶内长度相近再 padding | 追求训练效率 |

**方案 ② 详解（最常用 ✅）**：

```python
MAX_SEQ_LEN = 128  # 超参，根据数据分布选

if len(tokens) > MAX_SEQ_LEN:
    tokens = tokens[:MAX_SEQ_LEN]   # 截断
else:
    tokens = tokens + [PAD] * (MAX_SEQ_LEN - len(tokens))  # 填充
```

如何选这个值？先画出数据集的句子长度分布，取 95% 或 99% 分位数：

```
句子长度分布：
  ████████████████████████░░░░░   ← 95% 的句子 ≤ 64 词
  0                        64  128
                           ↑
                        选 64，只牺牲尾部 5% 的极长句，但节省大量计算
```

**方案 ③ 示意**：

```
桶1 (len 1-5):  ["你好", "我爱NLP", "今天天气好"]     → pad 到 5
桶2 (len 6-10): ["我非常喜欢自然语言处理", ...]        → pad 到 10
```

既满足张量矩形要求，又避免短句子填充过多无意义的 `<PAD>`。

---

## 4.3 💻 代码实战

### 示例 1：基础用法

```python
import torch
import torch.nn as nn

# 1. 定义 RNN 模型
rnn = nn.RNN(
    input_size=3,
    hidden_size=4,
    num_layers=1,
    batch_first=True,
    bidirectional=False
)

# 2. 准备输入
# batch=2, seq_len=4, input_size=3
input = torch.randn(2, 4, 3)

# 3. 前向传播
output, hn = rnn(input)

print(f"输出形状: {output.shape}")   # (2, 4, 4) → (batch, seq_len, hidden_size)
print(f"最后状态: {hn.shape}")       # (1, 2, 4) → (num_layers, batch, hidden_size)
```

### 示例 2：多层双向 RNN

```python
rnn = nn.RNN(
    input_size=3,
    hidden_size=4,
    num_layers=2,
    batch_first=True,
    bidirectional=True
)

input = torch.randn(2, 4, 3)    # (batch=2, seq_len=4, input_size=3)
output, hn = rnn(input)

print(f"输出形状: {output.shape}")   # (2, 4, 8)   → hidden_size×2(双向)
print(f"最后状态: {hn.shape}")       # (4, 2, 4)   → (2层×2方向, batch, hidden_size)
```

### 示例 3：取最后一个时间步的输出

```python
# 单向 RNN
output, hn = rnn(input)
last_output = output[:, -1, :]     # (batch, hidden_size) ← 取每个样本的最后一个 token 输出

# 双向 RNN 拼接
output, hn = rnn(input)
# output 最后维度已经是 [hᶠ, hᵇ] 拼接好了
last_output = output[:, -1, :]     # (batch, hidden_size*2)
```

### 示例 4：自定义初始隐藏状态

```python
# 手动指定 h₀（默认全 0）
h0 = torch.randn(1, 2, 4)    # (num_layers, batch, hidden_size)
output, hn = rnn(input, h0)
```

---

## 4.4 🔍 关键要点总结

| 知识点 | 一句话 |
|--------|--------|
| **RNN 核心** | 隐藏状态在时间步间传递，形成"记忆" |
| **核心公式** | hₜ = tanh(Wₓₕ·xₜ + Wₕₕ·hₜ₋₁ + b) |
| **参数共享** | 所有时间步使用同一套参数 |
| **多层 RNN** | 堆叠层数 = 更深层次的特征抽象 |
| **双向 RNN** | 正向+反向 → 同时看到过去和未来 |
| **batch_first** | 建议=True，输入形状 (batch, seq, feature) |
| **输出维度** | 双向时 hidden_size × 2 |
| **取序列表示** | output[:, -1, :] 取最后一个时间步 |

---

# 第五部分 · 本章总结

---

## 5.1 RNN 的优缺点

### 优点 ✅
- 可以处理**任意长度**的序列数据
- 模型大小**不随序列长度**增加（参数共享）
- 具备**时序记忆**能力
- 是后续 LSTM、GRU、Seq2Seq 的基础

### 缺点 ❌
- **串行计算**：必须一步步走，不能并行 → 训练慢
- **梯度消失/爆炸**：长序列时，反向传播的梯度连乘导致问题
- **长距离依赖困难**：虽然比普通网络强，但超过几十步还是记不住
- 这些问题由 **LSTM** 和 **Transformer** 解决

---

## 5.2 RNN 在课程中的位置

```
Day1 (NLP基础) → Day2 (RNN) → Day3~6 → Day7 → Day8~9 → Day11
                                  ↓        ↓         ↓
                             LSTM/GRU  Attention  Transformer
                                           ↓
                                      你现在在这里
```

> 💡 **RNN 是理解序列模型的"第一块基石"**。虽然现在主流是 Transformer，但 RNN 的很多思想（隐藏状态、门控机制、序列建模）直接影响了后续所有架构。

---

## ❓ 自测问题

<details>
<summary><b>Q1：RNN 的核心公式是什么？每个符号代表什么？</b></summary>

hₜ = tanh(Wₓₕ·xₜ + Wₕₕ·hₜ₋₁ + b)
- hₜ: 当前隐藏状态
- xₜ: 当前输入
- hₜ₋₁: 上一时刻隐藏状态
- Wₓₕ: 输入→隐藏权重
- Wₕₕ: 隐藏→隐藏权重
- b: 偏置
</details>

<details>
<summary><b>Q2：为什么 RNN 能处理变长序列？</b></summary>

因为参数是共享的（所有时间步用同一套 Wₓₕ, Wₕₕ, b），无论序列多长，参数量不变。序列长就多走几步，序列短就少走几步。
</details>

<details>
<summary><b>Q3：多层 RNN 和双向 RNN 有什么区别？</b></summary>

多层 RNN = 纵向堆叠，每层学习不同层次的特征（低级→高级）；双向 RNN = 横向增加反向处理，让每个 token 同时包含过去和未来的信息。两者可以组合使用。
</details>

<details>
<summary><b>Q4：在 PyTorch 中，设置 batch_first=True 时输入输出是什么形状？</b></summary>

输入: (batch_size, seq_len, input_size)
输出: (batch_size, seq_len, hidden_size × num_directions)
hₙ: (num_layers × num_directions, batch_size, hidden_size)
</details>

<details>
<summary><b>Q5：为什么通常用 output[:, -1, :] 取最后一个时间步的输出？</b></summary>

在分类或情感分析等任务中，最后一个时间步的隐藏状态"看完了"整个序列，包含了全部输入的信息，可以作为整个序列的总结表示。
</details>

<details>
<summary><b>Q6：RNN 的主要缺点是什么？这些缺点后来被谁解决了？</b></summary>

主要缺点：① 串行计算不能并行（Transformer 解决）；② 梯度消失/爆炸，长距离记忆差（LSTM 解决了一部分，Transformer 完全解决）。
</details>

---

## 📌 Anki 卡片建议

| # | 正面 | 背面 |
|---|------|------|
| 1 | RNN 全称和中文名 | Recurrent Neural Network，循环神经网络 |
| 2 | RNN 核心公式 | hₜ = tanh(Wₓₕ·xₜ + Wₕₕ·hₜ₋₁ + b) |
| 3 | RNN 为什么要用 tanh 激活？ | 把值压缩到 [-1,1]，防止隐藏状态无限增长 |
| 4 | 什么是"参数共享"？ | 同一层 RNN 在不同时间步使用相同的 Wₓₕ, Wₕₕ, b |
| 5 | 参数共享的好处？ | 参数量固定，能处理任意长度序列 |
| 6 | 双向 RNN 的结构 | 正向 RNN + 反向 RNN，输出拼接 |
| 7 | 双向 RNN 适合哪些任务？ | 需要上下文的：NER、词性标注、阅读理解 |
| 8 | batch_first=True 时 input 的形状 | (batch_size, seq_len, input_size) |
| 9 | batch_first=True 时 output 的形状 | (batch_size, seq_len, hidden_size × num_directions) |
| 10 | RNN 的三个主要缺点 | 串行计算慢、梯度消失/爆炸、长距离记忆差 |

---

> **📖 复习提示**：
> - **重点理解核心公式**：把公式和"新记忆 = 新信息 + 旧记忆"的直觉对应起来
> - **代码跑一遍**：打开 Python 跑一遍上面示例 1~4，观察每个变量的形状
> - **RNN 是后续 LSTM 的基础**：LSTM 就是在 RNN 公式上加了三道"门"，理解 RNN 是掌握 LSTM 的前提
