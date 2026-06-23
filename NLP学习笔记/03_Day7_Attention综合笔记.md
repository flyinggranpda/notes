# Day7 — Attention（注意力机制）综合笔记

> 📅 日期：2026-06-22
> 📚 来源：尚硅谷 AI 大模型 NLP 课程 Day7
> 📝 涵盖：字幕 7~12，共 6 个视频

---

# 第一部分 · 为什么需要 Attention

---

## 1.1 🔍 Seq2Seq 的瓶颈

在 Attention 出现之前，**Seq2Seq（序列到序列）** 框架有一个严重的问题。

### 传统 Seq2Seq 流程

```
输入: "我 爱 你"
        ↓
    编码器 RNN
        ↓
   [一个语义向量]    ← 整个句子的信息"压缩"在这里
        ↓
    解码器 RNN
        ↓
输出: "I  love  you"
```

### 🚨 问题

> 不管输入句子是 5 个词还是 50 个词，编码器都只输出**一个固定长度的向量**。

这就像让你听一段 5 秒的语音，复述出来没问题；但让你听一小时的讲座，然后**只用一个词总结全部内容**——信息严重丢失。

### 🍳 费曼类比

```
传统 Seq2Seq = 闭卷考试 📕
  把整本书塞进脑子里 → 考试时全靠回忆 → 书越厚越记不住

Seq2Seq + Attention = 开卷考试 📖
  答题时可以随时翻书，找到对应页码看原文
```

---

## 1.2 💡 Attention 的核心思想

> **Attention = 在生成每一步输出时，回头看输入的所有位置，选重点看**

```
Decoder 生成第 1 个词时：
  "I"  ← 重点关注输入的 "我"
  
Decoder 生成第 2 个词时：
  "love"  ← 重点关注输入的 "爱"
  
Decoder 生成第 3 个词时：
  "you"  ← 重点关注输入的 "你"
```

每一步不是靠"压缩了整句话的一个向量"硬猜，而是**每次去原文找当前最需要的信息**。

---

# 第二部分 · Attention 工作原理

---

## 2.1 📊 四步流程

Attention 的计算分为 4 步：

```
Step 1: 计算注意力分数
          Score = 评分函数(查询, 键)
          
Step 2: Softmax 归一化
          Attention权重 = Softmax(Score)
          
Step 3: 加权求和
          上下文向量 = Σ(Attention权重 × 值)
          
Step 4: 融合到解码器
          解码器输入 = 拼接(上下文向量, 解码器状态)
```

### 具体来说

在 Seq2Seq + Attention 的翻译任务中：

| 术语 | 实际是什么 | 类比 |
|------|-----------|------|
| **查询（Query）** | 解码器当前隐藏状态 hₜ | "我现在要翻译哪个词？" |
| **键（Key）** | 编码器所有位置的隐藏状态 H₁, H₂... | "原文每个词的代表" |
| **值（Value）** | 编码器所有位置的隐藏状态 H₁, H₂... | "原文每个词的具体信息" |
| **分数（Score）** | 查询和每个键的匹配程度 | "当前该看原文哪个词？" |
| **权重（Weight）** | Softmax 后的概率分布 | "每个词占多少注意力" |
| **上下文向量** | 所有值按权重加权求和 | "聚焦后的关键信息" |

---

## 2.2 🧮 计算过程详解

### Step 1：计算每个位置的分数

```
解码器当前状态:  hₜ            (hidden_size,)
编码器所有输出:  H₁, H₂, H₃, H₄  每个 (hidden_size,)

score₁ = 评分函数(hₜ, H₁)    
score₂ = 评分函数(hₜ, H₂)    ← 每个分数表示"当前翻译到这一步，
score₃ = 评分函数(hₜ, H₃)       原文对应的位置有多重要"
score₄ = 评分函数(hₜ, H₄)

结果: [score₁, score₂, score₃, score₄]  ← 每个位置一个分数
```

### Step 2：Softmax 转成概率

```
分数:    [0.5,  2.1,  0.3,  0.1]
                   ↓ Softmax
权重:    [0.12, 0.68, 0.13, 0.07]   ← 加起来 = 1

"第 2 个位置（爱）的权重最高 (0.68)，说明翻译到这一步最该关注它"
```

### Step 3：加权求和得到上下文向量

```
权重:    [0.12, 0.68, 0.13, 0.07]
            ×     ×     ×     ×
编码器输出: H₁ + H₂ + H₃ + H₄
            = 
上下文向量 = 0.12·H₁ + 0.68·H₂ + 0.13·H₃ + 0.07·H₄
          ≈ H₂  (因为 H₂ 的权重最大)
```

### Step 4：融合到解码器

```
解码器输入 = 拼接(上下文向量, 解码器上一时刻输出)

然后解码器用这个"带了原文信息"的输入，生成当前词
```

---

## 2.3 📐 形状变化（代码角度）

```python
# 假设
batch_size = 2
seq_len = 5          # 输入有 5 个词
hidden_size = 4      # 隐藏状态维度

# 编码器输出
encoder_outputs = (2, 5, 4)    # (batch, seq_len, hidden_size)

# 解码器当前时刻的隐藏状态
decoder_hidden = (2, 1, 4)     # (batch, 1, hidden_size)

# Step 1: 计算分数
# decoder_hidden 和 encoder_outputs 做矩阵乘法
scores = decoder_hidden @ encoder_outputs.transpose(1, 2)
# (2, 1, 4) × (2, 4, 5) → (2, 1, 5)
# 每个 batch 的每个解码时间步，对 5 个编码器位置各有一个分数

# Step 2: Softmax
attention_weights = F.softmax(scores, dim=-1)
# (2, 1, 5)  →  每行加起来 = 1

# Step 3: 加权求和
context = attention_weights @ encoder_outputs
# (2, 1, 5) × (2, 5, 4) → (2, 1, 4)
# 得到上下文向量 → 形状跟 decoder_hidden 一样

# Step 4: 拼接
decoder_input = torch.cat([context, decoder_hidden], dim=-1)
# (2, 1, 4) + (2, 1, 4) → (2, 1, 8)
```

**关键**：Attention 机制本身不改变维度，它相当于一个**信息路由器**——从 `seq_len` 个位置中把相关信息汇聚到上下文向量里。

---

# 第三部分 · 注意力评分函数

---

## 3.1 🔢 三种评分方式

注意力机制的关键在 Step 1 的评分函数。主要有 3 种：

| 评分函数 | 公式 | 参数量 | 计算量 |
|:--------:|:----:|:------:|:------:|
| **点积**（Dot） | `score = hₜ · Hₛ` | 0 | 最小 |
| **一般**（General） | `score = hₜᵀ · W · Hₛ` | hidden_size² | 中等 |
| **拼接**（Concat） | `score = vᵀ · tanh(W · [hₜ; Hₛ])` | (hidden_size×2)×hidden_size | 最大 |

### ① 点积注意力（Dot Product Attention）

```python
score = decoder_hidden @ encoder_outputs.transpose(-2, -1)
```

**特点**：
- 不需要额外参数（直接用隐藏状态做点积）
- 要求解码器和编码器的**维度一致**
- 计算最快，Transformer 用的就是这种（的 scaled 版本）

### ② 一般注意力（General Attention）

```python
W = nn.Linear(hidden_size, hidden_size, bias=False)
score = decoder_hidden @ W(encoder_outputs).transpose(-2, -1)
```

**特点**：
- 加了一个可学习的权重矩阵 W
- 允许编码器和解码器维度不一致时做映射
- 比点积更灵活，但参数更多

### ③ 拼接注意力（Concat Attention / Additive Attention）

```python
W = nn.Linear(hidden_size * 2, hidden_size)
v = nn.Linear(hidden_size, 1)

combined = torch.cat([decoder_hidden.expand_as(encoder_outputs), encoder_outputs], dim=-1)
score = v(torch.tanh(W(combined))).squeeze(-1)
```

**特点**：
- 把查询和键拼接起来再过 MLP
- 是 Bahdanau 等人原始论文用的方法
- 表达能力最强，但参数量最大、计算最慢

---

## 3.2 🆚 对比总结

```
表达能力:  点积 < 一般 < 拼接
计算速度:  点积 > 一般 > 拼接
参数数量:  点积(0) < 一般(W) < 拼接(W+v)

实际使用:
  机器翻译 Seq2Seq → 拼接注意力（Bahdanau）最早使用
  Transformer     → 缩放点积注意力（Scaled Dot-Product）
  一般注意力       → 介于两者之间，用得相对少
```

---

# 第四部分 · Attention + Seq2Seq 代码架构

---

## 4.1 🏗️ 整体结构

```
                    ┌──────────────────┐
                    │  输出："I love you" │
                    └────────┬─────────┘
                             ↑
           ┌─────────────────┴─────────────────┐
           │           解码器（GRU）              │
           │  每一步输入 = [上下文向量, 上一词向量]  │
           └─────────────────┬─────────────────┘
                             ↑
                    ┌────────┴────────┐
                    │   Attention 层   │
                    │  评分 → Softmax  │
                    │  → 加权求和      │
                    └────────┬────────┘
                   ┌─────────┴──────────┐
                   │  编码器输出 (H₁~H₄)  │
                   └─────────┬──────────┘
                             ↑
                   ┌─────────┴──────────┐
                   │    编码器（GRU）     │
                   └─────────┬──────────┘
                             ↑
                   ┌─────────┴──────────┐
                   │  输入："我 爱 你"     │
                   └────────────────────┘
```

### 每个组件的职责

| 组件 | 作用 | 输入 | 输出 |
|------|------|------|------|
| **编码器** | 读完整句原文 | token 序列 | 每个位置的隐藏状态 H₁~Hₙ |
| **Attention 层** | 计算当前该看原文哪里 | 解码器状态 + 编码器输出 | 上下文向量 + 注意力权重 |
| **解码器** | 逐词生成译文 | [上下文向量, 上一词向量] | 当前词的预测 |

---

## 4.2 💻 Attention 层的 PyTorch 实现

```python
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 拼接注意力所需的参数
        self.W = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
    
    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: (batch, hidden_size)
        encoder_outputs: (batch, seq_len, hidden_size)
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Step 1: 把 decoder_hidden 从 (batch, hidden) → (batch, seq_len, hidden)
        hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Step 2: 拼接 → 评分
        combined = torch.cat([hidden, encoder_outputs], dim=2)  # (batch, seq_len, 2*hidden)
        scores = self.v(torch.tanh(self.W(combined)))           # (batch, seq_len, 1)
        scores = scores.squeeze(2)                               # (batch, seq_len)
        
        # Step 3: Softmax 归一化
        attention_weights = F.softmax(scores, dim=1)             # (batch, seq_len)
        
        # Step 4: 加权求和得到上下文向量
        context = torch.bmm(attention_weights.unsqueeze(1),
                           encoder_outputs)                     # (batch, 1, hidden)
        context = context.squeeze(1)                             # (batch, hidden)
        
        return context, attention_weights
```

### ⚡ 关于 `torch.bmm`（Batch Matrix Multiply）

`bmm` 是 Attention 实现里最重要也最容易困惑的运算。

```python
# bmm = batch matrix multiply → 批量矩阵乘法
# 输入: (batch, n, m) × (batch, m, p) → (batch, n, p)

# 在 Attention 里的用法：
attention_weights = (batch, seq_len)        # 先 squeeze 成 2D
attention_weights = attention_weights.unsqueeze(1)  # → (batch, 1, seq_len)
encoder_outputs   = (batch, seq_len, hidden_size)

context = torch.bmm(attention_weights, encoder_outputs)
# (batch, 1, seq_len) × (batch, seq_len, hidden_size)
# → (batch, 1, hidden_size)
```

**bmm 的作用**：对 batch 里的**每一个样本独立做矩阵乘法**，互不干扰。

---

## 4.3 Teacher Forcing（教师强制）

训练 Seq2Seq 时的一个关键技巧：

```
推理时（生成时）：
  解码器上一步的输出 → 作为这一步的输入
  
训练时（Teacher Forcing）：
  真实的上一词 → 作为这一步的输入  ← 而不是用模型自己预测的
```

### 为什么需要 Teacher Forcing？

```
假设翻译 "我 爱 你" → "I love you"

不用 Teacher Forcing：
  第 1 步: 输入 <SOS> → 预测 "I" ✅
  第 2 步: 输入 "I" → 预测 "love" ✅
  第 3 步: 输入 "love" → 预测 "you" ✅
  
但如果第 1 步就错了：
  第 1 步: 输入 <SOS> → 预测 "He" ❌  ← 一步错，步步错！
  第 2 步: 输入 "He" → 预测 ???        ← 错误累积
```

**Teacher Forcing = 不走歪路**：训练时就算模型预测错了，也给它喂正确的词，让它专注于"每一步该学什么"，而不是被自己的错误带偏。

---

# 第五部分 · 训练与评估

---

## 5.1 🏋️ 训练流程

```
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. 编码器处理输入
        encoder_outputs, encoder_hidden = encoder(input_seqs)
        
        # 2. 解码器 + Attention（Teacher Forcing）
        decoder_input = target_seqs[:, 0]   # <SOS> 开始符
        decoder_hidden = encoder_hidden       # 编码器最后状态作初始
        
        for t in range(1, max_target_len):
            # Attention 层
            context, weights = attention(decoder_hidden, encoder_outputs)
            
            # 解码器
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, context)
            
            # 计算损失
            loss += criterion(decoder_output, target_seqs[:, t])
            
            # Teacher Forcing：喂真实词（而不是模型预测的）
            decoder_input = target_seqs[:, t]
        
        # 3. 反向传播
        loss.backward()
        optimizer.step()
```

---

## 5.2 🧪 推理（测试）

推理时不再用 Teacher Forcing，而是**用自己的预测**作为下一步输入：

```
推理流程：
  输入: "我 爱 你"
        ↓
  编码器: 读完所有词 → 输出 H₁ H₂ H₃
        ↓
  解码器 Step 1:
    Attention → 算权重（重点关注 "我"）
    预测 → "I" ✅
        ↓
  解码器 Step 2:
    输入上一步的预测 "I" → Attention → 关注 "爱"
    预测 → "love" ✅
        ↓
  解码器 Step 3:
    输入上一步的预测 "love" → Attention → 关注 "你"
    预测 → "you" ✅
        ↓
  解码器预测 <EOS> → 停止
```

---

## 5.3 📈 用 TensorBoard 观察训练

训练过程中可以观察 Attention 权重的可视化：

```
输入: "我  爱  你"
       ↓  ↓  ↓
输出 "I"  →  权重高亮在 "我"    ← 对齐
输出 "love" → 权重高亮在 "爱"
输出 "you"  → 权重高亮在 "你"
```

这是一个好的 Attention 模型——**对齐正确**，翻译准确。

---

# 第六部分 · 关键总结

---

## 6.1 💡 Attention 为什么是里程碑？

| 解决了什么问题 | 怎么解决的 |
|--------------|-----------|
| 长序列信息丢失 | 不再依赖单一语义向量，直接看原文每个位置 |
| 对齐问题（翻译时不知道对应哪个词） | 自动学习源语言和目标语言的对齐关系 |
| 可解释性 | 通过注意力权重能看到"模型在看哪里" |

---

## 6.2 📊 Seq2Seq vs Seq2Seq + Attention

| 对比维度 | 无 Attention | 有 Attention |
|:--------:|:------------:|:------------:|
| 短句翻译 | ✅ 还行 | ✅ 好 |
| 长句翻译 | ❌ 明显变差 | ✅ 几乎不受长度影响 |
| 语义向量压力 | ⭐⭐⭐ 全压在一个向量上 | ⭐ 分摊到各个位置 |
| 可解释性 | ❌ 黑箱，不知道看哪里 | ✅ 能看到权重分布 |
| 计算量 | 小 | 稍大（多一步评分） |

---

## 6.3 🔗 为 Transformer 做铺垫

Attention 机制的直接延续就是 **Transformer** 的核心——**Self-Attention**。

```
Seq2Seq Attention:    解码器 ←看→ 编码器      （跨序列）
Self-Attention:       每个词 ←看→ 所有词     （单序列内）
```

| | Seq2Seq Attention | Self-Attention（Transformer） |
|:---:|:---:|:---:|
| 查询是谁 | 解码器状态 | 序列里的每个词 |
| 键值是谁 | 编码器输出 | 序列里的每个词 |
| 看几遍 | 每步看一次 | 一次全都看 |
| 并行？ | ❌ 必须串行生成 | ✅ 可以并行计算 |

> 💡 **Attention 是 Transformer 的"前传"**：理解了 Attention，Transformer 的 Self-Attention 就是把"查询"换成"同一个序列里的每个位置"。

---

## ❓ 自测问题

<details>
<summary><b>Q1：传统 Seq2Seq 有什么问题？Attention 怎么解决的？</b></summary>

问题：编码器只输出一个固定长度的语义向量，长句信息丢失严重。  
解决：生成每一步时"回头看"原文所有位置，按重要程度加权求和。
</details>

<details>
<summary><b>Q2：Attention 的四步流程是什么？</b></summary>

① 评分函数算分数 → ② Softmax 归一化为权重 → ③ 加权求和得到上下文向量 → ④ 融合到解码器输入。
</details>

<details>
<summary><b>Q3：三种注意力评分函数分别是什么？</b></summary>

① 点积（Dot）：直接做点积，最快但要求维度一致  
② 一般（General）：中间加一个可学习矩阵 W  
③ 拼接（Concat）：拼接后过 MLP，最灵活但最慢
</details>

<details>
<summary><b>Q4：上下文向量（context vector）是怎么算出来的？</b></summary>

编码器所有输出 H₁~Hₙ 按注意力权重加权求和：context = Σ weightᵢ × Hᵢ
</details>

<details>
<summary><b>Q5：什么是 Teacher Forcing？为什么需要它？</b></summary>

训练时用真实的目标词作为解码器输入，而不是用模型自己预测的。防止错误累积，让模型每个时间步专注于学习正确的映射。
</details>

<details>
<summary><b>Q6：torch.bmm 是做什么的？</b></summary>

批量矩阵乘法（Batch Matrix Multiply）：对 batch 里每个样本独立做矩阵乘法。在 Attention 里用于把注意力权重和编码器输出相乘得到上下文向量。
</details>

<details>
<summary><b>Q7：Seq2Seq Attention 和 Transformer 的 Self-Attention 有什么区别？</b></summary>

Seq2Seq Attention 是"两个序列之间"的注意力（解码器看编码器），必须串行。  
Self-Attention 是"序列内部"的注意力（每个词看所有词），可以全部并行计算。
</details>

---

## 📌 Anki 卡片建议

| # | 正面 | 背面 |
|---|------|------|
| 1 | Attention 解决了 Seq2Seq 的什么核心问题？ | 固定语义向量导致长句信息丢失 |
| 2 | Attention 四步流程 | 评分 → Softmax → 加权求和 → 融合 |
| 3 | 上下文向量怎么算？ | 编码器所有输出按注意力权重加权求和 |
| 4 | 三种注意力评分函数 | 点积(Dot) / 一般(General) / 拼接(Concat) |
| 5 | 什么是 Teacher Forcing？ | 训练时喂真实词，不喂模型预测的词 |
| 6 | torch.bmm 的作用 | 批量矩阵乘法，batch 内每个样本独立做矩阵乘 |
| 7 | Seq2Seq Attention vs Self-Attention | 跨序列 vs 序列内部；串行 vs 并行 |

---

> **📖 复习提示**：Attention 是整个课程**承上启下最关键的一章**。
> - **承上**：解决 Seq2Seq 的瓶颈
> - **启下**：理解 Self-Attention → 理解 Transformer → 理解 GPT/BERT → 理解你现在做的 Agent
>
> 如果你能向别人讲清"Attention 的四步"和"为什么它比纯 Seq2Seq 强"，就到位了。接下来 Day8 的 Transformer 就是在这个基础上的扩展。
