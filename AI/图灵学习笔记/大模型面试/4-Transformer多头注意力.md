# Transformer 多头注意力机制（Multi-Head Attention）

## 1. 课程元信息

- **主题**：Transformer 源码中的多头注意力实现 — QKV 线性映射、分头、掩码分发、残差与层归一化
- **前置知识**：Transformer 整体架构、掩码机制、缩放点积注意力公式
- **核心目标**：理解多头注意力的代码实现流程 — 如何对 QKV 分头计算再拼接

---

## 2. 核心概念图谱

| 术语 | 英文 | 通俗解释 |
|------|------|---------|
| 线性层 | Linear Layer / NN.Linear | 全连接层：`output = weight × input + bias`，用于将 QKV 投影到不同子空间 |
| 模型维度 | d_model | Transformer 中向量的统一维度，原始论文默认 512 |
| 注意力头维度 | d_k / d_v | 每个注意力头的维度，`d_k = d_v = d_model / n_heads` |
| 头数 | n_heads | 将 QKV 拆分成多少个独立子空间，原始论文默认 8 |
| 分头 | Split Heads | 将 d_model 维度的向量拆成 n_heads 个 d_k 维的子向量 |
| 残差连接 | Residual Connection | 将层的输入和输出相加，缓解梯度消失 |
| 层归一化 | Layer Norm | 对每个样本的隐层做归一化，稳定训练 |
| 转置 | Transpose | 交换张量的两个维度，改变张量的"视角" |

---

## 3. 技术原理 / 流程拆解

### 3.1 多头注意力完整流程

```
输入 Q, K, V (形状相同: [batch, seq_len, d_model])
      ↓
Step 1: 线性映射（W_Q, W_K, W_V）
  Q = Linear(Q) → [batch, seq_len, d_model]
  K = Linear(K) → [batch, seq_len, d_k × n_heads]
  V = Linear(V) → [batch, seq_len, d_v × n_heads]
      ↓
Step 2: 分头（Split into n_heads）
  Q → [batch, seq_len, n_heads, d_k] → transpose → [batch, n_heads, seq_len, d_k]
  K → [batch, seq_len, n_heads, d_k] → transpose → [batch, n_heads, seq_len, d_k]
  V → [batch, seq_len, n_heads, d_v] → transpose → [batch, n_heads, seq_len, d_v]
      ↓
Step 3: 每个头独立计算 Scaled Dot-Product Attention
  每个头: Attention(Q_i, K_i, V_i) = softmax(Q_i · K_i^T / √d_k) · V_i
      ↓
Step 4: 拼接（Concat Heads）
  [batch, n_heads, seq_len, d_v] → transpose → [batch, seq_len, n_heads, d_v]
  → view → [batch, seq_len, n_heads × d_v] = [batch, seq_len, d_model]
      ↓
Step 5: 输出线性映射 + 残差连接 + Layer Norm
  output = Linear(concat) + residual_input
  output = LayerNorm(output)
```

### 3.2 为什么 W_Q / W_K 用 d_k 维度，而 W_V 用 d_v 维度？

```
W_Q 输出维度: n_heads × d_k    ← 因为 Q·K^T 需要 d_k 维度
W_K 输出维度: n_heads × d_k    ← 同上
W_V 输出维度: n_heads × d_v    ← V 只参与加权求和，不影响 softmax 分数
```

d_k 和 d_v 在原始论文中相等（d_k = d_v = d_model / n_heads），但概念上可以不同。

### 3.3 Transpose 的作用（为什么交换 head 和 seq_len 维度）

```
分头后的张量形状: [batch, seq_len, n_heads, d_k]
transpose(1, 2) 后: [batch, n_heads, seq_len, d_k]
```

**意义**：将"头"的维度提前，使每个头可以**并行计算**注意力。比喻：
- 10个大任务（batch）
- 每个大任务8个小任务（seq_len）
- 4个人（n_heads）并行处理

交换维度后，每个人直接领取任务并行执行，提高效率。

### 3.4 掩码分发（Mask Repeat）

```python
# 原始掩码形状: [batch, 1, seq_len, seq_len]（已含一个 singleton 维度）
# 每个头需要相同的掩码，因此需要复制
attention_mask = mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
# 复制后形状: [batch, n_heads, seq_len, seq_len]
```

原始掩码是针对整个序列的，分头后需要**为每个头复制一份相同的掩码**。

### 3.5 attention scores vs context 的区别

老师在源码中强调这两个返回值的不同：

| 返回值 | 含义 | 计算过程 |
|--------|------|---------|
| `attn` (attention scores) | **注意力分数矩阵** | softmax(Q·K^T / √d_k) — 每个位置对其他位置的关注权重 |
| `context` (context) | **注意力加权特征** | attn × V — 用注意力分数加权求和后的特征向量 |

> **简单区分**：`scores` = "每个位置看其他位置有多重要"，`context` = "按照这个重要性重新组合后的特征"。

---

## 4. 代码核心流程

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_k = d_model // n_heads  # 每个头的维度: 64
        self.n_heads = n_heads
        
        # Step 1: 定义 Q/K/V 的线性映射层
        self.W_Q = nn.Linear(d_model, n_heads * self.d_k)  # 512 → 512
        self.W_K = nn.Linear(d_model, n_heads * self.d_k)
        self.W_V = nn.Linear(d_model, n_heads * self.d_k)
        
        # 输出线性层
        self.linear = nn.Linear(n_heads * self.d_k, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, Q, K, V, mask=None):
        # 保存残差连接输入
        residual = Q
        batch_size = Q.size(0)
        
        # Step 2: 线性映射 + 分头 + 转置
        q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Step 3: 掩码分发（每个头复制一份）
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        
        # Step 4: 计算注意力（每个头独立计算）
        context, attn = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Step 5: 拼接（转置回来 + view 合并）
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_k
        )
        
        # Step 6: 输出映射 + 残差连接 + Layer Norm
        output = self.linear(context)
        return self.layer_norm(output + residual), attn
```

---

## 5. 避坑指南

| 注意点 | 说明 |
|--------|------|
| d_k 和 d_v 可以不等 | 但原始论文中相等，实践中也通常相等 |
| View 和 Transpose 的顺序 | 先 view 分头再 transpose 交换维度，顺序不能乱 |
| Contiguous 的必要性 | Transpose 后的张量内存不连续，需要 `.contiguous()` 才能 view |
| 掩码的 n_heads 维度复制 | `unsqueeze(1)` 插入头维度，`repeat` 复制到每个头 |
| 残差连接的输入是 Q | 第一次注意力时 Q=K=V=编码器输出；交叉注意力时 Q=解码器，K=V=编码器 |

---

## 6. 本节课思维导图

- Transformer 多头注意力机制
  - 线性映射（QKV）
    - W_Q / W_K：输出 n_heads × d_k
    - W_V：输出 n_heads × d_v
    - d_model = 512, n_heads = 8, d_k = d_v = 64
  - 分头（Split）
    - view → transpose
    - [batch, seq_len, n_heads, d_k] → [batch, n_heads, seq_len, d_k]
  - 掩码分发
    - unsqueeze(1) → repeat(1, n_heads, 1, 1)
  - 缩放点积注意力（每个头独立）
    - Score = Q·K^T / √d_k
    - Softmax → × V
  - 拼接（Concat）
    - transpose → contiguous → view
  - 输出层
    - Linear 映射回 d_model
    - 残差连接（+ residual）
    - Layer Norm
