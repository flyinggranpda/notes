# Transformer Encoder 部分

## 1. 课程元信息

- **主题**：Transformer 源码中的 Encoder 实现 — Encoder Layer 内部结构 + 整体 Encoder 组装
- **前置知识**：多头注意力、FFN、位置编码、Padding Mask
- **核心目标**：理解 Encoder Layer 和整体 Encoder 的关系，以及数据在其中的流转过程

---

## 2. 核心概念图谱

| 术语 | 英文 | 通俗解释 |
|------|------|---------|
| 编码器层 | Encoder Layer | Transformer 中一个完整的处理块 = 多头注意力 + FFN |
| 编码器 | Encoder | N 个 Encoder Layer 堆叠而成的整体 |
| 自注意力掩码 | Self-Attention Mask | Encoder 中用于屏蔽 `<pad>` 位置的 Padding Mask |
| 堆叠 | Stack / ModuleList | 将 N 个相同的 Encoder Layer 串联起来 |

---

## 3. 技术原理 / 流程拆解

### 3.1 Encoder Layer（单层结构）

```
Encoder Layer 输入: [batch, seq_len, d_model]
      ↓
┌─────────────────────────────────────┐
│  1. Multi-Head Self-Attention       │
│     Q = K = V = encoder_inputs      │
│     掩码: Padding Mask（屏蔽 pad）  │
│     输出 → 残差连接 + Layer Norm    │
├─────────────────────────────────────┤
│  2. Position-wise FFN               │
│     升维 (512→2048) → ReLU → 降维   │
│     输出 → 残差连接 + Layer Norm    │
└─────────────────────────────────────┘
      ↓
Encoder Layer 输出: [batch, seq_len, d_model]
```

**代码核心：**
```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
    
    def forward(self, encoder_inputs, self_attention_mask):
        # Step 1: 自注意力（Q=K=V=encoder_inputs）
        attention_output, _ = self.self_attention(
            encoder_inputs,      # Q
            encoder_inputs,      # K
            encoder_inputs,      # V
            self_attention_mask  # Padding Mask
        )
        
        # Step 2: FFN
        output = self.feed_forward(attention_output)
        return output
```

### 3.2 整体 Encoder（N 层堆叠）

```
原始数据 SRC（源语言词）
      ↓
Step 1: Embedding
  SRC_embedding = nn.Embedding(src_vocab_size, d_model)
  → [batch, src_length, d_model]
      ↓
Step 2: Positional Encoding
  output = PositionalEncoding(embedded)
  → [batch, src_length, d_model]
      ↓
Step 3: Padding Mask
  mask = get_attention_padding_mask(src_inputs, src_inputs)
  → 标记 pad 位置
      ↓
Step 4: N 层 Encoder Layer 堆叠
  for layer in self.layers:
      output = layer(output, mask)
  → [batch, src_length, d_model]
      ↓
Encoder 输出 → 传给 Decoder 的 Cross-Attention 作为 K 和 V
```

**代码中的维度变化过程：**
```python
class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, vocab_size, max_len):
        # 1. 词嵌入
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        # 2. 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        # 3. N 层堆叠
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
    
    def forward(self, src_inputs):
        # 输入形状: [batch_size, src_length]
        
        # Step 1: Embedding
        # [batch, src_len] → [batch, src_len, d_model]
        output = self.src_embedding(src_inputs)
        
        # Step 2: Positional Encoding（需两次 transpose 对齐维度）
        output = output.transpose(0, 1)        # [src_len, batch, d_model]
        output = self.positional_encoding(output)
        output = output.transpose(0, 1)        # [batch, src_len, d_model]
        
        # Step 3: Padding Mask
        mask = get_attention_padding_mask(src_inputs, src_inputs)
        # 形状: [batch, 1, src_len]
        
        # Step 4: 堆叠 N 层
        self_attentions = []
        for layer in self.layers:
            output = layer(output, mask)
            self_attentions.append(output)
        
        # 输出形状: [batch, src_len, d_model]
        return output, self_attentions
```

### 3.3 两次 Transpose 的原因

```
第一次 transpose(0,1):
  [batch, src_len, d_model] → [src_len, batch, d_model]
  原因：PositionalEncoding 期望第一维度是序列长度（方便按位置索引）

第二次 transpose(0,1):
  [src_len, batch, d_model] → [batch, src_len, d_model]
  原因：后续 Encoder Layer 期望第一维度是 batch（方便并行计算）
```

---

## 4. Encoder 的完整流程总结

```
SRC 输入（原始词索引）
    │
    ├── SRC Embedding（词嵌入）
    │       ↓
    ├── Positional Encoding（位置编码）
    │       ↓
    ├── Padding Mask（生成掩码）
    │       ↓
    ├── Encoder Layer 1（自注意力 + FFN）
    │       ↓
    ├── Encoder Layer 2（自注意力 + FFN）
    │       ↓
    ├── ...（堆叠 N 层）
    │       ↓
    └── Encoder Layer N
            ↓
    输出特征 → 传递给 Decoder（作为 K 和 V）
```

---

## 5. 避坑指南

| 注意点 | 说明 |
|--------|------|
| Encoder Layer 中 Q=K=V | 全部来自同一个 encoder_inputs（自注意力） |
| Encoder 没有 Subsequence Mask | 编码器可以看到完整序列，不需要防止偷看未来 |
| 两次 transpose 是为了 PositionalEncoding | 位置编码期望序列长度在第一维，编码器期望 batch 在第一维 |
| `src_length` vs `seq_len` | src_length 特指源语言的序列长度，是一种特殊的 seq_len |
| 输出 `[batch, src_len, d_model]` 传给 Decoder | Decoder 的 Cross-Attention 以这个输出作为 K 和 V |

---

## 6. Encoder vs Decoder 结构对比

| 对比维度 | Encoder Layer | Decoder Layer |
|---------|--------------|--------------|
| 注意力层数 | 1 层（自注意力） | 2 层（自注意力 + 交叉注意力） |
| Subsequence Mask | 无 | 有 |
| Padding Mask | 有 | 有 |
| QKV 来源 | Q=K=V=自身 | 自注意力: Q=K=V=Decoder；交叉: Q=Decoder, K=V=Encoder |
| 输出用途 | 传给 Decoder 做 K/V | 映射到词汇分布 |

---

## 7. 本节课思维导图

- Transformer Encoder
  - Encoder Layer（单层）
    - Multi-Head Self-Attention（Q=K=V=inputs）
    - Position-wise FFN
    - 残差连接 + Layer Norm
  - 整体 Encoder（N 层堆叠）
    - SRC Embedding
    - Positional Encoding（两次 transpose）
    - Padding Mask 生成
    - N 层 Encoder Layer 依次处理
    - 收集每层注意力权重
    - 输出传给 Decoder
  - 数据流转
    - [batch, src_len] → Embedding → [batch, src_len, d_model]
    → transpose → Positional Encoding → transpose
    → N 层 Layer → [batch, src_len, d_model]
  - 关键要点
    - Encoder 无 Subsequence Mask（可看全局）
    - 两次 transpose 对齐不同模块的维度期望
    - Encoder 输出作 Decoder 交叉注意力的 K/V
