# Transformer Decoder 部分

## 1. 课程元信息

- **主题**：Transformer 源码中的 Decoder 实现 — Decoder Layer 内部结构 + 整体 Decoder 组装 + 输出层
- **前置知识**：Encoder 架构、多头注意力、掩码机制、FFN
- **核心目标**：理解 Decoder Layer 中两层注意力的区别（Self-Attention vs Cross-Attention），以及 QKV 来源的差异

---

## 2. 核心概念图谱

| 术语 | 英文 | 通俗解释 |
|------|------|---------|
| 解码器自注意力 | Decoder Self-Attention | Q/K/V 全部来自 Decoder 自身，关注目标序列内部关系（带掩码） |
| 编码器-解码器注意力 | Encoder-Decoder Attention (Cross-Attention) | Q 来自 Decoder，K/V 来自 Encoder，建立输入和输出的关联 |
| 目标词嵌入 | Target Embedding | 将目标语言（如英文）的词转化为向量 |
| 输出线性层 | Output Linear Layer | 将 Decoder 输出映射到目标词表大小的向量 |
| Softmax 词汇分布 | Vocabulary Distribution | 每个位置下一个词的概率分布 |

---

## 3. 技术原理 / 流程拆解

### 3.1 Decoder Layer 内部结构

```
Decoder Layer 输入 (来自上一层或嵌入层)
      ↓
┌────────────────────────────────────────────┐
│  1. Masked Self-Attention（自注意力）       │
│     Q = K = V = Decoder 自身输出            │
│     作用：关注目标序列自身的关系             │
│     掩码：Subsequence Mask + Padding Mask   │
│     输出 → 残差连接 + Layer Norm           │
├────────────────────────────────────────────┤
│  2. Cross-Attention（交叉注意力）           │
│     Q = Decoder 输出（来自第一步）          │
│     K = V = Encoder 输出（来自编码器）      │
│     作用：建立"目标词"和"源句子"的关联      │
│     掩码：Padding Mask（只标 encoder 的pad）│
│     输出 → 残差连接 + Layer Norm           │
├────────────────────────────────────────────┤
│  3. Position-wise FFN                       │
│     和 Encoder 中的 FFN 完全一样            │
│     输出 → 残差连接 + Layer Norm           │
└────────────────────────────────────────────┘
      ↓
Decoder Layer 输出
```

### 3.2 QKV 来源对比

```
Encoder Layer 中的注意力（只有一种）：
  自注意力: Q = K = V = Encoder 自身输入

Decoder Layer 中的注意力（两种）：
  1. 自注意力:   Q = K = V = Decoder 自身输入
  2. 交叉注意力: Q = Decoder 输出, K = V = Encoder 输出
```

**代码中的体现：**
```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        # 第一层：Decoder 自注意力
        self.decoder_self_attention = MultiHeadAttention(d_model, n_heads)
        # 第二层：Encoder-Decoder 交叉注意力（注意命名差异）
        self.decoder_encoder_attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
    
    def forward(self, decoder_inputs, encoder_outputs, 
                self_mask, padding_mask):
        # Step 1: 自注意力（QKV 都来自 decoder）
        decoder_output, _ = self.decoder_self_attention(
            decoder_inputs, decoder_inputs, decoder_inputs, self_mask
        )
        
        # Step 2: 交叉注意力（Q 来自 decoder，KV 来自 encoder）
        decoder_output, _ = self.decoder_encoder_attention(
            decoder_output, encoder_outputs, encoder_outputs, padding_mask
        )
        
        # Step 3: FFN
        decoder_output = self.ffn(decoder_output)
        
        return decoder_output
```

### 3.3 训练阶段 vs 推理阶段的理解

```
训练阶段：
  Decoder 输入 = 完整的目标句子（带 <sos>）
  一次性看到整个目标序列（通过掩码防止偷看未来）
  并行计算所有位置的输出

推理阶段（生成时）：
  Decoder 输入 = 已生成的词
  逐个生成下一个词
  每一步只能看到已生成的词（自回归）
```

### 3.4 整体 Decoder 组装

```python
class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, ...):
        # 目标词嵌入 + 位置编码
        self.target_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # 堆叠 N 个 Decoder Layer
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, ...) for _ in range(n_layers)
        ])
    
    def forward(self, target_inputs, encoder_outputs, ...):
        # 词嵌入 + 位置编码 + dropout
        output = self.target_embedding(target_inputs)
        output = self.positional_encoding(output)
        
        # 逐层传递
        self_attentions = []
        cross_attentions = []
        for layer in self.layers:
            output, self_attn, cross_attn = layer(
                output, encoder_outputs, ...
            )
            self_attentions.append(self_attn)
            cross_attentions.append(cross_attn)
        
        return output, self_attentions, cross_attentions
```

### 3.5 输出层 — 映射到词汇分布

```python
class Transformer(nn.Module):
    def __init__(self, ...):
        self.encoder = Encoder(...)
        self.decoder = Decoder(...)
        # 输出线性层：d_model → 目标词表大小
        self.projection = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src_inputs, tgt_inputs):
        # 编码
        encoder_outputs = self.encoder(src_inputs)
        # 解码
        decoder_outputs, _, _ = self.decoder(tgt_inputs, encoder_outputs)
        # 映射到词表分布 [batch, seq_len, d_model] → [batch, seq_len, vocab]
        logits = self.projection(decoder_outputs)
        # 变形为 [batch × seq_len, vocab] 便于计算交叉熵
        return logits.view(-1, logits.size(-1))
```

---

## 4. 掩码的组合使用

Decoder 中**两种掩码同时起作用**：

```python
# Subsequence Mask（上三角）：防止看未来
subsequent_mask = get_subsequent_mask(seq)
# 形状: [batch, 1, seq_len, seq_len]

# Padding Mask：防止看 pad
padding_mask = get_padding_mask(seq_k, seq_k)
# 形状: [batch, 1, 1, seq_len]

# 两者合并：大于0的位置被屏蔽
combined_mask = (subsequent_mask + padding_mask) > 0
```

---

## 5. 避坑指南

| 注意点 | 说明 |
|--------|------|
| Decoder 输入的 QKV 来源要分清 | 自注意力中 QKV 都来自 Decoder；交叉注意力中 Q 来自 Decoder，KV 来自 Encoder |
| 训练时 Decoder 一次性输入完整句子 | 不是逐词输入，而是并行输入所有词（利用掩码防止偷看） |
| 交叉注意力标记 Encoder 的 pad，不标记 Decoder 的 | 因为 K 来自 Encoder，只需要告诉模型源语言哪些位置是无效的 |
| Decoder 输入和 Target 标签错一位 | Decoder 输入带 `<sos>`，Target 标签带 `<eos>`，模型预测下一个词 |
| 输出层 linear 的维度 | `d_model → tgt_vocab_size`，每个位置输出一个 vocab 大小的概率分布 |

---

## 6. 对比与思考

### 6.1 Encoder vs Decoder Layer

| 对比维度 | Encoder Layer | Decoder Layer |
|---------|--------------|--------------|
| 注意力层数 | 1 层（自注意力） | 2 层（自注意力 + 交叉注意力） |
| 自注意力掩码 | 无（可看全部） | 有（Subsequence Mask） |
| QKV 来源 | Q=K=V=Encoder 输入 | 自注意力: Q=K=V=Decoder；交叉: Q=Decoder, K=V=Encoder |
| 能否看到未来 | 能 | 不能 |
| 典型堆叠层数 | 6 | 6 |

### 6.2 原始论文默认超参数

| 参数 | 值 | 含义 |
|------|-----|------|
| d_model | 512 | 模型维度 |
| d_ff | 2048 | FFN 中间层维度（4×） |
| d_k / d_v | 64 | 每个注意力头的维度 |
| n_heads | 8 | 注意力头数 |
| n_layers | 6 | Encoder/Decoder 堆叠层数 |

---

## 7. 本节课思维导图

- Transformer Decoder
  - Decoder Layer（核心）
    - Masked Self-Attention（Q=K=V=Decoder）
    - Cross-Attention（Q=Decoder, K=V=Encoder）
    - Position-wise FFN
  - 整体 Decoder
    - Target Embedding + Positional Encoding
    - N 层 Decoder Layer 堆叠
    - 收集每层的自注意力和交叉注意力
  - 输出层
    - Linear(d_model → tgt_vocab_size)
    - view(-1, vocab_size) 便于损失计算
  - 两种掩码的组合
    - Subsequence Mask（上三角）
    - Padding Mask（标记 pad）
    - 相加后 >0 的位置被屏蔽
  - Encoder vs Decoder 对比
