# Transformer 前馈网络与填充掩码

## 1. 课程元信息

- **主题**：Transformer 源码中的 Position-wise FFN（用卷积实现）+ Padding Mask 机制
- **前置知识**：多头注意力、卷积神经网络基础
- **核心目标**：理解 FFN 先升维再降维的设计意图，以及 Padding Mask 如何在注意力中屏蔽无效位置

---

## 2. 核心概念图谱

| 术语 | 英文 | 通俗解释 |
|------|------|---------|
| Position-wise | Position-wise | 对序列中**每个位置独立**做相同的运算，不跨位置 |
| 前馈神经网络 | FFN (Feed-Forward Network) | 两层全连接（这里用卷积实现），先升维再降维 |
| d_ff | d_ff (Feed-Forward Dimension) | FFN 中间层的维度，通常是 d_model 的 4 倍（512→2048→512） |
| 卷积核 | Kernel | Conv1D 中滑动窗口的大小，这里 kernel_size=1（逐点运算） |
| 填充掩码 | Padding Mask | 标记 `<pad>` 填充位置，让注意力机制忽略这些无效位置 |
| 广播机制 | Broadcasting | PyTorch 自动将低维张量扩展至高维进行运算 |

---

## 3. 技术原理 / 流程拆解

### 3.1 Position-wise FFN 架构

```
输入: [batch, seq_len, d_model] (512维)
      ↓
Step 1: 升维（Conv1D: 512 → 2048）
  Conv1d(in_channels=512, out_channels=2048, kernel_size=1)
  → 每个位置的向量从512维扩展到2048维
      ↓
Step 2: ReLU 激活
  ReLU() → 引入非线性
      ↓
Step 3: 降维（Conv1D: 2048 → 512）
  Conv1d(in_channels=2048, out_channels=512, kernel_size=1)
  → 恢复到原始维度
      ↓
Step 4: 残差连接 + Layer Norm
  output = LayerNorm(input + FFN_output)
```

**为什么要先升维再降维？**
- 类似于"把字句→把字句+被字句+其他句式→把字句"的改写过程
- 升维到更高空间（2048维），让模型有更多参数空间来学习特征变换
- 再降维回原始空间（512维），保持与残差连接的维度匹配

**为什么用 Conv1D kernel=1 而不是 Linear？**
- `kernel_size=1` 的 Conv1D 等价于逐位置的 Linear
- 但 Conv1D 需要调整维度顺序：`[batch, seq_len, d_model]` → `[batch, d_model, seq_len]`（通道在前）
- 运算后再 transpose 回来

### 3.2 Padding Mask 机制

```
问题：batch 中句子长度不一，短句用 <pad> 补齐
     注意力机制会错误地把注意力分配到 <pad> 上

解决方案：生成一个布尔掩码矩阵，标记 <pad> 的位置
         在 softmax 之前，将掩码位置的值设为 -inf（softmax 后趋于 0）
```

**掩码判断逻辑：**
```python
# pad_token 通常为 0
# 判断 key 中哪些位置是 pad
mask = (key == 0)  # → 布尔张量: True=pad, False=有效
```

**广播机制的作用：**
```
原始掩码形状: [batch, key_length]（二维）
注意力分数形状: [batch, q_length, key_length]（三维）

通过广播机制，将二维掩码扩展为三维：
[batch, key_length] → unsqueeze(1) → [batch, 1, key_length]
→ 自动广播为 [batch, q_length, key_length]
```

**为什么只标记 K 不标记 Q？**
- 注意力分数 `Q·K^T` 中，K 对应被关注的"源"位置
- 只需要告诉模型哪些"源"位置是无效的（pad）
- 在 **Encoder 的自注意力**中：Q=K=编码器输入，标记自己的 pad
- 在 **Decoder 的交叉注意力**中：Q 来自解码器，K 来自编码器 → **只标记编码器的 pad**，解码器的 pad 信息在交叉注意力层用不到

> 老师强调："K 来自编码端，所以编码端的 pad 信息有用；Q 来自解码端，解码端的 pad 信息在交叉注意力层没有用到。"

---

## 4. 代码核心

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        # Step 1: 升维卷积 (512 → 2048)
        self.conv1 = nn.Conv1d(in_channels=d_model, 
                                out_channels=d_ff, 
                                kernel_size=1)
        # Step 2: 降维卷积 (2048 → 512)
        self.conv2 = nn.Conv1d(in_channels=d_ff, 
                                out_channels=d_model, 
                                kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, inputs):
        residual = inputs  # 保存残差
        
        # 调整维度: [batch, seq_len, d_model] → [batch, d_model, seq_len]
        output = inputs.transpose(1, 2)
        
        # 升维 + ReLU
        output = F.relu(self.conv1(output))
        # 降维
        output = self.conv2(output)
        
        # 调整回来: [batch, d_model, seq_len] → [batch, seq_len, d_model]
        output = output.transpose(1, 2)
        
        # 残差连接 + Layer Norm
        return self.layer_norm(output + residual)


def get_attention_padding_mask(seq_q, seq_k):
    """生成填充掩码，标记 K 中的 pad 位置"""
    # seq_k: [batch, key_length]
    # 检查哪些位置是 pad (值为0)
    mask = (seq_k == 0).unsqueeze(1)  # [batch, 1, key_length]
    # 广播机制自动扩展为 [batch, q_length, key_length]
    return mask
```

---

## 5. 避坑指南

| 注意点 | 说明 |
|--------|------|
| Conv1D 期望通道维度在前 | FFN 输入输出需要 transpose 交换 seq_len 和 d_model 维度 |
| FFN 是 Position-wise 但不是 Positional | 只对每个位置独立运算，不跨位置，所以不携带位置信息 |
| d_ff 默认是 d_model 的 4 倍 | 原论文 512→2048→512，这是可配置的超参数 |
| Padding Mask 中 pad_token 不一定是 0 | 取决于词表定义，但通常 `pad_idx=0` |
| 交叉注意力中 pad 信息来自编码器 | Q 来自解码器，K 来自编码器，掩码只标记编码器的 pad |
| Subsequence Mask + Padding Mask 同时使用 | 前者遮未来，后者遮 pad，两者做 element-wise 与操作 |

---

## 6. 对比与思考

### 6.1 FFN 中的 Conv1D vs Linear

| 方式 | 维度处理 | 本质 |
|------|---------|------|
| `nn.Linear(d_model, d_ff)` | 无需转置 | 标准的全连接层 |
| `nn.Conv1d(d_model, d_ff, 1)` | 需转置 `transpose(1,2)` | kernel=1 时等价于全连接，但 Conv1D 更通用 |

### 6.2 两种掩码的对比

| 类型 | Subsequence Mask | Padding Mask |
|------|-----------------|-------------|
| **遮蔽对象** | 未来位置 | 无效填充位 |
| **矩阵形状** | `[batch, seq_len, seq_len]` | `[batch, 1, seq_len]` → 广播 |
| **应用位置** | Decoder Self-Attention | Encoder + Decoder 所有注意力层 |
| **生成方式** | `torch.triu(ones)` | `seq_k == pad_idx` |

---

## 7. 本节课思维导图

- Position-wise FFN
  - 本质：每个位置独立的双层网络
  - 流程：升维 (512→2048) → ReLU → 降维 (2048→512) → 残差 + Layer Norm
  - 实现：Conv1D(kernel=1)，需 transpose 调整维度
  - d_ff = 4 × d_model
- Padding Mask
  - 作用：让注意力忽略 `<pad>` 位置
  - 判断：`seq_k == 0` → 布尔张量
  - 广播：`[batch, 1, key_len]` → `[batch, q_len, key_len]`
  - 只标记 K（编码器端），不标记 Q
  - 与 Subsequence Mask 同时作用于 Decoder
