# Transformer 位置编码（Positional Encoding）

## 1. 课程元信息

- **主题**：Transformer 源码中的正弦余弦位置编码实现
- **前置知识**：三角函数周期性、Transformer 整体架构
- **核心目标**：理解固定位置编码的公式推导、代码实现，以及为什么用正弦余弦能做到位置感知

---

## 2. 核心概念图谱

| 术语 | 英文 | 通俗解释 |
|------|------|---------|
| 位置编码 | Positional Encoding (PE) | 为序列中每个位置添加的唯一标识，让自注意力知道词的先后顺序 |
| 最大长度 | max_len | 预设的最大序列长度，超出部分无法编码（默认5000） |
| 正弦函数 | Sine | 用于编码**偶数位**（2i）的三角函数 |
| 余弦函数 | Cosine | 用于编码**奇数位**（2i+1）的三角函数 |
| 周期 | Period | 正弦/余弦函数的重复间隔，利用周期性编码相对位置 |
| 缓冲区 | Buffer | PyTorch 中不参与梯度更新的张量，适合存储固定的位置编码 |

---

## 3. 技术原理 / 流程拆解

### 3.1 位置编码公式

原始论文中的位置编码公式：

```
PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))   ← 偶数维度
PE(pos, 2i+1)   = cos(pos / 10000^(2i / d_model))   ← 奇数维度

其中：
- pos: 单词在句中的位置索引（从0开始）
- i:   维度索引（从0到d_model/2 - 1）
- d_model: 模型维度（默认512）
```

**为什么不同维度用不同的频率？**
- 低维（i 小）：频率高，相邻位置差异大，适合编码**绝对位置**
- 高维（i 大）：频率低，相邻位置差异小，适合编码**相对位置**
- 不同频率的组合使模型能同时感知绝对位置和相对位置

### 3.2 指数转换技巧

公式中需要计算 `pos / 10000^(2i / d_model)`，即 `pos × 10000^(-2i/d_model)`。当 i 很大时直接算 `10000^(2i/d_model)` 数值极大会溢出精度，因此通过对数转换简化：

```
10000^(-2i / d_model) = exp((-2i / d_model) × ln(10000))
                      = exp(2i × (-ln(10000) / d_model))
                      = position × div_term

其中 div_term = -ln(10000) / d_model    （预先算好，避免重复计算幂运算）
position = pos                           （位置索引 [0, 1, 2, ...]）
```

这样将幂运算转为乘法 + exp，数值稳定性更好。

### 3.3 奇偶维度的周期性

```
同一维度上相邻两个位置（pos 和 pos+1）：

偶数位 (2i): 使用 sin
奇数位 (2i+1): 使用 cos

sin(x + 1) 和 cos(x) 之间的关系：
  当维度差为1时（偶→奇），sin 和 cos 的值产生有规律的变化
  这种变化携带着"相对位置"的信息
```

**为什么正弦余弦搭配？**
- 利用三角函数的**周期性**：相隔 k 个位置的编码可以表示为当前位置编码的线性变换
- 模型可以学习到"相距 k 的位置"这个抽象概念

### 3.4 代码流程

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Step 1: 创建空的位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # Step 2: 生成位置序列 [max_len, 1] → 广播用
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # Step 3: 计算指数项（对数转换技巧）
        div_term = torch.exp(
            torch.arange(0, d_model, 2) *  # 0, 2, 4, ..., 510
            -(math.log(10000.0) / d_model)
        )
        
        # Step 4: 偶数位填 sin，奇数位填 cos
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数列
        
        # Step 5: 增加 batch 维度 → [1, max_len, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Step 6: 注册为缓冲区（不参与梯度更新）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x 形状: [seq_len, batch_size, d_model]
        # 将位置编码加到输入上（只取 x 的实际长度）
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
```

---

## 4. 代码关键点解读

| 代码行 | 作用 |
|--------|------|
| `torch.arange(0, max_len).unsqueeze(1)` | 生成 `[0, 1, 2, ..., max_len-1]` 作为位置索引 |
| `torch.arange(0, d_model, 2)` | 生成 `[0, 2, 4, ..., d_model-2]` 作为偶数维度索引 |
| `pe[:, 0::2]` | 取所有行的偶数列（0, 2, 4, ...） |
| `pe[:, 1::2]` | 取所有行的奇数列（1, 3, 5, ...） |
| `register_buffer` | 将 pe 注册为模型的一部分，但不会在反向传播中更新 |
| `self.pe[:x.size(0), :]` | 只取当前序列实际长度的编码（不一定用完 max_len） |

---

## 5. 避坑指南

| 注意点 | 说明 |
|--------|------|
| 位置从 0 开始 | pos=0 代表句子第一个词，不是 pos=1 |
| max_len 是硬上限 | 超过 max_len 的位置无法编码，需提前设置足够大 |
| 位置编码不做梯度更新 | 用 `register_buffer` 注册为固定值，不参与训练 |
| 维度匹配 | PE 输出需要和输入嵌入的 `[seq_len, batch, d_model]` 对齐 |
| 奇偶对应 2i 和 2i+1 | i 从 0 到 d_model/2 - 1，2i 对应偶数维度，2i+1 对应奇数维度 |
| 正弦余弦不是唯一方案 | 这是**固定编码**，还有可学习位置编码、相对位置编码等变体 |

---

## 6. 对比与思考

### 6.1 为什么需要位置编码？

```
自注意力的固有缺陷：
  "我打你" 和 "你打我" 的词完全一样，但意思完全相反
  自注意力不知道词的先后顺序

解决方案：
  给每个位置添加一个"位置指纹"→ 位置编码
```

### 6.2 固定正弦余弦 vs 可学习位置编码

| 方案 | 优点 | 缺点 |
|------|------|------|
| 正弦余弦（固定） | 可外推到更长序列（理论上），无需额外参数 | 灵活性低 |
| 可学习位置编码 | 适应具体任务，效果可能更好 | 不能处理超过 max_len 的序列 |
| RoPE（旋转位置编码） | 相对位置感知，外推性好 | 实现更复杂 |

---

## 7. 本节课思维导图

- Transformer 位置编码
  - 公式：PE(pos,2i)=sin, PE(pos,2i+1)=cos
  - 代码实现流程
    - 创建 `[max_len, d_model]` 零矩阵
    - 生成位置序列 + 维度索引
    - 指数转换（对数技巧）
    - 偶数位填 sin，奇数位填 cos
    - unsqueeze + transpose 对齐维度
    - register_buffer 固定
  - 关键设计
    - 不同维度不同频率（低维高频→绝对位置，高维低频→相对位置）
    - 奇偶搭配利用三角函数周期性
    - 默认 max_len=5000
    - 需与输入嵌入维度对齐
