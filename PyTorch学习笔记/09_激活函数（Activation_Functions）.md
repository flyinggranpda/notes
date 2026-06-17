# 🧠 第九部分：激活函数（Activation Functions）

> 📺 视频来源：PyTorch 深度学习 · 激活函数详解
> 🎯 核心目标：掌握四种常用激活函数及其在 PyTorch 中的使用
> 📝 风格：图像可视化 + 自动求导验证 + 代码实践

---

## 📌 一、为什么需要激活函数？

### 非线性的必要性

如果神经网络只有线性层（`nn.Linear`），无论堆叠多少层：

```
Linear → Linear → Linear
```

最终结果**等价于一个线性层**（线性组合的线性组合还是线性组合）。

> ❌ 没有激活函数 → 多层线性 = 单层线性 → **无法处理复杂问题**

激活函数的作用就是**引入非线性**，让神经网络有能力学习复杂的模式。

```
Linear → 激活函数 → Linear → 激活函数 → ...
```

每一层"线性变换 + 非线性激活"的组合，才是真正的"深度"。

### PyTorch 中的激活函数

PyTorch 提供了两种使用方式：

```python
# 方式一：基于张量直接调用函数
y = torch.sigmoid(x)     # 函数式
y = x.sigmoid()          # 方法式（如 x.sum(), x.mean() 一样）

# 方式二：作为神经网络层（后面会学）
# nn.Sigmoid(), nn.Tanh(), nn.ReLU(), nn.Softmax()
```

> 💡 在本文中我们使用**方法一**，因为激活函数本质上就是逐元素运算，像加减乘除一样直接调就完了。

---

## 📌 二、Sigmoid（S 型曲线）

### 2.1 表达式与图像

```
公式：σ(x) = 1 / (1 + e⁻ˣ)

值域：(0, 1)    ← 输出可以看作概率
中心：x=0 → σ(0) = 0.5
```

图像是一个 S 型曲线：
- 在 x∈[-6, 6] 范围内变化明显
- 在 x<-6 或 x>6 时几乎平了（梯度接近 0）

### 2.2 导数

```
σ'(x) = σ(x) × (1 - σ(x))
最大值：σ'(0) = 0.25
```

### 2.3 用 PyTorch + Autograd 画图（⭐ 重点）

这是一种很有趣的方式：**直接利用反向传播自动计算导数**，而不需要手动写导函数表达式。

```python
import torch
import matplotlib.pyplot as plt

# ① 定义数据（开启梯度追踪）
X = torch.linspace(-10, 10, 1000, requires_grad=True)

# ② 前向传播：Sigmoid
Y = torch.sigmoid(X)

# ③ 反向传播：自动算导数
Y.sum().backward()   # sum 让标量梯度 = 1，不改变导数

# ④ 画图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：原函数
axes[0].plot(X.detach().numpy(), Y.detach().numpy(), color='purple')
axes[0].axhline(y=1, color='gray', alpha=0.5)
axes[0].axhline(y=0.5, color='gray', alpha=0.5)
axes[0].set_title("Sigmoid")

# 右图：导函数（直接从梯度中取！）
axes[1].plot(X.detach().numpy(), X.grad.numpy(), color='purple')
axes[1].set_title("Sigmoid'")

# 美化：去掉上、右边框
for ax in [axes[0], axes[1]]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

plt.show()
```

> 🎯 **关键点**：`X.grad` 就是 Sigmoid 在每个 X 处的导数值！因为自动微分引擎已经帮我们算好了。

### 2.4 Sigmoid 的优缺点

| 优点 | 缺点 |
|------|------|
| ✅ 输出在 (0,1)，可解释为概率 | ❌ 梯度消失：导数最大值仅 0.25，深度网络中层层累乘 → 梯度几乎为 0 |
| ✅ 光滑可导 | ❌ 不是关于原点对称（输出均值 > 0） |
| ✅ 历史最悠久的激活函数 | ❌ 计算量相对较大（指数运算） |

---

## 📌 三、Tanh（双曲正切）

### 3.1 表达式与图像

```
公式：tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
            = 2 × sigmoid(2x) - 1

值域：(-1, 1)    ← 关于原点中心对称
```

Tanh 其实就是 Sigmoid 的**平移缩放版本**：
`sigmoid(2x)` 取值为 (0,1)，乘以 2 得 (0,2)，再减 1 得 (-1,1)。

### 3.2 导数

```
tanh'(x) = 1 - tanh²(x)
最大值：tanh'(0) = 1
```

### 3.3 代码画图

```python
X = torch.linspace(-5, 5, 1000, requires_grad=True)
Y = torch.tanh(X)          # PyTorch 自带 tanh

Y.sum().backward()

# 画原函数（左）和导函数（右）
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(X.detach().numpy(), Y.detach().numpy(), color='purple')
axes[0].axhline(y=1, color='gray', alpha=0.5)
axes[0].axhline(y=-1, color='gray', alpha=0.5)
axes[0].set_title("Tanh")

axes[1].plot(X.detach().numpy(), X.grad.numpy(), color='purple')
axes[1].set_title("Tanh'")
# ... 同样的边框美化 ...
plt.show()
```

### 3.4 Sigmoid vs Tanh 对比

| 对比维度 | Sigmoid | Tanh |
|---------|---------|------|
| 值域 | **(0, 1)** | **(-1, 1)** |
| 对称中心 | 0.5 | **0**（原点对称） |
| 导数最大值 | 0.25 | **1** |
| 有效变化区间 | [-6, 6] | **[-3, 3]**（区间更窄） |
| 梯度消失问题 | ❌ 严重 | ❌ 也有（深度网络仍会消失） |

> ⚠️ **Sigmoid 和 Tanh 都不适合深度神经网络的隐藏层**——因为它们的导数在两端接近 0，反向传播层层累乘后梯度消失。

---

## 📌 四、ReLU（深度网络首选）

### 4.1 表达式与图像

```
公式：ReLU(x) = max(0, x)

      ┌ 0,  x ≤ 0
      └ x,  x > 0
```

图像就是一个**折线**：
- x ≤ 0：水平线 y = 0
- x > 0：斜线 y = x（斜率 1）

### 4.2 导数

```
ReLU'(x) = 阶跃函数

      ┌ 0,  x ≤ 0
      └ 1,  x > 0

x = 0 处不可导，工程上规定导数为 0
```

### 4.3 为什么 ReLU 是深度网络的首选？

| 特性 | ReLU | 对比 Sigmoid/Tanh |
|------|------|------------------|
| **梯度消失** | ❌ **不会**！x>0 时导数恒为 1，不衰减 | ✅ 导数 < 1，多层累乘就没了 |
| **计算量** | ✅ 只需一次 max 比较 | ❌ 需要指数运算 |
| **稀疏性** | ✅ x≤0 输出为 0，相当于让部分神经元"休眠" | ❌ 输出全非 0 |
| **收敛速度** | ✅ 更快 | ❌ 较慢 |

### 4.4 代码画图

```python
X = torch.linspace(-5, 5, 1000, requires_grad=True)
Y = torch.relu(X)          # PyTorch 自带 ReLU

Y.sum().backward()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(X.detach().numpy(), Y.detach().numpy(), color='purple')
axes[0].set_title("ReLU")

axes[1].plot(X.detach().numpy(), X.grad.numpy(), color='purple')
axes[1].set_title("ReLU'")
# ... 边框美化 ...
plt.show()
```

### 4.5 注意点

- **ReLU 的"死亡"问题**：如果学习率太大，某些神经元可能永远输出 0（梯度一直是 0，再也激活不了）
- 解决方案：用 **LeakyReLU**、**ELU** 等变体，给 x<0 的部分一个很小的负斜率

---

## 📌 五、Softmax（多分类概率转换）

### 5.1 是什么

前面的 Sigmoid/Tanh/ReLU 都是**逐元素操作**——每个输入对应一个输出，互不影响。

而 **Softmax** 不一样，它把一组数值转换成**概率分布**（所有输出之和 = 1）：

```
公式：Softmax(xᵢ) = eˣⁱ / Σⱼ eˣʲ

对第 i 个元素：
  分母 = 所有 eˣʲ 的总和
  分子 = 自己的 eˣⁱ

结果：一组概率，和为 1
```

**典型用途**：多分类任务的**输出层**
- 输入：网络最后一层的原始得分（logits）
- 输出：每个类别的概率，取最大概率的类别作为预测结果

### 5.2 代码演示 + dim 参数详解

```python
import torch

# 假设分类问题的输出：3条数据，5个类别
X = torch.randn(3, 5)
print(X)
# tensor([[-1.03,  0.57,  0.79, -0.42, -1.18],
#         [ 1.31,  1.39, -0.89, -0.52, -2.84],
#         [ 0.85, -0.07,  1.01, -1.64,  0.55]])

# ❌ dim=0：按列计算概率
y_prob_0 = torch.softmax(X, dim=0)
print(y_prob_0.sum(dim=0))  # 每一列和为 1
# tensor([1.000, 1.000, 1.000, 1.000, 1.000])
# 这是"3分类"的概率！—— 一列就是一个 batch 的不同数据

# ✅ dim=1：按行计算概率（正确用法）
y_prob_1 = torch.softmax(X, dim=1)
print(y_prob_1.sum(dim=1))  # 每一行和为 1
# tensor([1.000, 1.000, 1.000])
```

### 5.3 dim 参数的理解

| dim | 操作方向 | 含义 |
|-----|---------|------|
| **`dim=0`** | 按列 | 同一列不同样本之间算概率（❌ 不常用） |
| **`dim=1`** | 按行 | **同一行（同一个样本）不同类别之间算概率（✅ 标准用法）** |

> 💡 **记忆方法**：一行 = 一条数据，对一条数据的不同类别输出算概率 → `dim=1`

### 5.4 Softmax 的偏导数

当输出层使用 Softmax + CrossEntropyLoss 时：
- 如果 i = j（对同一个类别的偏导）：`∂yᵢ/∂xⱼ = yᵢ(1 - yᵢ)`
- 如果 i ≠ j（对不同类别的偏导）：`∂yᵢ/∂xⱼ = -yᵢ·yⱼ`

结果是**一个矩阵**（雅可比矩阵），而不是一个数值，所以一般不会像 Sigmoid 那样画图表示。

---

## 📝 本章总结 + 对比速查表

### 🌳 知识树

```
激活函数（Activation Functions）
│
├── ① 为什么需要？
│   └── 引入非线性，否则多层线性 = 单层线性
│
├── ② Sigmoid（σ）
│   ├── 公式：1/(1+e⁻ˣ)
│   ├── 值域：(0, 1)，过 (0, 0.5)
│   ├── 导数：σ(x)(1-σ(x))，最大 0.25
│   └── 问题：梯度消失 ❌
│
├── ③ Tanh
│   ├── 公式：2σ(2x)-1
│   ├── 值域：(-1, 1)，原点对称
│   ├── 导数：1-tanh²(x)，最大 1
│   └── 问题：梯度消失 ❌（深度网络不用）
│
├── ④ ReLU ⭐（首选）
│   ├── 公式：max(0, x)
│   ├── 导数：x>0→1, x≤0→0
│   ├── 优点：无梯度消失、计算快、稀疏性
│   └── 注意："死亡 ReLU"问题（学习率别太大）
│
└── ⑤ Softmax（输出层专用）
    ├── 公式：eˣⁱ/Σeˣʲ → 概率分布
    ├── 多分类任务的输出层
    └── dim=1 → 对每个样本的各类别算概率
```

### 🚀 四种激活函数速查

| 函数 | 公式 | 值域 | 导数最大值 | 适用场景 | 梯度消失 |
|------|------|:---:|:--------:|---------|:-------:|
| **Sigmoid** | 1/(1+e⁻ˣ) | (0, 1) | 0.25 | 二分类输出层、浅层网络 | ❌ 严重 |
| **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | **1** | 浅层网络、RNN | ❌ 有 |
| **ReLU** ⭐ | max(0, x) | [0, ∞) | **1** | **隐藏层首选** | ✅ **无** |
| **Softmax** | eˣⁱ/Σeˣʲ | (0,1), 和为1 | — | **多分类输出层** | — |

### ⭐ 经验法则

> **隐藏层用 ReLU（或其变体），输出层根据任务决定：**
> - 回归 → 不用激活函数（或 Identity）
> - 二分类 → Sigmoid
> - 多分类 → Softmax

---

---