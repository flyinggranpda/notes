# 🧠 第十三部分：损失函数（Loss Functions）

> 📺 视频来源：PyTorch 深度学习 · 损失函数详解
> 🎯 核心目标：掌握分类任务和回归任务的常用损失函数及其 PyTorch 实现
> 📝 风格：理论公式 + 代码实操

---

## 📌 一、分类任务 vs 回归任务

损失函数根据**任务类型**不同分为两大类：

| 任务类型 | 输出层激活函数 | 常见损失函数 | 用途 |
|---------|:------------:|-------------|------|
| **二分类** | Sigmoid | **BCE Loss**（二元交叉熵） | 是/否、正/负 |
| **多分类** | Softmax | **CrossEntropyLoss**（交叉熵） | 猫/狗/鸟... |
| **回归** | 无（Identity） | **MSE / MAE / SmoothL1** | 预测连续值 |

---

## 📌 二、二分类损失 — BCE Loss（二元交叉熵）

### 2.1 公式回顾

```
BCE Loss = -1/N × Σ [ yᵢ × log(pᵢ) + (1-yᵢ) × log(1-pᵢ) ]
```

其中：
- `yᵢ` = 真实标签（0 或 1）
- `pᵢ` = 预测为正类（类别 1）的概率

**本质**：只看正确类别对应的预测概率，取对数后求平均。

### 2.2 输出层配合

二分类任务输出层用 **Sigmoid** 激活函数：输出一个概率值 p（正类概率），负类概率就是 1-p。

```
┌─ 输入数据 ─→ Linear(..., 1) ─→ Sigmoid ─→ p（正类概率）
```

### 2.3 代码演示

```python
import torch
import torch.nn as nn

# 1. 输入数据（经过前向传播后）
X = torch.randn(3, 2)                          # 3条数据，2个特征
y_pred = torch.sigmoid(X)                       # → 预测概率（3×2）

# 但注意：BCE 的输入通常只需 1 个输出（正类概率）
# 这里为演示，假设 3×2 是每个样本的2个类别概率

# 2. 真实标签（独热编码形式，或概率分布）
target = torch.tensor([[1.0, 0.0],    # 属于第0类
                       [0.0, 1.0],    # 属于第1类
                       [0.3, 0.7]])   # 分布概率

# 3. 定义损失函数
bce_loss = nn.BCELoss()

# 4. 计算损失
loss = bce_loss(y_pred, target)
print(f"BCE Loss: {loss.item():.4f}")
```

> ⚠️ **注意**：BCE Loss 要求 `input` 和 `target` **形状相同**。如果你的 Sigmoid 输出是 N×1（只输出正类概率），target 也必须是 N×1 的 0/1 标签。

### 2.4 参数说明

| 参数 | 说明 |
|------|------|
| `nn.BCELoss()` | 输入必须是经过 Sigmoid 的概率值 |
| `nn.BCEWithLogitsLoss()` | 输入可以是原始 logits（内部自带 Sigmoid），数值更稳定 |

> 💡 **推荐**：实际工程中用 `BCEWithLogitsLoss` 而不是 `BCELoss` + 手动 Sigmoid，可以避免数值不稳定的问题。

---

## 📌 三、多分类损失 — CrossEntropyLoss（交叉熵）

### 3.1 公式回顾

```
CrossEntropy = -1/N × Σᵢ log(pᵢ[ yᵢ ])
```

其中：
- `yᵢ` = 第 i 个样本的**真实类别编号**（如 0, 1, 2...）
- `pᵢ[yᵢ]` = 模型预测该样本属于类别 yᵢ 的概率

**本质**：把真实类别对应的预测概率拿出来取负对数，求平均。

### 3.2 PyTorch 的特殊设计

⚠️ **关键点**：`nn.CrossEntropyLoss` **内部已经集成了 Softmax**，所以：
- ❌ 不需要手动在输出层加 Softmax
- ✅ 直接把**网络最后一层的原始输出（logits）** 传进去

```
代码中：input = logits（未激活的原始值）
底层逻辑：input → log_softmax → NLLLoss → 最终损失
```

### 3.3 代码演示（场景一：真实标签为顺序编号）

```python
# 6条数据，8分类
input = torch.randn(6, 8)              # logits（原始输出）
target = torch.randint(0, 8, (6,))    # 真实标签编号：[3, 7, 0, 2, ...]

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(input, target)
print(f"CrossEntropy Loss: {loss.item():.4f}")
```

### 3.4 代码演示（场景二：真实标签为概率分布/独热编码）

```python
# 如果 target 传概率分布（形状与 input 相同）
input = torch.randn(6, 8)
target = torch.rand(6, 8)                       # 随机概率分布
# target = torch.softmax(torch.randn(6, 8), dim=1)  # 每行和为1

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(input, target)
```

### 3.5 CrossEntropyLoss vs BCE Loss 对比

| 对比维度 | BCE Loss | CrossEntropyLoss |
|---------|:-------:|:---------------:|
| **适用任务** | 二分类 | 多分类 |
| **输出层激活函数** | 需要 Sigmoid | **不需要**（内部自带 Softmax） |
| **input 形状** | 与 target 相同 | 可以不同 |
| **target 形状** | 与 input 相同（概率/独热） | N 维向量（类别编号）或 N×C 矩阵（概率） |

---

## 📌 四、回归损失 — MSE / MAE / Smooth L1

### 4.1 三种损失函数对比

| 损失函数 | 别名 | 公式 | 特点 |
|---------|:---:|------|------|
| **MSE**（均方误差） | L2 Loss | 1/N × Σ(y - ŷ)² | 光滑可导，但对异常值敏感 ❌ |
| **MAE**（平均绝对误差） | L1 Loss | 1/N × Σ\|y - ŷ\| | 对异常值鲁棒，但零点不可导 |
| **Smooth L1** | Huber Loss 变体 | 分区间定义 | 结合二者优点 ✅ |

### 4.2 Smooth L1 公式

```
           ┌ 0.5 × (y - ŷ)²,          |y - ŷ| < 1
SmoothL1 = ┤
           └ |y - ŷ| - 0.5,            |y - ŷ| ≥ 1
```

- 小误差时 → 用 L2（光滑可导）
- 大误差时 → 用 L1（降低异常值影响）
- 在 ±1 处平滑连接

### 4.3 代码演示

```python
import torch
import torch.nn as nn

# 数据
input = torch.randn(3, 5)    # 预测值
target = torch.randn(3, 5)   # 真实值

# 定义三种损失函数
mae_loss = nn.L1Loss()               # MAE
mse_loss = nn.MSELoss()              # MSE
smooth_l1_loss = nn.SmoothL1Loss()   # Smooth L1

# 计算
mae = mae_loss(input, target)
mse = mse_loss(input, target)
sl1 = smooth_l1_loss(input, target)

print(f"MAE:       {mae.item():.4f}")
print(f"MSE:       {mse.item():.4f}")  # 通常比 MAE 大（平方放大）
print(f"Smooth L1: {sl1.item():.4f}")  # 介于两者之间
```

### 4.4 reduction 参数

所有回归损失函数都支持 `reduction` 参数：

```python
nn.MSELoss(reduction='mean')   # 默认：求平均
nn.MSELoss(reduction='sum')    # 求和，不平均
nn.MSELoss(reduction='none')   # 不聚合，返回每个位置的损失
```

> 💡 用 `reduction='sum'` 相当于不做平均得到的"总误差"，`'none'` 可以看到每个数据点的损失。

### 4.5 Smooth L1 的 beta 参数

```python
# beta 控制 L1/L2 切换的阈值（默认 1.0）
nn.SmoothL1Loss(beta=1.0)   # 标准版本
nn.SmoothL1Loss(beta=0.5)   # 更小的阈值 → 更早切换到 L1
```

---

## 📝 本章总结 + 选择指南

### 🌳 知识树

```
损失函数（Loss Functions）
│
├── ① 分类问题
│   ├── 二分类 → BCE Loss（配合 Sigmoid）
│   │   └── 推荐：BCEWithLogitsLoss（数值更稳定）
│   └── 多分类 → CrossEntropyLoss（自带 Softmax）
│       └── 直接传 logits + 类别编号
│
└── ② 回归问题
    ├── MSE（L2 Loss）→ 光滑可导，对异常值敏感
    ├── MAE（L1 Loss）→ 鲁棒，零点不可导
    └── Smooth L1 → 二者结合，工程首选 ✅
```

### 🚀 损失函数选择速查

| 场景 | 推荐的损失函数 | 配合的激活函数 |
|------|:--------------:|:-------------:|
| **二分类** | `BCEWithLogitsLoss` ✅ | 不需要手动 Sigmoid |
| **多分类** | `CrossEntropyLoss` ✅ | 不需要手动 Softmax |
| **回归（一般）** | `MSELoss` | 无 |
| **回归（有异常值）** | `SmoothL1Loss` ✅ | 无 |
| **回归（需要鲁棒性）** | `L1Loss` | 无 |

### ⭐ 经验法则

> **分类任务**：二分类用 `BCEWithLogitsLoss`，多分类用 `CrossEntropyLoss`（都不需要手动加激活函数）。
>
> **回归任务**：一般用 `MSELoss`；如果数据有异常值（outliers），用 `SmoothL1Loss` 更安全。
>
> 损失函数的**绝对值不重要**，重要的是它能在训练过程中**持续下降**。
