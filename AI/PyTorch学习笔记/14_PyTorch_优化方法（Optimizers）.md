# 🧠 第十四部分：优化方法（Optimizers）

> 📺 视频来源：PyTorch 深度学习 · 优化器详解（动量法/学习率衰减/AdaGrad/RMSProp/Adam/AdamW）
> 🎯 核心目标：掌握 PyTorch 中各种优化器的原理与使用
> 📝 风格：等高线可视化 + 代码实操对比

---

## 📌 一、优化器概述

### 优化器解决了什么问题？

有了损失函数后，我们需要**最小化损失**。怎么做？用**梯度下降**更新参数。而优化器就是"如何更新参数"的具体实现。

```
参数_new = 参数_old - 学习率 × 梯度
```

### PyTorch 中的优化器

所有优化器都在 `torch.optim` 模块下：

| 优化器 | 特点 | 使用频率 |
|--------|------|:-------:|
| **SGD** | 基础梯度下降 | ⭐⭐ |
| **SGD + Momentum** | 带动量的 SGD，加速收敛 | ⭐⭐⭐ |
| **AdaGrad** | 自适应学习率（历史梯度平方和） | ⭐ |
| **RMSProp** | AdaGrad 改进版（指数移动平均） | ⭐⭐ |
| **Adam** ⭐ | 动量 + RMSProp 结合，工程首选 | ⭐⭐⭐⭐⭐ |
| **AdamW** ⭐ | Adam + 正确的权值衰减 | ⭐⭐⭐⭐⭐（最新推荐） |

---

## 📌 二、SGD + Momentum（动量法）

### 2.1 标准 SGD 的问题

标准 SGD 根据当前梯度更新参数，但会遇到：
- **震荡严重**：在等高线的"窄谷"中来回摆动
- **收敛慢**：尤其是平坦区域

### 2.2 动量法（Momentum）

动量法引入**历史梯度的加权和**（类似物理中的动量）：

```
v = momentum × v_old + gradient    ← 速度累积
w = w - lr × v                       ← 用速度更新参数
```

- 梯度方向一致时 → 加速前进
- 梯度方向相反时 → 减缓震荡

### 2.3 代码对比

```python
import torch
import torch.optim as optim

# 参数
params = torch.tensor([-7.0, 2.0], requires_grad=True)

# SGD（无动量）
optim_sgd = optim.SGD([params], lr=0.01)

# SGD + 动量（momentum=0.9）
optim_momentum = optim.SGD([params], lr=0.01, momentum=0.9)
```

> 💡 **只加了一个参数 `momentum=0.9` 就实现了动量法！**

### 2.4 等高线可视化效果

```
SGD（无动量）：       Momentum（0.9）：
   ╱╲                    ╱╲
 ╱    ╲               ╱    ╲
╱      ╲             ╱      ╲
 震荡大，收敛慢        平滑快速到达最小值
```

---

## 📌 三、学习率衰减（Learning Rate Decay）

### 3.1 为什么需要学习率衰减？

动量法有个问题：**容易冲过头**。一开始步子大没问题，但快接近最小值时还大步子就会来回震荡。

> 💡 **思路**：训练初期用大学习率快速下降，后期用小学习率精细调整。

### 3.2 PyTorch 的 LR Scheduler

学习率衰减通过 `torch.optim.lr_scheduler` 实现，它和优化器**绑定使用**：

```python
optimizer = optim.SGD([params], lr=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
```

**使用流程**：

```python
for epoch in range(epochs):
    # 1. 前向传播
    loss = forward(x)
    # 2. 反向传播
    loss.backward()
    # 3. 更新参数
    optimizer.step()
    # 4. 梯度清零
    optimizer.zero_grad()
    # 5. 更新学习率 ⭐
    scheduler.step()
```

### 3.3 三种衰减策略

#### (1) StepLR — 等间隔衰减

每隔固定的步数，学习率乘以 gamma：

```python
scheduler = StepLR(optimizer, step_size=20, gamma=0.7)
# 每 20 轮：lr = lr × 0.7
```

```
学习率变化：阶梯状下降
lr
│
│   ████
│       ████
│           ████
│               ████
└───────────────────→ epoch
    20    40    60
```

#### (2) MultiStepLR — 指定间隔衰减

在指定的轮次衰减：

```python
scheduler = MultiStepLR(optimizer, milestones=[50, 100, 200], gamma=0.7)
# 在第 50、100、200 轮时：lr = lr × 0.7
```

```
学习率变化：不等宽台阶
lr
│
│   ████████████
│               ██████████████
│                             ██████████████████████
└───────────────────────────────────→ epoch
    50          100             200
```

#### (3) ExponentialLR — 指数衰减

**每轮**都衰减：

```python
scheduler = ExponentialLR(optimizer, gamma=0.99)
# 每轮：lr = lr × 0.99
```

```
学习率变化：平滑曲线
lr
│
│   ╲
│    ╲
│     ╲
│      ╲
└───────────────────→ epoch
```

> ⚠️ ExponentialLR 的 gamma 要接近 1（如 0.99），否则学习率会太快衰减到 0。

### 3.4 查看当前学习率

```python
current_lr = optimizer.param_groups[0]['lr']
```

---

## 📌 四、自适应学习率（AdaGrad / RMSProp）

前面三种衰减策略是**固定的**（不管梯度多大，到了轮次就衰减）。那有没有一种方法让学习率**根据梯度自动调整**呢？

### 4.1 AdaGrad

**核心思路**：对每个参数单独调整学习率——历史上梯度大的参数，学习率衰减更快。

```
H = H + g²                       ← 历史梯度平方和累积
w = w - lr / √(H + ε) × g        ← 梯度大 → 分母大 → 步长小
```

**优点**：每个参数自适应，适合稀疏数据
**缺点**：H 只增不减，学习率会**单调衰减到 0**

```python
optimizer = optim.Adagrad([params], lr=0.9)
```

### 4.2 RMSProp

**改进点**：用**指数移动平均**代替简单的平方和累加，不再单调衰减。

```
H = α × H + (1-α) × g²          ← 指数移动平均（α=0.99）
w = w - lr / √(H + ε) × g
```

- 越久远的梯度权重越小
- 学习率不会衰减到 0

```python
optimizer = optim.RMSprop([params], lr=0.1, alpha=0.99)
```

### AdaGrad vs RMSProp 对比

| 对比维度 | AdaGrad | RMSProp |
|---------|:------:|:-------:|
| **H 的计算** | 简单累加 g² | 指数移动平均 |
| **学习率** | 单调衰减到 0 ❌ | 不会衰减到 0 ✅ |
| **适用场景** | 稀疏数据 | 通用 |
| **超参数** | 无额外参数 | alpha（默认 0.99） |

---

## 📌 五、Adam 与 AdamW（工程首选）

### 5.1 Adam = Momentum + RMSProp

Adam 融合了两种思想：
- **动量法**：累积历史梯度（一阶矩估计）
- **RMSProp**：自适应学习率（二阶矩估计）

```
m = β₁ × m + (1-β₁) × g          ← 动量项（一阶矩）
v = β₂ × v + (1-β₂) × g²         ← 自适应项（二阶矩）

# 偏差修正（解决初始偏差）
m_hat = m / (1 - β₁ᵗ)
v_hat = v / (1 - β₂ᵗ)

w = w - lr × m_hat / (√(v_hat) + ε)
```

**默认超参数**（原始论文推荐，基本不用改）：

| 参数 | 默认值 | 含义 |
|------|:-----:|------|
| **lr** | 0.001 | 学习率 |
| **betas** | (0.9, 0.999) | β₁（动量系数）, β₂（自适应系数） |
| **eps** | 1e-8 | 防止除零 |

```python
optimizer = optim.Adam([params], lr=0.001)
# 使用默认的 betas=(0.9, 0.999)
```

> ⭐ **Adam 几乎可以直接用默认参数**，不需要像 SGD 那样精细调参。这就是它成为"工程首选"的原因。

### 5.2 AdamW — Adam 的正确权值衰减

**为什么需要 AdamW？**

`weight_decay`（权值衰减/正则化）是每个优化器都有的参数：

```python
optimizer = optim.Adam([params], lr=0.001, weight_decay=0.01)
```

但 Adam 的实现方式有问题：权值衰减被加到了梯度里，然后参与了 Adam 的自适应计算，导致**学习率衰减加速**，效果打了折扣。

**AdamW 的改进**：把权值衰减从梯度计算中分离出来，直接作用在参数更新上：

```
标准 Adam：   梯度 = 原始梯度 + weight_decay × w       ← 参与自适应
AdamW：       参数更新 = Adam更新 - lr × weight_decay × w  ← 独立处理
```

```python
# 官方推荐使用 AdamW 替代 Adam
optimizer = optim.AdamW([params], lr=0.001, weight_decay=0.01)
```

> 📌 PyTorch 官方从 2019 年起推荐使用 **AdamW 替代 Adam**。

### 5.3 Adam vs SGD 可视化对比

```
等高线下降轨迹：

SGD（lr=0.9）：     震荡大，但若调参精细也能收敛
    ╱╲╱╲╱╲

Momentum（0.9）：   平滑快速，但可能冲过
    ╱╲__╱╲__╱

Adam（lr=0.001）：  快速平滑到达，默认参数就表现优秀
    ╱_____╱___

AdamW：            与 Adam 类似（无 weight_decay 时几乎一样）
```

---

## 📝 本章总结 + 选择指南

### 🌳 知识树

```
优化方法（Optimizers）
│
├── ① SGD（基础）
│   └── +Momentum → 加速收敛、减缓震荡
│
├── ② 学习率衰减（LR Scheduler）
│   ├── StepLR → 等间隔阶梯下降
│   ├── MultiStepLR → 指定节点下降
│   └── ExponentialLR → 平滑指数衰减
│
├── ③ 自适应学习率
│   ├── AdaGrad → 历史梯度平方和（单调衰减 ❌）
│   └── RMSProp → 指数移动平均（不衰减 ✅）
│
└── ④ Adam / AdamW ⭐（融合动量+自适应）
    ├── Adam → 工程首选，默认参数就表现优秀
    └── AdamW → 正确的权值衰减，官方推荐
```

### 🚀 优化器选择速查

| 场景 | 推荐优化器 | 学习率 |
|------|:---------:|:-----:|
| **快速实验/默认选择** | **Adam** ✅ | 0.001 |
| **需要权值衰减（正则化）** | **AdamW** ✅ | 0.001 |
| **数据稀疏（NLP/推荐）** | **Adam** ✅ | 0.001 |
| **CV 领域/精细调参** | SGD + Momentum | 0.01~0.1 |
| **需要自适应学习率** | RMSProp | 0.001~0.1 |

### ⭐ 经验法则

> **对于初学者和大多数工程任务：**
> 1. 首先用 **Adam**（默认学习率 0.001）
> 2. 需要正则化时换成 **AdamW**
> 3. 如果追求极致效果，再精调 **SGD + Momentum**
>
> **关于学习率衰减：**
> - Adam 自身已有自适应机制，通常**不需要额外加 LR Scheduler**
> - SGD/Momentum 配合 LR Scheduler 效果更好
