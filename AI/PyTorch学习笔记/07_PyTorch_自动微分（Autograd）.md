# 🧠 第七部分：PyTorch 自动微分（Autograd）

---

## 📌 回顾：为什么需要自动微分？

还记得我们之前在**反向传播与计算图**笔记中学过的吗？

```
训练神经网络的"三部曲"：
① 前向传播（Forward） → 计算预测值 ŷ
② 计算损失（Loss） → 看预测值和真实值差多少
③ 反向传播（Backward）→ 算梯度 → 更新参数（W, b）
```

反向传播的核心就是**链式求导法则**。如果手工实现，每一层都要手写求导公式，非常痛苦。

**PyTorch 的自动微分（Autograd）就是来救场的**——你只需要搭好计算图，调一个 `.backward()`，所有梯度自动算好！

---

## ⚙️ 一、自动微分原理

### 1.1 Tensor 的 3 个关键属性

每个张量（Tensor）内部包含了自动微分所需的一切：

```
Tensor 对象
    ├── .data          ← 存储张量数据本身
    ├── .grad          ← 存储计算出的梯度
    ├── .grad_fn       ← 记录这个张量是怎么算出来的（计算函数）
    └── .requires_grad ← 是否开启梯度追踪（布尔值）
```

```python
import torch

# 创建一个张量，查看默认属性
x = torch.tensor([1.0, 2.0, 3.0])
print(x.requires_grad)        # False ← 默认为 False
```

### 1.2 `requires_grad` — 开启梯度追踪

只有设置了 `requires_grad=True` 的张量，PyTorch 才会追踪它的所有操作：

```python
# 开启梯度追踪
w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, 1, requires_grad=True)

print(w.requires_grad)        # True
print(w.grad)                 # None ← 还没计算梯度
```

> 输入数据（X, Y）一般不需要梯度，参数（W, b）需要梯度。

### 1.3 `grad_fn` — 记录计算操作

如果一个张量是通过运算得到的，它会记录自己是怎么算出来的：

```python
x = torch.tensor([1.0])
w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, 1, requires_grad=True)

z = w * x + b                 # 通过乘法和加法得到
print(z.grad_fn)              # <AddBackward0> ← 记录了这是"加法"操作
```

> **传播规则**：参与运算的张量中，只要有一个 `requires_grad=True`，结果张量的 `requires_grad` 也会自动变成 True，并且 `grad_fn` 会被记录。

### 1.4 动态计算图

PyTorch 使用的是**动态计算图**（跟 TensorFlow 1.x 的静态图不同）：

```
静态图（TensorFlow 1.x）：
    先搭图 → 再编译 → 再运行（僵化，不好调试）

动态图（PyTorch）✅：
    边算边搭图，每一步都记录 grad_fn（灵活，好调试）
```

> 每做一次前向传播，计算图就动态构建一次。反向传播后，计算图会自动释放以节省内存。

---

## 🔄 二、前向传播与计算图构建

### 2.1 完整的前向传播流程

以一个简单的线性模型为例：`ŷ = Wx + b`，损失函数用 MSE。

```python
import torch
import torch.nn as nn

# ========== 1. 定义数据 ==========
X = torch.tensor([[10.0]])        # 输入 ⭐ 形状 (1,1) ← 和 W, Z 保持一致
Y = torch.tensor([[3.0]])         # 真实标签 ⭐ 形状 (1,1) ← 和 Z 保持一致

# 输入数据不需要梯度 ❌
print(f"X.requires_grad: {X.requires_grad}")   # False
print(f"X.shape: {X.shape}")                    # torch.Size([1, 1])

# ========== 2. 初始化参数 ==========
W = torch.randn(1, 1, requires_grad=True)      # 权重 ← 需要梯度 ✅
B = torch.randn(1, 1, requires_grad=True)      # 偏置 ← 需要梯度 ✅

print(f"W.requires_grad: {W.requires_grad}")   # True
print(f"B.requires_grad: {B.requires_grad}")   # True

# ========== 3. 前向传播 ==========
Z = W @ X + B                 # ⭐ 矩阵乘法 @：线性模型 ŷ = Wx + b
print(f"Z: {Z}")
print(f"Z.shape: {Z.shape}")                    # torch.Size([1, 1]) ← 和 X, Y 一致 ✅
print(f"Z.requires_grad: {Z.requires_grad}")   # True ← 自动开启
print(f"Z.grad_fn: {Z.grad_fn}")                # <AddBackward0>

# ========== 4. 计算损失 ==========
loss_fn = nn.MSELoss()        # 均方误差损失函数
loss = loss_fn(Z, Y)          # 计算损失值 ⭐ Z(1,1) 和 Y(1,1) 形状一致 ✅
print(f"loss: {loss}")
print(f"loss.requires_grad: {loss.requires_grad}")   # True
print(f"loss.grad_fn: {loss.grad_fn}")                # <MseLossBackward0>
```

### 2.2 叶子节点 vs 非叶子节点

```
计算图结构：
                  W    B
                  \   /
      X  ──┐      \ /
            ├── [✕] ──→ Z ──→ [MSE Loss] ──→ loss
      W ───┘               ↑                        ↑
                            │                        │
                      非叶子节点                 根节点
                      （中间结果）             （反向传播起点）

叶子节点：X, Y, W, B（最初定义的，不依赖计算）
非叶子节点：Z（通过运算得到的）
根节点：loss（前向终点，反向起点）
```

```python
# 查看是否是叶子节点
print(f"X.is_leaf: {X.is_leaf}")    # True
print(f"W.is_leaf: {W.is_leaf}")    # True
print(f"Z.is_leaf: {Z.is_leaf}")    # False ← 通过计算得到的
print(f"loss.is_leaf: {loss.is_leaf}")  # False
```

---

## ⏪ 三、反向传播

### 3.1 `.backward()` — 一行代码算梯度

```python
# ========== 5. 反向传播 ==========
loss.backward()                # ← 就这一行！所有梯度自动算好

# ========== 6. 查看梯度 ==========
print(f"W.grad: {W.grad}")    # 权重 W 的梯度
print(f"B.grad: {B.grad}")    # 偏置 B 的梯度
```

> **要求**：`loss` 必须是一个**标量张量**（只有一个元素），才能调 `.backward()`。

### 3.2 完整代码示例（一键运行）

```python
import torch
import torch.nn as nn

# 1. 数据 ⭐ 统一用 (1,1) 形状
X = torch.tensor([[10.0]])   # (1,1)
Y = torch.tensor([[3.0]])    # (1,1)

# 2. 参数（开启梯度）
W = torch.randn(1, 1, requires_grad=True)
B = torch.randn(1, 1, requires_grad=True)

# 3. 前向传播
Z = W @ X + B                # ⭐ 矩阵乘法 @ 结果 (1,1) ← 和 Y 一致 ✅

# 4. 损失
loss_fn = nn.MSELoss()
loss = loss_fn(Z, Y)

# 5. 反向传播 ← 核心！
loss.backward()

# 6. 查看梯度
print(f"W 的梯度: {W.grad.item():.4f}")
print(f"B 的梯度: {B.grad.item():.4f}")
```

---

## 🌿 四、叶子节点 vs 非叶子节点 — 深入理解

### 4.1 核心区别

| | 叶子节点 | 非叶子节点 |
|---|---|---|
| **定义** | 初始定义的张量（数据/参数） | 通过运算得到的张量 |
| **举例** | X, Y, W, B | Z, loss |
| **`is_leaf`** | True | False |
| **梯度保留** | ✅ **梯度会保留**，可查看 | ❌ **梯度自动释放**，不能查看 |
| **`requires_grad`** | 可手动设 | 自动继承 |
| **可原地修改？** | ❌ 不行 | ✅ 可以 |

### 4.2 为什么非叶子节点的梯度会被释放？

```python
# 反向传播后，非叶子节点的 grad 被自动释放
loss.backward()
print(Z.grad)   # None ← 被释放了！
```

**原因**：非叶子节点（中间节点）的作用只是帮我们把梯度传递到叶子节点。传完了就没用了，释放掉可以节省内存。

### 4.3 如果想保留非叶子节点的梯度？

```python
Z.retain_grad()               # 强制保留 Z 的梯度
loss.backward()
print(Z.grad)                 # ✅ 现在就能看到了
```

### 4.4 叶子节点的梯度累积

叶子节点的梯度在反向传播后**不会自动清零**，而是会**累积**：

```python
loss.backward()
print(f"第1次: W.grad = {W.grad}")

loss.backward()               # 再次反向传播
print(f"第2次: W.grad = {W.grad}")  # 梯度累积了！翻倍了？！
```

#### 为什么 PyTorch 选择不清零？

这不是疏忽，而是**有意为之**，有两个原因：
  
**原因 ①：为了支持"梯度累积"技巧**

有时候显存不够一次喂太多数据，可以**分批次算梯度，累加起来再一次性更新参数**：

```python
# 显存不够，分 4 个小批次
for i in range(4):
    x_batch = data[i * 8:(i + 1) * 8]    # 每次只喂 8 条
    y_batch = label[i * 8:(i + 1) * 8]
    
    loss = model(x_batch, y_batch)
    loss.backward()   # 梯度会累积到 W.grad 上 ↗️
    # 这里不清零！

# 累积了 4 次的梯度 → 等效于一次喂 32 条数据
optimizer.step()      # 用累积的梯度更新一次参数
optimizer.zero_grad() # 清空，准备下一轮
```

> 就像**往桶里接水**——backward 是"倒一杯水"，倒 4 次后桶里有 4 杯，最后一次性更新。如果每次自动清零，这个技巧就用不了。

**原因 ②：PyTorch 的设计哲学——显式优于隐式**

```python
# TensorFlow 的做法：自动清零（隐式）
optimizer.minimize(loss)  # 梯度算完自动清，你没得选

# PyTorch 的做法：手动清零（显式）
loss.backward()            # 梯度累积
optimizer.step()           # 更新参数
optimizer.zero_grad()      # 手动清 ← 让你意识到这步的存在
```

PyTorch 希望开发者**明确知道**自己什么时候该清零，而不是框架替你决定。

#### 每次迭代前必须清零

大多数情况下，你并不需要累积梯度，所以训练循环的标准写法是：

```python
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()       # ← 不清就废了！梯度会翻倍累积
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

```python
# 清空梯度（训练循环中标准做法）
optimizer.zero_grad()         # 后面学优化器时再细讲
# 等价于手动：
# W.grad.zero_()
# B.grad.zero_()
```

---

## 📝 五、本章总结

### 知识地图

```
自动微分（Autograd）
    │
    ├── ① 三个关键属性
    │      ├── .data          ← 数据
    │      ├── .grad          ← 梯度（反向传播后才有）
    │      ├── .grad_fn       ← 计算函数（中间节点有）
    │      └── .requires_grad ← 是否开启追踪
    │
    ├── ② 前向传播
    │      ├── 定义数据（X, Y）← 不需要梯度
    │      ├── 定义参数（W, B）← 需要梯度 ✅
    │      ├── 计算预测值 Z = Wx + B
    │      └── 计算损失 loss = loss_fn(Z, Y)
    │
    ├── ③ 反向传播
    │      └── loss.backward()  ← 自动算梯度！
    │
    ├── ④ 叶子节点 vs 非叶子节点
    │      ├── 叶子：X, Y, W, B → 梯度保留 ✅
    │      ├── 非叶子：Z, loss → 梯度自动释放 ❌
    │      ├── retain_grad() → 强制保留非叶子梯度
    │      └── 叶子梯度会累积 → 需 optimizer.zero_grad()
    │
    └── ⑤ 动态计算图
           └── 边算边搭，反向后自动释放
```

### 关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| **`requires_grad`** | 是否开启梯度追踪，参数设为 True |
| **`.grad`** | 存储计算出的梯度（叶子节点） |
| **`.grad_fn`** | 记录张量是由什么运算得到的 |
| **`backward()`** | 从损失开始反向传播，自动算梯度 |
| **叶子节点** | 初始定义的张量，梯度会保留 |
| **非叶子节点** | 中间计算结果，梯度自动释放 |
| **`retain_grad()`** | 强制保留非叶子节点的梯度 |
| **动态计算图** | 边计算边构建，灵活易调试 |
| **梯度累积** | 叶子节点梯度多次反向会叠加，需清零 |

### 自动微分的完整流程（必记！）

```python
import torch
import torch.nn as nn

# ① 定义数据 ⭐ (1,1) 形状，与 Z 保持一致
X = torch.tensor([[10.0]])   # (1,1)
Y = torch.tensor([[3.0]])    # (1,1)

# ② 初始化参数（开启梯度）
W = torch.randn(1, 1, requires_grad=True)
B = torch.randn(1, 1, requires_grad=True)

# ③ 前向传播
Z = W @ X + B                # ⭐ 矩阵乘法 @  (1,1)
loss_fn = nn.MSELoss()
loss = loss_fn(Z, Y)

# ④ 反向传播 ⭐
loss.backward()

# ⑤ 查看梯度
print(f"W.grad = {W.grad}")
print(f"B.grad = {B.grad}")

# ⑥ 清空梯度（下次迭代前）
# optimizer.zero_grad()  ← 后面学
```

---

## 📌 五、detach 分离张量（切断梯度计算）

### 为什么要 detach？

在训练神经网络时，计算图中的中间结果（包括最终的 loss）往往不只是"算完就放那"——我们可能还要拿它做比较、累加、可视化等**额外操作**。

问题来了：如果所有操作都开启了梯度追踪，这些额外操作也会被纳入计算图，反向传播时就会**干扰原始的梯度计算**。

> 💡 **直觉理解**：想象你修了一条水管（计算图），水流（梯度）要沿着管子回流。但你在管子中间又接了一根分支管去做别的事——这根分支管的水流会干扰主干的水压。`detach` 就是把分支管**封死**，让它不影响主干。

### detach 是什么？

`detach()` 会返回一个**新的张量**，和原始张量**数值完全相同**，但：
- ❌ **丢失** `grad_fn`（不知道自己是怎么算出来的）
- ❌ **关闭** `requires_grad`（不再追踪梯度）

```
原始张量 X ──→ detach() ──→ 新张量 Y
  │                              │
  │ requires_grad=True          │ requires_grad=False
  │ grad_fn = ...               │ grad_fn = None
  │ 数值 = 2.0                  │ 数值 = 2.0（一样！）
```

### 代码演示

```python
import torch

# 定义叶子节点 X，开启梯度追踪
X = torch.tensor(2.0, requires_grad=True)

# 用 detach 分离出 Y
Y = X.detach()

# 对比 X 和 Y
print(X)              # tensor(2., requires_grad=True)
print(Y)              # tensor(2.)
print(Y.requires_grad) # False  ← 梯度开关已关闭！
```

### detach 出来的 Y 和 X 是什么关系？

| 属性 | X | Y（detach 出来） |
|------|---|-----------------|
| **数值** | 2.0 | 2.0 ✅ 相同 |
| **requires_grad** | True | False ❌ 不同 |
| **grad_fn** | None（叶子） | None ❌ 不同 |
| **是否同一对象** | — | ❌ 不同（`id()` 不同） |
| **底层数据存储** | — | ✅ 共享（`data_ptr()` 相同） |

> ⚠️ **注意**：X 和 Y 是**不同的对象**，但**共享底层数据**。不过由于 X 是叶子节点且开启了梯度，所以**不能对 X 做原地修改**（in-place operation）。

### detach 在计算图中的效果

```python
import torch

X = torch.tensor(2.0, requires_grad=True)
Y = X.detach()   # Y 被分离，梯度追踪关闭

# 两条分支分别做平方运算
Z1 = X ** 2      # 基于 X → 有梯度追踪
Z2 = Y ** 2      # 基于 Y → 无梯度追踪

print(Z1.requires_grad)  # True
print(Z2.requires_grad)  # False
print(Z1.grad_fn)        # PowBackward0
print(Z2.grad_fn)        # None

# Z1 可以反向传播
Z1.backward()
print(X.grad)  # tensor(4.)  ← 2X = 2×2 = 4 ✅

# Z2 不能反向传播！
# Z2.backward()  ← ❌ 报错！没有 grad_fn
```

**关键结论**：即使 Y 是从 X 分离出来的，Y 后续的操作**完全不影响** X 的梯度计算。X 的梯度仍然是 4，和没有 Y 这条分支时**完全一样**。

> 🎯 **一句话总结**：`detach()` 把张量从计算图中"摘"出来，之后它做的任何操作都不会影响原始计算图的梯度。

---

## 📌 六、detach 对梯度计算的影响（进阶验证）

### 更复杂的场景

上面是简单案例，再看一个更复杂的：**detach 出来的结果又参与了后续计算**。

```
计算图结构：

    X（叶子，requires_grad=True）
    │
    ├──→ Y = X²（正常计算）
    │      │
    │      └──→ U = Y.detach()（分离，关闭梯度）
    │             │
    │             └──→ Z = U × X（U 作为常数参与）
    │
    └────────────────────────→（X 也直接参与 Z 的计算）
```

数学上：`Z = U × X`，而 `U = X²`（数值上），所以 `Z = X³`？

**不是！** 因为 U 是 detach 出来的，它被当成**常数**，所以：
- `Z = U × X`（U 是常数）
- `∂Z/∂X = U`（把 U 当常数求导）

### 代码验证

```python
import torch

# 定义 2×2 全一矩阵，开启梯度
X = torch.ones(2, 2, requires_grad=True)

# Y = X²
Y = X ** 2
print(Y)
# tensor([[1., 1.],
#         [1., 1.]], grad_fn=<PowBackward0>)

# U = Y.detach()，分离出来
U = Y.detach()
print(U.requires_grad)  # False
print(U.grad_fn)        # None

# Z = U × X
Z = U * X
print(Z.requires_grad)  # True  ← 只要有一个输入开了梯度，结果就是 True
print(Z.grad_fn)        # <MulBackward0>
```

### 反向传播的注意点

```python
# ❌ Z 是矩阵，不能直接 backward！
# Z.backward()  ← 报错：grad can be implicitly created only for scalar outputs

# ✅ 先求和变成标量，再 backward
Z.sum().backward()

# 查看梯度
print(X.grad)
# tensor([[1., 1.],
#         [1., 1.]])  ← 就是 U 的值！
```

### 为什么梯度是全 1？

| 情况 | 公式 | 对 X 求导 | X=1 时的梯度 |
|------|------|-----------|-------------|
| **U 当常数**（detach） | Z = U × X | ∂Z/∂X = **U** | U = [[1,1],[1,1]] |
| **U 不 detach**（正常） | Z = X² × X = X³ | ∂Z/∂X = **3X²** | 3×1 = [[3,3],[3,3]] |

验证一下不用 detach 的情况：

```python
X = torch.ones(2, 2, requires_grad=True)
Y = X ** 2
Z = Y * X        # 不 detach，直接用 Y
Z.sum().backward()
print(X.grad)
# tensor([[3., 3.],
#         [3., 3.]])  ← 3X² = 3×1 = 3 ✅
```

> 🎯 **核心结论**：`detach` 出来的张量在后续计算中被当成**常数**，不会把它的"来源"带入梯度计算。这就是 detach 对梯度计算的影响。

### 速查对比

| 操作 | `requires_grad` | `grad_fn` | 对上游梯度的影响 |
|------|:-:|:-:|------|
| `X`（原始） | ✅ True | None（叶子） | 正常传递 |
| `Y = X.detach()` | ❌ False | ❌ None | **完全切断** |
| `Z = Y * X` | ✅ True | MulBackward0 | Y 分支不回传，只从 X 分支回传 |

---

## 📌 七、detach vs data（推荐的分离方式）

### 两个看起来一样的东西

你可能会想：之前说 `detach` 的本质是把数据拿出来，其他属性全扔掉。那 `Y.data` 不就是 Y 的底层数据吗？**二者有什么区别？**

```python
import torch

X1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
X2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 模拟神经网络：经过激活函数
Y1 = X1.sigmoid()
Y2 = X2.sigmoid()

# 两种方式"分离"数据
Z1 = Y1.data       # ① 直接取底层 data 属性
Z2 = Y2.detach()   # ② 用 detach 方法

print(Z1)  # tensor([0.7311, 0.8808, 0.9526])
print(Z2)  # tensor([0.7311, 0.8808, 0.9526])
print(Z1.requires_grad)  # False
print(Z2.requires_grad)  # False
```

**表面上看完全一样**：数据相同、都不追踪梯度、都没有 grad_fn。甚至底层内存也是共享的。

### 但二者的本质区别

> 🏭 **仓库管理员比喻**：
> - `Y.data` → 你**偷偷钻到仓库里**，直接修改底层货物（data）。仓库管理员（autograd 引擎）**完全不知情**。
> - `Y.detach()` → 你**先跟管理员打报告**：我分离了一份出来，我要改它。管理员记录在案。

这个区别在**你不修改分离后的数据**时看不出来，但一旦你**改了底层数据**，后果截然不同。

### 危险实验：修改分离后的数据

```python
# 模拟神经网络
X1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
X2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

Y1 = X1.sigmoid()  # 前向传播
Y2 = X2.sigmoid()

# 用两种方式分离
Z1 = Y1.data        # data 属性
Z2 = Y2.detach()    # detach 方法

# ⚠️ 修改分离出来的数据（原地操作）
Z1.zero_()          # Y1.data 全部改成 0
Z2.zero_()          # Y2.detach() 全部改成 0

# 因为内存共享，Y1 和 Y2 的数据也被改了！
print(Y1)  # tensor([0., 0., 0.], grad_fn=<SigmoidBackward0>)
print(Y2)  # tensor([0., 0., 0.], grad_fn=<SigmoidBackward0>)
```

### 反向传播时的致命差异

```python
# ⚠️ 注意：先关闭 data 修改后的梯度计算
# Y1.sum().backward()
# print(X1.grad)  # ✅ 能算！但结果是 tensor([0., 0., 0.])
#                 # 因为 Y1 数据全变 0，梯度也是 0
#                 # 你以为是正确的，其实已经乱了！

# Y2.sum().backward()
# print(X2.grad)  # ❌ 直接报错！
# RuntimeError: one of the variables needed for gradient computation
# has been modified by an inplace operation
```

| 方式 | 修改底层数据后反向传播 | 结果 |
|------|----------------------|------|
| **`Y.data`** | ✅ 能算 | 算出**错误梯度**，毫无警告 ❌ 极危险 |
| **`Y.detach()`** | ❌ 报错 | 直接报错提醒你修过数据 ✅ 安全 |

> ⚠️ **为什么 data 更危险？**
>
> 你用 `data` 改了底层数据，autograd 引擎完全不知道。它以为一切正常，**默默地算出错误的梯度**。你发现不了，训练出来的模型就废了。
>
> 而 `detach` 分离出来的变量被 autograd **跟踪管理**，你改了它，autograd 知道，反向传播时**直接报错叫停**，提醒你代码有 bug。

### 总结对比

| 对比维度 | `Y.data`（属性） | `Y.detach()`（方法） |
|---------|:---------------:|:------------------:|
| 返回类型 | 张量 | 张量 |
| 共享底层数据 | ✅ 是 | ✅ 是 |
| 关闭 `requires_grad` | ✅ 是 | ✅ 是 |
| 丢失 `grad_fn` | ✅ 是 | ✅ 是 |
| **被 autograd 跟踪** | ❌ 否 | ✅ **是** |
| 修改后反向传播 | 能算但结果错误 ❌ | 直接报错 ✅ |
| **推荐使用** | ❌ 不推荐 | ✅ **官方推荐** |

> 🎯 **一句话结论**：分离张量**永远用 `detach()`** 而不是 `.data`。除非你 100% 确定不会修改分离出来的数据，否则 `.data` 可能悄无声息地毁掉你的训练结果。

---

## 📝 本章总结（自动微分完整版）

### 🌳 知识树

```
PyTorch 自动微分（Autograd）
│
├── ① 基础概念
│   ├── requires_grad → 梯度追踪开关
│   ├── .grad → 存储梯度值
│   └── .grad_fn → 记录计算来源
│
├── ② 计算图
│   ├── 动态构建（边算边搭）
│   ├── 叶子节点 vs 非叶子节点
│   └── 反向后自动释放
│
├── ③ 反向传播
│   ├── backward() → 从标量开始
│   ├── 链式求导法则
│   └── 梯度累积 → 需 zero_grad()
│
├── ④ 梯度管理
│   ├── retain_grad() → 保留非叶子梯度
│   └── optimizer.zero_grad() → 清空
│
└── ⑤ detach 分离
│   ├── 返回新张量，数值相同
│   ├── 关闭 requires_grad
│   ├── 丢失 grad_fn
│   ├── 底层数据共享（data_ptr 相同）
│   ├── 后续计算中被当成常数
│   └──
└── ⑥ detach vs data
    ├── 表面一样：数据相同、不追踪梯度
    ├── data 修改后反向传播 → 无声算出错误梯度 ❌
    ├── detach 修改后反向传播 → 直接报错 ✅
    └── 推荐：永远用 detach()
```

### 关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| **`requires_grad`** | 是否开启梯度追踪，参数设为 True |
| **`.grad`** | 存储计算出的梯度（叶子节点） |
| **`.grad_fn`** | 记录张量是由什么运算得到的 |
| **`backward()`** | 从损失开始反向传播，自动算梯度 |
| **叶子节点** | 初始定义的张量，梯度会保留 |
| **非叶子节点** | 中间计算结果，梯度自动释放 |
| **`retain_grad()`** | 强制保留非叶子节点的梯度 |
| **动态计算图** | 边计算边构建，灵活易调试 |
| **梯度累积** | 叶子节点梯度多次反向会叠加，需清零 |
| **`detach()`** | 分离张量，切断梯度追踪，后续视为常数 |
| **`.data`（属性）** | 取出底层数据，**不被 autograd 跟踪**，修改后反向传播无警告 — ❌ 不推荐 |
| **detach vs data** | 都能分离数据，但 `detach` 被 autograd 跟踪更安全，`data` 可能无声产生错误梯度 |

### 自动微分的完整流程（必记！）

```python
import torch
import torch.nn as nn

# ① 定义数据 ⭐ (1,1) 形状，与 Z 保持一致
X = torch.tensor([[10.0]])   # (1,1)
Y = torch.tensor([[3.0]])    # (1,1)

# ② 初始化参数（开启梯度）
W = torch.randn(1, 1, requires_grad=True)
B = torch.randn(1, 1, requires_grad=True)

# ③ 前向传播
Z = W @ X + B
loss_fn = nn.MSELoss()
loss = loss_fn(Z, Y)

# ④ 反向传播 ⭐
loss.backward()

# ⑤ 查看梯度
print(f"W.grad = {W.grad}")
print(f"B.grad = {B.grad}")

# ⑥ 清空梯度（下次迭代前）
# optimizer.zero_grad()  ← 后面学
```

---

> **下一篇预告**：接下来是**神经网络模块（torch.nn）**——用 PyTorch 搭建真正的神经网络模型！

---