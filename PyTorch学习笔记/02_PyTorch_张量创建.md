# 🧠 第二部分：PyTorch 张量操作

---

## 📌 前置准备

在开始之前，先确保你的 PyCharm / VS Code 已经切换到装有 PyTorch 的虚拟环境（安装过程见第一部分）：

```
PyCharm → 右下角 Python Interpreter → 选择你创建的 conda 环境
           (比如 pytorch_env 或 pytorch_2.6.0GPU)
```

然后引入 PyTorch：

```python
import torch
import numpy as np  # 后面会拿来做对比
```

> ⚠️ **常见报错**：如果 import torch 报红/报错 → 说明解释器没切对，去设置里改一下。

---

## 🎯 一、创建张量的两大类场景

我们为什么要创建张量？其实就两种需求：

| 场景 | 举例 | 用什么方法 |
|------|------|-----------|
| **我知道里面存什么数据** | 比如权重矩阵初始化为 `[[1,2],[3,4]]` | **按内容创建** |
| **我只知道形状，不关心内容** | 比如偏置初始化为全 0，或者随机初始化 | **按形状创建** |

这节课我们就围绕这两种场景展开 👇

---

## 📦 二、按内容创建张量 — `torch.tensor()`（小写 t）

### 2.1 基本用法

`torch.tensor(数据)` — 你给什么数据，它就创建什么张量。

#### ① 传入一个标量（0 维）

```python
# 传入一个数 → 标量张量（零维）
tensor1 = torch.tensor(10)
print(tensor1)
print(f"形状: {tensor1.shape}")
print(f"数据类型: {tensor1.dtype}")
```

输出：
```
tensor(10)
形状: torch.Size([])      ← 空的，说明零维
数据类型: torch.int64      ← 传入整数 → int64
```

> 标量张量就像一个"装了数的盒子"，没有"长宽高"的概念，就是零维。

#### ② 传入一个列表（1 维 → 向量）

```python
# 传入一个列表 → 一维张量（向量）
tensor2 = torch.tensor([1, 2, 3])
print(tensor2)
print(f"形状: {tensor2.shape}")   # torch.Size([3])
print(f"数据类型: {tensor2.dtype}") # torch.int64
```

输出：
```
tensor([1, 2, 3])
形状: torch.Size([3])
数据类型: torch.int64
```

#### ③ 传入二维列表（2 维 → 矩阵）

```python
# 传入二维列表 → 二维张量（矩阵）
tensor3 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor3)
print(f"形状: {tensor3.shape}")   # torch.Size([2, 3])
```

输出：
```
tensor([[1, 2, 3],
        [4, 5, 6]])
形状: torch.Size([2, 3])
```

#### ④ 传入高维列表（3 维及以上）

```python
# 3 维：2 个 2×3 的矩阵叠在一起
tensor4 = torch.tensor([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
])
print(tensor4)
print(f"形状: {tensor4.shape}")   # torch.Size([2, 2, 3])
```

输出：
```
tensor([[[ 1,  2,  3],
         [ 4,  5,  6]],

        [[ 7,  8,  9],
         [10, 11, 12]]])
形状: torch.Size([2, 2, 3])
```

### 2.2 从 NumPy 数组创建

```python
# 先创建一个 numpy 数组
np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

# 直接传进 torch.tensor()
tensor_from_np = torch.tensor(np_array)
print(tensor_from_np)
print(f"形状: {tensor_from_np.shape}")
print(f"数据类型: {tensor_from_np.dtype}")
```

输出：
```
tensor([[1., 2., 3.],
        [4., 5., 6.]])
形状: torch.Size([2, 3])
数据类型: torch.float64    ← 注意！从 numpy 来，类型跟着 numpy 走
```

### 2.3 ⚠️ 重点：数据类型（dtype）的规则

这是初学者最容易搞混的地方，记住一条规则：

> **`torch.tensor()` 的 dtype 跟着你传入的数据走。**

| 传入的数据 | 默认 dtype | 说明 |
|-----------|-----------|------|
| 整数（如 `10`） | `int64` | 64 位整数（因为现在电脑都是 64 位系统） |
| 浮点数（如 `1.0`） | **`float32`** ⭐ | **和 NumPy 不一样！** |
| 从 NumPy 数组来 | 跟着 NumPy 的类型走 | NumPy 默认 `float64`，所以过来也是 `float64` |

**和 NumPy 的关键区别：**

```python
# NumPy：浮点数默认 float64
np_a = np.array([1.0, 2.0, 3.0])
print(np_a.dtype)  # float64

# PyTorch：浮点数默认 float32
pt_a = torch.tensor([1.0, 2.0, 3.0])
print(pt_a.dtype)  # float32  ← 不一样！
```

> 💡 为什么 PyTorch 用 `float32` 而不是 `float64`？
> - `float32`（32 位）占 4 个字节，`float64`（64 位）占 8 个字节
> - 深度学习模型动辄上亿参数，用 `float64` 显存直接翻倍，根本跑不动
> - `float32` 精度已经够用了，所以 PyTorch 默认用 `float32` ✅

---

## 📦 三、按形状创建张量 — `torch.Tensor()`（大写 T）

### 3.1 为什么需要按形状创建？

前面按内容创建，你得把所有数据都写进去。但想象一下：

- 一个 `10 × 20 × 30` 的张量，你要手动写 6000 个数？😱
- 偏置向量初始化为全 0，你更关心**形状**，不关心具体值
- 随机初始化，你只需要指定形状，值让电脑随机生成

**所以：更多时候我们只需要指定形状，不关心具体内容。**

### 3.2 基本用法

```python
# 创建一个形状为 (3, 2, 4) 的张量
t = torch.Tensor(3, 2, 4)
print(t)
print(f"形状: {t.shape}")       # torch.Size([3, 2, 4])
print(f"数据类型: {t.dtype}")   # torch.float32
```

输出示例：
```
tensor([[[-4.1574e+21,  4.5868e-41,  0.0000e+00,  0.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]],

        [[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]],

        [[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]]])
形状: torch.Size([3, 2, 4])
数据类型: torch.float32
```

> ⚠️ **注意**：这里面的值看起来像 0，其实不是！它只是"碰巧"是 0。因为 `torch.Tensor()` 只是**开辟一块内存空间**，里面的值是**未初始化的垃圾值**（系统残留数据）。真正想要全 0，得用后面学的 `torch.zeros()`。

### 3.3 大写 T 也能传入内容

```python
# 传入内容也可以
t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
print(t)
print(f"数据类型: {t.dtype}")  # float32 ！注意！
```

输出：
```
tensor([[1., 2., 3.],
        [4., 5., 6.]])
数据类型: torch.float32
```

> ⚠️ **关键区别**：就算你传的全是整数，`torch.Tensor()` 也**强制转成 float32**！而 `torch.tensor()` 会保留整数类型。

### 3.4 一个有趣的坑：传单个数字

```python
# 小写 t → 传 10 → 创建标量张量
a = torch.tensor(10)
print(f"小写: {a}, 形状: {a.shape}")   # 标量，形状 []

# 大写 T → 传 10 → 创建形状为 (10,) 的一维张量
b = torch.Tensor(10)
print(f"大写: {b}, 形状: {b.shape}")   # 一维数组，形状 [10]
```

```
小写: tensor(10), 形状: torch.Size([])
大写: tensor([...]), 形状: torch.Size([10])
```

> **原因**：大写 `Tensor()` 默认你传的是**形状参数**，所以 `10` 被理解成"创建一个长度为 10 的一维张量"。

---

## ⚔️ 四、核心对比：`torch.tensor()` vs `torch.Tensor()`

| 对比维度 | `torch.tensor()`（小写） | `torch.Tensor()`（大写） |
|---------|------------------------|------------------------|
| **主要用途** | ✅ **按内容创建** | ✅ **按形状创建** |
| **传入内容** | 可以，dtype 跟着数据走 | 也可以，但强制转 float32 |
| **传入形状** | ❌ 不支持 | ✅ 支持 |
| **默认 dtype** | 整数→int64，浮点→float32 | **一律 float32** |
| **典型场景** | 你知道具体值，比如手动设参数 | 你不知道具体值，只关心形状 |
| **底层本质** | 一个函数（方法），返回 Tensor 对象 | Tensor 类的构造函数 |

```python
# 总结对比
torch.tensor([1, 2, 3])     # int64  ✅ 整数保留
torch.Tensor([1, 2, 3])     # float32 ⚠️ 强制转浮点

torch.tensor(3, 2)          # ❌ 报错！不支持传多个数作形状
torch.Tensor(3, 2)          # ✅ 形状 (3, 2) 的张量

torch.tensor(10)            # ✅ 标量张量，值是 10
torch.Tensor(10)            # ✅ 一维张量，长度 10（值是垃圾值）
```

---

## 📝 五、指定数据类型（dtype）

前面学的 `tensor()` 和 `Tensor()` 的 dtype 都是固定的。但实际中我们经常需要**手动指定 dtype**，有两种方法。

### 方法一：`tensor()` 加 `dtype` 参数

```python
# 小写 tensor + dtype 参数
t1 = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"{t1}, dtype: {t1.dtype}")  # float32

t2 = torch.tensor([1, 2, 3], dtype=torch.int16)
print(f"{t2}, dtype: {t2.dtype}")  # int16
```

### 方法二：用特定的 Tensor 类直接创建

大写 Tensor 是一类"家族"，有各种专门的类型：

```python
# 直接用类型名创建
torch.IntTensor([1, 2, 3])     # int32（4字节）
torch.LongTensor([1, 2, 3])    # int64（8字节）→ 默认整数类型
torch.ShortTensor([1, 2, 3])   # int16（2字节）
torch.ByteTensor([1, 2, 3])    # uint8（1字节，无符号）

torch.FloatTensor([1, 2, 3])   # float32（单精度）→ 默认浮点类型
torch.DoubleTensor([1, 2, 3])  # float64（双精度）
torch.HalfTensor([1, 2, 3])    # float16（半精度）

torch.BoolTensor([1, 0, 1])    # 布尔类型 → True / False
```

### ⚠️ 关键区别

```python
# 小写 tensor：dtype 跟着内容自动判断
torch.tensor([1, 2, 3])        # int64（默认整数）
torch.tensor([1.0, 2.0, 3.0])  # float32（默认浮点）

# 大写 Tensor 子类：固定类型，内容自动转换
torch.IntTensor([1, 2, 3])     # int32（不是 int64！）
torch.FloatTensor([1, 2, 3])   # float32（即使传整数也转）
```

### 常见 dtype 速查表

| dtype | 字节数 | 对应大写类 | 说明 |
|-------|-------|-----------|------|
| `torch.int64` | 8 | `LongTensor` | **默认整数**（Python int 默认） |
| `torch.int32` | 4 | `IntTensor` | 相当于 C/Java 的 int |
| `torch.int16` | 2 | `ShortTensor` | 短整型 |
| `torch.uint8` | 1 | `ByteTensor` | 无符号字节 |
| `torch.float32` | 4 | `FloatTensor` | **默认浮点**（深度学习常用） |
| `torch.float64` | 8 | `DoubleTensor` | 双精度（NumPy 默认） |
| `torch.float16` | 2 | `HalfTensor` | 半精度（节省显存） |
| `torch.bool` | 1 | `BoolTensor` | 布尔（非零→True） |

> 💡 **记忆技巧**：大的默认类型不显示标注，小的非默认类型会显示标注。
> ```python
> torch.tensor([1, 2, 3])        # tensor([1, 2, 3])  ← 默认 int64，不显示
> torch.IntTensor([1, 2, 3])     # tensor([1, 2, 3], dtype=torch.int32)  ← 非默认，显示
> ```

---

## 📏 六、指定区间创建张量

有时候我们需要**按一定规律生成一系列数**，比如等差数列、等间隔采样等。

### 6.1 `torch.arange()` — 等差数列（跟 Python 的 range 一模一样）

```python
# 基本用法：arange(start, end, step)
# 注意：包含 start，不包含 end

a = torch.arange(10, 30, 2)   # 10 到 30，步长 2
print(a)                       # [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
print(f"形状: {a.shape}")       # [10]
print(f"数据类型: {a.dtype}")   # int64（整数）

# 简写：只给 end → 从 0 开始，步长 1
b = torch.arange(10)
print(b)                       # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

> 跟 Python 的 `range()` 和 NumPy 的 `np.arange()` **用法完全一样**，没有任何学习成本。

### 6.2 `torch.linspace()` — 等间隔采样（包含头尾）

```python
# linspace(start, end, steps)
# 在 [start, end] 区间内生成 steps 个数，包含 start 也包含 end

a = torch.linspace(10, 30, 5)   # 10 到 30，生成 5 个数
print(a)                         # [10, 15, 20, 25, 30]
print(f"数据类型: {a.dtype}")    # float32

b = torch.linspace(10, 30, 6)   # 步长自动变成 4
print(b)                         # [10, 14, 18, 22, 26, 30]
```

> **arange vs linspace 的区别**：
> - `arange`：指定**步长**，不关心个数
> - `linspace`：指定**个数**，自动算步长

### 6.3 `torch.logspace()` — 对数等间隔采样

```python
# logspace(start, end, steps, base=10)
# 先生成 linspace(start, end, steps)，然后取 base 的该次方

a = torch.logspace(1, 3, 3)         # base=10，生成 [10^1, 10^2, 10^3]
print(a)                             # [10, 100, 1000]

b = torch.logspace(1, 3, 3, base=2) # base=2，生成 [2^1, 2^2, 2^3]
print(b)                             # [2, 4, 8]
```

> 💡 **名字由来**：对这个结果取 `log`（以 base 为底），得到的就是你传入的线性区间。

### 速查对比

| 方法 | 效果 | 包含 end？ | 默认 dtype |
|------|------|-----------|-----------|
| `arange(10, 30, 2)` | 步长固定，个数不定 | ❌ 不包含 | int64 |
| `linspace(10, 30, 5)` | 个数固定，步长自动 | ✅ 包含 | float32 |
| `logspace(1, 3, 3)` | linspace + 指数运算 | ✅ 包含 | float32 |

---

## 🎨 七、按数值填充张量

实际写代码时最常用的创建方法！**指定形状，自动填充数值**。

### 7.1 核心四件套

| 方法 | 作用 | 示例 | 默认 dtype |
|------|------|------|-----------|
| `torch.zeros(3, 2)` | 全 0 | 3×2 的零矩阵 | float32 |
| `torch.ones(3, 2)` | 全 1 | 3×2 的一矩阵 | float32 |
| `torch.full((3, 2), 6)` | 全填指定值 | 3×2 全是 6 | **跟着填充值走** |
| `torch.empty(3, 2)` | 不初始化 | 值是垃圾值（系统残留） | float32 |

```python
# 全 0
z = torch.zeros(3, 2)
print(z)              # 全是 0.（浮点数）
print(z.dtype)        # float32

# 全 1
o = torch.ones(3, 2)
print(o)              # 全是 1.（浮点数）

# 填指定值
f = torch.full((3, 2), 6)     # 注意：形状要传元组或列表
print(f)              # 全是 6（整数！因为 6 是整数）
print(f.dtype)        # int64

f2 = torch.full((3, 2), 6.0)  # 传浮点数
print(f2.dtype)       # float32

# 空张量（垃圾值）
e = torch.empty(3, 2)
print(e)              # 里面的值不确定，可能是很大的数或 0
```

> ⚠️ `torch.full()` 的 dtype 规则：
> - `torch.full((3,2), 6)` → int64（传整数就整数）
> - `torch.full((3,2), 6.0)` → float32（传浮点就浮点）

### 7.2 四件套的 "like" 版本

如果你有一个现成的张量，想创建**形状相同**的新张量：

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 形状 (2, 3)

z = torch.zeros_like(a)   # 形状跟 a 一样，全 0
o = torch.ones_like(a)    # 形状跟 a 一样，全 1
f = torch.full_like(a, 9) # 形状跟 a 一样，全 9
e = torch.empty_like(a)   # 形状跟 a 一样，垃圾值
```

> **形状和 dtype 都跟着原张量走**，省得你手动指定了。

### 7.3 特殊：单位矩阵 `torch.eye()`

```python
# 方阵：4×4 的单位矩阵
I = torch.eye(4)
print(I)
# tensor([[1., 0., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 1., 0.],
#         [0., 0., 0., 1.]])
print(I.dtype)    # float32

# 非方阵：4×6，主对角线是 1，其余是 0
I2 = torch.eye(4, 6)
print(I2)
# tensor([[1., 0., 0., 0., 0., 0.],
#         [0., 1., 0., 0., 0., 0.],
#         [0., 0., 1., 0., 0., 0.],
#         [0., 0., 0., 1., 0., 0.]])
```

### 速查

```python
torch.zeros(3, 2)                # 全 0
torch.ones(3, 2)                 # 全 1
torch.full((3, 2), 5)            # 全填 5
torch.empty(3, 2)                # 垃圾值
torch.eye(4)                     # 单位矩阵
torch.zeros_like(x)              # 跟 x 形状一样的全 0
torch.ones_like(x)               # 跟 x 形状一样的全 1
torch.full_like(x, 5)            # 跟 x 形状一样的全 5
torch.empty_like(x)              # 跟 x 形状一样的垃圾值
```

---

## 🎲 八、随机生成张量

这是**最最重要**的创建方式！神经网络的权重初始化基本都用随机生成。

### 8.1 各种随机分布

#### ① 均匀分布 `torch.rand()`

```python
# 0~1 均匀分布（默认）
r1 = torch.rand(2, 3)         # 2×3，值在 [0, 1) 之间
print(r1)
# tensor([[0.2345, 0.8765, 0.1234],
#         [0.5432, 0.9876, 0.3456]])
```

#### ② 指定范围的整数均匀分布 `torch.randint()`

```python
# randint(low, high, size)
r2 = torch.randint(0, 10, (2, 3))  # 0~9 的整数，2×3
print(r2)
# tensor([[3, 7, 0],
#         [9, 2, 5]])
```

> 包含 low（0），不包含 high（10）。

#### ③ 标准正态分布 `torch.randn()`

```python
# 标准正态分布：均值=0，标准差=1
r3 = torch.randn(2, 3)
print(r3)
# tensor([[ 0.4321, -0.9876,  1.2345],
#         [-0.3456,  0.5678, -1.2345]])
# 有正有负，集中在 0 附近
```

#### ④ 一般正态分布 `torch.normal()`

```python
# normal(mean, std, size)
r4 = torch.normal(5, 2, (2, 3))  # 均值=5，标准差=2，2×3
print(r4)
# tensor([[4.23, 6.78, 3.12],
#         [7.89, 2.34, 5.67]])
# 大部分值在 5±2 范围内（3~7）
```

### 8.2 "like" 版本

跟数值填充一样，随机生成也有 like 版本：

```python
a = torch.rand(2, 3)              # 先创建一个现成张量

r1 = torch.rand_like(a)           # 形状和 dtype 跟 a 一样，0~1 均匀
r2 = torch.randint_like(a, 0, 10) # 形状跟 a 一样，0~9 整数（需指定范围）
r3 = torch.randn_like(a)          # 形状跟 a 一样，标准正态
```

> ⚠️ 注意：没有 `normal_like`，但可以用 `shape` 属性手动实现：
> ```python
> r4 = torch.normal(5, 2, a.shape)  # 跟 a 形状一样的正态分布
> ```

### 8.3 随机排列 `torch.randperm()`

不生成随机数，而是**把一组固定的数打乱顺序**：

```python
# 生成 0 到 N-1 的随机排列
rp = torch.randperm(10)
print(rp)
# tensor([3, 7, 0, 9, 2, 5, 1, 8, 4, 6])
# 每次运行结果都不一样（乱序）
```

> **应用场景**：训练时打乱数据顺序，防止模型学到数据顺序的规律。

### 8.4 随机数种子（让随机可重复）

有时候我们希望**随机结果可以复现**（比如调试时），就可以设置随机种子：

```python
# 查看当前随机种子
print(torch.random.initial_seed())  # 每次启动不一样

# 设置随机种子（让结果可复现）
torch.manual_seed(42)  # 42 是一个经典数字，随便设

# 设置之后，每次生成的随机数都一样
a = torch.rand(3)
print(a)  # 每次运行结果都一样了！

b = torch.rand(3)
print(b)  # 也和第一次运行的结果一样
```

> 💡 **为什么用 42？** 这是《银河系漫游指南》里的梗——"生命、宇宙以及一切终极问题的答案"。很多程序员习惯用 42 作随机种子。

### 随机创建速查

```python
torch.rand(2, 3)                    # [0,1) 均匀分布
torch.randint(0, 10, (2, 3))        # [low, high) 整数均匀
torch.randn(2, 3)                   # 标准正态分布（均值0，方差1）
torch.normal(5, 2, (2, 3))          # 一般正态分布（均值5，方差2）
torch.randperm(10)                  # 0~9 的随机排列
torch.manual_seed(42)               # 设置随机种子，让结果可复现

# like 版本
torch.rand_like(x)
torch.randint_like(x, 0, 10)
torch.randn_like(x)
```

---

## 🧠 九、类比理解：张量的"维度"

刚接触张量时，很多人对维度很晕。这里用生活化的类比帮你建立直觉：

| 维度 | 名称 | 类比 | PyTorch 形状 |
|-----|------|------|-------------|
| 0 维 | **标量** 🎯 | 一个数，比如温度 25°C | `torch.Size([])` |
| 1 维 | **向量** 📏 | 一排数，比如一周气温 `[25,26,27,28,29,30,31]` | `torch.Size([7])` |
| 2 维 | **矩阵** 📊 | 一张表格，比如 3 行 4 列的成绩单 | `torch.Size([3, 4])` |
| 3 维 | **张量** 🧊 | 一个立方体，比如 5 张 RGB 图片（5×28×28） | `torch.Size([5, 28, 28])` |
| 4 维 | **高维张量** 🚀 | 比如 100 张彩色图片的批处理（100×3×28×28） | `torch.Size([100, 3, 28, 28])` |

> **记忆口诀**：从外往里数中括号，一层中括号 = 一维。

```python
# 0 维（标量）
torch.tensor(10)                    # 没有中括号

# 1 维（向量）
torch.tensor([1, 2, 3])             # 1 层中括号

# 2 维（矩阵）
torch.tensor([[1, 2], [3, 4]])      # 2 层中括号

# 3 维
torch.tensor([[[1,2],[3,4]], [[5,6],[7,8]]])  # 3 层中括号
```

---

## ⚠️ 十、常见问题 & 避坑指南

### Q1：我该用大写 T 还是小写 t？

> **建议**：
> - 创建指定内容的张量 → 用 `torch.tensor()` ✅
> - 创建指定形状的张量 → 用 `torch.zeros()` / `torch.ones()` / `torch.randn()` ✅（不推荐直接 `Tensor()`）

### Q2：为什么 PyTorch 的 float 默认是 float32，而 NumPy 是 float64？

> 深度学习模型参数非常多，用 float64 显存占用翻倍，大部分场景 float32 精度已经足够。这是**为性能做的取舍**。

### Q3：`torch.Tensor(3, 2)` 和 `torch.Tensor([3, 2])` 一样吗？

```python
torch.Tensor(3, 2)     # 形状 (3, 2) — 3行2列
torch.Tensor([3, 2])   # 形状 (2,) — 长度为2的一维数组，值是 [3, 2]
```

> 不一样！前者传的是多个整数作形状参数，后者传的是一个列表作内容。

### Q4：`arange`、`linspace`、`logspace` 怎么区分？

| 方法 | 记忆法 | 核心参数 |
|------|-------|---------|
| `arange` | a + range = **步长**等差数列 | start, end, **step** |
| `linspace` | linear + space = **线性等分** | start, end, **steps**(个数) |
| `logspace` | log + space = **对数等分** | start, end, steps, **base** |

### Q5：什么时候需要设置随机种子？

> **调试时**——确保每次运行得到相同结果，方便排查 bug。
> **正式训练时**——一般不设，让模型每次学到不同的东西。

### Q6：`zeros_like` 和 `zeros` 有什么区别？

```python
torch.zeros(3, 2)           # 自己指定形状
torch.zeros_like(a)          # 从 a 的形状复制过来（省事）
```

---

## 📝 十一、本章总结（完整版）

### 知识地图

```
张量创建（共 7 种方式）
    │
    ├── ① torch.tensor(数据)      ← 按内容，dtype 跟着数据走
    │
    ├── ② torch.Tensor(形状)      ← 按形状，dtype 固定 float32（不推荐）
    │
    ├── ③ 指定 dtype
    │      ├── tensor(..., dtype=torch.float32)
    │      └── torch.FloatTensor / torch.IntTensor / ...
    │
    ├── ④ 指定区间
    │      ├── torch.arange(start, end, step)     ← 步长固定
    │      ├── torch.linspace(start, end, steps)  ← 个数固定
    │      └── torch.logspace(start, end, steps)  ← 对数级
    │
    ├── ⑤ 按数值填充
    │      ├── zeros / ones / full / empty
    │      ├── zeros_like / ones_like / full_like / empty_like
    │      └── eye（单位矩阵）
    │
    ├── ⑥ 随机生成
    │      ├── rand（均匀）、randint（整数均匀）
    │      ├── randn（正态）、normal（一般正态）
    │      └── randperm（随机排列）
    │
    └── ⑦ 随机种子
           └── torch.manual_seed(n)  ← 让随机可复现
```

### 关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| **`torch.tensor()`** | 按内容创建，dtype 跟着数据走（小写 t） |
| **`torch.Tensor()`** | 按形状创建，dtype 固定 float32（大写 T，不推荐） |
| **`torch.zeros/ones`** | 按形状创建全 0/全 1 张量，最常用 ✅ |
| **`torch.full`** | 按形状创建，全部填指定值 |
| **`torch.empty`** | 按形状创建，不初始化（垃圾值） |
| **`torch.eye`** | 创建单位矩阵 |
| **`torch.arange`** | 等差数列（跟 Python range 一样） |
| **`torch.linspace`** | 等间隔采样（包含头尾） |
| **`torch.rand`** | [0,1) 均匀分布随机张量 |
| **`torch.randint`** | [low, high) 整数均匀分布 |
| **`torch.randn`** | 标准正态分布随机张量 |
| **`torch.normal`** | 一般正态分布（指定均值、方差） |
| **`torch.randperm`** | 0~N-1 的随机排列 |
| **`torch.manual_seed`** | 设置随机种子（结果可复现） |
| **`xxx_like`** | 按某张量的形状创建新张量（省事） |

### 一句话记住各种创建方式

> **小写 tensor 按内容，大写 Tensor 按形状；指定类型加 dtype，区间 arange linspace；数值填充 zeros ones，随机生成 rand randn；like 后缀省长度，seed 种子定乾坤。**

---

---