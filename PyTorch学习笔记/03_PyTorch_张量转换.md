# 🧠 第三部分：PyTorch 张量转换

---

## 📌 前置说明

张量创建好之后，经常需要做各种转换：
- **元素类型转换**：int64 → float32 / float64 → bool ...
- **数据结构转换**：Tensor ↔ NumPy ndarray（注意内存共享！）
- **标量转换**：Tensor ↔ Python 标量（一个数）

---

## 🔄 一、张量元素类型转换

### 1.1 为什么要做类型转换？

创建张量时我们可以指定 dtype，但有时候创建好了发现类型不对，或者后面需要改类型。

比如：
```python
t = torch.tensor([1, 2, 3])  # int64，但我想把它变成浮点数
```

### 1.2 方法一：`tensor.type(dtype)`

```python
t = torch.tensor([1, 2, 3])       # 默认 int64
print(t.dtype)                     # torch.int64

# 转换成 float32
t = t.type(torch.float32)
print(t)                           # tensor([1., 2., 3.])
print(t.dtype)                     # torch.float32

# 转换成 float64（双精度）
t = t.type(torch.float64)
print(t.dtype)                     # torch.float64
```

### 1.3 方法二：直接调对应方法

```python
t = torch.tensor([1, 2, 3])

# 浮点系列
t.float()    # → float32（单精度，深度学习常用）
t.double()   # → float64（双精度）
t.half()     # → float16（半精度，省显存）

# 整数系列
t.int()      # → int32（4字节）
t.long()     # → int64（8字节，默认整数）
t.short()    # → int16（2字节）
t.byte()     # → uint8（1字节，无符号）

# 布尔
t.bool()     # → bool（非零→True，零→False）
```

**示例：**
```python
t = torch.tensor([1, 2, 3])
print(t.long())      # tensor([1, 2, 3])       ← int64（不显示 dtype）
print(t.float())     # tensor([1., 2., 3.])     ← float32（不显示 dtype）
print(t.double())    # tensor([1., 2., 3.], dtype=torch.float64)  ← 非默认，显示 dtype
print(t.half())      # tensor([1., 2., 3.], dtype=torch.float16)
print(t.bool())      # tensor([True, True, True])
```

> 💡 **记忆规律**：默认类型不显示 dtype（int64、float32），非默认类型会显示。

### 1.4 特殊：复数类型

PyTorch 也支持科学计算中的复数：

```python
t = torch.tensor([1, 2, 3], dtype=torch.complex64)
print(t)  # tensor([1.+0.j, 2.+0.j, 3.+0.j])
```

### 类型转换速查

```python
tensor.type(torch.float32)    # 通用方式
tensor.float()                # → float32
tensor.double()               # → float64
tensor.half()                 # → float16
tensor.int()                  # → int32
tensor.long()                 # → int64
tensor.short()                # → int16
tensor.byte()                 # → uint8
tensor.bool()                 # → bool
tensor.to(torch.complex64)    # → 复数
```

---

## 🔗 二、Tensor ↔ NumPy ndarray 转换

这是**最常用**的转换！PyTorch 和 NumPy 的互转非常方便，但有个关键概念：**内存共享**。

### 2.1 Tensor → NumPy：`.numpy()`

```python
import torch
import numpy as np

# 设置打印精度，方便对比
torch.set_printoptions(precision=6)
np.set_printoptions(precision=6)

# 创建一个 tensor
t1 = torch.rand(3, 2)       # 3×2 随机张量
print(t1)

# 转成 ndarray
n1 = t1.numpy()
print(n1)                   # 数据跟 t1 完全一样
```

### ⚠️ 2.2 内存共享（重点！）

`.numpy()` 转换后的 ndarray 和原 tensor **共享同一块内存**：

```python
t1 = torch.rand(3, 2)
n1 = t1.numpy()

# 改 ndarray → tensor 也跟着变！
n1[1, 0] = 999
print(n1)
print(t1)   # tensor 也被改了！😱

# 改 tensor → ndarray 也跟着变！
t1[:, 0] = 0
print(t1)
print(n1)   # ndarray 也被改了！
```

> **底层原理**：`t1` 和 `n1` 是两个不同的 Python 对象（`type()` 不同，`id()` 不同），但它们底层指向**同一块数据内存**。就像两个门牌号指向同一个房间，一个改房间里的东西，另一个也能看见。

### 2.3 避免内存共享：`.copy()`

如果你希望转换后两者独立，加一个 `.copy()`：

```python
t1 = torch.rand(3, 2)
n1 = t1.numpy()       # 共享内存
n2 = t1.numpy().copy()  # 复制一份，不共享！

# 改 tensor
t1[:, 0] = 10
print(n1)   # 跟着变了（共享）
print(n2)   # 没变！（独立副本 ✅）
```

### 2.4 NumPy → Tensor：`torch.from_numpy()`

```python
# 创建一个 ndarray
n1 = np.random.randn(3)       # 3 个标准正态分布随机数
print(n1)
print(n1.dtype)               # float64（NumPy 默认）

# 转成 tensor（同样共享内存！）
t1 = torch.from_numpy(n1)
print(t1)
print(t1.dtype)               # float64（跟着 NumPy 走）
```

> ⚠️ **注意**：`from_numpy()` 转换过来的 dtype 是 **float64**，不是 PyTorch 默认的 float32！因为它跟着 NumPy 的数据类型走。

### 2.5 同样有内存共享

```python
n1 = np.random.randn(3)
t1 = torch.from_numpy(n1)

# 改 ndarray
n1[0] = 10
print(t1)   # tensor 也跟着变了！

# 改 tensor
t1[1] = 5
print(n1)   # ndarray 也跟着变了！
```

### 2.6 避免共享的两种方法

**方法一：转之前 copy**

```python
n1 = np.random.randn(3)
t1 = torch.from_numpy(n1.copy())  # 先 copy ndarray 再转
# 现在 t1 和 n1 独立了
```

**方法二：转之后 clone**

```python
n1 = np.random.randn(3)
t1 = torch.from_numpy(n1)
t2 = t1.clone()     # 克隆一份（tensor 的 copy 叫 clone）

# 验证：改 t2 不影响 n1
t2[0] = 1.5
print(n1)   # 没变 ✅
```

### 2.7 第三种转换方式：`torch.tensor()`（默认不共享）

这是最简单直接的方式——之前学的 `torch.tensor()` 也可以把 NumPy 转成 Tensor，而且**默认不共享内存**：

```python
n1 = np.random.randn(3)

# 用 torch.tensor() 创建（不是 from_numpy）
t1 = torch.tensor(n1)
# t1 和 n1 默认就是独立的！

# 验证
n1[0] = 10
print(t1)   # 没变 ✅ 互不影响
```

### 转换方式对比

| 转换方式 | 方向 | 内存共享？ | 推荐场景 |
|---------|------|-----------|---------|
| `tensor.numpy()` | Tensor → NumPy | ✅ 共享 | 快速查看/分析数据 |
| `tensor.numpy().copy()` | Tensor → NumPy | ❌ 独立 | 需要独立副本 |
| `torch.from_numpy(ndarray)` | NumPy → Tensor | ✅ 共享 | 快速加载 NumPy 数据 |
| `torch.from_numpy(n.copy())` | NumPy → Tensor | ❌ 独立 | 需要独立副本 |
| `tensor.clone()` | Tensor 克隆 | ❌ 独立 | 创建 tensor 的独立副本 |
| `torch.tensor(ndarray)` | NumPy → Tensor | ❌ 独立 ✅ **最省心** | 直接创建独立 tensor |

---

## 📏 三、Tensor ↔ 标量转换

### 3.1 标量 → Tensor

这个我们之前学创建时已经会了：

```python
# 用一个数创建张量
s = torch.tensor(10)
print(s)           # tensor(10)
print(s.shape)     # torch.Size([])  ← 零维
print(s.dtype)     # torch.int64

# 注意区别：
a = torch.tensor(10)   # 标量张量（零维，没有中括号）
b = torch.tensor([10]) # 一维张量（长度为1的数组）
print(a.shape)         # torch.Size([])
print(b.shape)         # torch.Size([1])
```

### 3.2 Tensor → 标量：`.item()`

把张量里的**唯一一个元素**取出来变成 Python 标量：

```python
t = torch.tensor(3.14)
value = t.item()
print(value)      # 3.14
print(type(value)) # <class 'float'>  ← 不再是 tensor 了
```

### ⚠️ 要求：必须只有一个元素

```python
t1 = torch.tensor(10)           # ✅ 零维，1个元素
print(t1.item())                # 10

t2 = torch.tensor([10])          # ✅ 一维，长度1
print(t2.item())                 # 10

t3 = torch.tensor([[10]])        # ✅ 二维，1×1
print(t3.item())                 # 10

t4 = torch.tensor([1, 2, 3])     # ❌ 3个元素
print(t4.item())                 # 报错！ValueError
```

> **`.item()` 只适用于只有一个元素的张量**。不管它是 0 维、1 维还是 n 维，只要元素总数是 1，就能用。

### 3.3 实际用途

训练神经网络时，`loss` 通常是一个标量张量，用 `.item()` 把它取出来记录日志：

```python
# 训练时的典型用法
loss = torch.tensor(0.5234)   # 假设这是模型算出来的 loss
print(f"Loss: {loss.item():.4f}")  # Loss: 0.5234
```

---

## 📝 四、本章总结

### 知识地图

```
张量转换
    │
    ├── ① 元素类型转换
    │      ├── tensor.type(dtype)      ← 通用方法
    │      ├── tensor.float() / .int() / .long() / .bool() ...
    │      └── tensor.to(dtype)         ← 另一种通用方式
    │
    ├── ② Tensor ↔ NumPy
    │      ├── tensor.numpy()           ← Tensor → NumPy（共享内存）
    │      ├── torch.from_numpy(n)      ← NumPy → Tensor（共享内存）
    │      ├── .copy() / .clone()       ← 避免共享的两种方式
    │      └── torch.tensor(n)          ← NumPy → Tensor（默认不共享 ✅）
    │
    └── ③ Tensor ↔ 标量
           ├── torch.tensor(数值)        ← 标量 → Tensor
           └── tensor.item()            ← Tensor → 标量（必须只有一个元素）
```

### 关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| **`tensor.type(dtype)`** | 通用类型转换方法 |
| **`tensor.float()`** | 转 float32（单精度） |
| **`tensor.double()`** | 转 float64（双精度） |
| **`tensor.long()`** | 转 int64（长整型） |
| **`tensor.int()`** | 转 int32（普通整数） |
| **`tensor.bool()`** | 转布尔类型（非零→True） |
| **`tensor.numpy()`** | Tensor → NumPy（⚠️ 共享内存） |
| **`torch.from_numpy()`** | NumPy → Tensor（⚠️ 共享内存） |
| **`.copy()` / `.clone()`** | 复制数据，避免内存共享 |
| **`torch.tensor(n)`** | NumPy → Tensor（✅ 默认不共享） |
| **`tensor.item()`** | Tensor → Python 标量（仅1个元素） |

---

---