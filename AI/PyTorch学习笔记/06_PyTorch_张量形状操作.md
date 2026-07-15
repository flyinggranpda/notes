# 🧠 第六部分：PyTorch 张量形状操作

---

## 📌 前置说明

搭建神经网络时，**张量形状操作**是每天都要用的功能。主要分为4大类：

```
形状操作
    ├── ① 交换维度（维度重新排列）
    ├── ② 调整形状（元素个数不变，形状改变）
    ├── ③ 增删维度（增加或删除大小为 1 的维度）
    └── ④ 拼接和堆叠（多个张量合成一个）
```

---

## 🔀 一、交换维度

把张量的维度重新排列。比如把行变成列、把矩阵和通道互换等。

### 1.1 `t.T` — 翻转所有维度（即将弃用）

```python
t = torch.randint(1, 10, (3, 2, 6))   # (3, 2, 6)
print(t.T.shape)                       # (6, 2, 3) ← 所有维度全翻转！
```

> ⚠️ `.T` 把 0→2、1→1、2→0，**全部翻转**，即将弃用，不推荐使用。

### 1.2 `t.mT` — 只转置最后两个维度（矩阵转置）

```python
t = torch.randint(1, 10, (3, 2, 6))   # (3, 2, 6)
print(t.mT.shape)                      # (3, 6, 2) ← 只交换最后两维

# 等价于调换维度 1 和 2
```

> ✅ **推荐**：高维张量做矩阵转置时用 `mT`。

### 1.3 `t.transpose(dim1, dim2)` — 交换任意两个维度

```python
t = torch.randint(1, 10, (3, 2, 6))

# 交换维度 1 和 2（行↔列）
t1 = t.transpose(1, 2)
print(t1.shape)         # (3, 6, 2)

# 交换维度 0 和 2（矩阵号↔列）
t2 = t.transpose(0, 2)
print(t2.shape)         # (6, 2, 3)

# 交换维度 0 和 1（矩阵号↔行）
t3 = t.transpose(0, 1)
print(t3.shape)         # (2, 3, 6)
```

### 1.4 `t.permute(*dims)` — 任意排列所有维度（大招 🎯）

这是最灵活的，直接指定所有维度的新顺序：

```python
t = torch.randint(1, 10, (3, 2, 6))   # 原始: (0, 1, 2) = (3, 2, 6)

# 变成 (2, 6, 3) → 维度顺序: (1, 2, 0)
t1 = t.permute(1, 2, 0)
print(t1.shape)         # (2, 6, 3)

# 变成 (6, 3, 2) → 维度顺序: (2, 0, 1)
t2 = t.permute(2, 0, 1)
print(t2.shape)         # (6, 3, 2)
```

> **`permute()` 是维度交换的王牌**，`T`、`mT`、`transpose()` 能做的事它都能做，而且一步到位。

### 交换维度方法对比

| 方法 | 功能 | 推荐 |
|------|------|------|
| `t.T` | 翻转所有维度 | ❌ 即将弃用 |
| `t.mT` | 只转置最后两维 | ✅ 矩阵转置用 |
| `t.transpose(d1, d2)` | 交换任意两维 | ✅ 简单交换用 |
| `t.permute(dims)` | 任意排列所有维度 | ✅ **万能大招，记这个就够了** |

---

## 🔧 二、调整形状

**元素个数不变，改变排列方式。** 比如把 3×2×6（36个元素）变成 4×9 或 6×6。

### 2.1 `reshape()` — 最常用的调整形状方法

```python
t = torch.randint(1, 10, (3, 2, 6))   # 36 个元素
print(t.shape)                          # torch.Size([3, 2, 6])

# 变成二维：4 行 9 列
t2 = t.reshape(4, 9)
print(t2.shape)                         # torch.Size([4, 9])

# 变成一维：长度为 36（向量化）
t3 = t.reshape(-1)
print(t3.shape)                         # torch.Size([36])

# 用 -1 让 PyTorch 自动算维度
t4 = t.reshape(4, -1)                   # 4 × ? = 36 → ? = 9
print(t4.shape)                         # torch.Size([4, 9])
```

> **`-1` 表示"让 PyTorch 自动计算这个维度的大小"**，前提是其他维度乘起来能整除总元素数。

### 2.2 `view()` — 共享内存版本的 reshape

`view()` 跟 `reshape()` 功能一样，但有两个关键点：

| | `reshape()` | `view()` |
|---|---|---|
| **内存** | 可能返回副本或视图 | ✅ **始终共享内存** |
| **连续性要求** | ❌ 不需要连续 | ✅ **必须内存连续** |

```python
t = torch.randint(1, 10, (3, 2, 6))

# 初始张量内存是连续的
print(t.is_contiguous())    # True

# view 可以用
t1 = t.view(6, 6)           # ✅ 连续内存，可以用

# 转置后内存就不连续了
t2 = t.transpose(0, 1)
print(t2.is_contiguous())   # False

# ❌ view 不能用
# t3 = t2.view(-1)          # 报错！内存不连续

# ✅ 先强制连续，再用 view
t3 = t2.contiguous().view(-1)
print(t3.shape)             # torch.Size([36])
```

> **规律**：初始创建的张量内存总是连续的。做转置、交换维度等操作后，内存可能变得不连续。

### 调整形状速查

```python
t.reshape(4, 9)     # 变成 4×9
t.reshape(-1)       # 展平成一维
t.reshape(4, -1)    # 4 行，列自动算
t.view(4, 9)        # 同 reshape，但共享内存
t.contiguous()      # 强制内存连续
```

---

## 📏 三、增删维度

**直接增加或删除大小为 1 的维度。**

### 3.1 `unsqueeze(dim)` — 增加维度（膨胀）

```python
t = torch.tensor([1, 2, 3])        # 形状 (3,) — 一维
print(t.shape)

# 在第 0 维增加 → (1, 3)
t1 = t.unsqueeze(0)
print(t1.shape)                     # torch.Size([1, 3])
print(t1)                           # tensor([[1, 2, 3]])

# 在第 1 维增加 → (3, 1)
t2 = t.unsqueeze(1)
print(t2.shape)                     # torch.Size([3, 1])
print(t2)                           # tensor([[1], [2], [3]])

# 也可以用负数索引
t3 = t.unsqueeze(-1)                # 等同 unsqueeze(1)
t4 = t.unsqueeze(-2)                # 等同 unsqueeze(0)
```

> **增加维度后，数据不变，只是多了一层中括号。**

### 3.2 `squeeze(dim)` — 删除维度（压缩）

```python
t = torch.tensor([[[1, 2, 3]]])    # 形状 (1, 1, 3)

# 删除所有大小为 1 的维度
t1 = t.squeeze()
print(t1.shape)                     # torch.Size([3])

# 只删除指定维度
t2 = t.squeeze(0)
print(t2.shape)                     # torch.Size([1, 3])

# 不能删除大小不为 1 的维度！
t3 = torch.randn(2, 3)
# t3.squeeze(0)                     # ❌ 维度 0 大小为 2，删不掉！
```

> **`squeeze()` 只能删除大小（size）为 1 的维度**。如果大小不是 1，调用无效。

### 增删维度速查

```python
t.unsqueeze(0)      # 在第 0 维前加一维
t.unsqueeze(-1)     # 在最后加一维
t.squeeze()         # 删掉所有大小为 1 的维度
t.squeeze(0)        # 只删第 0 维（如果它大小为 1）
```

---

## 🧩 四、拼接和堆叠

把多个张量合成一个。

### 4.1 `torch.cat()` — 拼接（不增加维度）

**要求**：除了拼接维度外，其他维度大小必须相同。

```python
t1 = torch.randint(1, 10, (3, 2, 6))    # (3, 2, 6)
t2 = torch.randint(1, 10, (3, 1, 6))    # (3, 1, 6)

# 沿 dim=1 拼接：把行合在一起
t3 = torch.cat([t1, t2], dim=1)
print(t3.shape)                         # torch.Size([3, 3, 6])
# 注意：2 + 1 = 3 行

# ❌ 沿 dim=0 拼接不行！因为 dim=1 维度大小不同（2 ≠ 1）
# torch.cat([t1, t2], dim=0)            # 报错！
```

> **`cat` 不改变维度个数**，只在某个维度上"叠加"数据。

### 4.2 `torch.stack()` — 堆叠（增加新维度）

**要求**：所有张量的形状必须完全相同。

```python
t1 = torch.randint(1, 10, (3, 4, 6))    # (3, 4, 6)
t2 = torch.randint(1, 10, (3, 4, 6))    # (3, 4, 6)

# 沿 dim=0 堆叠 → 新增一个维度放在最前面
t3 = torch.stack([t1, t2], dim=0)
print(t3.shape)                         # torch.Size([2, 3, 4, 6])
# 多出来的 2 是因为有 2 个张量

# 沿 dim=1 堆叠
t4 = torch.stack([t1, t2], dim=1)
print(t4.shape)                         # torch.Size([3, 2, 4, 6])

# 沿 dim=2 堆叠
t5 = torch.stack([t1, t2], dim=2)
print(t5.shape)                         # torch.Size([3, 4, 2, 6])
```

> **`stack` 会增加一个维度**，新增维度的大小 = 你堆叠的张量个数。

### cat vs stack 对比

| | `cat()` | `stack()` |
|---|---|---|
| **维度变化** | 不变 | **增加一维** |
| **形状要求** | 拼接维可以不同，其他必须相同 | **必须完全相同** |
| **结果形状** | 拼接维叠加 | 新增一维 大小=张量个数 |
| **典型场景** | 合并数据 | 把多个样本合成一个 batch |

---

## 📝 五、本章总结

### 知识地图

```
张量形状操作
    │
    ├── ① 交换维度
    │      ├── t.T              ← 全翻转（不推荐）
    │      ├── t.mT             ← 矩阵转置（后两维）
    │      ├── t.transpose(d1,d2) ← 交换任意两维
    │      └── t.permute(dims)  ← 大招！任意排列 ✅
    │
    ├── ② 调整形状
    │      ├── t.reshape(形状)  ← 最常用 ✅
    │      ├── t.view(形状)     ← 共享内存（需连续）
    │      ├── t.reshape(-1)    ← 展平成一维
    │      └── t.contiguous()   ← 强制内存连续
    │
    ├── ③ 增删维度
    │      ├── t.unsqueeze(dim) ← 增加维度（膨胀）
    │      └── t.squeeze(dim)   ← 删除维度（压缩，仅 size=1）
    │
    └── ④ 拼接和堆叠
           ├── torch.cat([t1,t2], dim)  ← 不增维，叠加
           └── torch.stack([t1,t2], dim) ← 增一维，堆叠
```

### 关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| **`permute()`** | 维度交换的万能方法，直接指定新顺序 |
| **`mT`** | 只转置最后两维（矩阵转置） |
| **`reshape()`** | 改变形状，元素个数不变 |
| **`view()`** | 同 reshape，但要求内存连续且共享内存 |
| **`contiguous()`** | 强制内存连续（之后才能用 view） |
| **`unsqueeze()`** | 增加一个大小为 1 的维度 |
| **`squeeze()`** | 删除大小为 1 的维度 |
| **`cat()`** | 拼接，不增加维度 |
| **`stack()`** | 堆叠，增加一个新维度 |

---

---