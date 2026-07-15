# 🧠 第十二部分：设备管理（device）与 Sequential 快捷搭建

> 📺 视频来源：PyTorch 深度学习 · 设备设置 · Sequential 快捷定义
> 🎯 核心目标：掌握 CPU/GPU 设备管理，学会用 Sequential 快速搭建神经网络
> 📝 风格：实操对比 + 最佳实践

---

## 📌 一、什么是设备（device）？CPU vs GPU

### device 的概念

在 PyTorch 中，**张量和模型都可以放在不同的设备上运行**：

| 设备 | 说明 | 计算速度 |
|------|------|:-------:|
| **CPU** | 中央处理器，通用的计算单元 | 较慢 |
| **GPU**（CUDA） | 图形处理器，擅长**并行计算** | **快 10~100 倍** |

深度学习训练的核心是**大量矩阵运算**→ GPU 的并行架构天然适合 → **加速训练几倍到几十倍**。

### 查看设备

```python
import torch

# 检查 GPU 是否可用
print(torch.cuda.is_available())  # True 表示 GPU 版本安装成功
# 查看 GPU 数量
print(torch.cuda.device_count())

# 查看当前张量默认设备
x = torch.randn(3, 5)
print(x.device)  # cpu
```

---

## 📌 二、创建张量时指定设备

创建张量时，通过 `device` 参数直接指定：

```python
# 创建在 CPU 上（默认）
x_cpu = torch.randn(3, 5)

# 创建在 GPU 上（需要安装 CUDA 版本）
x_gpu = torch.randn(3, 5, device='cuda')
# 等价于
x_gpu = torch.randn(3, 5, device=torch.device('cuda'))

# 指定具体的 GPU 编号（多显卡时）
x_gpu0 = torch.randn(3, 5, device='cuda:0')
```

> 💡 **提示**：`'cuda'` 默认等于 `'cuda:0'`。如果有多个 GPU，可以用 `'cuda:1'`、`'cuda:2'` 等。

---

## 📌 三、迁移设备 — `to()` 方法

如果张量已经创建好了，可以用 `.to()` 方法迁移到另一个设备：

```python
# 创建在 CPU 上
x = torch.randn(3, 5)
print(x.device)  # cpu

# 迁移到 GPU
x = x.to('cuda')
print(x.device)  # cuda:0

# 再迁移回 CPU
x = x.to('cpu')
print(x.device)  # cpu
```

### `to()` 还能转换数据类型

```python
x = torch.randn(3, 5)
x = x.to(dtype=torch.int64)   # 转换数据类型
x = x.to(device='cuda', dtype=torch.float32)  # 同时转设备和类型
```

> 🎯 **类比**：`to()` 就像张量的"搬家"工具，可以把张量搬到任何设备上。

---

## 📌 四、将模型迁移到 GPU

### 4.1 模型的参数默认在 CPU

```python
import torch.nn as nn

model = nn.Linear(3, 4)
for param in model.parameters():
    print(param.device)  # cpu（默认）
```

### 4.2 把模型迁移到 GPU

```python
model = model.to('cuda')   # 模型所有参数 → GPU
```

### 4.3 前向传播时设备必须一致

> ⚠️ **关键规则**：**数据和模型必须在同一个设备上**，否则报错。

```python
# ❌ 错误示例
model = nn.Linear(3, 4).to('cuda')
x = torch.randn(10, 3)           # 默认 CPU
y = model(x)  # RuntimeError：Expected all tensors to be on the same device!

# ✅ 正确做法
x = torch.randn(10, 3).to('cuda')  # 数据也放到 GPU
y = model(x)                        # 正常工作
```

### 4.4 创建层时直接指定设备

`nn.Linear` 创建时也可以直接指定设备：

```python
# 直接在 GPU 上创建层
layer = nn.Linear(3, 4, device='cuda')

# 查看参数设备
print(layer.weight.device)  # cuda:0
```

> 💡 但更推荐的做法是用 `model.to('cuda')` 整体迁移，而不是每个层单独设。

---

## 📌 五、设备统一管理（全局变量 + is_available）

### 5.1 为什么需要统一管理？

如果代码里到处写 `device='cuda'`，以后想换回 CPU 运行就要改几十个地方 → **用全局变量统一管理**。

### 5.2 标准写法（工程最佳实践）

```python
# 定义一个全局变量，统一管理设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)  # cuda 或 cpu
```

### 5.3 使用全局变量

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建数据
x = torch.randn(10, 3).to(device)

# 创建模型
model = NNModel().to(device)

# 前向传播
y = model(x)
```

以后如果想切换设备：
- **用 GPU**：自动检测，不用改
- **用 CPU**：改一行都行，把 `'cuda'` 改成 `'cpu'`（或者直接卸载 CUDA 版）

> 🎯 **一句话总结**：永远不要硬编码 `device='cuda'`，用 `'cuda' if torch.cuda.is_available() else 'cpu'`。

---

## 📌 六、Sequential 顺序容器（快捷搭建神经网络）

### 6.1 为什么需要 Sequential？

之前我们自定义模型（继承 `nn.Module`）很灵活，但对于**简单的顺序结构**：

```
Linear → Tanh → Linear → ReLU → Linear → Softmax
```

这种**一层接一层的顺序结构**，可以用 `nn.Sequential` 更简洁地定义。

### 6.2 两种方式对比

**方式一：继承 nn.Module（我们之前的方式）**

```python
class NNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 4)
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.softmax(self.out(x), dim=1)
        return x
```

**方式二：nn.Sequential（快捷方式）**

```python
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),          # 激活函数也被当成"层"
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Softmax(dim=1)
)
```

> **关键点**：激活函数用**大写**（`nn.Tanh()`、`nn.ReLU()`、`nn.Softmax()`）——它们是类，作为层放入 Sequential。

### 6.3 把激活函数也包成"层"

之前我们用 `torch.tanh(x)` 是**函数式调用**。在 Sequential 中，必须使用**类形式的激活函数**：

| 函数式（之前用） | 类形式（Sequential 中用） | 说明 |
|:---:|:---:|------|
| `torch.tanh(x)` | `nn.Tanh()` | 双曲正切 |
| `torch.relu(x)` | `nn.ReLU()` | ReLU |
| `torch.sigmoid(x)` | `nn.Sigmoid()` | Sigmoid |
| `torch.softmax(x, dim)` | `nn.Softmax(dim)` | Softmax |

> 💡 类形式的激活函数也是一个 `nn.Module`，只是内部没有任何参数（参数量 = 0）。

### 6.4 前向传播和查看结构

```python
# 1. 定义数据
X = torch.randn(10, 3)

# 2. 创建模型
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Softmax(dim=1)
)

# 3. 前向传播（和 nn.Module 一样！）
y_pred = model(X)
print(y_pred.shape)   # torch.Size([10, 2])
print(y_pred.sum(dim=1))  # 每行和为 1
```

### 6.5 torchsummary 查看结构

```python
from torchsummary import summary

summary(model, input_size=(3,), batch_size=10, device='cpu')
```

输出：
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                [-1, 10, 4]              16
              Tanh-2                [-1, 10, 4]               0
            Linear-3                [-1, 10, 4]              20
              ReLU-4                [-1, 10, 4]               0
            Linear-5                [-1, 10, 2]              10
           Softmax-6                [-1, 10, 2]               0
================================================================
Total params: 46
Trainable params: 46
Non-trainable params: 0
----------------------------------------------------------------
```

注意：
- 现在是 **6 层**（3 个 Linear + 3 个激活函数层）
- 激活函数层的 **Param # = 0**（没有参数）
- 总参数量仍然是 **46**，和之前完全一样

---

## 📌 七、Sequential 的参数初始化（apply 方法）

### 7.1 Sequential 初始化的问题

Sequential 定义很简洁，但**不能在定义的同时做参数初始化**（因为初始化需逐个层指定策略）。

```python
# ❌ 没办法在 Sequential 创建时做 Xavier/Kaiming 初始化
model = nn.Sequential(
    nn.Linear(3, 4),   # 想初始化这里用 Xavier
    nn.Tanh(),
    nn.Linear(4, 4),   # 想初始化这里用 Kaiming
    ...
)
# 嗯...怎么初始化呢？
```

### 7.2 解决方案：`model.apply()` 方法

`apply()` 方法会把传入的函数**应用到模型的每一个子模块（层）上**。

```python
def init_weights(layer):
    """参数初始化函数：对 Linear 层做初始化"""
    if isinstance(layer, nn.Linear):
        # 初始化权重
        nn.init.xavier_uniform_(layer.weight)
        # 初始化偏置（给一个很小的常数）
        nn.init.constant_(layer.bias, 0.01)

# 创建模型
model = nn.Sequential(...)

# 应用初始化（注意：传函数名，不要加括号）
model.apply(init_weights)
```

### 7.3 原理说明

```python
# apply 底层大致逻辑：
for sub_module in model.modules():   # 递归遍历所有子模块
    init_weights(sub_module)          # 对每个模块应用函数

# 在我们的例子中，sub_module 依次是：
#   nn.Linear(3,4) → init_weights(Linear)
#   nn.Tanh()       → init_weights(Tanh)   ← Tanh 不是 Linear，跳过
#   nn.Linear(4,4) → init_weights(Linear)
#   nn.ReLU()       → init_weights(ReLU)   ← 不是 Linear，跳过
#   nn.Linear(4,2) → init_weights(Linear)
#   nn.Softmax()    → init_weights(Softmax) ← 不是 Linear，跳过
```

### 7.4 完整示例

```python
import torch.nn as nn
import torch.nn.init as init

def init_weights(layer):
    if isinstance(layer, nn.Linear):
        init.xavier_uniform_(layer.weight)
        init.constant_(layer.bias, 0.01)

model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Softmax(dim=1)
)

# 对模型应用初始化
model.apply(init_weights)

# 前向传播测试
X = torch.randn(10, 3)
y_pred = model(X)
print(y_pred)
```

### 7.5 继承式 vs Sequential 对比

| 对比维度 | 继承 nn.Module（手动定义） | nn.Sequential（顺序容器） |
|---------|:------------------------:|:------------------------:|
| **定义方式** | 自定义类，写 `__init__` + `forward` | 像列表一样依次传入层 |
| **灵活性** | ✅ **高**（可自定义复杂 forward 逻辑） | ❌ 仅支持**简单顺序**结构 |
| **参数初始化** | ✅ 在 `__init__` 中逐层指定 | ⚠️ 需额外用 `apply()` 统一处理 |
| **代码简洁度** | ❌ 较繁琐 | ✅ **非常简洁** |
| **适合场景** | 复杂网络（残差连接、多分支等） | **简单顺序**网络（MLP 等） |

---

## 📝 本章总结 + 完整代码

### 🌳 知识树

```
设备管理与 Sequential 搭建
│
├── ① 设备（device）
│   ├── CPU vs GPU
│   ├── 创建时指定：device='cuda'
│   ├── 迁移：.to('cuda')
│   └── 统一管理：torch.device(...)
│
├── ② 设备最佳实践
│   ├── model = model.to(device)
│   ├── X = X.to(device)
│   └── device = 'cuda' if cuda.is_available() else 'cpu'
│
├── ③ nn.Sequential（快捷搭建）
│   ├── 像列表一样定义各层
│   ├── 激活函数用类形式：nn.Tanh() / nn.ReLU() / nn.Softmax()
│   └── model(x) 自动顺序前向传播
│
└── ④ apply() 参数初始化
    ├── 定义函数取每个层
    ├── isinstance 判断层类型
    ├── 只对 Linear 层做初始化
    └── model.apply(init_weights)
```

### 🔥 完整可运行代码

```python
import torch
import torch.nn as nn
import torch.nn.init as init
from torchsummary import summary

# ============ 1. 设备管理 ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============ 2. 定义初始化函数 ============
def init_weights(layer):
    if isinstance(layer, nn.Linear):
        init.xavier_uniform_(layer.weight)
        init.constant_(layer.bias, 0.01)

# ============ 3. 用 Sequential 搭建模型 ============
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Softmax(dim=1)
).to(device)

# 参数初始化
model.apply(init_weights)

# ============ 4. 前向传播 ============
X = torch.randn(10, 3).to(device)
y_pred = model(X)

print(f"输出形状: {y_pred.shape}")
print(f"预测结果:\n{y_pred}")

# ============ 5. 查看模型结构 ============
summary(model, input_size=(3,), batch_size=10, device='cpu')
```

### 关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| **`device`** | 张量/模型运行在哪个设备上（CPU 或 GPU） |
| **`tensor.to(device)`** | 把张量迁移到指定设备 |
| **`model.to(device)`** | 把模型所有参数迁移到指定设备 |
| **`torch.cuda.is_available()`** | 检查 GPU 是否可用 |
| **全局 device 变量** | `'cuda' if cuda.is_available() else 'cpu'` |
| **`nn.Sequential`** | 按顺序排列层，自动完成前向传播 |
| **`nn.Tanh()`、`nn.ReLU()`** | 激活函数的类形式，可作为层放入 Sequential |
| **`model.apply(fn)`** | 对每个子模块应用函数 fn |
| **`isinstance(layer, nn.Linear)`** | 判断层是否是线性层（用于初始化） |

> 🎯 **现在你学会了两种搭建方式**：
> - **继承 nn.Module** — 灵活强大，适合复杂网络
> - **nn.Sequential** — 简洁快捷，适合简单顺序网络
>
> 以后写代码的时候，别忘了 **device 统一管理**那一步！