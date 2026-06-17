# 🧠 深度学习笔记 — PyTorch 从入门到张量操作

> 📝 本笔记为 PyTorch 学习系列合集，适合深度学习初学者
> 🎯 风格：代码实践 + 通俗讲解

---

## 📋 目录

### 第一部分：PyTorch 入门与安装
- **一、** PyTorch 是什么？为什么要学它？
- **二、** 安装前的准备工作（CPU vs GPU）
- **三、** PyTorch 安装实战（命令行版）
- **四、** 建议：用虚拟环境安装（推荐做法）
- **五、** GPU 版本安装全流程
- **六、** CUDA Toolkit 安装（可选）
- **七、** 常见问题 & 避坑指南
- **八、** 本章总结 + 速查命令 + 安装验证

### 第二部分：PyTorch 张量创建
- **一、** 创建张量的两大类场景
- **二、** 按内容创建 — `torch.tensor()`
- **三、** 按形状创建 — `torch.Tensor()`
- **四、** 核心对比：`tensor()` vs `Tensor()`
- **五、** 指定数据类型（dtype）
- **六、** 指定区间创建（arange / linspace / logspace）
- **七、** 按数值填充（zeros / ones / full / empty / eye）
- **八、** 随机生成（rand / randint / randn / normal / randperm / seed）
- **九、** 类比理解：张量的"维度"
- **十、** 常见问题 & 避坑指南
- **十一、** 本章总结

### 第三部分：PyTorch 张量转换
- **一、** 张量元素类型转换（type / float / int / long / bool）
- **二、** Tensor ↔ NumPy 转换（内存共享 & 避免共享）
- **三、** Tensor ↔ 标量转换（item）
- **四、** 本章总结

### 第四部分：PyTorch 张量数值计算
- **一、** 基本运算（加减乘除、取负、幂、平方根、指数、对数）
- **二、** 哈达玛积（对应位置元素相乘）
- **三、** 矩阵乘法（二维 & 高维张量、广播机制）
- **四、** 节省内存（原地操作、切片赋值实现节省内存）
- **五、** 统计函数（sum / mean / std / var / max / min / argmax / unique / sort）
- **六、** 本章总结

### 第五部分：PyTorch 张量索引
- **一、** 简单索引（整数索引）
- **二、** 范围索引（切片）
- **三、** 列表索引（花式索引）
- **四、** 布尔索引（条件筛选、行/列筛选 + 转置）
- **五、** 本章总结

### 第六部分：PyTorch 张量形状操作
- **一、** 交换维度（T / mT / transpose / permute）
- **二、** 调整形状（reshape / view + 内存连续性）
- **三、** 增删维度（unsqueeze / squeeze）
- **四、** 拼接和堆叠（cat / stack）
- **五、** 本章总结

### 第七部分：PyTorch 自动微分（Autograd）
- **一、** 自动微分原理（requires_grad / grad / grad_fn）
- **二、** 前向传播与计算图构建
- **三、** 反向传播（backward）
- **四、** 叶子节点 vs 非叶子节点
- **五、** detach 分离张量（切断梯度计算）
- **六、** detach 对梯度计算的影响（进阶验证）
- **七、** detach vs data（推荐的分离方式）
- **八、** 本章总结

### 第八部分：线性回归实战案例
- **一、** 整体思路与流程
- **二、** 准备数据（生成样本 + Dataset + DataLoader）
- **三、** 定义模型、损失函数与优化器
- **四、** 模型训练（五步核心流程）
- **五、** 记录损失与画图
- **六、** 调参优化与效果对比
- **七、** 本章总结 + 完整代码

### 第九部分：激活函数（Activation Functions）
- **一、** 为什么需要激活函数？
- **二、** Sigmoid（S 型曲线 + 自动求导画图）
- **三、** Tanh（双曲正切 + 对比 Sigmoid）
- **四、** ReLU（深度网络首选 + 不可导点的处理）
- **五、** Softmax（多分类概率转换 + dim 参数详解）
- **六、** 本章总结 + 对比速查表

### 第十部分：全连接层与参数初始化
- **一、** 全连接层（nn.Linear）详解
- **二、** weight 和 bias 的形状与含义
- **三、** 常数初始化（zeros / ones / constant / eye）
- **四、** 随机初始化（normal / uniform）
- **五、** 工程初始化（Xavier / Kaiming）
- **六、** 默认初始化机制
- **七、** 本章总结 + 速查表

### 第十一部分：自定义神经网络模型
- **一、** nn.Module 基类（所有网络的基石）
- **二、** 搭建三层神经网络（3→4 Tanh → 4 ReLU → 2 Softmax）
- **三、** 参数初始化实践（Xavier + Kaiming 组合）
- **四、** Dropout 正则化（随机关闭神经元）
- **五、** 查看模型参数（4种方法）
- **六、** 查看模型结构与参数数量（torchsummary）
- **七、** 本章总结 + 完整代码

### 第十二部分：设备管理（device）与 Sequential 快捷搭建
- **一、** 什么是设备（device）？CPU vs GPU
- **二、** 创建张量时指定设备
- **三、** 迁移设备 — `to()` 方法
- **四、** 将模型迁移到 GPU
- **五、** 设备统一管理（全局变量 + is_available）
- **六、** Sequential 顺序容器（快捷搭建神经网络）
- **七、** Sequential 的参数初始化（apply 方法）
- **八、** 本章总结 + 完整代码

### 第十三部分：损失函数（Loss Functions）
- **一、** 分类任务 vs 回归任务
- **二、** 二分类损失 — BCE Loss（二元交叉熵）
- **三、** 多分类损失 — CrossEntropyLoss（交叉熵 + Softmax）
- **四、** 回归损失 — MSE / MAE / Smooth L1
- **五、** 本章总结 + 选择指南

### 第十四部分：优化方法（Optimizers）
- **一、** 优化器概述
- **二、** SGD + Momentum（动量法）
- **三、** 学习率衰减（StepLR / MultiStepLR / ExponentialLR）
- **四、** 自适应学习率（AdaGrad / RMSProp）
- **五、** Adam 与 AdamW（工程首选）
- **六、** 本章总结 + 选择指南

---

---

# 🧠 第一部分：PyTorch 入门与安装

> 📺 视频来源：PyTorch 基本介绍 · CPU/GPU 安装 · CUDA 配置
> 🎯 适合人群：深度学习初学者
> 📝 风格：代码实践 + 通俗讲解

---

## 📌 一、PyTorch 是什么？为什么要学它？

### 1.1 从"手写"到"调库"

前面我们学神经网络的时候，是不是手动实现了前向传播、反向传播、梯度下降？那些代码帮助我们**理解原理**。

但真正到实际工程中，没人会手写这些东西 —— 就像你不会自己造轮子一样 🛞

**工程应用 = 调库**，深度学习有自己的专业框架。

之前学机器学习我们用 `scikit-learn`（sklearn），但深度学习的特点是：

| 机器学习 | 深度学习 |
|---------|---------|
| 数据维度低 | **高维张量**（图片、视频、文本） |
| 没有反向传播 | **自动求梯度**是核心 |
| sklearn 就够用 | 需要专业的深度学习框架 |

### 1.2 PyTorch 的前世今生

```
        Torch 库（底层 C 语言，接口用 Lua）
                  ↓
     2016 年 Facebook 给它做了 Python 接口
                  ↓
           PyTorch = Python + Torch
                  ↓
       现在：业界最火的深度学习框架 🔥
```

**名字拆解**：
- **Py** = Python（你会的那个 Python）
- **Torch** = 火炬（前身是一个叫 Torch 的科学计算框架）

### 1.3 PyTorch vs TensorFlow（为啥选 PyTorch？）

在深度学习发展史上，有两个最著名的框架：

| 对比维度 | PyTorch 🔥 | TensorFlow 🔵 |
|---------|-----------|-------------|
| **开发团队** | Facebook（Meta） | 谷歌 |
| **上手难度** | 🟢 **超简单**，跟 numpy 几乎一样 | 🔴 偏底层，像学一门新语言 |
| **计算图** | 动态计算图（灵活，好调试） | 静态计算图（先建图再编译，僵化） |
| **API 设计** | 简洁直观 | 经常被吐槽，2.0 大改版还不兼容旧版 |
| **学术界** | 📚 主流！论文代码都用 PyTorch | 越来越少 |
| **工业界** | 🔥 迎头赶上，生态越来越好 | 曾经的主流，但逐渐被替代 |

> **结论**：现在学深度学习，**默认直接学 PyTorch 就行了**，不用再学 TensorFlow 了。

### 1.4 PyTorch 的两大核心能力

```
PyTorch
    ├── 🎯 跟 NumPy 几乎一样的张量操作（tensor）
    └── 🚀 自动微分系统（autograd）—— 自动算梯度！
```

- **张量（Tensor）**：就是多维数组，跟 NumPy 的 ndarray 几乎一模一样
- **自动微分（Autograd）**：帮你自动算反向传播的梯度 —— 再也不用手写链式法则了！

---

## 🚀 二、安装前的准备工作

### 2.1 CPU 版本 vs GPU 版本

PyTorch 分两个版本：

```
PyTorch 版本
    ├── CPU 版本 🐢 → 用 CPU 算，数据在内存里
    └── GPU 版本 🚀 → 用显卡算，数据在显存里
```

**一定要搞清楚**：深度学习的核心是**大规模矩阵运算**，这种场景显卡（GPU）有天然优势。

### 2.2 CPU 和 GPU 的"神类比" 🎯

> **CPU = 一个博士生** 🎓
> **GPU = 100 个小学生** 👦👧👦👧...

| 任务 | 谁厉害？ |
|------|---------|
| 处理复杂任务（打开浏览器、运行游戏） | 🟢 **博士生**（CPU）厉害 |
| 跑 20 以内的加减乘除四则运算 | 🟢 **100 个小学生**（GPU）厉害！并行计算 |

深度学习就是要做**海量的简单矩阵运算**，所以 GPU 的并行计算能力完胜！

### 2.3 到底装哪个版本？

```mermaid
graph TD
    A[我要装 PyTorch] --> B{有独立显卡吗？}
    B -->|没有 →| C[装 CPU 版本]
    B -->|有 →| D{是 N 卡还是 A 卡？}
    D -->|N 卡 NVIDIA| E[装 GPU 版本 🚀 推荐！]
    D -->|A 卡 AMD| F{操作系统？}
    F -->|Windows| G[装 CPU 版本<br>（ROCm 不支持 Windows）]
    F -->|Linux| H[装 GPU 版本<br>（选 ROCm 平台）]
```

> 💡 **重要提示**：初学者阶段，CPU 和 GPU 版本**几乎没有差别**。我们做的案例模型都很小，CPU 跑得飞快。只有到训练大模型的时候，差别才体现出来。甚至真到大模型阶段，你自己的显卡也带不动，得上云租服务器。

---

## 🔧 三、PyTorch 安装实战（命令行版）

### 3.1 第一步：去官网找安装命令

打开 [pytorch.org](https://pytorch.org)，你会看到一个配置表格：

```
PyTorch Build:  2.8.0 （稳定版，选这个）
Your OS:        Windows / Linux / Mac
Package:        Pip （推荐）
Language:       Python （你用的）
Compute Platform: CPU / CUDA 12.6 / CUDA 12.8 / CUDA 12.9
```

选好之后，页面会自动生成安装命令，复制粘贴就能装。

### 3.2 安装 CPU 版本（最简单）

```bash
# CPU 版本 — 一行命令搞定
pip install torch torchvision torchaudio
```

**就这么简单！** 没有任何其他参数。

> 从 PyTorch 2.8.0 开始，`torchaudio` 已经合并到 `torch` 里了，但 `torchvision`（图像处理）还得单独装。

### 3.3 安装 GPU 版本（稍微麻烦一点）

GPU 安装的核心就一句话：

> **选一个不超过你显卡支持的最高 CUDA 版本的 PyTorch 版本**

怎么选？往下看 👇

---

## 🛠️ 四、建议：用虚拟环境安装（推荐做法）

### 4.1 为什么要用虚拟环境？

> **虚拟环境 = 一个独立的 Python 小房间** 🏠
> 每个项目都有自己的房间，互不干扰。

直接装在 base 环境也可以，但推荐**新建一个虚拟环境**，因为：
- PyTorch 比较大，集中管理方便
- 不同项目可能需要不同版本的 PyTorch
- 环境隔离，互不影响

### 4.2 创建虚拟环境

```bash
# 用 conda 创建虚拟环境（名字自己取，比如 pytorch_env）
conda create -n pytorch_env python=3.12

# 建议指定 Python 版本（3.12），不然默认装最新版容易乱
```

> 💡 Python 版本建议指定 `3.12`，稳定兼容性好。如果不指定，conda 会装最新版（当前是 3.13），有些包可能还没适配。

### 4.3 激活虚拟环境

```bash
# Windows 下激活
conda activate pytorch_env

# 看到命令提示符前面变成了 (pytorch_env) → 说明进入成功
```

### 4.4 配置国内镜像源（下载加速）

默认 pip 源在国外，下载慢可以配成国内源：

```bash
# 配置清华镜像源（推荐）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4.5 在虚拟环境中安装 PyTorch

```bash
# 进入虚拟环境后，直接执行你在官网上复制的命令
# CPU 版：
pip install torch torchvision torchaudio

# GPU 版（以 CUDA 11.8 为例）：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4.6 安装常用库

新建的虚拟环境里只有最基本的包，你常用的库需要自己装：

```bash
# 建议至少装上 jupyter notebook，方便后面做测试
pip install jupyter notebook

# 其他常用库（需要时再装）
pip install pandas matplotlib scikit-learn
```

### 4.7 在 IDE 中切换解释器

如果你用 PyCharm 或 VS Code：

```
PyCharm 设置：
File → Settings → Project → Python Interpreter
    → 选择你刚创建的 conda 环境 (pytorch_env)
    
或者新建项目时：
New Project → Custom Environment → Select Existing → Conda → 选 pytorch_env
```

---

## 🎯 五、GPU 版本安装全流程（重点！）

### 5.1 三步走战略

```
第 1 步：查显卡型号 → 确定计算能力
第 2 步：查最高 CUDA 版本
第 3 步：选对应 PyTorch 版本 → 执行安装命令
```

### 5.2 第 1 步：查看显卡型号

**方法**：打开电脑的「设备管理器」→「显示适配器」

你会看到类似这样的信息：
```
NVIDIA GeForce RTX 3050 Ti   ← 你的显卡型号
```

### 5.3 第 2 步：查计算能力

去 NVIDIA 官网查你的显卡**计算能力**：
- 打开这个网页：https://developer.nvidia.com/cuda-gpus
- 找到你的显卡型号，看对应的 **Compute Capability**
- **要求 >= 3.0**（最近几年买的电脑都满足）

### 5.4 第 3 步：查最高支持的 CUDA 版本 ⭐

**推荐方法 — 用 nvidia-smi 命令：**

```bash
# 打开命令行（CMD 或 Anaconda Prompt 都行）
nvidia-smi
```

你会看到类似这样的输出：

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI  xxx.xx    Driver Version: xxx.xx       CUDA Version: 12.3       |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   52°C    P0    N/A /  60W|      0MiB /  4096MiB |      0%      Default |
|-------------------------------+----------------------+----------------------+
|                                                                              |
| Processes:                                                                   |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

**重点看这一行：**
```
CUDA Version: 12.3   ← 你的显卡支持的最高 CUDA 版本
```

> 💡 **备用方法**：打开「NVIDIA 控制面板」→ 左下角「系统信息」→「组件」标签 → 找 `CUDA 64.dll`

### 5.5 第 4 步：根据 CUDA 版本选择 PyTorch 版本

**原则：** PyTorch 基于的 CUDA 版本 ≤ 你显卡支持的 CUDA 版本

| 你的 CUDA 版本 | 可以选 PyTorch 基于的 CUDA 版本 |
|---------------|-------------------------------|
| 12.6 | 12.6 ✅（完美匹配）、12.4、11.8 ... |
| 12.3 | 11.8 ✅（不超过 12.3 就行） |
| 12.9 | 12.9 ✅（完美匹配） |
| 13.0 | 12.9（因为 PyTorch 最新只到 12.9） |

> **CUDA 向前兼容**：高版本 CUDA 支持低版本编译的程序。所以你显卡是 12.3，选 11.8 的 PyTorch 完全没问题。

**举例：**

如果你的显卡最高支持 CUDA 12.3：
```bash
# PyTorch 2.7.0 有基于 CUDA 11.8 的版本（≤ 12.3，可以用）
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
```

如果你的显卡最高支持 CUDA 12.6+：
```bash
# 直接选最新版，完美匹配
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 5.6 nvidia-smi 的其他有用信息

`nvidia-smi` 不只可以查 CUDA 版本，以后跑模型的时候你还会经常用它来看性能：

| 指标 | 含义 | 怎么看 |
|-----|------|-------|
| **GPU-Util** | GPU 利用率 | 0% 说明没跑东西，100% 说明跑满了 |
| **Memory-Usage** | 显存占用 | 比如 `2048MiB / 4096MiB` = 用了 2G，总共 4G |
| **Temp** | GPU 温度 | 一般 40-80°C 正常，太高要注意散热 |
| **Fan** | 风扇转速 | 温度高了风扇会转得快 |
| **Perf** | 性能状态 | P0（最大性能）~ P12（最小性能） |
| **PWR** | 功率 | 当前功率 / 最大功率 |
| **Processes** | 正在跑的进程 | 能看到哪个程序在占显卡 |

---

## 📦 六、CUDA Toolkit 安装（可选）

### 6.1 我需要装 CUDA Toolkit 吗？

```
你现在用 PyTorch 需要 CUDA 吗？
    ├── 只是用 PyTorch 训练模型 → ❌ 不需要！
    └── 想做 CUDA 编程（直接写 GPU 代码）→ ✅ 需要
```

> **好消息**：现在的 PyTorch 版本已经**内嵌了 CUDA 支持**，相关的 DLL 文件都打包好了。
> 早期版本需要先装 CUDA Toolkit，现在不需要了！

### 6.2 如果想装，怎么装？

去 [NVIDIA CUDA 官网](https://developer.nvidia.com/cuda-toolkit-archive)：

1. 选择你需要的 CUDA 版本（比如 11.8 或 12.3）
2. 选择操作系统（Windows x86_64）
3. 选择安装方式（推荐 Local，约 3GB）
4. 下载 exe 文件，一路「下一步」傻瓜式安装

安装完成后验证：
```bash
nvcc --version
# 输出类似：nvcc: NVIDIA (R) Cuda compiler driver
#          release 11.8, V11.8.89
```

---

## ⚠️ 七、常见问题 & 避坑指南

### Q1：我该装最新版还是历史版本？

> ✅ **推荐装最新版**。除非你的显卡不支持最新版对应的 CUDA 版本。

### Q2：Windows + A 卡（AMD）能装 GPU 版吗？

> ❌ **目前不行**。ROCm（AMD 的 CUDA）还不支持 Windows。只能装 CPU 版，或者换 Linux 系统。

### Q3：装 GPU 版比 CPU 版大很多吗？

> ✅ 是的。GPU 版因为包含了 CUDA 支持，包会大一些。但下载一次就行了。

### Q4：下载太慢怎么办？

```bash
# 可以加国内镜像源加速
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
```

或者下载 whl 文件离线安装（让下载好的同学分享给你）。

### Q5：torchvision 和 torchaudio 是什么？

| 包名 | 用途 | 说明 |
|-----|------|------|
| **torch** | PyTorch 核心 | 张量操作、神经网络、自动微分 |
| **torchvision** | 计算机视觉 | 图像处理、数据集、预训练模型 |
| **torchaudio** | 语音/音频 | 音频处理（2.8.0 起合并到 torch 了） |

---

## 📝 八、本章总结

### 知识地图

```
PyTorch = 当前最火的深度学习框架 🔥
    ↓
分 CPU 版（简单）和 GPU 版（快，但稍麻烦）
    ↓
GPU 安装三步走：
    ① nvidia-smi 查 CUDA 版本
    ② 官网选不超过该版本的 PyTorch
    ③ 复制命令，pip install 搞定
    ↓
CUDA Toolkit → 可装可不装（PyTorch 已内嵌）
    ↓
CPU 版一行命令搞定 → 初学者也能用
```

### 速查命令大全

```bash
# 查看显卡信息
nvidia-smi

# 查看 CUDA Toolkit 版本
nvcc --version

# 查看已安装的 PyTorch 信息
pip show torch

# 安装 CPU 版 PyTorch（最新版）
pip install torch torchvision torchaudio

# 安装 GPU 版 PyTorch（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 GPU 版 PyTorch（CUDA 12.6）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 查看 conda 有哪些虚拟环境
conda env list

# 创建新虚拟环境（Python 3.12）
conda create -n pytorch_env python=3.12

# 激活虚拟环境
conda activate pytorch_env

# 配置清华镜像源（加速下载）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 安装后验证

装完后，有几种方法可以验证是否安装成功：

#### 方法一：pip 查看（快速检查）

```bash
# 在命令行直接查看
pip show torch
```

输出示例：
```
Name: torch
Version: 2.7.0+cu118
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
...
```

> 看到 `Version: 2.7.0+cu118` 中的 `+cu118` 就说明装的是 CUDA 11.8 的 GPU 版本。

#### 方法二：在 Python 中 import（最靠谱）

打开命令行，激活你的虚拟环境，然后直接敲 `python` 进入交互模式：

```bash
conda activate pytorch_env
python
```

然后在 Python 里 import：

```python
import torch  # 如果卡一下才出结果 → 说明加载成功！
             # 如果直接报错 → 说明没装好
```

#### 方法三：完整验证脚本（推荐）

创建一个 `.py` 文件或在 Jupyter Notebook 中运行：

```python
import torch

# 1. 查看 PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 2. 创建一个张量
x = torch.tensor([1, 2, 3])
print(f"张量: {x}")

# 3. 检查 GPU 是否可用
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"显卡名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
    
    # 4. 查看 cuDNN 版本（深度学习加速库）
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
```

输出示例：
```
PyTorch 版本: 2.7.0+cu118
张量: tensor([1, 2, 3])
CUDA 是否可用: True
显卡名称: NVIDIA GeForce RTX 3050 Ti
CUDA 版本: 11.8
cuDNN 版本: 90100
```

> 🎉 看到 `CUDA 是否可用: True` 就说明 GPU 版安装成功了！
>
> `cuDNN` 是基于 CUDA 的深度神经网络加速库，**PyTorch 已经内嵌了**，不用单独安装。

---

---

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

# 🧠 第四部分：PyTorch 张量数值计算

---

## 📌 前置说明

张量创建和转换搞定后，自然要拿它来做计算。数值计算分为两大类：

```
张量计算
    ├── 数值计算（本节内容）
    │     针对每个元素做独立运算
    │     → 加减乘除、幂、开方、指数、对数、矩阵乘法
    │
    └── 聚合运算（后面讲）
          把一组数聚合成一个结果
          → 求和、求均值、求最大值等
```

---

## ➕ 一、基本运算

### 1.1 四则运算

加减乘除可以直接用运算符，也可以调方法：

```python
import torch

# 创建一个 2×3 的随机整数张量
t1 = torch.randint(1, 10, (2, 3))   # 1~9 的整数
print(t1)
# tensor([[3, 1, 4],
#         [7, 4, 8]])

# ===== 加法 =====
# 运算符方式
print(t1 + 10)   # 每个元素 +10，不改变 t1

# 方法调用（不改变原数据）
t2 = t1.add(10)
print(t2)        # 返回新张量

# 方法调用（原地修改，加下划线）
t1.add_(10)      # ← 下划线版本，直接改 t1 本身！
print(t1)        # t1 已经被改了
```

> **加下划线和不下划线的区别：**
> - `tensor.add(x)` → 返回新结果，原 tensor 不变
> - `tensor.add_(x)` → **原地修改**原 tensor，不返回新结果

```python
# 其他运算同理
t1 - 5           # 减法
t1 * 2           # 乘法
t1 / 3           # 除法
t1.add(10)       # 加
t1.sub(5)        # 减
t1.mul(2)        # 乘
t1.div(3)        # 除
t1.sub_(5)       # 原地减
t1.mul_(2)       # 原地乘
```

### 1.2 取负

```python
t = torch.tensor([1, -2, 3, -4])

# 运算符
print(-t)                    # tensor([-1,  2, -3,  4])

# 方法
print(t.neg())               # 同上，不改变原数据
t.neg_()                     # 原地取负
```

### 1.3 幂运算

```python
t = torch.tensor([1, 2, 3])

# 运算符：**
print(t ** 2)                # tensor([1, 4, 9])

# 方法：pow
print(t.pow(2))              # tensor([1, 4, 9])
t.pow_(2)                    # 原地求平方
print(t)                     # tensor([1, 4, 9])
```

### 1.4 平方根

```python
t = torch.tensor([1, 4, 9], dtype=torch.float32)

# 方法
print(t.sqrt())              # tensor([1., 2., 3.])

# ⚠️ 注意：开方结果默认是 float32
# 如果原张量是 int64，不能原地开方！
t2 = torch.tensor([1, 4, 9])     # int64
# t2.sqrt_()                    # ❌ 报错！int64 不能存浮点结果

# 需要先转类型
t2 = t2.float()
t2.sqrt_()                       # ✅ 可以了
```

### 1.5 指数运算

```python
t = torch.tensor([1, 2, 3], dtype=torch.float32)

# exp：e 的多少次方（e ≈ 2.718）
print(t.exp())               # tensor([2.7183, 7.3891, 20.0855])
# 验证：e^1 ≈ 2.718, e^2 ≈ 7.389, e^3 ≈ 20.086
```

### 1.6 对数运算

```python
t = torch.tensor([1, 2, 3], dtype=torch.float32)

# log：以 e 为底的自然对数
print(t.log())               # tensor([0.0000, 0.6931, 1.0986])
# 验证：ln(1)=0, ln(2)≈0.693, ln(3)≈1.099
```

### 基本运算速查

| 运算 | 运算符 | 方法（不原地） | 方法（原地） |
|------|-------|--------------|------------|
| 加法 | `+` | `tensor.add(x)` | `tensor.add_(x)` |
| 减法 | `-` | `tensor.sub(x)` | `tensor.sub_(x)` |
| 乘法 | `*` | `tensor.mul(x)` | `tensor.mul_(x)` |
| 除法 | `/` | `tensor.div(x)` | `tensor.div_(x)` |
| 取负 | `-tensor` | `tensor.neg()` | `tensor.neg_()` |
| 幂 | `**` | `tensor.pow(n)` | `tensor.pow_(n)` |
| 平方根 | — | `tensor.sqrt()` | `tensor.sqrt_()` |
| 指数 | — | `tensor.exp()` | `tensor.exp_()` |
| 自然对数 | — | `tensor.log()` | `tensor.log_()` |

---

## ✖️ 二、哈达玛积（Hadamard Product）

### 2.1 什么是哈达玛积？

> **两个形状相同的张量，对应位置元素相乘。** 也叫**元素级乘法**。

```python
# 两个形状相同的张量
t1 = torch.randint(0, 10, (2, 3))   # 2×3
t2 = torch.randint(0, 10, (2, 3))   # 2×3

print(t1)
# tensor([[3, 1, 4],
#         [7, 4, 8]])

print(t2)
# tensor([[6, 8, 5],
#         [1, 9, 2]])

# 哈达玛积：对应位置相乘
print(t1 * t2)
# tensor([[18,  8, 20],
#         [ 7, 36, 16]])

# 也可以用方法
print(t1.mul(t2))     # 一样
```

### 2.2 高维张量的哈达玛积

不仅限于 2 维，任意维度的张量都可以：

```python
# 3 维张量
t1 = torch.randint(0, 10, (3, 2, 4))   # 3×2×4
t2 = torch.randint(0, 10, (3, 2, 4))   # 3×2×4

result = t1 * t2                        # 对应位置相乘
print(result.shape)                     # torch.Size([3, 2, 4]) ← 形状不变
```

> **规则**：两个张量形状必须完全一样，或者满足**广播条件**（后面讲）。

### 2.3 原地修改

```python
t1.mul_(t2)   # 原地乘，t1 被直接修改
```

### 哈达玛积 vs 矩阵乘法

| | 哈达玛积（元素级乘） | 矩阵乘法 |
|---|-------------------|---------|
| **符号** | `*` | `@` |
| **方法** | `tensor.mul()` | `torch.matmul()` |
| **形状要求** | 形状相同即可 | 需满足矩阵乘法规则 |
| **结果形状** | 跟原形状一样 | 根据矩阵乘法规则变化 |

---

## 🔗 三、矩阵乘法

### 3.1 二维矩阵乘法

```python
# 3×2 矩阵 × 2×4 矩阵 = 3×4 矩阵
t1 = torch.tensor([[1, 2], [3, 4], [5, 6]])      # 3×2
t2 = torch.tensor([[7, 8, 9, 0], [2, 3, 5, 7]])  # 2×4

# 方式一：@ 运算符
result = t1 @ t2
print(result)
print(result.shape)   # torch.Size([3, 4])

# 方式二：torch.matmul()
result = torch.matmul(t1, t2)
print(result.shape)   # torch.Size([3, 4])

# 方式三：torch.mm()（仅限二维！）
result = torch.mm(t1, t2)
print(result.shape)   # torch.Size([3, 4])
```

### 3.2 高维张量的矩阵乘法

对于 3 维及以上的张量，可以理解为**矩阵的数组**：

```python
# 三维：2 个矩阵的数组
t1 = torch.randn(2, 3, 4)   # 2个 3×4 的矩阵
t2 = torch.randn(2, 4, 5)   # 2个 4×5 的矩阵

# 对应位置矩阵相乘
result = t1 @ t2
print(result.shape)         # torch.Size([2, 3, 5])
# 等价于：2个 (3×4 @ 4×5 = 3×5) 矩阵
```

> **高维矩阵乘法的原则：**
> 1. **最后两个维度**做矩阵乘法（满足行 × 列规则）
> 2. **前面所有维度**必须相同（或满足广播条件）
> 3. 结果 = 前面维度不变 + 矩阵乘法后的形状

**举例：4 维张量的矩阵乘法**

```python
t1 = torch.randn(4, 2, 3, 5)   # 4×2 个 3×5 的矩阵
t2 = torch.randn(4, 2, 5, 6)   # 4×2 个 5×6 的矩阵

result = t1 @ t2
print(result.shape)             # torch.Size([4, 2, 3, 6])
# 前面维度 (4,2) 相同
# 后面 (3×5) @ (5×6) = (3×6)
```

### 3.3 广播在矩阵乘法中的应用

如果前面维度有一方为 1，可以广播：

```python
t1 = torch.randn(4, 2, 3, 5)
t2 = torch.randn(1, 2, 5, 6)   # 第一个维度是 1！

result = t1 @ t2
print(result.shape)             # torch.Size([4, 2, 3, 6])
# t2 的维度 1 被广播成 4，匹配 t1
```

### 三种方法的区别

| 方法 | 适用维度 | 说明 |
|------|---------|------|
| `@` 运算符 | 任意维度 | 最简洁，推荐 ✅ |
| `torch.matmul()` | 任意维度 | 功能同 @ |
| `torch.mm()` | **仅限 2 维** | 专门的二维矩阵乘法 |

---

## 📝 四、本章总结

### 知识地图

```
张量数值计算
    │
    ├── ① 基本运算
    │      ├── 加减乘除（+ - * /）← 运算符 & 方法
    │      ├── 取负（neg）
    │      ├── 幂（pow）、平方根（sqrt）
    │      ├── 指数（exp）、对数（log）
    │      └── 原地修改（加下划线 _）
    │
    ├── ② 哈达玛积
    │      ├── 对应位置元素相乘
    │      ├── 符号：* 或 .mul()
    │      └── 形状必须相同（或可广播）
    │
    └── ③ 矩阵乘法
           ├── 二维：@ / matmul / mm
           ├── 高维：最后两维做矩阵乘法，前面维度相同
           ├── 广播：前面维度为 1 时可广播
           └── mm 只适用于二维
```

### 关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| **运算符 vs 方法** | `+` 等价于 `.add()`，`*` 等价于 `.mul()` |
| **原地操作 `_`** | 加下划线的方法直接改原数据（如 `tensor.add_(10)`） |
| **哈达玛积** | 对应位置元素相乘，形状不变 |
| **矩阵乘法** | `@` 或 `matmul()`，需满足行×列规则 |
| **高维矩阵乘** | 最后两维做矩阵乘法，前面维度要匹配 |
| **`torch.mm()`** | 只支持二维矩阵乘法 |
| **开方注意** | sqrt 返回 float 类型，int 张量不能原地开方 |

---

## 💾 四、节省内存

### 4.1 为什么要节省内存？

做张量运算时，有些操作会创建**新的内存空间**来存结果。如果一个张量之后不用了，完全可以把结果直接覆盖到它上面，避免开辟新内存：

```python
x = torch.randint(1, 10, (3, 2, 4))
y = torch.randint(1, 10, (3, 2, 4))

# ❌ 这种操作会创建新对象
x = x * y
print(id(x))  # ID 变了 → 新对象

# ✅ 原地操作，不创建新对象（元素级乘法可用）
x.mul_(y)
print(id(x))  # ID 不变 → 同一对象
```

### 4.2 元素级运算的原地操作

对应位置的元素运算（哈达玛积）有下划线版本：

```python
x.mul_(y)      # ✅ 原地乘，ID 不变
x.add_(y)      # ✅ 原地加
```

### 4.3 矩阵乘法的困境

**矩阵乘法没有 `matmul_()` 方法！** `@` 和 `matmul()` 都没有下划线版本：

```python
x = torch.randint(1, 10, (3, 2, 4))
y = torch.randint(1, 10, (3, 4, 5))

# ❌ 没有 matmul_ 方法
# x.matmul_(y)    # 报错！
```

### 4.4 解决方案一：直接赋值（不节省内存）

```python
x = x @ y
print(id(x))   # ID 变了 → 新对象
```

### 4.5 解决方案二：切片赋值（可以节省内存，有条件）

原理：**如果结果张量的形状 ≤ 原张量的形状（且最后一个维度为 1），就可以通过切片赋值填回去。**

```python
# 假设原形状 (3, 2, 4)，运算后得到 (3, 2, 1)
x = torch.randint(1, 10, (3, 2, 4))
y = torch.randint(1, 10, (3, 4, 1))   # y 的最后一个维度是 1

result = x @ y
print(result.shape)   # torch.Size([3, 2, 1])

# 切片赋值：把 (3,2,1) 广播填充到 (3,2,4)
x[:] = x @ y          # ← 切片赋值，不改变 ID！
print(x.shape)        # torch.Size([3, 2, 4])
print(id(x))          # ✅ ID 不变！
```

> **原理**：`x[:] = ...` 是切片赋值，它不会创建新对象，而是把结果通过**广播**填回原张量。但前提是结果张量的形状**必须能广播成原形状**（通常要求最后一维为 1）。

### 4.6 什么时候能用切片赋值节省内存？

```python
# ✅ 能节省内存：结果最后一维为 1，可以广播填回原形状
x = torch.randn(3, 2, 4)
y = torch.randn(3, 4, 1)
x[:] = x @ y      # OK, (3,2,1) 广播成 (3,2,4)

# ❌ 不能：结果形状跟原形状不匹配
x = torch.randn(3, 2, 4)
y = torch.randn(3, 4, 5)
# x[:] = x @ y    # 报错！(3,2,5) 无法填回 (3,2,4)
```

### 节省内存速查

| 方法 | 是否节省内存 | 适用场景 |
|------|------------|---------|
| `x.mul_(y)` | ✅ 原地 | 元素级乘法（哈达玛积） |
| `x = x @ y` | ❌ 新对象 | 矩阵乘法直接赋值 |
| `x[:] = x @ y` | ✅ 有条件 | 结果广播后能填回原形状（最后一维通常为 1） |

---

## 📊 五、统计函数（聚合运算）

### 5.1 什么是统计函数？

之前的数值计算是**一对一**（每个元素独立运算），统计函数是**多对一**（把一组数聚合成一个结果）：

```
数值计算：tensor([1,2,3]) + 1 = tensor([2,3,4])    ← 一对一
统计函数：tensor([1,2,3]).sum() = tensor(6)          ← 多对一
```

### 5.2 dim 参数的核心概念

**所有统计函数几乎都支持 `dim` 参数**，指定沿哪个维度做聚合。用一句话理解：

> **`dim=k` 就是把第 k 个维度消掉（合并）** 🎯

```python
# 形状：(3, 2, 4)
# dim=0 → 消掉 3，得到 (2, 4) — 矩阵对应位置元素求和
# dim=1 → 消掉 2，得到 (3, 4) — 每列求和（行合并）
# dim=2 → 消掉 4，得到 (3, 2) — 每行求和（列合并）
```

### 5.3 求和：`sum()`

```python
t = torch.randint(1, 10, (3, 2, 4))   # 3×2×4
print(t)

# 全局求和（所有元素加起来）
print(t.sum())       # tensor(92)  ← 标量

# 沿 dim=0 求和：消掉 3，得到 (2, 4)
print(t.sum(dim=0).shape)   # torch.Size([2, 4])

# 沿 dim=1 求和：消掉 2，得到 (3, 4)
print(t.sum(dim=1).shape)   # torch.Size([3, 4])

# 沿 dim=2 求和：消掉 4，得到 (3, 2)
print(t.sum(dim=2).shape)   # torch.Size([3, 2])
```

> 💡 **记忆技巧**：把形状写出来，`dim=k` 就是消掉第 k 个维度。

### 5.4 均值：`mean()`

```python
t = torch.randint(1, 10, (3, 2, 4)).float()   # ⚠️ mean 要求浮点数！

# 全局均值
print(t.mean())           # tensor(3.833)  ← 所有元素平均

# 沿 dim=1（列均值）
print(t.mean(dim=1).shape)   # torch.Size([3, 4])
```

> ⚠️ **`mean()` 要求输入必须是浮点类型**。如果是整数张量，需要先 `.float()`。

### 5.5 标准差：`std()` 和 方差：`var()`

```python
t = torch.randint(1, 10, (3, 2, 4)).float()

# 全局标准差 / 方差
print(t.std())     # 所有元素的标准差
print(t.var())     # 所有元素的方差

# 按维度
print(t.std(dim=2).shape)   # torch.Size([3, 2])  ← 每行算一个标准差
```

### 5.6 最大值/最小值：`max()` / `min()`

```python
t = torch.randint(1, 10, (3, 2, 4))

# 全局最大
print(t.max())     # tensor(9)

# 按维度（返回值和索引！）
result = t.max(dim=2)
print(result.values)    # 每行的最大值
print(result.indices)   # 最大值所在的索引号
```

> **注意**：`max(dim=k)` 返回的是一个命名元组，包含 `values` 和 `indices`。

### 5.7 最大/最小索引：`argmax()` / `argmin()`

只返回**最大值所在的索引**，不关心值本身。**这是分类任务中最常用的！**

```python
t = torch.randint(1, 10, (3, 2, 4))

# 全局最大索引
print(t.argmax())        # 展平后最大值的索引

# 按维度
print(t.argmax(dim=2))   # 每行最大值的列索引
```

> 💡 **典型应用**：多分类任务中，取概率最大的类别索引作为预测标签。

### 5.8 去重：`unique()`

把张量中重复的元素去掉，返回排序后的结果：

```python
t = torch.tensor([[1, 2, 3], [1, 2, 4]])
print(t.unique())        # tensor([1, 2, 3, 4])  ← 去重 + 排序
```

### 5.9 排序：`sort()`

返回排序后的值和对应的**原索引**：

```python
t = torch.randint(1, 10, (3, 4))
print(t)

# 默认按行升序（dim=-1，即最后一个维度）
result = t.sort()
print(result.values)     # 排序后的值
print(result.indices)    # 原位置的索引

# 降序
result = t.sort(descending=True)
print(result.values)     # 每行从大到小
```

> **`sort()` 返回的 `indices`** 可以用来还原原始顺序，或者在排序后追踪每个元素原来的位置。

### 统计函数速查

| 函数 | 作用 | 是否要求 float | 返回值 |
|------|------|--------------|--------|
| `t.sum()` | 求和 | ❌ | 标量或张量 |
| `t.mean()` | 均值 | ✅ **必须** | 标量或张量 |
| `t.std()` | 标准差 | ✅ **必须** | 标量或张量 |
| `t.var()` | 方差 | ✅ **必须** | 标量或张量 |
| `t.max()` | 最大值 | ❌ | 值或(values, indices) |
| `t.min()` | 最小值 | ❌ | 值或(values, indices) |
| `t.argmax()` | 最大索引 | ❌ | 索引张量 |
| `t.argmin()` | 最小索引 | ❌ | 索引张量 |
| `t.unique()` | 去重 | ❌ | 排序后的唯一值 |
| `t.sort()` | 排序 | ❌ | (values, indices) |

---

## 📝 六、本章总结

### 知识地图

```
张量数值计算 & 统计函数
    │
    ├── ① 基本运算
    │      ├── 加减乘除（+ - * /）
    │      ├── 取负、幂、平方根、指数、对数
    │      └── 原地操作（加 _ 后缀）
    │
    ├── ② 哈达玛积（* / mul）
    │
    ├── ③ 矩阵乘法（@ / matmul / mm）
    │      ├── 二维：行 × 列
    │      ├── 高维：最后两维乘法，前面维度相同
    │      └── 广播：维度为 1 时可广播
    │
    ├── ④ 节省内存
    │      ├── 元素级：mul_() 原地操作 ✅
    │      ├── 矩阵乘：无原地方法，用切片赋值 x[:] = ...
    │      └── 条件：结果必须能广播填回原形状
    │
    └── ⑤ 统计函数（聚合运算）
           ├── sum / mean / std / var
           ├── max / min / argmax / argmin
           ├── unique（去重）
           └── sort（排序）
```

### 关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| **`dim` 参数** | `dim=k` 就是消掉第 k 个维度（降维聚合） |
| **均值要求 float** | `mean()` / `std()` / `var()` 输入必须浮点 |
| **`max(dim=k)`** | 返回 (values, indices) 两个结果 |
| **`argmax()`** | 只取最大值的索引，分类任务超常用 |
| **`sort()`** | 返回 (values, indices)，可指定升序/降序 |
| **`unique()`** | 去重 + 排序 |
| **节省内存** | 元素级用 `_()`，矩阵乘用切片赋值 `x[:] = ...` |

---

---

# 🧠 第五部分：PyTorch 张量索引

---

## 📌 前置说明

张量创建好之后，经常需要从中**选取特定的元素、行、列或子区域**。PyTorch 的索引方式跟 NumPy / Python 基本一样，有 4 种方式：

```
张量索引
    ├── ① 简单索引（整数索引）
    ├── ② 范围索引（切片 slice）
    ├── ③ 列表索引（花式索引 fancy indexing）
    └── ④ 布尔索引（条件筛选）
```

---

## 🎯 一、简单索引（整数索引）

直接用整数下标访问元素，跟 Python 列表和 NumPy 完全一样：

```python
import torch

t = torch.randint(1, 10, (3, 2, 4))   # 3×2×4
print(t)
# tensor([[[8, 6, 8, 6],
#          [8, 8, 8, 9]],
#
#         [[9, 9, 8, 6],
#          [7, 6, 9, 5]],
#
#         [[3, 7, 5, 1],
#          [4, 6, 7, 1]]])

# 取第一个矩阵的第零行
print(t[0, 0])       # tensor([8, 6, 8, 6])

# 取第二个矩阵的第一行第三列
print(t[1, 1, 2])    # tensor(9)

# 取第一个矩阵
print(t[0])          # 形状 (2, 4)
```

> 记住：**索引从 0 开始**，跟 Python 完全一致。

---

## 📏 二、范围索引（切片）

跟 Python 的 `start:end:step` 完全一样：

```python
t = torch.randint(1, 10, (3, 5, 4))   # 3×5×4

# 取所有矩阵
print(t[:])            # 全部

# 取前两个矩阵（0 到 2，不包含 2）
print(t[:2])

# 取所有矩阵的第 1 到第 3 行（不包含 3）
print(t[:, 1:3])

# 取所有矩阵的所有行，每行取第 0 和第 2 列（步长 2）
print(t[:, :, ::2])

# 取第 0 个矩阵，第 2 行到最后一行
print(t[0, 2:])

# 反序（步长为 -1）
print(t[:, :, ::-1])   # 每行元素反序
```

> 💡 **切片返回的是视图（view），不是副本**，修改切片会影响原张量。

---

## 🔢 三、列表索引（花式索引）

用列表（或张量）指定要取的**坐标组合**。有两种用法：

### 3.1 用列表取多个行

```python
t = torch.randint(1, 10, (3, 5, 4))

# 取第 0 个矩阵的第 1 行和第 2 行
print(t[0, [1, 2]])   # 形状 (2, 4)

# 取所有矩阵的第 0 行和第 2 行
print(t[:, [0, 2]])   # 形状 (3, 2, 4)
```

### 3.2 坐标列表：一一对应取元素

这是列表索引最核心的用法——**多组坐标同时取**：

```python
t = torch.randint(1, 10, (3, 5, 4))

# 取坐标 (0, 1, 3) 和 (1, 2, 1) 两个元素
result = t[[0, 1], [1, 2], [3, 1]]
print(result)   # tensor([5, 7])  ← 两个元素
```

> 原理：三个列表对应三个维度，第 i 组坐标是 `(list1[i], list2[i], list3[i])`。

**直观理解：**
```python
# 列表索引：[矩阵号], [行号], [列号]
# 取的坐标就是：
#   (list1[0], list2[0], list3[0]) → 第 1 个元素
#   (list1[1], list2[1], list3[1]) → 第 2 个元素
t[[0, 1], [1, 2], [3, 1]]   # → (0,1,3) 和 (1,2,1) 两个元素
```

### 3.3 列表嵌套：一对多广播

```python
t = torch.randint(1, 10, (3, 5, 4))

# [[0, 1], [1, 2], [3, 1]]   → 一一对应，取 2 个元素
# [0, 1] 作为一个列表               → 一对多
# 相当于先取矩阵 0 的所有行，再取矩阵 1 的所有行
# 更复杂的组合需要多练习
```

---

## ✅ 四、布尔索引（条件筛选）

### 4.1 基本用法

先用条件表达式得到一个布尔张量（mask），再把它作为索引：

```python
t = torch.randint(1, 10, (3, 5, 4))

# 做条件判断 → 得到布尔张量
mask = t > 5
print(mask)       # 全是 True / False

# 布尔索引：提取所有 > 5 的元素
print(t[mask])    # tensor([8, 6, 8, 6, ...])  ← 一维列表
```

> 筛出来的结果会**展平成一维**，因为无法保持原形状（不知道哪些位置是 True）。

### 4.2 筛选符合条件的矩阵

```python
t = torch.randint(1, 10, (3, 5, 4))

# 矩阵的第 (1, 2) 个元素 > 5 的矩阵
mask = t[:, 1, 2] > 5     # 形状 (3,)
print(mask)                # tensor([True, False, True])

# 用这个 mask 选矩阵
print(t[mask])             # 形状 (2, 5, 4) ← 第 0 和第 2 个矩阵
```

### 4.3 筛选符合条件的行

```python
t = torch.randint(1, 10, (3, 5, 4))

# 每行的首元素 > 5 的行
mask = t[:, :, 0] > 5      # 形状 (3, 5)
print(mask)

# 应用 mask 选出对应行
print(t[mask])             # 所有符合条件的行
```

### 4.4 筛选符合条件的列（需要转置）

⚠️ 布尔索引只能按**前面的维度**做筛选。要筛选列，需要先转置：

```python
t = torch.randint(1, 10, (3, 5, 4))

# 选出第二个元素 > 5 的列
# 先把矩阵的最后两维转置（行列互换）
t2 = t.mT                  # 形状 (3, 4, 5) ← 最后两维调换

# 再按行筛选（原来的列变成行了）
mask = t2[:, :, 1] > 5     # 形状 (3, 4)
result = t2[mask]          # 选出符合条件的"行"

# 如果想恢复成列的形式，再转置回来
result = result.mT         # 或 .T（二维）
```

> **`t.mT`** 只转置最后两个维度（矩阵转置），适合高维张量。
> **`t.T`** 会翻转所有维度，通常不适用于高维张量。

### 布尔索引速查

| 用法 | 代码 | 说明 |
|------|------|------|
| 所有大于某值的元素 | `t[t > 5]` | 结果展平成一维 |
| 选符合条件的矩阵 | `t[t[:,1,2] > 5]` | mask 长度 = 矩阵个数 |
| 选符合条件的行 | `t[t[:,:,0] > 5]` | mask 形状 = (矩阵数, 行数) |
| 选符合条件的列 | `t.mT[t.mT[:,:,1] > 5]` | 先转置再选，再转回来 |

---

## 📝 五、本章总结

### 知识地图

```
张量索引
    │
    ├── ① 简单索引（整数）
    │      t[0], t[0, 1], t[0, 1, 2]
    │
    ├── ② 范围索引（切片）
    │      t[:2], t[:, 1:3], t[:, :, ::2], t[::-1]
    │
    ├── ③ 列表索引（花式）
    │      ├── 取多行：t[0, [1, 2]]
    │      ├── 坐标列表一一对应：t[[0,1], [1,2], [3,1]]
    │      └── 列表嵌套：一对多广播
    │
    └── ④ 布尔索引（条件）
           ├── 基本：t[t > 5] → 展平
           ├── 选矩阵：t[t[:,1,2] > 5]
           ├── 选行：t[t[:,:,0] > 5]
           └── 选列：t.mT[mask].mT（先转置）
```

### 关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| **简单索引** | 用整数下标，跟 Python 一样 |
| **范围索引** | `start:end:step` 切片，返回视图（view） |
| **列表索引** | 用列表指定坐标，一一对应或多个行 |
| **布尔索引** | 条件筛选，筛出符合条件的元素 |
| **`t.mT`** | 只转置最后两个维度（矩阵转置） |
| **`t.T`** | 翻转所有维度（高维不推荐） |

---

---

---

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
X = torch.tensor(10.0)        # 输入（标量）
Y = torch.tensor(3.0)         # 真实标签

# 输入数据不需要梯度 ❌
print(f"X.requires_grad: {X.requires_grad}")   # False

# ========== 2. 初始化参数 ==========
W = torch.randn(1, 1, requires_grad=True)      # 权重 ← 需要梯度 ✅
B = torch.randn(1, 1, requires_grad=True)      # 偏置 ← 需要梯度 ✅

print(f"W.requires_grad: {W.requires_grad}")   # True
print(f"B.requires_grad: {B.requires_grad}")   # True

# ========== 3. 前向传播 ==========
Z = W * X + B                 # 线性模型：ŷ = Wx + b
print(f"Z: {Z}")
print(f"Z.requires_grad: {Z.requires_grad}")   # True ← 自动开启
print(f"Z.grad_fn: {Z.grad_fn}")                # <AddBackward0>

# ========== 4. 计算损失 ==========
loss_fn = nn.MSELoss()        # 均方误差损失函数
loss = loss_fn(Z, Y)          # 计算损失值
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

# 1. 数据
X = torch.tensor(10.0)
Y = torch.tensor(3.0)

# 2. 参数（开启梯度）
W = torch.randn(1, 1, requires_grad=True)
B = torch.randn(1, 1, requires_grad=True)

# 3. 前向传播
Z = W * X + B

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

**所以每次迭代前需要清零：**

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

# ① 定义数据
X = torch.tensor(10.0)
Y = torch.tensor(3.0)

# ② 初始化参数（开启梯度）
W = torch.randn(1, 1, requires_grad=True)
B = torch.randn(1, 1, requires_grad=True)

# ③ 前向传播
Z = W * X + B
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

# ① 定义数据
X = torch.tensor(10.0)
Y = torch.tensor(3.0)

# ② 初始化参数（开启梯度）
W = torch.randn(1, 1, requires_grad=True)
B = torch.randn(1, 1, requires_grad=True)

# ③ 前向传播
Z = W * X + B
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

# 🧠 第八部分：线性回归实战案例

> 📺 视频来源：PyTorch 线性回归案例 · 完整流程
> 🎯 核心目标：用神经网络实现线性回归，跑通训练全流程
> 📝 风格：完整代码 + 逐行讲解

---

## 📌 一、整体思路与流程

### 从理论到实战

前面我们学完了 PyTorch 的基础操作和自动微分，现在来做第一个**完整的实战项目**——**线性回归**。

> 💡 线性回归本质上就是 `Y = WX + B`，而神经网络的一个全连接层（Linear Layer）做的也是 `Y = XWᵀ + B`。所以用一个**单层神经网络**就能实现线性回归！

### 完整训练流程

```
① 准备数据 ──→ ② 构建模型 ──→ ③ 定义损失函数和优化器
                                              ↓
              ⑥ 迭代训练 ←──────────────────── ④ 前向传播计算输出
                  ↓                                    ↓
              ⑦ 调参优化                            ⑤ 计算损失
                  ↓                                    ↓
              ⑧ 画图看效果                          ⭐ 反向传播
                                                       ↓
                                                   更新参数
```

### 核心五步（每一轮每批数据都要做）

| 步骤 | 代码 | 说明 |
|------|------|------|
| ① 前向传播 | `y_pred = model(x)` | 算预测值 |
| ② 计算损失 | `loss = loss_fn(y_pred, y)` | 算预测和真实差距 |
| ③ 反向传播 | `loss.backward()` | 算所有参数的梯度 |
| ④ 更新参数 | `optimizer.step()` | 用梯度更新 W 和 B |
| ⑤ 梯度清零 | `optimizer.zero_grad()` | 清掉旧梯度，准备下一轮 |

---

## 📌 二、准备数据

### 2.1 生成随机样本数据

我们不知道真实的线性数据长什么样，那就**自己编**一个：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ─── ① 生成随机样本点 ───
# 100个随机点，每个点1个特征
X = torch.randn(100, 1)

# 预设真实参数（这是"本质规律"）
w_real = torch.tensor([2.5])   # 斜率
b_real = torch.tensor([5.2])   # 截距

# 随机噪声（让点不在直线上，更真实）
noise = torch.randn(100, 1) * 0.1

# 真实目标值 Y = WX + B + noise
Y = X @ w_real + b_real + noise
```

> 🎯 **理解**：我们假装 `Y = 2.5X + 5.2` 是真实规律，然后加了一点随机抖动。模型训练的目的就是**从数据中反推**这个 W 和 B。

### 2.2 构建 Dataset 和 DataLoader

```python
# ─── ② 构建数据集 ───
dataset = TensorDataset(X, Y)   # 把 X,Y 打包成数据集

# ─── ③ 构建数据加载器 ───
dataloader = DataLoader(
    dataset=dataset,
    batch_size=10,    # 每批10个样本
    shuffle=True      # 每轮都打乱数据
)
```

| 概念 | 说明 |
|------|------|
| **`TensorDataset`** | 把 X 和 Y 打包成 (特征, 标签) 对 |
| **`DataLoader`** | 自动按 batch_size 切分、打乱、迭代 |
| **`shuffle=True`** | 每个 epoch 开始前自动洗牌，引入随机性 |

---

## 📌 三、定义模型、损失函数与优化器

### 3.1 模型：一个线性层（全连接层）

```python
# ─── ④ 定义模型 ───
model = nn.Linear(in_features=1, out_features=1)
```

> 💡 `nn.Linear(1, 1)` 表示输入1个特征，输出1个值。内部自动创建了权重 `W`（1×1）和偏置 `B`（1），这正是 `Y = WX + B`。

### 3.2 损失函数：均方误差 MSE

```python
# ─── ⑤ 定义损失函数 ───
loss_fn = nn.MSELoss()   # 回归问题的标配
```

### 3.3 优化器：随机梯度下降 SGD

```python
# ─── ⑥ 定义优化器 ───
optimizer = optim.SGD(model.parameters(), lr=0.001)
```

| 组件 | 代码 | 作用 |
|------|------|------|
| 模型 | `nn.Linear(1, 1)` | `Y = WX + B` 的结构 |
| 损失函数 | `nn.MSELoss()` | 衡量预测值和真实值的差距 |
| 优化器 | `optim.SGD(...)` | 如何用梯度更新参数（学习率控制步长） |

---

## 📌 四、模型训练（五步核心流程）

### 双重循环结构

```python
# ─── ⑦ 模型训练 ───
epochs = 100

for epoch in range(epochs):           # 外层：遍历轮次
    for x_batch, y_batch in dataloader:  # 内层：遍历每批数据

        # ① 前向传播（计算预测值）
        y_pred = model(x_batch)

        # ② 计算损失
        loss = loss_fn(y_pred, y_batch)

        # ③ 反向传播（计算梯度）⭐
        loss.backward()

        # ④ 更新参数（梯度下降）
        optimizer.step()

        # ⑤ 梯度清零（清掉本轮梯度，准备下批）⭐
        optimizer.zero_grad()

# 查看训练后的参数
print(f"斜率 W: {model.weight.item():.4f}")
print(f"截距 B: {model.bias.item():.4f}")
```

### 五步流程详解

| 步骤 | 代码 | 通俗理解 | 类比 |
|------|------|---------|------|
| ① 前向传播 | `y_pred = model(x)` | 用当前参数算预测值 | 猜测一下答案 |
| ② 计算损失 | `loss = loss_fn(y_pred, y)` | 预测值和真实值差多少 | 看看错了多少 |
| ③ 反向传播 | `loss.backward()` | 自动算每个参数的梯度 | 找改进方向 |
| ④ 更新参数 | `optimizer.step()` | 按梯度方向更新 W 和 B | 往前走一步 |
| ⑤ 梯度清零 | `optimizer.zero_grad()` | 清空旧梯度，用新的 | 清空小黑板 |

> ⚠️ **梯度清零的位置**：可以在反向传播**前**或**后**，但**绝不能**插在 ③ 和 ④ 之间（否则刚算的梯度就被清了，更新了个寂寞）。

### 首次训练效果

训练 100 轮，学习率 0.001 后的结果：
```
斜率 W: 1.86    ← 目标 2.5，还差不少
截距 B: 4.43    ← 目标 5.2，也差不少
```

**为什么不准？** 学习率太小 + 轮数不够。下面我们画图看看问题出在哪。

---

## 📌 五、记录损失与画图

### 5.1 在训练中记录损失

```python
loss_history = []      # 记录每轮的平均损失

for epoch in range(epochs):
    total_loss = 0.0
    num_batches = 0

    for x_batch, y_batch in dataloader:
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        total_loss += loss.item()   # 累加损失
        num_batches += 1

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 记录本轮平均损失
    avg_loss = total_loss / num_batches
    loss_history.append(avg_loss)
```

### 5.2 画两个子图

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：损失下降曲线
axes[0].plot(loss_history)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")

# 右图：原始散点 + 拟合直线
axes[1].scatter(X, Y)        # 原始数据点
# 用训练好的模型画直线
x_plot = torch.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot = model(x_plot).detach()
axes[1].plot(x_plot, y_plot, color='red', linewidth=2)
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")

plt.show()
```

> 📊 左图看收敛：loss 不断下降说明模型在学习
> 📊 右图看拟合：红色直线穿过散点的中心说明拟合成功

---

## 📌 六、调参优化与效果对比

### 参数调整方案

| 问题 | 解决方案 | 效果 |
|------|---------|------|
| 训练不足 | 增大 `epochs`（100 → 1000） | 更多训练，更好拟合 |
| 学习率太小 | 增大 `lr`（0.001 → 0.01） | 更快收敛 |
| 数据噪声大 | 调整 `noise` 系数 | 更真实或更清晰 |

### 最佳效果示例

```python
# 调参后
model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 1000
```

训练结果：
```
斜率 W: 2.5391    ← 接近真实 2.5 ✅
截距 B: 5.1638    ← 接近真实 5.2 ✅
```

> ⚠️ 注意：因为有随机噪声，不可能完全等于 2.5 和 5.2。噪声越大，偏差越大，这是理论上的极限。

### 损失下降曲线解读

```
loss
│
│   ╲
│    ╲
│     ╲_____  ← 约 100-200 轮时就已接近最小值
│           ╲
│            ╲____  ← 后面几乎水平
│
└───────────────────→ epoch
```

学习率 0.01 时大约 100~200 轮就收敛了，后面几乎是一条平线。这说明模型已经学得差不多了。

---

## 📝 本章总结 + 完整代码

### 🌳 知识树

```
PyTorch 线性回归实战
│
├── ① 准备数据
│   ├── 生成样本：X = randn, Y = WX + B + noise
│   ├── TensorDataset：打包 X,Y
│   └── DataLoader：自动分批 + 打乱
│
├── ② 构建模型
│   ├── nn.Linear(1, 1) → Y = WX + B
│   └── 一个全连接层就能实现线性回归
│
├── ③ 定义损失和优化器
│   ├── nn.MSELoss() → 回归损失
│   └── optim.SGD(model.parameters(), lr) → 梯度下降
│
├── ④ 训练五步（⭐ 核心）
│   ├── y_pred = model(x)     前向传播
│   ├── loss = loss_fn(...)   计算损失
│   ├── loss.backward()       反向传播 → 算梯度
│   ├── optimizer.step()      更新参数
│   └── optimizer.zero_grad() 梯度清零
│
└── ⑤ 评估与可视化
    ├── 记录 loss_history → 画损失曲线
    ├── 画散点 + 拟合直线
    └── 对比 W, B 与真实值
```

### 🔥 完整可运行代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ============ 1. 准备数据 ============
X = torch.randn(100, 1)
w_real, b_real = torch.tensor([2.5]), torch.tensor([5.2])
noise = torch.randn(100, 1) * 0.1
Y = X @ w_real + b_real + noise

dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# ============ 2. 构建模型 ============
model = nn.Linear(1, 1)

# ============ 3. 定义损失和优化器 ============
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ============ 4. 训练 ============
epochs = 1000
loss_history = []

for epoch in range(epochs):
    total_loss, num_batches = 0.0, 0

    for x_batch, y_batch in dataloader:
        # 前向传播
        y_pred = model(x_batch)
        # 计算损失
        loss = loss_fn(y_pred, y_batch)
        total_loss += loss.item()
        num_batches += 1
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()

    avg_loss = total_loss / num_batches
    loss_history.append(avg_loss)

# ============ 5. 评估 ============
print(f"训练结果：W = {model.weight.item():.4f}, B = {model.bias.item():.4f}")
print(f"真实值：  W = {w_real.item()}, B = {b_real.item()}")

# 画图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(loss_history)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Loss 下降曲线")

axes[1].scatter(X, Y, alpha=0.6)
x_plot = torch.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot = model(x_plot).detach()
axes[1].plot(x_plot, y_plot, 'r-', linewidth=2)
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")
axes[1].set_title("拟合效果")

plt.tight_layout()
plt.show()
```

### 关键收获

| 新概念 | 一句话解释 |
|--------|-----------|
| **`nn.Linear(in, out)`** | 全连接层，实现 `Y = XWᵀ + B` |
| **`nn.MSELoss()`** | 均方误差损失，回归任务标配 |
| **`optim.SGD(...)`** | 随机梯度下降优化器 |
| **`optimizer.step()`** | 用梯度更新所有参数 |
| **`optimizer.zero_grad()`** | 清空所有参数的梯度（防累积） |
| **`TensorDataset`** | 把 X, Y 打包成数据集 |
| **`DataLoader`** | 自动分批、打乱、迭代 |
| **`model.parameters()`** | 获取模型所有可训练参数 |
| **`model.weight` / `model.bias`** | 直接访问线性层的 W 和 B |

> 🎯 现在你已经跑通了 PyTorch 训练的第一个完整项目！虽然很简单，但**训练流程在任何复杂的神经网络中都是一模一样的**：前向传播 → 算损失 → 反向传播 → 更新参数 → 梯度清零。

---

---

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

# 🧠 第十部分：全连接层与参数初始化

> 📺 视频来源：PyTorch 深度学习 · 全连接层详解 + 参数初始化
> 🎯 核心目标：掌握 `nn.Linear` 的用法和多种参数初始化方法
> 📝 风格：代码实操 + 理论对比

---

## 📌 一、全连接层（nn.Linear）详解

### 全连接层 = 仿射层 = 线性层

在 PyTorch 中，全连接层通过 `nn.Linear` 实现，本质上就是之前我们手写的**仿射层（Affine）**：

```
Y = X × Wᵀ + B
```

| 概念 | 手写版本 | PyTorch 版本 |
|------|---------|-------------|
| 层定义 | `class Affine:` | `nn.Linear(in, out)` |
| 权重 | `self.W` | `layer.weight` |
| 偏置 | `self.B` | `layer.bias` |
| 前向传播 | `Y = X @ W + B` | `layer(X)` |

### 基本用法

```python
import torch.nn as nn

# 创建一个全连接层：输入5个特征，输出2个特征
linear = nn.Linear(in_features=5, out_features=2)

# 参数
print(linear.weight)  # 权重，形状 [2, 5]
print(linear.bias)    # 偏置，形状 [2]
print(linear.weight.requires_grad)  # True ← 默认开启梯度追踪！
```

### 可选参数

```python
nn.Linear(
    in_features,      # 输入神经元个数
    out_features,     # 输出神经元个数
    bias=True,        # 是否使用偏置（False = 过原点）
    device=None,      # CPU 或 GPU
    dtype=None        # 数据类型，默认 float32
)
```

---

## 📌 二、weight 和 bias 的形状与含义

### 形状规则

| 参数 | 形状 | 原因 |
|------|------|------|
| **`weight`** | `[out_features, in_features]` | **保存的是 W 的转置**，方便计算 |
| **`bias`** | `[out_features]` | 每个输出节点有一个偏置 |

> ⚠️ **注意**：`weight` 的形状是 `[out, in]`，不是 `[in, out]`！PyTorch 存的是转置。

### 为什么存转置？

我们手写的时候 `Y = X @ W + B`，其中 `W` 的形状是 `[in, out]`。

而 PyTorch 存的是 `Wᵀ`（形状 `[out, in]`），因为实际计算时：
```
Y = X @ Wᵀ + B   （等价于 X @ W + B）
```

这样做是为了**计算效率更高**——在反向传播时，梯度计算天然就需要 W 的转置。

```python
linear = nn.Linear(5, 2)
print(linear.weight.shape)  # torch.Size([2, 5])  ← [out, in]
print(linear.bias.shape)    # torch.Size([2])       ← [out]
```

### 默认参数值

创建 `nn.Linear` 时，参数默认已经有一个**随机的初始值**（不是 0），并且 `requires_grad=True`。

> 至于默认初始化的具体方案，我们会在第六节详细展开。

---

## 📌 三、常数初始化

有时我们需要手动控制参数的初始值，PyTorch 在 `torch.nn.init` 模块中提供了全套初始化方法。

> ⚠️ **注意**：所有初始化方法都带**下划线**（`_`），表示**原地修改**传入的张量。

### 3.1 全零初始化 `zeros_()`

```python
import torch.nn as nn

linear = nn.Linear(5, 2)

# 把 bias 全部初始化为 0
nn.init.zeros_(linear.bias)
print(linear.bias)  # tensor([0., 0.])
```

### 3.2 全一初始化 `ones_()`

```python
nn.init.ones_(linear.bias)
print(linear.bias)  # tensor([1., 1.])
```

### 3.3 指定常数 `constant_()`

```python
# 把 bias 全部初始化为 10
nn.init.constant_(linear.bias, val=10.0)

# 把 weight 全部初始化为 3
nn.init.constant_(linear.weight, val=3.0)
```

### 3.4 单位矩阵初始化 `eye_()`

```python
# 只对 weight 有效（需要是矩阵）
nn.init.eye_(linear.weight)
# 非方阵时，取最大主对角线
```

> ⚠️ **注意**：常数初始化（尤其是全零）**不适合权重 W**。如果所有权重初始化为相同值，反向传播时更新量也相同，所有神经元学习到同样的特征——这就是**对称性问题**。

---

## 📌 四、随机初始化

### 4.1 正态分布 `normal_()`

```python
# 标准正态分布：均值 0，标准差 1
nn.init.normal_(linear.weight, mean=0.0, std=1.0)

# 自定义参数
nn.init.normal_(linear.weight, mean=5.0, std=1.0)
```

### 4.2 均匀分布 `uniform_()`

```python
# 均匀分布范围 [a, b]
nn.init.uniform_(linear.weight, a=0.0, b=10.0)
```

---

## 📌 五、工程初始化（Xavier / Kaiming）

在实际工程中，我们不会拍脑袋给均值和方差——而是根据**激活函数的类型**选择对应的初始化策略。

### 5.1 Xavier 初始化（适用 Sigmoid / Tanh）

> 也叫 **Glorot 初始化**，目标是让每一层的**输入方差和输出方差相近**，缓解梯度消失/爆炸。

```python
# Xavier 正态分布（默认：mean=0, std=√(2/(n_in + n_out))）
nn.init.xavier_normal_(linear.weight)

# Xavier 均匀分布（范围：±√(6/(n_in + n_out))）
nn.init.xavier_uniform_(linear.weight)
```

**适用场景**：激活函数为 Sigmoid、Tanh 等 S 型曲线（浅层网络）

### 5.2 Kaiming 初始化（适用 ReLU）

> 也叫 **He 初始化**，专门为 ReLU 及其变体设计，方差是 Xavier 的**两倍**。

```python
# Kaiming 正态分布（默认：mean=0, std=√(2/n_in)）
nn.init.kaiming_normal_(linear.weight)

# Kaiming 均匀分布（范围：±√(6/n_in)）
nn.init.kaiming_uniform_(linear.weight)
```

**适用场景**：隐藏层使用 ReLU 或其变体（深度网络的首选）

### 两种初始化的对比

| 对比维度 | Xavier（Glorot） | Kaiming（He） |
|---------|:--------------:|:------------:|
| 适用激活函数 | Sigmoid, Tanh | **ReLU** 及其变体 |
| 方差公式（正态） | 2/(n_in + n_out) | **2/n_in** |
| 范围公式（均匀） | ±√(6/(n_in + n_out)) | **±√(6/n_in)** |
| 特点 | 考虑输入和输出 | 只考虑输入，方差更大 |

### 可视化对比

```
Xavier 正态：数据集中在 0 附近，范围较小
    ╱╲
  ╱    ╲
╱        ╲
───┴───→

Kaiming 正态：数据分布更广，方差更大
    ╱╲
  ╱    ╲
╱        ╲
────┴────→
```

---

## 📌 六、默认初始化机制

如果不手动初始化，PyTorch 的 `nn.Linear` 底层默认使用了什么方案？

```python
# 创建时自动调用了 reset_parameters() 方法
linear = nn.Linear(5, 2)
```

查看 PyTorch 源码，`reset_parameters()` 做了两件事：

| 参数 | 默认初始化方案 | 公式 |
|------|:------------:|------|
| **`weight`** | **Kaiming 均匀分布** | `U(-√(1/n_in), +√(1/n_in))` |
| **`bias`** | **均匀分布** | `U(-√(1/n_in), +√(1/n_in))` |

```python
# 等价于以下代码
nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5))  # a 是 ReLU 的负半轴斜率参数
fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
bound = 1 / math.sqrt(fan_in)
nn.init.uniform_(linear.bias, -bound, bound)
```

> 🎯 **结论**：默认初始化已经考虑了 ReLU 的特性（Kaiming），对于大多数情况直接用就行。只有当你需要特定的激活函数+初始化组合时，才需要手动设置。

---

## 📝 本章总结 + 速查表

### 🌳 知识树

```
全连接层与参数初始化
│
├── ① 全连接层 nn.Linear
│   ├── Y = X × Wᵀ + B
│   ├── weight: [out, in]（存转置）
│   └── bias: [out]
│
├── ② 常数初始化（不常用）
│   ├── zeros_() → 全 0（不适合 W）
│   ├── ones_() → 全 1
│   ├── constant_() → 指定常数
│   └── eye_() → 单位矩阵
│
├── ③ 随机初始化
│   ├── normal_() → 正态分布
│   └── uniform_() → 均匀分布
│
├── ④ Xavier（Glorot）→ Sigmoid / Tanh
│   ├── xavier_normal_()
│   └── xavier_uniform_()
│
├── ⑤ Kaiming（He）→ ReLU ⭐
│   ├── kaiming_normal_()
│   └── kaiming_uniform_()
│
└── ⑥ 默认方案
    ├── weight: Kaiming 均匀分布
    └── bias: 均匀分布
```

### 🚀 初始化方法速查表

| 初始化方法 | 代码 | 适用场景 | 特点 |
|-----------|------|---------|------|
| **全零** | `nn.init.zeros_()` | bias（很少用于 weight） | 导致对称性问题 ❌ |
| **全一** | `nn.init.ones_()` | 极少用 | 同对称问题 ❌ |
| **常数** | `nn.init.constant_(t, val)` | 特殊需求 | — |
| **正常随机** | `nn.init.normal_(t, mean, std)` | 参数手动调 | 无理论保障 |
| **均匀随机** | `nn.init.uniform_(t, a, b)` | 参数手动调 | 无理论保障 |
| **Xavier 正态** | `nn.init.xavier_normal_()` | **Sigmoid / Tanh** | 方差 = 2/(in+out) |
| **Xavier 均匀** | `nn.init.xavier_uniform_()` | **Sigmoid / Tanh** | 范围 = ±√(6/(in+out)) |
| **Kaiming 正态** ⭐ | `nn.init.kaiming_normal_()` | **ReLU 及其变体** | 方差 = 2/in |
| **Kaiming 均匀** ⭐ | `nn.init.kaiming_uniform_()` | **ReLU 及其变体**（默认方案） | 范围 = ±√(6/in) |

### ⭐ 经验法则

> **全连接层参数初始化 = 根据激活函数选方案：**
> - Hidden layer with **ReLU** → `kaiming_uniform_()`（**或直接用默认，默认就是这个**）
> - Hidden layer with **Sigmoid/Tanh** → `xavier_uniform_()`
> - **Bias** 一般初始化为 0

---

---

# 🧠 第十一部分：自定义神经网络模型

> 📺 视频来源：PyTorch 深度学习 · 自定义模型 · Dropout · 参数查看
> 🎯 核心目标：学会用 nn.Module 搭建神经网络，掌握 Dropout 和各种参数查看方法
> 📝 风格：从搭积木到完整实践

---

## 📌 一、nn.Module 基类（所有网络的基石）

### 一切皆模块

在 PyTorch 中，**所有层、所有模块、整个神经网络**都继承自 `nn.Module`。

```
                nn.Module（基类）
              ┌───────┼───────┐
              │       │       │
         nn.Linear  nn.Dropout  nn.BatchNorm1d
              │       │
              └───┬───┘
                  │
         自定义模型（也是 Module）
```

**核心思想**：搭积木
- 一个 `nn.Linear` 是一个 Module
- 多个 Module 组合起来，还是一个 Module
- 整个神经网络还是一个 Module

### 自定义模型必须实现两个方法

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        """初始化：定义各层结构"""
        super().__init__()           # 调用父类构造
        self.layer1 = nn.Linear(3, 4)  # 定义层
        self.layer2 = nn.Linear(4, 4)

    def forward(self, x):
        """前向传播：定义数据流动路径"""
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x
```

### Module 的关键特性

| 特性 | 说明 |
|------|------|
| **`__init__`** | 定义各层结构、初始化参数 |
| **`forward`** | 定义前向传播逻辑（数据如何流动） |
| **`model(x)`** | 把模型对象当函数调用 → 自动调 `forward` |
| **`model.parameters()`** | 自动管理**所有层的所有参数**（可迭代） |
| **`model.state_dict()`** | 返回所有参数的状态字典 |
| `model.train()` | 切换到训练模式（Dropout 生效） |
| `model.eval()` | 切换到评估模式（Dropout 关闭） |

> 💡 `model(x)` 等价于 `model.forward(x)`，但前者还会处理一些额外的逻辑（如注册 hooks）。所以**永远用 `model(x)` 而不是手动调 `forward`**。

---

## 📌 二、搭建三层神经网络

### 网络结构图

```
输入层（3个神经元）
  │
  ├── W1: 3×4 矩阵
  ├── B1: 长度4向量
  ↓
第一个隐藏层（4个神经元）
  │ 激活函数：Tanh
  │
  ├── W2: 4×4 矩阵
  ├── B2: 长度4向量
  ↓
第二个隐藏层（4个神经元）
  │ 激活函数：ReLU
  │
  ├── W3: 4×2 矩阵
  ├── B3: 长度2向量
  ↓
输出层（2个神经元）
  │ 激活函数：Softmax（dim=1）
  ↓
二分类概率 [p₁, p₂]，和 = 1
```

### 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.init as init

class NNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 定义三个全连接层
        self.linear1 = nn.Linear(3, 4)   # 3→4
        self.linear2 = nn.Linear(4, 4)   # 4→4
        self.out = nn.Linear(4, 2)       # 4→2

        # 参数初始化（下一节详细讲）
        init.xavier_normal_(self.linear1.weight)   # Tanh → Xavier
        init.kaiming_normal_(self.linear2.weight)  # ReLU → Kaiming

    def forward(self, x):
        # 第一层：线性 → Tanh
        x = self.linear1(x)
        x = torch.tanh(x)

        # 第二层：线性 → ReLU
        x = self.linear2(x)
        x = torch.relu(x)

        # 输出层：线性 → Softmax
        x = self.out(x)
        y = torch.softmax(x, dim=1)  # 对每个样本的2个类别算概率

        return y
```

### 测试前向传播

```python
# 1. 生成数据：10条数据，每条3个特征
X = torch.randn(10, 3)

# 2. 创建模型
model = NNModel()

# 3. 前向传播
y_pred = model(X)   # 等价于 model.forward(X)

print(y_pred.shape)  # torch.Size([10, 2])
print(y_pred)
# tensor([[0.57, 0.43],
#         [0.62, 0.38],
#         ...])   ← 每行和为1
```

---

## 📌 三、参数初始化实践

结合上一部分的知识，在自定义模型中进行初始化：

| 层 | 激活函数 | 推荐初始化 | 代码 |
|----|---------|-----------|------|
| 第一层 | **Tanh** | **Xavier 正态** | `init.xavier_normal_(self.linear1.weight)` |
| 第二层 | **ReLU** | **Kaiming 正态** | `init.kaiming_normal_(self.linear2.weight)` |
| 输出层 | Softmax | **默认（Kaiming 均匀）** | 不用手动设置 |

```python
# 等价于上面 __init__ 中的初始化
init.xavier_normal_(self.linear1.weight)
init.kaiming_normal_(self.linear2.weight)
# self.out 使用默认初始化（PyTorch 自动处理）
# bias 默认随基初始化，一般不需要手动改
```

> ⚠️ **注意**：`linear.weight` 的形状是 `[out, in]`（转置存储），但初始化公式中的 `n_in` 和 `n_out` PyTorch 内部会自动识别。

---

## 📌 四、Dropout 正则化

### 4.1 原理

Dropout（随机失活）在进行前向传播时，**以概率 p 随机关闭一些神经元**（输出置为 0）。

```
关闭前：[a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈, a₉, a₁₀]
                      ↓ p=0.5 关闭一半
关闭后：[0, a₂, 0, 0, a₅, a₆, 0, a₈, 0, a₁₀] × 2倍缩放
```

**缩放机制**：为了让输出的数学期望不变，剩下的神经元要乘以 **1/(1-p)**。
- p=0.5 → 剩下的一半 ×2
- p=0.2 → 剩下的 ×1.25

> 💡 训练时 Dropout 生效，**评估时 Dropout 不生效**（`model.eval()` 自动关闭）。

### 4.2 代码演示

```python
import torch
import torch.nn as nn

# 1. 定义数据：10个神经元
X = torch.randint(1, 10, (1, 10), dtype=torch.float32)
print(X)
# tensor([[5., 6., 5., 8., 2., 5., 9., 7., 3., 9.]])

# 2. 创建 Dropout 层（p=0.5，即关闭一半）
dropout = nn.Dropout(p=0.5)

# 3. 前向传播
Y = dropout(X)
print(Y)
# tensor([[ 0., 12.,  0.,  0.,  4.,  0., 18.,  0.,  6.,  0.]])
# 关掉了一半 → 剩下的翻倍（×2）
```

### 4.3 在神经网络中使用

```python
class NNModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 4)
        self.out = nn.Linear(4, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = self.dropout(x)       # 第一个隐藏层后接 Dropout
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)       # 第二个隐藏层后也接 Dropout
        y = torch.softmax(self.out(x), dim=1)
        return y
```

### Dropout 参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| **`p`** | 关闭神经元的概率 | 0.2 ~ 0.5（越大正则化越强）|
| **`inplace`** | 是否原地修改（默认 False） | 一般用默认 |

---

## 📌 五、查看模型参数（4种方法）

### 5.1 笨办法：逐层手动访问

```python
# 需要知道每一层的属性名，非常麻烦
print(model.linear1.weight)
print(model.linear1.bias)
print(model.linear2.weight)
print(model.linear2.bias)
print(model.out.weight)
print(model.out.bias)
```

### 5.2 进阶：`model.parameters()`

```python
# 自动管理所有层的所有参数
for param in model.parameters():
    print(param)
```

**输出顺序**：按照模型定义中层的顺序（linear1.weight → bias → linear2.weight → bias → out.weight → bias）

但**看不到参数名称**，不知道谁是谁。

### 5.3 推荐：`model.named_parameters()`

```python
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
# linear1.weight: torch.Size([4, 3])
# linear1.bias:   torch.Size([4])
# linear2.weight: torch.Size([4, 4])
# linear2.bias:   torch.Size([4])
# out.weight:     torch.Size([2, 4])
# out.bias:       torch.Size([2])
```

> ✅ 既有名称又有参数形状，一目了然！

### 5.4 最简洁：`model.state_dict()`

```python
print(model.state_dict())
# OrderedDict([
#     ('linear1.weight', tensor([...])),
#     ('linear1.bias', tensor([...])),
#     ('linear2.weight', tensor([...])),
#     ...
# ])
```

返回一个**有序字典**，一行代码展示所有参数。

| 方法 | 优点 | 缺点 |
|------|------|------|
| 手动访问 | 精准访问某一层 | 代码繁琐，需知道属性名 |
| `parameters()` | 自动管理 | 无名称，分不清谁是谁 |
| `named_parameters()` ⭐ | **有名称有参数** | 需 for 循环 |
| `state_dict()` ⭐ | **一行搞定，清晰直观** | 输出较长 |

---

## 📌 六、查看模型结构与参数数量（torchsummary）

当模型变大（比如 784→50→10），手动查看参数就不现实了。我们需要专业的工具。

### 6.1 安装

```bash
pip install torchsummary
```

### 6.2 使用

```python
from torchsummary import summary

# 传入模型、输入特征数、batch_size、设备
model = NNModel()
summary(model, input_size=(3,), batch_size=10, device='cpu')
```

### 6.3 输出解读

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                [-1, 4, 4]              16
            Linear-2                 [-1, 4, 4]              20
            Linear-3                 [-1, 2, 4]              10
================================================================
Total params: 46
Trainable params: 46
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.00
----------------------------------------------------------------
```

| 列 | 含义 |
|----|------|
| **Layer (type)** | 层的类型（这里都是 Linear） |
| **Output Shape** | 经过该层后的输出形状 |
| **Param #** | **该层的参数个数（权重 + 偏置）** |
| **Total params** | 模型总参数量 |
| **Trainable params** | 可训练参数数量 |

### 6.4 参数个数计算

| 层 | 权重参数 | 偏置参数 | 总参数 |
|----|:-------:|:-------:|:-----:|
| Linear1 (3→4) | 3×4=12 | 4 | **16** |
| Linear2 (4→4) | 4×4=16 | 4 | **20** |
| Linear3 (4→2) | 4×2=8 | 2 | **10** |
| **合计** | — | — | **46** |

### 6.5 参数规模的单位

| 单位 | 含义 | 示例 |
|:---:|------|------|
| **K** | 千（10³） | 小网络 |
| **M** | 百万（10⁶） | 中等网络 |
| **B** | **十亿（10⁹）** | 大语言模型 |

> 真实的 LLM 参数规模：DeepSeek-R1 满血版 = **671B** = **6710 亿参数**

---

## 📝 本章总结 + 完整代码

### 🌳 知识树

```
自定义神经网络模型
│
├── ① nn.Module（基类）
│   ├── __init__：定义层结构
│   ├── forward：定义前向传播
│   └── model(x) → 自动调 forward
│
├── ② 搭建三层神经网络
│   ├── 3→4 Tanh → 4 ReLU → 2 Softmax
│   └── 初始化：Xavier + Kaiming 组合
│
├── ③ Dropout 正则化
│   ├── nn.Dropout(p)
│   ├── 训练时关闭 + 缩放
│   └── 评估时自动关闭
│
├── ④ 参数查看方法
│   ├── 手动访问（麻烦）
│   ├── parameters()（无名称）
│   ├── named_parameters()（有名称 ✅）
│   └── state_dict()（字典 ✅）
│
└── ⑤ 结构查看（torchsummary）
    ├── summary(model, input_size)
    ├── Output Shape / Param #
    └── 参数规模：K → M → B
```

### 🔥 完整可运行代码

```python
import torch
import torch.nn as nn
import torch.nn.init as init
from torchsummary import summary

# ============ 1. 自定义三层神经网络 ============
class NNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 4)
        self.out = nn.Linear(4, 2)
        self.dropout = nn.Dropout(p=0.5)

        # 参数初始化
        init.xavier_normal_(self.linear1.weight)   # Tanh → Xavier
        init.kaiming_normal_(self.linear2.weight)  # ReLU → Kaiming
        # self.out 用默认初始化

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = self.dropout(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)
        y = torch.softmax(self.out(x), dim=1)
        return y

# ============ 2. 测试前向传播 ============
X = torch.randn(10, 3)
model = NNModel()
y_pred = model(X)
print(f"输出形状: {y_pred.shape}")
print(f"每行概率和: {y_pred.sum(dim=1)}")  # 应该都是 1.0

# ============ 3. 查看参数 ============
print("\n=== named_parameters() ===")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

print("\n=== state_dict() ===")
print(model.state_dict())

# ============ 4. 查看模型结构 ============
print("\n=== torchsummary ===")
summary(model, input_size=(3,), batch_size=10, device='cpu')
```

### 关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| **`nn.Module`** | 所有神经网络层的基类 |
| **`__init__`** | 定义各层和参数初始化 |
| **`forward`** | 定义数据流动路径 |
| **`model(x)`** | 自动调用 forward 进行前向传播 |
| **`nn.Dropout(p)`** | 以概率 p 随机关闭神经元 + 缩放 |
| **`model.eval()`** | 评估模式，Dropout 关闭 |
| **`model.parameters()`** | 获取所有参数的迭代器 |
| **`model.named_parameters()`** | 带名称的参数迭代器 |
| **`model.state_dict()`** | 所有参数的状态字典 |
| **`torchsummary.summary()`** | 查看模型结构、每层输出形状和参数量 |

> 🎯 现在你已经掌握了用 `nn.Module` 搭建任意神经网络的能力！把层像积木一样组合、初始化参数、加 Dropout、查看结构和参数，全部都会了。

---

---

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

---

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

---

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
