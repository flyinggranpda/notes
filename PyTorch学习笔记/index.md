# 🧠 PyTorch 深度学习笔记

> 📝 本笔记为 PyTorch 学习系列合集，适合深度学习初学者
> 🎯 风格：代码实践 + 通俗讲解

---

## 📋 目录

- **[入门与安装](01_PyTorch_入门与安装.md)** — PyTorch 简介、CPU/GPU安装、CUDA配置、常见问题
- **[张量创建](02_PyTorch_张量创建.md)** — tensor/Tensor、dtype、arange/linspace、zeros/ones、随机生成
- **[张量转换](03_PyTorch_张量转换.md)** — 类型转换、Tensor↔NumPy、Tensor↔标量、内存共享
- **[张量数值计算](04_PyTorch_张量数值计算.md)** — 基本运算、哈达玛积、矩阵乘法、节省内存、统计函数
- **[张量索引](05_PyTorch_张量索引.md)** — 简单索引、切片、列表索引、布尔索引
- **[张量形状操作](06_PyTorch_张量形状操作.md)** — 交换维度(T/transpose/permute)、reshape/view、unsqueeze/squeeze、cat/stack
- **[自动微分(Autograd)](07_PyTorch_自动微分（Autograd）.md)** — requires_grad/grad/grad_fn、计算图、backward、叶子节点、detach
- **[线性回归实战案例](08_线性回归实战案例.md)** — 完整流程：数据生成→Dataset/DataLoader→模型定义→训练五步→画图
- **[激活函数](09_激活函数（Activation_Functions）.md)** — Sigmoid、Tanh、ReLU、Softmax + Autograd自动求导画图
- **[全连接层与参数初始化](10_全连接层与参数初始化.md)** — nn.Linear详解、weight/bias形状、常数/随机/Xavier/Kaiming初始化
- **[自定义神经网络模型](11_自定义神经网络模型.md)** — nn.Module继承、三层网络搭建、Dropout、参数查看(4种)、torchsummary
- **[设备管理与Sequential](12_设备管理（device）与_Sequential_快捷搭建.md)** — CPU/GPU、device统一管理、nn.Sequential快捷搭建、apply初始化
- **[损失函数](13_PyTorch_损失函数（Loss_Functions）.md)** — BCE Loss、CrossEntropyLoss、MSE/MAE/SmoothL1、选择指南
- **[优化方法](14_PyTorch_优化方法（Optimizers）.md)** — SGD+Momentum、学习率衰减(StepLR/MultiStepLR/ExponentialLR)、AdaGrad、RMSProp、Adam/AdamW

---

> 💡 每个笔记文件都是独立的，可直接打开阅读。推荐按编号顺序学习。
