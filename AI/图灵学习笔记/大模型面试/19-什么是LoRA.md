# 什么是 LoRA（Low-Rank Adaptation）

## 1. 课程元信息

- **主题**：LoRA 的核心原理、数学推导、代码实现与局限性
- **前置知识**：PEFT、Transformer、矩阵低秩分解
- **核心目标**：理解 LoRA 如何通过低秩分解大幅减少可训练参数，同时保持接近全参数微调的效果

---

## 2. 核心概念图谱

| 术语 | 英文 | 通俗解释 |
|------|------|---------|
| 低秩适配 | LoRA (Low-Rank Adaptation) | 将权重矩阵拆成两个小矩阵训练，大幅减少参数量 |
| 低秩分解 | Low-Rank Decomposition | 将大矩阵拆为两个小矩阵相乘的近似表示 |
| 秩 | Rank (r) | 分解后的小矩阵维度，r << d，通常 8 或 16 |
| 缩放因子 | Alpha (α) | 控制 LoRA 增量权重的缩放比例 |
| 冻结 | Frozen | 原始权重 W₀ 在训练中不更新 |
| 热插拔 | Hot-swappable | 不同任务的 LoRA 权重可随时更换 |

---

## 3. 技术原理 / 流程拆解

### 3.1 核心数学原理

```
原始权重矩阵: W₀ (d × d) — 冻结，不参与训练
LoRA 分解:    ΔW = B × A
              B: d × r,  A: r × d,  r << d

微调后的权重: W = W₀ + α · B × A

训练时：只更新 B 和 A（参数量从 d² 降为 2dr）
推理时：B×A 可合并回 W₀，无额外推理开销
```

### 3.2 参数效率对比（以 d=512, r=8 为例）

```
全参数微调：512 × 512 = 262,144 个参数
LoRA：      512×8 + 8×512 = 8,192 个参数
节省：      262,144 / 8,192 ≈ 32 倍
```

### 3.3 代码实现

```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=1.0):
        super().__init__()
        # 冻结原始权重
        self.W0 = nn.Linear(in_dim, out_dim)
        self.W0.weight.requires_grad = False
        
        # LoRA 低秩分解
        self.A = nn.Parameter(torch.randn(rank, out_dim))   # r × d
        self.B = nn.Parameter(torch.randn(in_dim, rank))     # d × r
        self.alpha = alpha
    
    def forward(self, x):
        # W₀ 冻结 + B×A 训练
        return self.W0(x) + self.alpha * (x @ self.B @ self.A)
```

### 3.4 LoRA 可应用的模块

Transformer 中几乎所有含矩阵运算的模块都可以用 LoRA：

```
输入 Embedding    → ✅ 可应用 LoRA
位置编码          → ✅ 可应用 LoRA
QKV 投影          → ✅ 最常用（Q/K/V 各一个 LoRA）
注意力输出投影    → ✅ 可应用 LoRA
FFN 两层全连接    → ✅ 可应用 LoRA
```

---

## 4. LoRA 的核心优势

| 优势 | 说明 |
|------|------|
| **显存需求大幅降低** | 可训练参数减少 90%~99% |
| **性能接近全参数微调** | 差距在 1% 以内 |
| **推理无额外开销** | B×A 可合并回 W₀ |
| **可热插拔** | 不同任务维护不同 LoRA 权重，切换无需重载模型 |
| **可回滚** | W₀ 冻结，不满意只需丢弃当前 B×A |

---

## 5. 局限性

| 局限 | 说明 |
|------|------|
| **不是全参数微调替代品** | 性能"接近"但不是"等于" |
| **跨任务兼容性** | 多任务合并时可能有冲突 |
| **秩的选择** | r 太小→表达能力不足，r 太大→失去了效率优势 |

---

## 6. 面试建议

讲 LoRA 时顺带提及其变体，可显著提升回答深度：

| 变体 | 核心改进 | 提出方 |
|------|---------|--------|
| **LoRI** | 冻结 A + 稀疏化 B，参数再减 95% | 蚂蚁+清华 |
| **QLoRA** | 量化 + LoRA，4-bit 训练 | 华盛顿大学 |
| **AdaLoRA** | 自适应分配秩到不同层 | 微软 |
| **DoRA** | 权重分解后应用 LoRA | — |

---

## 7. 本节课思维导图

- LoRA（低秩适配）
  - 原理：W = W₀ + B × A（W₀ 冻结，B/A 训练）
  - 效率：d² → 2dr，可省 30 倍+
  - 优势
    - 显存需求低
    - 性能接近全参数微调（1% 以内）
    - 推理无额外开销
    - 热插拔 + 可回滚
  - 应用范围：Transformer 全模块
  - 局限：不是全参数替代品，跨任务兼容性
  - 面试提升：顺带讲 LoRI / QLoRA / AdaLoRA 等变体
