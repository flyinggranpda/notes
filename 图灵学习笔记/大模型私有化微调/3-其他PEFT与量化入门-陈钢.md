# 其他 PEFT 方法与量化入门 — 陈钢老师

## 1. 课程元信息

- **课程主题**：（1）其他 PEFT 方法（Prompt Tuning / P-Tuning / Prefix Tuning）；（2）量化入门
- **主讲老师**：陈钢
- **课程定位**：十节课系列第三节。上节课讲完了 LoRA，这节课讲跟 LoRA **平级**的其他 PEFT 方法，最后给量化开了个头
- **前置知识**：
  - PEFT 概念（参数高效微调）
  - LoRA 原理（低秩分解 + 相加）
  - Embedding（词向量化）的基本概念
- **时长**：约 2 小时 15 分
- **课堂互动**：老师新增了两个环节——课前用一句话总结上节课内容、课中投票确认跟进情况

---

## 2. 核心概念图谱

### 2.1 三种微调方式的层次关系

```
微调（Fine-Tuning）
  ├── 全量微调（Full Fine-Tuning）— 所有参数，最慢最费
  └── PEFT（参数高效微调）
        ├── LoRA（低秩适配）— 调模型内部权重矩阵 ⭐ 最主流
        ├── Prompt Tuning（提示词微调）— 只调输入层的 embedding
        ├── P-Tuning（P 微调）— embedding 可以插在输入中间任意位置
        └── Prefix Tuning（前缀微调）— 同时调 embedding + 部分 KV 层
```

### 2.2 Prompt Tuning / Soft Prompt 概念

| 中文术语 | 英文术语 | 通俗解释 |
|---------|---------|---------|
| 硬提示词 | Hard Prompt | 人写的一段文字（如"你是一个有用的助手"），是提示词工程 |
| 软提示词 | Soft Prompt | **机器学出来的**一组数字（token ID 序列），不是人能读懂的文字 |
| 提示词微调 | Prompt Tuning | 只微调输入层最前面的几个 token embedding，不改模型内部参数 |
| P 微调 | P-Tuning | 跟 Prompt Tuning 类似，但 embedding 不仅能放前面，还能往句子中间插 |
| 前缀微调 | Prefix Tuning | 不仅调 embedding 层，还调 KV Cache 的一部分 |
| 可调 embedding | Trainable Embedding | 留空几个 token 位置，让神经网络自己通过学习找到最佳值 |
| KV Cache | Key-Value Cache | Transformer 推理时缓存之前的 K 和 V 矩阵，避免重复计算 |

### 2.3 量化基础概念

| 中文术语 | 英文术语 | 通俗解释 |
|---------|---------|---------|
| 量化 | Quantization | 把模型参数从高精度（如 FP16 16bit）降到低精度（如 INT4 4bit），让模型变小 |
| QLoRA | Quantized LoRA | **量化 + LoRA 的组合**，在量化后的模型上做 LoRA 微调 |
| load_in_4bit | — | Unsloth 里的一个参数，设为 True 就开启 4bit 量化 |
| 采样 | Sampling | 把连续信号变成离散的时间点（如每秒取一个值） |
| 离散化 | Discretization | 把连续的数值限制到有限的几个数值上（只能取规定好的值） |

---

## 3. 技术原理 / 流程拆解

### 3.1 三种微调方式全景

```
老师原话："这个微调方式有很多，大家用脑子记住大框架就行，不用记在纸上。"

全量微调（Full FT）：所有参数都调 → 最准但最慢
PEFT（参数高效微调）：只调一小部分 → 快
  ├── LoRA          : 调 Transformer 内部层的权重（低秩分解）
  ├── Prompt Tuning  : 只调输入层最前面的 embedding（极小，几 KB）
  ├── P-Tuning       : embedding 能插到输入任意位置
  └── Prefix Tuning  : 调 embedding + KV 层

其中 LoRA 是当前使用最多的 PEFT 方法。
其他方法（Prompt/P/ Prefix）比 LoRA 更轻量，但能力上限也更低。
```

### 3.2 Hard Prompt vs Soft Prompt 的区别

```
Hard Prompt（提示词工程）：
  "你是一个专业的法律顾问，精通劳动法、劳动合同法……"
  → 人写的文字 → 输入给模型 → 每个人都能看懂

Soft Prompt（提示词微调）：
  [0.12, -0.87, 0.33, ..., 0.01]  ← 一堆不能读懂的 token ID
  → 机器学出来的数学表达 → 只有机器能"看懂"
  → 反解回中文通常是一堆胡话乱码
```

**Hard Prompt 的困境：**
```
你写了一段提示词，效果还行
但你不知道这是不是最好的
你试来试去，效果很难再提升了
→ 需要机器来帮你"找"最好的提示词
```

**Soft Prompt 的解决方案：**
```
留空几个 token 位（初始化为 0）
→ 让大模型在这些空位上做梯度下降
→ 找到一组"最适合当前任务"的 token ID
→ 这组 token ID（矩阵）就是 Soft Prompt
```

### 3.3 Hard Prompt 示例（情绪识别小技巧）

```
任务：判断一句话是正面情绪还是负面情绪

Hard Prompt 做法（一个巧妙的小技巧）：
  用户说： "I will definitely watch it again"（我绝对会再看一遍）
  我们在前面拼上这句话，让模型续写：
    "I will definitely watch it again. It is ___"
  模型因为是序列生成器，为了让句子通顺，它大概率会写：
    "It is great" （而不是 "It is bad"）
  查词典：great → 正面词 → 判断这句话是正面情绪
```

**这个技巧展示了：** 不需要微调，光靠巧妙的提示词设计，就能让 Base 模型（不会问答的续写器）完成特定任务。但问题是：**当你找不到合适的提示词时怎么办？**

### 3.4 Soft Prompt 的工作原理

```
左边（Hard Prompt）：               右边（Soft Prompt）：
  用户输入：I will watch again         用户输入：I will watch again
  手动写死：It is ...                   空出 3 个 token：[?, ?, ?]
                                           ↓
                                       初始化为 0
                                           ↓
                                       训练（梯度下降）
                                           ↓
                                       找到最佳值 [0.12, -0.87, 1.23]
                                           ↓
                                       存下来（33KB），下次直接用
```

**Soft Prompt 的本质：**
```
Prompt Tuning 不去碰模型内部的任何参数
它只调输入层最前面的几个 embedding 值
→ 相当于找到了一组"能让模型更好回答问题"的输入前缀
→ 比 LoRA 更轻量（几 KB vs LoRA 的几 MB）
→ 但能力上限 = 提示词工程的上限
```

### 3.5 三种 Soft Prompt 方法的区别

| 方法 | 调什么 | 插在哪里 | 复杂度 |
|------|--------|---------|--------|
| **Prompt Tuning** | 输入 embedding | **最前面**（跟人类写提示词的位置一样） | 最低 |
| **P-Tuning** | 输入 embedding | **任意位置**（可以插在句子中间） | 中等 |
| **Prefix Tuning** | embedding + **KV 层参数** | 最前面 + Transformer 各层 | 最高（但比 LoRA 轻） |

**老师原话：**
> "Prompt Tuning 就是把人类写系统提示词的位置空出来，让机器学。P-Tuning 觉得放在前面还不够，中间可能更好。Prefix Tuning 说我把 KV 也拉上一起调。这些都是很自然的创新——有了第一个点子，后面两个就顺理成章了。"

### 3.6 LoRA vs Soft Prompt 的核心区别

```
LoRA：模型向任务靠拢
  "我要改变大模型本身，让它更擅长回答医疗问题"
  → 改模型内部的权重矩阵
  → 效果更强（能调整说话风格/语气/知识）
  → 文件较大（几 MB ~ 几十 MB）

Soft Prompt：任务向模型靠拢
  "我不改大模型，我只改变提问方式"
  → 只改输入层的 embedding
  → 效果较弱（上限 = 提示词工程上限）
  → 文件极小（几 KB）

例子：
  想调一个猫娘（说话带喵呜）→ LoRA 能做到 → Soft Prompt 做不到
  想识别一段话是不是抱怨 → 两者都能做
```

### 3.7 什么时候用 Soft Prompt？

**三种典型场景：**

```
场景 1：成本敏感的在线微调服务
  假设你在 OpenAI 工作，要实现"用户在线微调"功能
  每个用户微调后都要占独显存？
  → Soft Prompt 每个用户只需多存几 KB
  → 推理时拼上即可，成本极低

场景 2：酒馆式多角色对话
  几十个不同角色（椅子、机器人、猫娘……）
  不可能每个角色存一个 8GB 模型
  → 不同角色用不同的 Soft Prompt（几 KB）
  → 同一个基座模型 + 不同前缀 = 不同角色

场景 3：文本分类等简单任务
  判断一段话的情感/是否投诉/是否违规
  → Soft Prompt 足够用
  → 比 LoRA 更快更轻
```

### 3.8 Soft Prompt 的极限

> **老师原话：** "它的上限就是你有一个天下最会写提示词的人帮你写提示词，写出来那个提示词就是它的上限了。但它也突破不了提示词工程的能力上限。"

```
能做的事：文本分类、情感分析、简单指令遵循
不能做的事：改变说话风格、注入新知识、复杂行为定制
           （这些需要 LoRA 或全量微调）
```

### 3.9 量化入门

**量化的动机：把模型变小**

```
DeepSeek R1 671B（未量化）：404 GB → 没有消费级显卡能跑
DeepSeek R1 1.58bit（量化后）：~100 GB → 企业成本大幅降低
```

**量化的本质：减少每个参数占用的位数**

```
原来的参数：0.723456789  ← FP32（32 bit = 4 bytes）
量化后的参数：0.75        ← INT4（4 bit = 0.5 bytes）
                         → 模型大小缩小到原来的 1/8
```

**量化与降秩的区别（老师强调）：**
```
降秩（LoRA）：100 个数 → 变成 10 个数（N 变少）
量化：         还是 100 个数，但每个数变小了（精度变低）

降秩 = 矩阵变小（维度减少）
量化 = 精度变低（每个值的分辨率降低）
```

**在 Unsloth 中使用量化：**
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen2.5-7B",
    load_in_4bit=True,  # ← 这节课只讲这一个参数！
)
```

**量化效果对比（来自 QLoRA 论文）：**
```
16bit LoRA 微调后模型大小：29 GB
4bit QLoRA 微调后模型大小：7.5 GB
相差约 4 倍 = 16 / 4
```

---

## 4. 案例 / 代码实战复盘

### 4.1 案例一：不写代码的微调方式

- **教学目的**：知道微调不止一种玩法，开拓视野
- **要回答的核心问题**：除了 Unsloth 框架，还有哪些微调方式？

**老师展示了 3 种不同层次的微调方式：**

| 方式 | 代表 | 优点 | 缺点 |
|------|------|------|------|
| 完全不写代码 | 魔搭在线训练 / Llama Factory UI | 点点点就行 | 无法理解底层原理 |
| **使用框架（本节重点）** | **Unsloth / HuggingFace PEFT** | **概念清楚，承上启下** | 需要写代码 |
| 企业内部自研 | DeepSeek 内部训练代码 | 性能最高 | 概念不抽象，难迁移 |

**OpenAI 微调 API：**
- 三步：提供数据集 → 上传 → 创建 job
- 微调后价格翻倍（如 GPT-4.1：$1.25/M → $3/M）
- 不需要自己买 GPU（在 OpenAI 云端训练）

**苹果 Create ML：**
- macOS 自带的机器学习工具（`⌘+Space` 搜索 "Create ML"）
- 可以在本地 Mac 上做训练（不一定要 GPU 云）
- 微软和 Windows 也有类似工具

> **⭐ 选择框架微调的原因**：承上启下。理解了框架微调，往上走（不用代码）或往下走（企业自研）都轻松。

### 4.2 案例二：用 HuggingFace PEFT 做投诉识别

- **教学目的**：展示 Prompt Tuning 的完整代码流程
- **要回答的核心问题**：Soft Prompt 代码怎么写？怎么用？

**任务**：判断一条推特是不是在投诉（Complaint Detection）

```
数据：Twitter Complaints 数据集（已标注）
输入：一条推特文本
输出：complain（是在投诉）/ not complain（不是投诉）
```

**安装依赖：**
```bash
pip install peft transformers datasets
```

**代码流程：**

```python
# Step 0: 选择基座模型
# 这里是文本分类任务，选分类模型（如 RoBERTa/BERT），不是对话模型（如 Qwen/LLaMA）
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# 也可以用更轻量的： "distilbert-base-uncased"

# Step 1: 初始化 Prompt Tuning 配置
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model

prompt_config = PromptTuningConfig(
    task_type=TaskType.SEQ_CLS,               # 任务类型：序列分类
    prompt_tuning_init=PromptTuningInit.TEXT, # 用文本初始化
    prompt_tuning_init_text="Classify this tweet as complaint or not",
    num_virtual_tokens=8,                     # 8 个可训练的虚拟 token
    tokenizer_name_or_path=model_name,
)

# Step 2: 加载分词器和基座模型
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained(model_name)
model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(model_name)  # type: ignore[no-untyped-call]

# Step 3: 把 Prompt Tuning 加到模型上
model = get_peft_model(model, prompt_config)

# Step 4: 加载并处理数据集
from datasets import load_dataset

dataset = load_dataset("twitter_complaints")  # 推特投诉数据集

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")

dataloader = DataLoader(tokenized_dataset, batch_size=16, shuffle=True)

# Step 5: 设置训练参数并训练（约 50 分钟）
num_epochs = 3
optimizer = AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Step 6: 保存 Prompt Tuning 权重（仅 33KB！）
model.save_pretrained("./model")

# Step 7: 加载保存的权重做推理
from peft import PeftModel

loaded_model = PeftModel.from_pretrained(
    base_model=AutoModelForSequenceClassification.from_pretrained(model_name),
    model_id="./model",
)

test_text = "I have no water, and bill is already paid. Could you do something about this?"
inputs = tokenizer(test_text, return_tensors="pt")
output = loaded_model(**inputs)
predicted_label = "complain" if output.logits.argmax().item() == 1 else "not complain"
print(predicted_label)  # 输出: complain
```

**提示词微调 vs LoRA 的代码对比：**

| 步骤 | LoRA (Unsloth) | Prompt Tuning (HuggingFace) |
|------|---------------|---------------------------|
| 配置 | `lora_config = {...}` | `prompt_config = {...}` |
| 加适配器 | `get_peft_model(model, lora_config)` | `get_peft_model(model, prompt_config)` |
| 训练 | `Trainer.train()` | 手动写训练循环 |
| 保存 | ~30 MB（LoRA 权重） | **~33 KB（几组 token ID）** |
| 推理 | 加载 LoRA 权重 + 原始模型 | 加载 soft prompt + 原始模型 |

**测试结果：**
```
输入："I have no water, and bill is already paid. Could you do something about this?"
→ 输出：complain ✓
```

> **⭐ 核心道理：** Prompt Tuning 训练完的文件只有 33KB。相比 LoRA 的几十 MB，小了 1000 倍。但它的能力也有限——只能改善"怎么问"，不能改变"模型本身的能力"。

### 4.3 案例三：量化入门演示

- **教学目的**：直观感受量化带来的模型大小变化
- **要回答的核心问题**：量化到底能省多少空间？

```python
# 这就是量化的全部——一个参数
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen2.5-7B",
    load_in_4bit=True,   # ← 量化开关
)
```

| 精度 | 7B 模型大小 | 能否跑在消费卡上 |
|------|------------|----------------|
| FP16（16bit）| ~14 GB | RTX 3090 (24G) ✅ |
| INT4（4bit）| ~3.5 GB | RTX 3060 (12G) ✅ |
| 1.58bit | ~1.3 GB | 手机芯片也能跑 |

**量化在 Unsloth 中的实际效果（来自 QLoRA）：**
```
同样的 LoRA 微调（r=16）：
  不量化（16bit）：最终模型 29 GB
  量化后（4bit）： 最终模型 7.5 GB
  → 缩小到原来的 1/4
```

---

## 5. 避坑指南（隐性知识提取）

### 5.1 PEFT 方法选择避坑

| 问题 | 原因 | 建议 |
|------|------|------|
| 不知道该用哪个 PEFT | 方法太多，概念混乱 | 记住一句话：**大多数情况用 LoRA 就够了** |
| Soft Prompt 效果不够 | 它只动了输入层，能力有限 | 调猫娘这类任务必须用 LoRA |
| HuggingFace PEFT 训练慢 | 没有 Unsloth 那样的优化 | HuggingFace PEFT 约 50 分钟，LoRA 约 5 分钟 |
| 以为 Soft Prompt 能替代 LoRA | 不理解两者的能力边界 | LoRA 改模型，Soft Prompt 改输入 |
| Soft Prompt 反解回中文 | 你试图把学到的 token ID 翻译成人话 | **反解出来是一堆胡话乱码**，不用管它长什么样 |

### 5.2 HuggingFace PEFT 使用避坑

| 问题 | 原因 | 建议 |
|------|------|------|
| 训练前环境初始化慢 | 要装很多依赖 | 提前跑 `env_init.sh` 脚本 |
| 训练到一半报错 | 可能是某个包版本不对 | 用 `pip install peft transformers datasets` |
| 模型保存位置找不到 | 默认存到 workspace/model | `cd` 到 model 目录查看 |
| Prompt Tuning 初始化用啥 | 用一段文字初始化 | 给一段和任务相关的描述文字即可 |

### 5.3 其他避坑

| 问题 | 原因 | 建议 |
|------|------|------|
| Unsloth 不支持 Prompt Tuning | 它专为 LoRA 优化 | 想玩 Soft Prompt 需要换 HuggingFace PEFT |
| 量化和降秩搞混 | 两个概念不同但都涉及"变小" | 量化=精度变低，降秩=维度减少 |
| 以为量化要写很多代码 | 其实就一个参数 | `load_in_4bit=True` |
| Mac 找不到 Create ML | 藏在开发者工具里 | `⌘+Space` 搜索 "Create ML" |
| 微调后的 OpenAI API 太贵 | 专有模型+独占显存 | 理解为什么贵：每个用户独享一个模型实例 |

### 5.4 老师经验总结

1. **"PEFT 里用 LoRA 就够了，Prompt Tuning 能做的事 LoRA 都能做"** — 这就是为什么课程重点讲 LoRA
2. **"Soft Prompt 反解回中文多半是胡话，不用管"** — 模型找到的最佳数字表达可能没有人类语言对应物
3. **"量化就是让模型变小——同样的 8B 模型，量化前 14G，量化后 3.5G"** — 一个参数解决问题
4. **"代码的价值很低，真正的价值在数据集"** — 开源模型代码就几百行，核心是模型和数据
5. **"选框架微调是因为它承上启下"** — 理解了概念，往上走不用代码、往下走企业自研都轻松

---

## 6. 对比与思考

### 6.1 LoRA vs Soft Prompt 对比

| 维度 | LoRA | Prompt Tuning |
|------|------|---------------|
| **调什么** | 模型内部的权重矩阵 | 输入层的 embedding |
| **文件大小** | 几 MB ~ 几十 MB | **几 KB（小 1000 倍）** |
| **能力上限** | 能改变知识/风格/行为 | **上限 = 提示词工程上限** |
| **能否调猫娘** | ✅ 可以 | ❌ 不可以 |
| **训练速度** | 快（5 分钟） | 慢（50 分钟，HuggingFace 实现） |
| **推理开销** | 加载额外权重 | 几乎为零 |
| **适合场景** | 需要改变模型行为的场景 | 文本分类等简单任务 |
| **框架** | Unsloth ⭐ | HuggingFace PEFT |

### 6.2 三种 Soft Prompt 对比

| 方法 | 可调位置 | 调什么 | 复杂度 | 创新点 |
|------|---------|--------|--------|--------|
| **Prompt Tuning** | 输入最前面 | embedding | 低 | 让机器找提示词 |
| **P-Tuning** | 任意位置 | embedding | 中 | 能插在句子中间 |
| **Prefix Tuning** | 输入 + 各层 | embedding + KV | 高 | 调更多层效果更好 |

### 6.3 三种微调代码对比

| 方式 | 代表 | 写代码？| 需要 GPU？| 理解原理？|
|------|------|---------|----------|---------|
| 不用代码 | 魔搭在线/Llama Factory | ❌ | ❌（云服务）| ❌ |
| **用框架 ⭐** | **Unsloth / HF PEFT** | **✅ 少量** | **✅ 需要** | **✅ 最推荐** |
| 企业自研 | DeepSeek 内部代码 | ✅ 大量 | ✅ 需要 | ✅ 深入 |

### 6.4 课程全景图

```
第一课：微调 101（Overview）
  → 微调是什么 + 三步走 + 三大框架

第二课：PEFT 上（LoRA 深入）
  → LoRA 原理 + 6 个参数详解 + 变体

第三课（本节）：PEFT 下 + 量化入门
  → Prompt Tuning / P-Tuning / Prefix Tuning
  → 量化动机 + load_in_4bit 参数

接下来：
  第四课：量化深入
  第五课：多模态上（CNN）
  第六课：多模态下（Transformer）
  最佳实践 → 实战项目
```

### 6.5 课后思考与延伸

1. **自己对比一下：**
   - 用 LoRA 做一个文本分类任务
   - 用 Prompt Tuning 做同一个任务
   - 看效果和文件大小的差异

2. **问问自己：**
   - 如果我在 OpenAI 工作，要实现"用户在线微调"功能，我会怎么设计？
   - 酒馆（角色扮演对话）这样的应用，用 Soft Prompt 是不是最合适？

3. **下节课预告（量化深入）**：
   - 量化的原理（采样 + 离散化 + 量化误差）
   - 不同量化精度对比（FP16/INT8/INT4/NF4）
   - QLoRA 的实现细节

4. **面试导向**：
   - Prompt Tuning 和 LoRA 的区别
   - 什么时候用 Soft Prompt 而不是 LoRA
   - 量化能带来什么好处？（模型变小、推理变快）

---

## 7. 本节课思维导图

- 其他 PEFT 方法与量化入门
  - 课前
    - 3 种微调方式（不用代码 / 框架 / 企业自研）
    - 选框架微调：承上启下
    - Mac/Windows 本地工具（Create ML）
    - OpenAI 微调 API（三步 + 价格翻倍）
  - 回顾上节（LoRA）
    - R = 秩（信息量）
    - LoRA = 降秩 + 相加
    - 为什么微调能降秩（信息量小）
    - 为什么原始模型不能降（信息量大）
  - Other PEFT Methods
    - PEFT 的分类框架
      - 全量（Full FT） vs PEFT
      - PEFT 包含：LoRA / Prompt Tuning / P-Tuning / Prefix Tuning
    - Hard Prompt vs Soft Prompt
      - Hard = 人写的文字（提示词工程）
      - Soft = 机器学出来的 token ID
      - 反解回中文 = 胡话乱码
    - Hard Prompt 示例
      - 情绪识别小技巧（it is ___ 让模型续写）
      - 优点：简单有效
      - 缺点：找不到最佳提示词
    - Soft Prompt 原理
      - 空出 token 位 → 梯度下降找最佳值
      - 不调模型内部参数
    - 三种方法对比
      - Prompt Tuning：只加在最前面
      - P-Tuning：可以插在中间
      - Prefix Tuning：调 embedding + KV
    - LoRA vs Soft Prompt
      - LoRA：模型向任务靠拢（改模型）
      - Soft Prompt：任务向模型靠拢（改输入）
      - 适用场景：Soft Prompt 适合简单分类
      - 不适合：说话风格/知识注入
  - 代码实战：HuggingFace PEFT
    - Prompt Tuning 做投诉识别
    - 流程：配置 → 加载模型 → 训练 → 保存
    - 保存文件：33KB（超小）
    - 训练时长：约 50 分钟
  - 量化入门
    - 动机：让模型变小
    - DeepSeek 671B：404GB → 100GB（1.58bit)
    - 量化 vs 降秩：精度变低 vs 维度减少
    - Unsloth 量化参数：load_in_4bit
    - QLoRA 效果：29GB → 7.5GB
  - 课后
    - 下节：量化深入
    - 面试重点：PEFT 方法选择、量化优势
