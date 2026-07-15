# Transformer 词表与数据预处理

## 1. 课程元信息

- **主题**：Transformer 源码中的数据预处理 — 词表（Vocabulary）与文本张量化
- **前置知识**：Transformer 整体架构、张量（Tensor）基本概念
- **核心目标**：理解原始文本如何通过词表映射，最终转化为 PyTorch 可以处理的张量格式

---

## 2. 核心概念图谱

| 术语 | 英文 | 通俗解释 |
|------|------|---------|
| 词表 | Vocabulary / Vocab | 包含所有词汇及其对应序号的字典。每个词有一个唯一整数ID |
| 源语言词表 | SRC Vocab | 输入语言的词表（如中文→英文任务中的中文词表） |
| 目标语言词表 | TGT Vocab | 输出语言的词表（如中文→英文任务中的英文词表） |
| 输入批次 | Input Batch | 经过切分和映射后的输入数据批次 |
| 输出批次 | Output Batch | 解码器使用的目标数据批次 |
| 目标批次 | Target Batch | 用于计算损失的标签数据批次 |
| 切分 | Split | 将句子按空格或字符拆分为独立的词/字 |
| 张量化 | Tensor Conversion | 将整数序列通过 `torch.LongTensor` 转换为张量 |

---

## 3. 技术原理 / 流程拆解

### 3.1 文本 → 张量的完整流程

```
原始文本句子
      ↓ Step 1: Split（切分）
["我", "吃", "饭"]    ["我", "喝", "水"]
      ↓ Step 2: Vocab Lookup（查词表）
  [1, 2, 3]             [1, 4, 5]
      ↓ Step 3: 组装为批次
  [[1, 2, 3], [1, 4, 5]]
      ↓ Step 4: torch.LongTensor
  tensor([[1, 2, 3],
          [1, 4, 5]])
```

### 3.2 代码解读

```python
def make_batch(sentences):
    # SRC: 源语言词表（如中文词表）
    # TGT: 目标语言词表（如英文词表）
    
    # 对句子按空格切分，查词表转为序号
    input_batch = [SRC_vocab[n] for n in sentences[0].split()]
    output_batch = [TGT_vocab[n] for n in sentences[1].split()]
    target_batch = [TGT_vocab[n] for n in sentences[2].split()]
    
    # 转换为 PyTorch 张量
    return torch.LongTensor(input_batch), \
           torch.LongTensor(output_batch), \
           torch.LongTensor(target_batch)
```

### 3.3 三个批次各自的角色（以中译英为例）

| 批次 | 词表 | 内容 | 用途 |
|------|------|------|------|
| `input_batch` | SRC（中文词表） | "我 吃 饭" | **编码器输入** |
| `output_batch` | TGT（英文词表） | "I eat rice" | **解码器输入**（带 `<sos>` 前缀） |
| `target_batch` | TGT（英文词表） | "I eat rice" | **计算损失**的标签（带 `<eos>` 后缀） |

---

## 4. 避坑指南

| 注意点 | 说明 |
|--------|------|
| Split 方式因语言而异 | 英文按空格 split，中文需要分词（示例中简化处理为字级别） |
| 词表必须包含特殊标记 | 如 `<pad>`（填充）、`<sos>`（起始）、`<eos>`（结束）、`<unk>`（未知词） |
| 词表 ID 必须连续 | 从 0 或 1 开始的连续整数，否则 embedding 层会出问题 |
| SRC 和 TGT 词表是独立的 | 源语言和目标语言各有一个词表，大小可以不同 |
| Batch 内句子需等长 | 不同长度的句子需要通过 padding 对齐到统一长度 |

---

## 5. 对比与思考

### 5.1 翻译任务的三元组数据设计

```python
sentences = [
    ["我 吃 饭"],           # sentence[0] → SRC → input_batch
    ["I eat rice"],          # sentence[1] → TGT → output_batch（解码器输入）
    ["I eat rice"]           # sentence[2] → TGT → target_batch（损失计算标签）
]
```

- `output_batch` 和 `target_batch` 来自同一句话，但偏移一位（`output_batch` 带 `<sos>`，`target_batch` 带 `<eos>`）
- 解码器看到 `<sos> I eat` 预测 `I eat rice`，输出层与 `target_batch` 计算交叉熵损失

---

## 6. 本节课思维导图

- Transformer 词表与数据预处理
  - 词表（Vocabulary）
    - SRC Vocab（源语言词表）
    - TGT Vocab（目标语言词表）
  - 三行数据的角色
    - sentences[0] → input_batch（编码器输入）
    - sentences[1] → output_batch（解码器输入）
    - sentences[2] → target_batch（损失计算标签）
  - 文本 → 张量的四个步骤
    - Split（切分）
    - Vocab Lookup（查序号）
    - 组装批次
    - torch.LongTensor（张量化）
  - 核心代码：`make_batch(sentences)`
