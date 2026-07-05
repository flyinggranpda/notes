# 🔧 上下文压缩与 PDF 高级处理

> 🎯 第 9 课补全了后索引的最后一块拼图：上下文压缩  
> 🎯 同时深入讲解了 PDF 中的表格识别等复杂文档处理  
> 💻 注重实战代码，解决真实场景中的文档处理难题

---

## 📌 一、上下文压缩（Contextual Compression）

### 1.1 它解决什么问题

**痛点**：检索到的文档太多、太长，但大部分内容跟问题无关。

```
用户问："公司的营收构成有哪些？"
      ↓
向量检索到 5 篇文档，每篇 500 字
      ↓
但实际上只有每篇中的 1~2 句话与营收构成有关
      ↓
把 2500 字全部喂给大模型 → 费 token、响应慢、干扰答案
```

### 1.2 核心思路

> **检索后，不直接返回原始文档，而是先做两道工序：**
> 1. **过滤**：文档与问题不相关 → 丢弃
> 2. **压缩**：相关文档中的冗余内容 → 精简

```
原始检索结果：
  doc1(800字, 6段) → rediction → 只保留与问题相关的2段(200字)
  doc2(600字, 4段) → 过滤 → 整篇与问题无关 → 丢弃
  doc3(900字, 7段) → 压缩 → 精炼成 150 字摘要
      ↓
最终喂给 LLM 的上下文：350 字（原来是 2300 字）
```

### 1.3 上下文压缩需要两个组件

```
上下文压缩 = 基础检索器（Base Retriever） + 文档压缩器（Document Compressor）
                    ↓                          ↓
              负责从向量库搜文档          负责对文档做过滤/压缩
```

### 1.4 代码实现

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


def create_compression_retriever(vector_db, llm_client):
    """
    创建带上下文压缩的检索器
    """
    # Step 1: 基础检索器（从向量库搜文档）
    base_retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    
    # Step 2: 文档压缩器（用 LLM 提取关键信息）
    compressor = LLMChainExtractor.from_llm(llm_client)
    
    # Step 3: 组合成压缩检索器
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever


# ============ 使用对比 ============
def compare_with_without_compression(query, compression_retriever, base_retriever):
    """对比有无上下文压缩的效果"""

    # 不使用压缩
    print("=" * 55)
    print("🔴 不使用上下文压缩：")
    print("=" * 55)
    docs_raw = base_retriever.get_relevant_documents(query)
    total_chars_raw = sum(len(d.page_content) for d in docs_raw)
    print(f"  检索到 {len(docs_raw)} 篇文档")
    print(f"  总字符数: {total_chars_raw}")
    print(f"  第1篇文档前100字: {docs_raw[0].page_content[:100]}...")

    # 使用压缩
    print("\n" + "=" * 55)
    print("🟢 使用上下文压缩：")
    print("=" * 55)
    docs_compressed = compression_retriever.get_relevant_documents(query)
    total_chars_compressed = sum(len(d.page_content) for d in docs_compressed)
    print(f"  压缩后保留 {len(docs_compressed)} 篇文档")
    print(f"  总字符数: {total_chars_compressed}")
    print(f"  压缩比: {total_chars_compressed / total_chars_raw * 100:.1f}%")
    
    return docs_compressed
```

### 1.5 上下文压缩的处理逻辑

压缩器内部的工作流程：

```
输入文档："A公司主要从事小家电的制造与销售。产品线包括厨房电器、"
          "个护电器、生活电器三大品类。其中厨房电器占营收的45%。"
          "公司2023年实现营业收入199.7亿元..."

查询："A公司的营收构成是什么？"
              ↓
压缩器分析：只保留与"营收构成"相关的句子
              ↓
输出压缩结果："产品线包括厨房电器、个护电器、生活电器三大品类。"
              "其中厨房电器占营收的45%。"
              "2023年实现营业收入199.7亿元。"
              （从 200 字 → 75 字，压缩比 37.5%）
```

### 1.6 压缩策略选择

| 压缩策略 | 做法 | 优点 | 缺点 | 推荐场景 |
|---------|------|------|------|---------|
| **LLMChainExtractor** | 用 LLM 提取关键内容 | 精度最高 | 慢、贵 | 对质量要求极高 |
| **LLMChainFilter** | 只判断文档是否相关 | 速度快 | 不压缩内容 | 只需要过滤的场景 |
| **EmbeddingsFilter** | 用向量相似度过滤 | 最快最便宜 | 精度一般 | 大规模文档过滤 |

### 1.7 权衡分析：压缩到底值不值？

上下文压缩有一个明显的代价：**它需要额外的 LLM 调用，既慢又费 token。**

```
假设检索到 10 篇文档，每篇 500 字：

🔴 不压缩：
  基础检索 → 5000 字直接喂给回答 LLM
  LLM 调用: 1 次
  输入 token: ~5000
  耗时: ~2 秒

🟡 用 LLMChainExtractor 压缩：
  基础检索 → 对每篇文档调 LLM 提取 → 压缩到 800 字 → 喂给回答 LLM
  LLM 调用: 1 + 10 + 1 = 12 次
  输入 token: ~10000（提取输入）+ 800（回答输入）
  耗时: ~15 秒
```

| 对比 | 不压缩 | LLMChainExtractor 压缩 |
|------|--------|----------------------|
| LLM 调用次数 | 1 次 | 12 次 |
| 总 token 消耗 | ~5000 | ~10800 |
| 耗时 | ~2 秒 | ~15 秒 |
| 回答质量 | 可能被无关内容干扰 | 更聚焦、更准确 |

**所以结论很明确：上下文压缩确实更慢、更贵。它不是"默认打开"的功能，而是有取舍的优化手段。**

### 1.8 什么时候该用，什么时候不该用

| 你的情况 | 建议 |
|---------|------|
| **刚起步，基础 RAG 还没调通** | ❌ 不要用，先让基础流程跑通 |
| **检索结果太长（>5000 字）** | ✅ 可以用 EmbeddingsFilter，又快又便宜 |
| **回答质量常被无关内容干扰** | ✅ 考虑 LLMChainExtractor（精度优先） |
| **用户等不了 15 秒** | ❌ 别用压缩，改用 RRF 重排 |
| **token 预算很紧** | ❌ 压缩本身的消耗 > 压缩省下的 |
| **需要高吞吐量** | ❌ 用 EmbeddingsFilter 或干脆不用 |
| **医疗/法律等精度优先场景** | ✅ 必须用 LLMChainExtractor |

### 1.9 实用建议：从 RRF 开始，不行再加压缩

上下文压缩不是唯一的"提纯"手段。推荐的做法是**按成本从低到高逐步尝试**：

```
① RRF 融合排序（几乎零成本）
   → 重排后取 Top-K，已经能过滤掉大部分无关文档
   ↓ 还不够？
② EmbeddingsFilter（纯计算，无 LLM 调用）
   → 用向量相似度阈值过滤低分文档
   ↓ 还不够？
③ LLMChainFilter（一次判断，不压缩内容）
   → 用 LLM 判断每篇文档是否相关
   ↓ 还不够？
④ LLMChainExtractor（最重，但最准）
   → 用 LLM 从文档中提取相关内容
```

> **一句话总结上下文压缩的取舍：** 精度、速度、成本三个只能选两个。绝大多数场景下，RRF 重排 + EmbeddingsFilter 已经足够好，LLMChainExtractor 只是"最后的大招"。

---

## 📄 二、PDF 高级处理：表格识别

### 2.1 为什么需要特殊处理

**普通的 PDF 解析只能提取纯文本，但遇到表格就出问题：**

```
原始表格：
┌──────────────┬────────┬────────┐
│   产品类别     │ 营收(亿) │ 占比   │
├──────────────┼────────┼────────┤
│ 厨房电器      │  89.8  │  45%   │
│ 个护电器      │  60.2  │  30%   │
│ 生活电器      │  49.7  │  25%   │
└──────────────┴────────┴────────┘

普通 PDF 解析的结果（乱成一团）：
"产品类别营收(亿)占比厨房电器89.845%个护电器60.230%生活电器49.725%"
```

### 2.2 解决方案

需要安装两个额外的库来增强 PDF 解析能力：

```bash
# 这两个库能识别 PDF 中的表格、图片布局
pip install pdfplumber          # 处理表格
pip install layoutparser        # 识别文档布局（表格/图片/文本区域）
```

或者使用更强大的方案：

```bash
pip install unstructured        # 支持多种文档类型
pip install pdf2image          # PDF 转图片
pip install pytesseract         # OCR 识别
pip install rapidocr-onnxruntime  # 轻量级 OCR
```

### 2.3 带表格识别的 PDF 处理流程

```
PDF 文档
   ↓
[PDF 解析器] → 识别页面中的元素类型
   ├── 文本段落 → 直接提取文本
   ├── 表格数据 → 用表格解析器提取结构化数据
   ├── 图片内容 → OCR 识别
   └── 标题/目录 → 单独处理为元数据
   ↓
统一输出为结构化文档
```

### 2.4 摘要索引处理表格

**核心思路**：不直接把表格数据喂给 LLM，而是先对表格做摘要，再用摘要做检索。

```
原始表格数据
   ↓
[表格摘要生成]
   ↓
包含摘要的索引
   ↓
用户提问 → 先检索到摘要 → 再从内存中取出原始表格 → 喂给 LLM
```

### 2.5 多级检索架构

```python
class MultiLevelRetriever:
    """
    多级检索器：摘要索引 + 原始文档存储
    先通过摘要快速匹配，再取出完整文档喂给 LLM
    """
    
    def __init__(self, vector_db, llm_client):
        self.vector_db = vector_db    # 存摘要的向量库
        self.storage = {}             # 存原始文档的内存
        self.llm = llm_client
    
    def add_text(self, text_id: str, content: str):
        """添加文本数据"""
        # 生成摘要
        summary = self.llm.chat(f"请对以下内容生成一句话摘要：\n{content}")
        # 摘要存向量库
        self.vector_db.add_texts(
            texts=[summary],
            metadatas=[{"type": "text", "original_id": text_id}]
        )
        # 原始文档存内存
        self.storage[f"text_{text_id}"] = content
    
    def add_table(self, table_id: str, table_data: str):
        """添加表格数据"""
        # 对表格生成摘要
        summary = self.llm.chat(f"请对以下表格内容生成一句话摘要：\n{table_data}")
        # 摘要存向量库
        self.vector_db.add_texts(
            texts=[summary],
            metadatas=[{"type": "table", "original_id": table_id}]
        )
        # 原始表格存内存
        self.storage[f"table_{table_id}"] = table_data
    
    def retrieve(self, query: str, top_k: int = 5):
        """检索：先查摘要，再取原文"""
        # 向量检索摘要
        results = self.vector_db.similarity_search(query, k=top_k)
        
        # 根据摘要的 metadata 取原始数据
        original_docs = []
        for r in results:
            doc_type = r.metadata["type"]
            original_id = r.metadata["original_id"]
            key = f"{doc_type}_{original_id}"
            if key in self.storage:
                original_docs.append(self.storage[key])
        
        return original_docs
```

---

## 🔄 三、RAG 整体回顾

### 3.1 高级 RAG 完整技术栈

```
                   用户提问
                      │
                      ▼
            ┌─────────────────────┐
            │   预索引优化         │
            │  ├─ 查询重写         │
            │  ├─ 查询扩展         │
            │  ├─ 假设性问题(HyDE) │
            │  └─ 元数据过滤       │
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │   检索过程           │
            │  ├─ 向量检索         │
            │  ├─ 关键词检索(BM25) │
            │  └─ 混合检索         │
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │   后索引优化         │
            │  ├─ RRF 融合排序    │
            │  ├─ 上下文压缩      │ ← 本课
            │  ├─ 文档过滤        │
            │  └─ 自我反思(Self-RAG)│
            └─────────┬───────────┘
                      │
                      ▼
                  生成回答
```

### 3.2 各种技术的最佳应用时机

| 技术 | 最佳时机 | 核心代价 |
|------|---------|---------|
| 查询重写 | 用户问得模糊时 | 1 次 LLM 调用 |
| 查询扩展 | 需要全面覆盖时 | 1 次 LLM + 多路检索 |
| 元数据过滤 | 文档有明确属性时 | 1 次 LLM |
| HyDE | 问题口语化、文档书面化 | 1 次 LLM |
| 混合检索 | 需要精确术语匹配 | 维护两套索引 |
| RRF 融合 | 多路检索后需要统一排序 | 几乎无 |
| **上下文压缩** | **检索结果太长太杂时** | **N 次 LLM 调用** |
| 自我反思 | 对精度要求极高 | N 次 LLM 调用 |

---

## 📝 本章总结

### 关键概念速查

| 概念 | 一句话解释 |
|------|-----------|
| **上下文压缩** | 检索后对文档做过滤+精简，只保留与问题相关的内容 |
| **文档压缩器** | 用 LLM 或 embedding 对文档做过滤/提取/摘要的组件 |
| **表格识别** | 用 pdfplumber/unstructured 等工具识别 PDF 中的表格结构 |
| **摘要索引** | 先对文档生成摘要，用摘要做检索，匹配到后再取原文 |

### 一句话总结

> **上下文压缩 = "搜到之后不直接用，先筛一遍再喂给 LLM"。**  
> 它和后索引中的 RRF 重排配合使用，一个负责排序，一个负责精简。🚀
