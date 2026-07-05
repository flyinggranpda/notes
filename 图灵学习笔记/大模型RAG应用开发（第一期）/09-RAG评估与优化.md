# 📊 RAG 评估体系与评估指标

> 🎯 怎么判断你写的 RAG 系统是好是坏？这一课讲**具体的评估方法和指标**  
> 🎯 不靠感觉，靠数据：精确度、召回率、忠诚度、相关性四个维度  
> 💻 所有评估逻辑都有对应的代码实现

---

## 📌 一、为什么要做 RAG 评估？

### 1.1 没有评估 = 靠感觉

```python
# 常见情况：
print("我感觉我这个 RAG 效果还不错...")   # ❌ 不能量化
print("好像回答得挺准的...")               # ❌ 无法对比
print("比之前好一点？")                    # ❌ 无法知道好在哪里
```

### 1.2 有了评估 = 靠数据

```python
# 理想情况：
results = evaluate_rag()
print(f"精确度: {precision:.2%}")     # 检索得准不准？
print(f"召回率: {recall:.2%}")        # 该搜的都搜到了吗？
print(f"忠诚度: {fidelity:.2%}")      # 回答基于检索结果吗？
print(f"相关性: {relevance:.2%}")     # 回答与问题相关吗？
# → 哪里分数低优化哪里 ✅
```

---

## 🔢 二、四个核心评估指标

### 2.1 精确度（Precision）

> **"检索到的文档中，有多少是和问题真正相关的？"**

```
问题："恐龙灭绝的原因是什么？"

检索到 4 篇文档：
  doc1: "小行星撞击地球导致恐龙灭绝"      ← 相关 ✅
  doc2: "恐龙化石的发现与考古历史"        ← 不相关 ❌  （没回答灭绝原因）
  doc3: "火山喷发对恐龙灭绝的影响"        ← 相关 ✅
  doc4: "恐龙的生活习性和繁殖方式"        ← 不相关 ❌

精确度 = 相关文档数 / 检索到的总文档数 = 2 / 4 = 50%
```

**公式**：
```
精确度 = 检索到的相关文档数 / 检索到的总文档数
```

```python
def calculate_precision(retrieved_docs, relevant_docs):
    """
    精确度：检索到的文档中真正有用的比例
    
    参数:
        retrieved_docs: 检索器返回的文档 ID 列表
        relevant_docs: 与问题真正相关的文档 ID 列表（人工标注）
    
    返回:
        精确度 (0~1)
    """
    if len(retrieved_docs) == 0:
        return 0.0
    
    # 统计检索到的文档中有多少是相关的
    true_positives = len(set(retrieved_docs) & set(relevant_docs))
    
    precision = true_positives / len(retrieved_docs)
    return precision
```

**优化方向**：
- 增加检索数量（多搜一些，增加命中概率）
- 做重排（RRF），把相关文档排前面

---

### 2.2 召回率（Recall）

> **"所有相关的文档中，检索引擎找到了多少？"**

```
数据库中共有 10 篇关于"恐龙灭绝"的相关文档
检索器只找到了其中的 6 篇

召回率 = 找到的相关文档数 / 全部相关文档数 = 6 / 10 = 60%
```

**公式**：
```
召回率 = 检索到的相关文档数 / 数据库中全部相关文档数
```

```python
def calculate_recall(retrieved_docs, all_relevant_docs):
    """
    召回率：该搜到的都搜到了吗？
    
    参数:
        retrieved_docs: 检索器返回的文档 ID 列表
        all_relevant_docs: 数据库中全部与问题相关的文档 ID 列表
                          （需要人工标注或使用标准答案集）
    
    返回:
        召回率 (0~1)
    """
    if len(all_relevant_docs) == 0:
        return 0.0
    
    true_positives = len(set(retrieved_docs) & set(all_relevant_docs))
    
    recall = true_positives / len(all_relevant_docs)
    return recall
```

**优化方向**：
- 使用混合检索（向量+关键词互补）
- 使用查询扩展（多问题检索，增加覆盖面）
- 调整分块大小（chunk size 太小可能错过关键信息）

---

### 2.3 忠诚度（Fidelity）

> **"大模型的回答，是基于检索到的文档还是自己瞎编的？"**

```
检索到的文档："A公司2023年营收199.7亿元，其中厨房电器占45%"

大模型回答 A："A公司2023年营收199.7亿元，厨房电器是主要收入来源。"
→ ✅ 忠诚度高（完全基于检索结果）

大模型回答 B："A公司2023年营收200亿元，主要产品是空调和冰箱。"
→ ❌ 忠诚度低（200亿≠199.7亿，空调冰箱不在检索结果中）
```

```python
def calculate_fidelity(query, retrieved_docs, llm_answer, llm_client):
    """
    忠诚度：LLM 的回答是否基于检索到的文档？
    做法：让另一个 LLM 判断回答中的关键信息是否能在检索文档中找到支撑。
    """
    docs_text = "\n\n".join(retrieved_docs)
    
    prompt = f"""
    问题：{query}
    
    检索到的参考文档：
    {docs_text}
    
    LLM 的回答：
    {llm_answer}
    
    请逐条判断回答中的每个关键信息是否能在检索文档中找到出处。
    如果所有关键信息都有出处，忠诚度为 1。
    如果有部分信息查不到出处，忠诚度按比例计算。
    
    只输出一个 0~1 之间的数字（如 0.85）："""
    
    fidelity_score = float(llm_client.chat(prompt).strip())
    return fidelity_score
```

**优化方向**：
- 在提示词中强调"严格按照检索到的内容回答"
- 确保检索到的数据本身是正确无误的
- 使用拒绝回答机制（如果检索不到，直接说不知道）

---

### 2.4 答案相关性（Answer Relevance）

> **"大模型的回答，跟用户问的问题相关吗？"**

```
用户问："A公司2023年的营收构成是什么？"

回答 A："A公司2023年营收199.7亿元，厨房电器占45%，个护电器占30%，生活电器占25%。"
→ ✅ 高度相关，直接回答用户问题

回答 B："A公司是一家小家电制造企业，成立于2005年..."
→ ❌ 低度相关，没有回答用户的核心问题
```

```python
def calculate_answer_relevance(query, llm_answer, llm_client):
    """
    答案相关性：回答是否解决了用户的问题？
    做法：让 LLM 评估回答与问题的匹配程度
    """
    prompt = f"""
    用户问题：{query}
    LLM 回答：{llm_answer}
    
    评估标准（0~1）：
    - 回答是否直接针对问题？
    - 回答是否覆盖了问题的核心？
    - 回答中是否有无关信息？
    
    只输出一个 0~1 之间的数字："""
    
    relevance_score = float(llm_client.chat(prompt).strip())
    return relevance_score
```

### 2.5 四个指标的关系

```
                     精确度（检索准不准）
                         ↑
                ┌─────────┴──────────┐
                │      检索质量       │
                │  (精确度 + 召回率)  │
                └─────────┬──────────┘
                         ↓
     ┌────────────────────┼────────────────────┐
     │                    │                    │
     ▼                    ▼                    ▼
  忠诚度             答案相关性            用户满意度
（基于检索吗？）    （回答到点了吗？）    （最终体验）
```

---

## 📋 三、完整的评估流程

### 3.1 准备评估数据集

```python
class RAGEvaluationDataset:
    """
    RAG 评估数据集
    
    每条数据包含：
    - question: 用户问题
    - ground_truth: 标准答案（人工标注）
    - relevant_doc_ids: 相关的文档 ID 列表
    - all_relevant_ids: 数据库中全部相关文档 ID（用于算召回率）
    """
    
    def __init__(self):
        self.examples = []
    
    def add_example(self, question, ground_truth, relevant_doc_ids, all_relevant_ids):
        self.examples.append({
            "question": question,
            "ground_truth": ground_truth,
            "relevant_doc_ids": relevant_doc_ids,
            "all_relevant_ids": all_relevant_ids
        })


# 构建评估数据集
eval_dataset = RAGEvaluationDataset()

eval_dataset.add_example(
    question="恐龙灭绝的原因是什么？",
    ground_truth="主流观点是小行星撞击地球导致恐龙灭绝。",
    relevant_doc_ids=["doc_001", "doc_003", "doc_007"],
    all_relevant_ids=["doc_001", "doc_003", "doc_005", "doc_007", "doc_010"]
)

eval_dataset.add_example(
    question="A公司的主要营收来源是什么？",
    ground_truth="厨房电器是A公司的主要营收来源，占总营收的45%。",
    relevant_doc_ids=["doc_101", "doc_103"],
    all_relevant_ids=["doc_101", "doc_103", "doc_105"]
)
```

### 3.2 执行评估

```python
def evaluate_rag_system(rag_system, eval_dataset, llm_client):
    """
    完整评估 RAG 系统
    """
    results = {
        "precision": [],
        "recall": [],
        "fidelity": [],
        "relevance": []
    }
    
    for example in eval_dataset.examples:
        question = example["question"]
        
        # Step 1: 让 RAG 系统回答
        answer, retrieved_doc_ids = rag_system.query(question)
        
        # Step 2: 计算四个指标
        precision = calculate_precision(retrieved_doc_ids, example["relevant_doc_ids"])
        recall = calculate_recall(retrieved_doc_ids, example["all_relevant_ids"])
        fidelity = calculate_fidelity(question, retrieved_doc_ids, answer, llm_client)
        relevance = calculate_answer_relevance(question, answer, llm_client)
        
        # Step 3: 记录结果
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["fidelity"].append(fidelity)
        results["relevance"].append(relevance)
        
        print(f"\n📝 问题: {question[:30]}...")
        print(f"  精确度: {precision:.2f} | 召回率: {recall:.2f}")
        print(f"  忠诚度: {fidelity:.2f} | 相关性: {relevance:.2f}")
    
    # Step 4: 计算平均值
    print("\n" + "=" * 50)
    print("📊 整体评估结果：")
    for metric, scores in results.items():
        avg_score = sum(scores) / len(scores)
        print(f"  {metric}: {avg_score:.2%}")
    
    return results
```

### 3.3 评估结果解读与优化方向

| 分数低的指标 | 可能的原因 | 优化方向 |
|------------|-----------|---------|
| **精确度低** | 检索器返回了太多无关文档 | 调整 top_k、加 RRF 重排、加元数据过滤 |
| **召回率低** | 应该搜到的文档没搜到 | 用混合检索、多问题查询、调整分块 |
| **忠诚度低** | 大模型喜欢自己编 | 强化提示词约束、数据本身质量差 |
| **相关性低** | 回答答非所问 | 优化提示词、检查检索到的内容是否准确 |

---

## 🛠️ 四、在线 RAG 应用平台简介

### 4.1 什么是 RAG 平台

RAG 平台是**可视化的工作流工具**，让你通过拖拽配置的方式搭建 RAG 应用，而不需要写代码。

```
这类平台通常提供：
├── 文档上传与管理
├── 向量数据库配置
├── 检索策略选择（向量/混合/多问题）
├── 大模型接入（OpenAI / 本地模型）
└── 可视化分析仪表盘
```

### 4.2 常见平台

| 平台 | 特点 | 适用场景 |
|------|------|---------|
| **Dify** | 开源、支持中文、工作流编排 | 个人/中小企业快速搭建 |
| **FastGPT** | 开源、专注知识库 | 企业知识库问答 |
| **LangSmith** | LangChain 官方、监控调试 | 开发者调试 RAG 流程 |
| **RAGFlow** | 开源、文档解析强 | 复杂文档处理场景 |

### 4.3 什么时候用平台，什么时候自己写

```
用平台（Dify/FastGPT）：
├── 快速验证想法
├── 非技术人员使用
├── 不需要深度定制
└── 快速上线 MVP

自己写代码：
├── 需要深度定制检索策略
├── 需要集成自己的数据管道
├── 对性能有极致要求
└── 需要和现有系统深度集成
```

---

## 📝 本章总结

### 核心公式

| 指标 | 公式 | 关注点 |
|------|------|--------|
| **精确度** | 检索到的相关文档 / 检索到的总文档 | **检索准不准** |
| **召回率** | 检索到的相关文档 / 全部相关文档 | **搜得全不全** |
| **忠诚度** | LLM 回答基于检索结果的程度 | **有没有瞎编** |
| **相关性** | LLM 回答与问题的匹配程度 | **答得对不对** |

### 一句话总结

> **没有评估的 RAG 系统就是"盲人摸象"——你不知道哪里好、哪里差。**  
> 精确度和召回率评估**检索**，忠诚度评估**回答的诚实度**，相关性评估**最终效果**。🚀
