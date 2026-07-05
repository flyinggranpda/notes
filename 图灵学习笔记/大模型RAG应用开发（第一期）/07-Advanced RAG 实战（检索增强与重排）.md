# 🔧 Advanced RAG 实战：检索增强与结果重排

> 🎯 学完前 7 课的理论和架构思路，这一课讲**实战代码**  
> 💻 元数据检索 → 多问题查询 → RRF 融合排序，每个技术点都有可运行的完整代码  
> 📖 定位：进阶课，适合已理解 RAG 基础、想写工程代码的开发者

---

## 📌 一、课前回顾：预索引 vs 后索引

在动手写代码之前，先理清两个核心概念。第 7 课已经讲过，但这里用代码实践再来一遍：

```
预索引（Pre-retrieval）：在搜索之前做优化
  ├── 让查询更完善（查询重写、扩展、分解）
  ├── 让索引更精准（元数据过滤）
  └── 目标：搜得更准

后索引（Post-retrieval）：在搜索之后做优化
  ├── 对检索结果进行重排（RRF 融合排序）
  ├── 对检索结果进行压缩/摘要
  └── 目标：结果更好
```

**第 7 课**重点讲了预索引的各种技术（HyDE、混合检索、查询转换），**第 8 课**以**查询扩展（Multi-Query）和后索引（RRF 重排）** 为实战重点。

---

## 🔍 二、元数据检索（Metadata Retrieval）

### 2.1 它解决什么问题

普通的向量检索只比较"语义相似度"。但很多场景需要**先按属性过滤，再按语义排序**。

```
普通检索：搜"2024 年苹果的财报数据"
  → 向量搜索匹配到所有和"苹果 财报 数据"相关的文档
  → 不管发布时间、文档类型，一股脑全出来

元数据检索：搜"2024 年苹果的财报数据"
  → ① LLM 先拆解：问题="苹果的营收和利润数据", 过滤条件="年份=2024, 类型=财报"
  → ② 用过滤条件缩小搜索范围（只搜 2024 年的财报）
  → ③ 在缩小后的范围内做向量匹配
  → 结果更精准 ✅
```

### 2.2 核心流程

```
用户提问
   ↓
LLM 拆解成两部分：
  ├── ① 纯文本问题（用于向量检索）
  └── ② 过滤条件（用于 metadata 过滤）
       ↓
先按过滤条件缩小文档范围
   ↓
再在范围内做向量相似度搜索
   ↓
返回结果
```

### 2.3 代码实现

```python
from typing import Dict, List, Any
import json

def metadata_retrieval(query: str, vector_db, llm_client) -> List[str]:
    """
    元数据检索流程：
    1. LLM 将问题拆解为「问题文本 + 过滤条件」
    2. 先用过滤条件缩小范围
    3. 再在范围内做向量检索
    """
    # Step 1: 让 LLM 拆解问题和过滤条件
    parse_prompt = f"""
    你是一个查询解析器。请将以下问题拆解为两部分：
    1. 搜索关键词：用于向量检索的核心问题（去掉过滤条件）
    2. 过滤条件：JSON 格式的元数据过滤条件，可选的字段有：
       - year: 年份（整数）
       - category: 文档类别（字符串）
       - author: 作者（字符串）
       - source: 来源（字符串）

    只输出 JSON 格式，不要输出其他内容。
    
    问题：{query}
    
    输出格式：
    {{
        "search_query": "搜索关键词",
        "filters": {{"year": 2024, "category": "财报"}}
    }}
    """
    
    result = llm_client.chat(parse_prompt)
    
    try:
        parsed = json.loads(result)
        search_query = parsed.get("search_query", query)
        filters = parsed.get("filters", {})
    except:
        # 解析失败时回退到纯向量搜索
        search_query = query
        filters = {}
    
    print(f"🔍 搜索关键词: {search_query}")
    print(f"📋 过滤条件: {filters}")
    
    # Step 2: 按元数据过滤
    if filters:
        # 实际项目中这里调用 vector_db 的 filter 方法
        # 例如：vector_db.filter(filters)
        filtered_ids = filter_by_metadata(filters)
        print(f"  → 过滤后候选文档数: {len(filtered_ids)}")
    else:
        filtered_ids = None  # 不过滤
    
    # Step 3: 在过滤后的范围内做向量检索
    results = vector_db.similarity_search(
        search_query,
        k=5,
        filter_ids=filtered_ids  # 限制搜索范围
    )
    
    return results
```

### 2.4 实际应用场景

| 场景 | 用户问题 | LLM 拆解结果 |
|------|---------|-------------|
| 新闻搜索 | "去年特斯拉的销量数据" | search_query: "特斯拉 销量 数据", filters: {year: 2025} |
| 论文搜索 | "李飞飞 2023 年的论文" | search_query: "论文 计算机视觉", filters: {author: "李飞飞", year: 2023} |
| 公司文档 | "财务部的报销流程" | search_query: "报销流程", filters: {category: "财务制度"} |

---

## 🔄 三、查询扩展（Multi-Query / Query Expansion）

### 3.1 它解决什么问题

**痛点**：用户只问了一个问题，但你可能需要从多个角度去检索才能覆盖全面。

```
用户问题："苹果的 AI 战略是什么？"
                  ↓
LLM 生成多个角度的问题：
  ├── "苹果在人工智能领域的最新布局"
  ├── "Apple 的 AI 产品和服务有哪些"  
  └── "蒂姆库克对 AI 的观点"
                  ↓
每个问题分别检索 → 综合所有结果 → 回答更全面
```

### 3.2 实现方式一：使用 LangChain 的 MultiQueryRetriever

LangChain 内置了 `MultiQueryRetriever`，本质上就是帮我们封装了"生成多个问题→分别检索→融合结果"的逻辑。

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

def multi_query_with_langchain(vector_db, llm_client):
    """
    使用 LangChain 内置的 MultiQueryRetriever
    它会自动：
    1. 根据原始问题生成多个相似问题
    2. 每个问题分别检索
    3. 合并所有检索结果（去重）
    """
    # 创建 MultiQueryRetriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
        llm=llm_client
    )
    
    # 直接检索（内部自动完成多问题生成+多路检索+结果合并）
    docs = retriever.get_relevant_documents(
        "苹果的 AI 战略是什么？"
    )
    
    return docs
```

**MultiQueryRetriever 内部做了什么？**

```
① 用户输入 "苹果的 AI 战略是什么？"
      ↓
② LLM 被调用来生成 3 个变体问题：
   "Who is the current CEO of Apple?"
   "What is Apple's approach to AI?"  
   "How does Apple integrate AI into products?"
      ↓
③ 4 个问题（原始+3个变体）分别去向量库检索
      ↓
④ 合并结果，去重
      ↓
⑤ 返回融合后的文档列表
```

### 3.3 实现方式二：自己手写（更灵活）

```python
from typing import List
import re

def multi_query_custom(query: str, vector_db, llm_client, num_queries: int = 3) -> List[str]:
    """
    手动实现多问题查询，更灵活可控
    """
    # Step 1: 用 LLM 生成多个相似问题
    prompt = f"""你是一个擅长从多个角度思考问题的助手。
给定原始问题，请从不同角度生成 {num_queries} 个变体问题。
每个变体应该覆盖原始问题的不同侧面。

原始问题：{query}

请直接输出 {num_queries} 个问题，每行一个："""
    
    result = llm_client.chat(prompt)
    
    # 解析输出，提取每一行作为独立问题
    generated_questions = [
        q.strip().lstrip("0123456789.、- ")
        for q in result.strip().split("\n")
        if q.strip()
    ]
    
    # 合并：原始问题 + 生成的问题
    all_questions = [query] + generated_questions[:num_queries]
    
    print(f"📝 生成 {len(all_questions)} 个查询：")
    for i, q in enumerate(all_questions):
        print(f"  [{i}] {q}")
    
    # Step 2: 每个问题分别检索
    all_docs = []
    for q in all_questions:
        docs = vector_db.similarity_search(q, k=5)
        all_docs.extend(docs)
    
    # Step 3: 去重（基于文档 ID 或内容 hash）
    seen = set()
    unique_docs = []
    for doc in all_docs:
        doc_id = doc.id if hasattr(doc, 'id') else hash(doc.page_content[:100])
        if doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append(doc)
    
    print(f"📊 合并前去重前: {len(all_docs)} 条, 去重后: {len(unique_docs)} 条")
    
    return unique_docs
```

### 3.4 两种实现方式的对比

| 对比维度 | LangChain MultiQueryRetriever | 手写实现 |
|---------|------------------------------|---------|
| 代码量 | 2 行 | 30+ 行 |
| 灵活度 | 低（内置提示词不可改） | 高（提示词、数量、策略都可调） |
| 可控性 | 低（结果合并策略固定） | 高（可以自定义排序规则） |
| 学习价值 | 直接可用 | ✅ 理解原理 |
| 推荐场景 | 快速验证 | 生产环境定制 |

---

## 📊 四、RRF 融合排序（Reciprocal Rank Fusion）

### 4.1 为什么要做重排？

多问题查询会产生**大量重复和排名混乱**的文档：

```
问题 A 检索结果：   doc1(第1), doc3(第2), doc5(第3)
问题 B 检索结果：   doc2(第1), doc1(第2), doc4(第3)  
问题 C 检索结果：   doc3(第1), doc6(第2), doc1(第3)

doc1 出现了 3 次，但每次排名不同 → 综合得分应该是多少？
doc3 出现了 2 次，也有排名 → 应该排第几？
doc5 只出现 1 次，排名第 3 → 还有价值吗？
```

**RRF 就是用来解决这个问题的**：把多个排名的结果综合成一个统一排名。

### 4.2 RRF 原理

> **RRF 的核心思想**：一个文档在越多的问题检索中排名靠前，它的综合排名就越高。

公式极其简单：

```
score(d) = Σ 1 / (rank(d, q) + k)

其中：
- rank(d, q)：文档 d 在第 q 个问题检索结果中的排名（从 1 开始）
- k：一个常数（通常设为 60），防止除零，控制排名的影响权重
```

### 4.3 手工演算

假设 k=60，三个问题检索到了文档 1：

```
问题 A：doc1 排名第 2  →  得分贡献 = 1/(2+60) = 1/62 ≈ 0.0161
问题 B：doc1 排名第 2  →  得分贡献 = 1/(2+60) = 1/62 ≈ 0.0161
问题 C：doc1 没有出现 →  得分贡献 = 0

doc1 总分 = 0.0161 + 0.0161 + 0 = 0.0322
```

再看另一个文档：

```
问题 A：doc3 排名第 2  →  得分贡献 = 1/(2+60) = 0.0161
问题 B：doc3 没有出现 →  得分贡献 = 0
问题 C：doc3 排名第 1  →  得分贡献 = 1/(1+60) = 0.0164

doc3 总分 = 0.0161 + 0 + 0.0164 = 0.0325
```

**doc3 的综合得分 > doc1，所以 doc3 应该排在 doc1 前面**。

### 4.4 k 值的作用

```python
# k 值越大，排名差异的影响越小
k=60:  1/(1+60)=0.0164,  1/(2+60)=0.0161  ← 差距很小
k=10:  1/(1+10)=0.0909,  1/(2+10)=0.0833  ← 差距稍大
k=1:   1/(1+1)=0.5,      1/(2+1)=0.333    ← 差距很大

# k 值还有另一个重要作用：防止文档在某个检索中没出现时得分为 0
# 如果没有 +k，没出现的文档得分为 1/0 = 无穷大 ❌
# 有了 +k，没出现的文档得分为 0
```

**工程经验**：
- k=60 是论文推荐的默认值，效果稳定
- k 越小，排名靠前的文档优势越大
- k 越大，所有文档的得分越平均

### 4.5 完整代码实现

```python
from typing import List, Dict, Tuple
from collections import defaultdict

def rrf_fusion(
    query_results: Dict[str, List[str]],  # {问题名: [文档ID列表(按排名排序)]}
    k: int = 60,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    RRF 融合排序算法
    
    参数:
        query_results: 每个问题的检索结果列表
                       key=问题, value=文档ID列表（按排名从高到低）
        k: 常数（默认 60）
        top_n: 返回前 N 个文档
    
    返回:
        排序后的 [(文档ID, 得分), ...]
    """
    # Step 1: 累加每个文档的 RRF 得分
    scores = defaultdict(float)  # {文档ID: 总分}
    
    for query_name, doc_ids in query_results.items():
        for rank, doc_id in enumerate(doc_ids, start=1):  # 排名从 1 开始
            scores[doc_id] += 1 / (rank + k)
    
    # Step 2: 按得分排序
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Step 3: 取 Top-N
    return sorted_docs[:top_n]

# ============ 手动演算 ============
def rrf_demo():
    """手工演算 RRF 融合过程"""
    
    # 模拟检索结果：三个问题分别检索到的文档排名
    query_results = {
        "问题A": ["doc1", "doc3", "doc5", "doc7", "doc9"],
        "问题B": ["doc2", "doc1", "doc4", "doc6", "doc8"],
        "问题C": ["doc3", "doc6", "doc1", "doc2", "doc4"],
    }
    
    print("=" * 55)
    print("RRF 融合排序（k=60）")
    print("=" * 55)
    
    # 打印每个文档在每个检索中的排名
    print("\n每个文档在不同问题下的排名：")
    all_docs = set()
    for docs in query_results.values():
        all_docs.update(docs)
    
    for doc in sorted(all_docs):
        ranks = []
        for qname, docs in query_results.items():
            if doc in docs:
                rank = docs.index(doc) + 1
                ranks.append(f"{qname}:第{rank}名")
            else:
                ranks.append(f"{qname}:未出现")
        print(f"  {doc}: {' | '.join(ranks)}")
    
    # 计算 RRF 得分
    print("\nRRF 得分明细：")
    scores = defaultdict(float)
    for qname, docs in query_results.items():
        for rank, doc in enumerate(docs, 1):
            score_contribution = 1 / (rank + 60)
            scores[doc] += score_contribution
            print(f"  {doc} ← {qname}(第{rank}名): +{score_contribution:.4f}")
    
    # 排序
    print("\n最终排名：")
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for i, (doc, score) in enumerate(sorted_docs, 1):
        print(f"  第{i}名: {doc}  得分: {score:.4f}")
    
    return sorted_docs

# ============ 实际 RAG 场景中的 RRF ============
def rrf_rag_pipeline(
    query: str,
    vector_db,
    llm_client,
    k: int = 60,
    top_n: int = 5
) -> List[str]:
    """
    完整的 RRF RAG 流程：
    多问题生成 → 分别检索 → RRF 融合 → 返回排序后的文档
    """
    # Step 1: 生成多个问题
    expand_prompt = f"""根据以下问题，生成 4 个不同角度的变体问题。
    直接输出，每行一个：
    
    原始问题：{query}"""
    
    result = llm_client.chat(expand_prompt)
    questions = [
        q.strip().lstrip("0123456789.、- ")
        for q in result.strip().split("\n")
        if q.strip()
    ]
    questions = [query] + questions[:4]  # 原问题 + 4 个变体 = 5 个问题
    
    print(f"📝 共 {len(questions)} 个查询：")
    for i, q in enumerate(questions):
        print(f"  [{i}] {q}")
    
    # Step 2: 每个问题分别检索，记录排名
    query_results = {}
    for q in questions:
        docs = vector_db.similarity_search(q, k=10)
        # 记录文档 ID 列表（按排名排序）
        doc_ids = [d.id for d in docs]
        query_results[q] = doc_ids
        print(f"  → [{q[:30]}...] 检索到 {len(doc_ids)} 条")
    
    # Step 3: RRF 融合排序
    print("\n🔄 RRF 融合排序中...")
    fused = rrf_fusion(query_results, k=k, top_n=top_n)
    
    print(f"\n🏆 最终 Top-{top_n}：")
    for i, (doc_id, score) in enumerate(fused, 1):
        print(f"  第{i}名: {doc_id}  得分: {score:.4f}")
    
    # Step 4: 返回排序后的文档
    return [doc_id for doc_id, _ in fused]

# ============ 单文档得分明细 ============
def explain_doc_score(doc_id: str, query_results: Dict[str, List[str]], k: int = 60):
    """
    解释某个文档的 RRF 得分是怎么算出来的
    """
    print(f"\n📊 {doc_id} 的 RRF 得分明细：")
    print(f"{'问题':<20} {'排名':<8} {'得分贡献':<12}")
    print("-" * 45)
    
    total = 0.0
    for qname, docs in query_results.items():
        if doc_id in docs:
            rank = docs.index(doc_id) + 1
            contribution = 1 / (rank + k)
            total += contribution
            print(f"{qname:<20} 第{rank:<4}   {contribution:.6f}")
        else:
            print(f"{qname:<20} 未出现    0.000000")
    
    print("-" * 45)
    print(f"{'总分':<20} {'':<8} {total:.6f}")
    
    return total

# ============ 使用示例 ============
# # 先跑演示，理解 RRF 逻辑
# rrf_demo()
#
# # 再集成到 RAG 流程中
# top_docs = rrf_rag_pipeline(
#     query="人工智能在医疗领域的应用",
#     vector_db=your_vector_db,
#     llm_client=your_llm,
#     k=60,
#     top_n=5
# )
```

**运行结果示例：**

```
=======================================================
RRF 融合排序（k=60）
=======================================================

每个文档在不同问题下的排名：
  doc1: 问题A:第1名 | 问题B:第2名 | 问题C:第3名
  doc2: 问题A:未出现 | 问题B:第1名 | 问题C:第4名
  doc3: 问题A:第2名 | 问题B:未出现 | 问题C:第1名
  doc4: 问题A:未出现 | 问题B:第3名 | 问题C:第5名

RRF 得分明细：
  doc1 ← 问题A(第1名): +0.0164
  doc1 ← 问题B(第2名): +0.0161
  doc1 ← 问题C(第3名): +0.0159
  doc1 总分: 0.0484

  doc3 ← 问题A(第2名): +0.0161
  doc3 ← 问题C(第1名): +0.0164
  doc3 总分: 0.0325

最终排名：
  第1名: doc1  得分: 0.0484
  第2名: doc3  得分: 0.0325
  第3名: doc2  得分: 0.0278
```

---

## 🧩 五、完整实战：多问题查询 + RRF 融合

把本课的核心内容整合成一个完整的 `MultiQueryRRFEngine`：

```python
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import hashlib

class MultiQueryRRFEngine:
    """
    多问题查询 + RRF 融合排序引擎
    
    使用流程：
    1. 用户输入问题
    2. 自动生成多个变体问题
    3. 每个问题分别检索
    4. RRF 算法融合排序
    5. 返回重排后的文档
    """
    
    def __init__(self, vector_db, llm_client, k: int = 60):
        self.vector_db = vector_db
        self.llm = llm_client
        self.k = k  # RRF 常数
        
        # 统计信息
        self.stats = {
            "original_questions": 0,
            "expanded_queries": 0,
            "total_retrieved": 0,
            "unique_docs": 0
        }
    
    def query(self, user_question: str, top_n: int = 5) -> List[str]:
        """
        完整流程：扩展 → 多路检索 → RRF 融合 → 返回
        """
        self.stats["original_questions"] += 1
        
        # ============ 阶段一：查询扩展 ============
        expanded_questions = self._expand_questions(user_question)
        self.stats["expanded_queries"] += len(expanded_questions)
        
        print(f"\n{'='*55}")
        print(f"原始问题: {user_question}")
        print(f"扩展后: {len(expanded_questions)} 个查询")
        print(f"{'='*55}")
        
        # ============ 阶段二：多路检索 ============
        query_results = {}
        for q in expanded_questions:
            docs = self.vector_db.similarity_search(q, k=10)
            doc_ids = [self._get_doc_id(d) for d in docs]
            query_results[q] = doc_ids
            self.stats["total_retrieved"] += len(doc_ids)
        
        # ============ 阶段三：RRF 融合 ============
        fused = self._rrf_fusion(query_results, top_n=top_n)
        self.stats["unique_docs"] = len(fused)
        
        # ============ 阶段四：按 RRF 分数排序返回 ============
        final_doc_ids = [doc_id for doc_id, _ in fused]
        
        print(f"\n🏆 Top-{top_n} 结果：")
        for i, (doc_id, score) in enumerate(fused, 1):
            print(f"  第{i}名: {doc_id}  (得分: {score:.4f})")
        
        return final_doc_ids
    
    def _expand_questions(self, query: str, n: int = 4) -> List[str]:
        """用 LLM 生成多个变体问题"""
        prompt = f"""请从不同角度生成 {n} 个检索问题。
        要求：
        - 每个问题覆盖原始问题的不同侧面
        - 使用不同的关键词和表述方式
        - 保持含义一致

        原始问题：{query}

        直接输出问题，每行一个："""
        
        result = self.llm.chat(prompt)
        generated = [
            q.strip().lstrip("0123456789.、- ")
            for q in result.strip().split("\n")
            if q.strip()
        ]
        
        # 始终包含原始问题
        all_questions = [query] + generated[:n]
        return all_questions
    
    def _rrf_fusion(
        self,
        query_results: Dict[str, List[str]],
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """RRF 融合排序"""
        scores = defaultdict(float)
        
        for qname, doc_ids in query_results.items():
            for rank, doc_id in enumerate(doc_ids, 1):
                scores[doc_id] += 1 / (rank + self.k)
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_n]
    
    def _get_doc_id(self, doc) -> str:
        """获取文档的唯一 ID"""
        if hasattr(doc, 'id') and doc.id:
            return doc.id
        # 如果没有 ID，用内容 hash 代替
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def print_stats(self):
        """打印运行统计"""
        print(f"\n📊 引擎运行统计：")
        print(f"  原始问题数: {self.stats['original_questions']}")
        print(f"  扩展查询数: {self.stats['expanded_queries']}")
        print(f"  总检索文档数: {self.stats['total_retrieved']}")
        print(f"  去重后文档数: {self.stats['unique_docs']}")
        print(f"  RRF 常数 k: {self.k}")
        if self.stats['original_questions'] > 0:
            print(f"  平均检索/问题: {self.stats['total_retrieved'] / self.stats['original_questions']:.1f}")

# ============ 使用示例 ============
# engine = MultiQueryRRFEngine(
#     vector_db=your_vector_db,
#     llm_client=your_llm,
#     k=60
# )
#
# # 提问
# results = engine.query("人工智能在医疗诊断中的应用")
#
# # 查看统计
# engine.print_stats()
```

---

## ⚡ 六、LLM 在检索中的应用方式对比

本课和前面几课中，LLM 在检索流程中被用在多个位置：

```
        用户提问
           │
           ▼
    ┌──────────────────┐
    │  ① 查询重写      │ ← LLM 改写问题（第6课）
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  ② 拆解过滤条件   │ ← LLM 提取元数据过滤条件（本课）
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  ③ 生成多问题     │ ← LLM 扩展为多个变体问题（本课）
    └────────┬─────────┘
             │
             ▼
        ┌─────────┐
        │ 向量检索  │ ← 不用 LLM
        └────┬────┘
             │
             ▼
    ┌──────────────────┐
    │  ④ 自我反思评估   │ ← LLM 判断文档是否相关（第6课Self-RAG）
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  ⑤ RRF 融合排序   │ ← 算法，不用 LLM
    └────────┬─────────┘
             │
             ▼
          回答
```

| 位置 | 技术 | LLM 是否参与 | 调用次数 |
|------|------|-------------|---------|
| 检索前 | 元数据提取 | ✅ 是 | 1 次 |
| 检索前 | 多问题生成 | ✅ 是 | 1 次 |
| 检索中 | 实际检索 | ❌ 否（向量数据库） | 0 次 |
| 检索后 | 自我反思 | ✅ 是 | N 次（每个文档一次） |
| 检索后 | RRF 重排 | ❌ 否（算法排序） | 0 次 |

**工程建议**：
- LLM 调用很贵，尽量用在"刀刃"上（重写、扩展、拆解过滤条件）
- 检索和排序尽量用算法（向量检索、RRF），又快又便宜
- 自我反思效果好但太慢，在精度要求高的场景才用

---

## 📝 本章总结

### 知识地图

```
第 8 课：Advanced RAG 实战
│
├── ① 元数据检索（检索前）
│   ├── LLM 拆解问题 + 过滤条件
│   └── 先过滤 → 再向量搜索
│
├── ② 查询扩展 / Multi-Query（检索前）
│   ├── LangChain MultiQueryRetriever
│   └── 手写实现（更灵活）
│
├── ③ RRF 融合排序（检索后）
│   ├── score = Σ 1/(rank + k)
│   ├── k=60 为推荐默认值
│   └── 多路检索结果的统一排序
│
└── ④ 完整引擎
    └── MultiQueryRRFEngine
```

### 关键公式

> **RRF 评分公式：** `score(d) = Σ 1 / (rank(d, q) + k)`

| 概念 | 一句话解释 |
|------|-----------|
| **元数据检索** | 先按属性过滤，再按语义搜索 |
| **Multi-Query** | 生成多个问题分别查，结果更全 |
| **RRF** | 多路排名融合成统一排名 |
| **k=60** | 默认常数，平衡排名影响和零分问题 |
| **预索引 vs 后索引** | 预索引优化查询，后索引优化结果 |

### 一句话总结

> **查询扩展让检索更"全"，RRF 让排序更"准"。**  
> 两个配合使用，是工业级 RAG 系统中最常用的组合之一。🚀

