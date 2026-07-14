# 🔧 Advanced RAG 技术实现：从架构到代码

> 📝 **文档摘要：** 本文详细讲解 Advanced RAG 各项技术的具体代码实现。涵盖文档摘要索引（对文档生成摘要后用摘要做检索）、HyDE（假设性文档嵌入——先生成假设答案再用于检索，缩小语义鸿沟）、混合检索（向量检索 + BM25 关键词检索加权融合，附完整 BM25 公式推导和 alpha 调参指南）、查询转换（查询重写/查询分解/Step-Back 后退一步 prompting）、RAPTOR 递归文档树（自底向上构建多层树结构，支持分层检索）、以及三种高级分块策略（固定大小/语义分块/代理式分块）。最后将这些技术整合成一个完整的 AdvancedRAGEngine 综合引擎，并提供常见组合模式和避坑指南。

> 🎯 上一章讲了"有哪些高级架构思路"，这一章讲"具体怎么实现"  
> 💻 每个技术点都有完整可运行的代码  
> 📖 适合人群：已理解 RAG 基本架构，想知道具体怎么写的开发者

---

## 📌 一、从第六章到第七章

### 1.1 第六章我们学了什么

上一章介绍了 **5 种高级 RAG 架构思路**（T-RAG、C-RAG、Self-RAG、Fusion RAG、Rewrite RAG），这些是"设计思想"层面的东西。

### 1.2 第七章学什么

这一章聚焦 **具体的技术实现手段**：

```
第六章（思路层）             第七章（技术层）
─────────────────           ─────────────────
T-RAG（拆解问题）            → 查询分解（Query Decomposition）
C-RAG（纠正检索）            → 检索评估 + 混合检索
Self-RAG（自我反思）         → 文档过滤 + 重排序
Fusion RAG（多问法融合）     → HyDE + 混合检索
Rewrite RAG（重写问题）      → 查询转换（Query Translation）

还有更多：
                             文件摘要索引（Summarization Index）
                             递归文档树（RAPTOR）
                             高级分块策略（Chunking）
```

> **一句话总结：第六章是"用什么武器"，第七章是"武器怎么造"。**

---

## 🔍 二、文档摘要索引（Document Summary Index）

### 2.1 它解决什么问题

**痛点**：一篇长文档有 10 个段落，embedding 时只能把每个段落单独编码。但用户问的问题可能涉及整篇文档的主题，单个段落很难检索到。

**打个比方**：

```
一本书的目录 vs 某一页的内容
├── 目录（摘要）：让你一眼知道这本书讲什么  ← 我们想要这个
└── 第 137 页（单一段落）：太细了，找不到  ← 普通 chunk 的问题
```

### 2.2 核心思想

```
文档入库时：
  文档 → 生成摘要（1-2 句话） → 用摘要做索引
                                ↓
用户提问时：
  问题 → 向量检索摘要 → 匹配到文档 → 返回完整文档做上下文
```

### 2.3 代码实现

```python
from typing import List, Dict
import numpy as np

class DocumentSummaryIndex:
    """
    文档摘要索引：先对每个文档生成摘要，用摘要做检索。
    匹配到摘要后，返回对应完整文档作为 LLM 的上下文。
    """
    
    def __init__(self, llm_client, embedding_model):
        self.llm = llm_client
        self.embed = embedding_model
        self.documents: List[Dict] = []  # 存储 {id, content, summary, summary_embedding}
    
    def add_document(self, doc_id: str, content: str):
        """添加一篇文档，自动生成摘要并嵌入"""
        # Step 1: 生成摘要
        summary = self.llm.chat(f"用一句话概括以下内容的要点：\n{content}")
        
        # Step 2: 嵌入摘要（存向量）
        summary_embedding = self.embed.embed(summary)
        
        # Step 3: 存储
        self.documents.append({
            "id": doc_id,
            "content": content,
            "summary": summary,
            "embedding": summary_embedding
        })
        
        print(f"✅ 已添加文档 [{doc_id}]")
        print(f"   摘要: {summary}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """用问题检索摘要，返回匹配的完整文档"""
        query_embedding = self.embed.embed(query)
        
        # 计算所有摘要与查询的相似度
        scores = []
        for doc in self.documents:
            sim = self._cosine_similarity(query_embedding, doc["embedding"])
            scores.append((sim, doc))
        
        # 取 Top-K
        scores.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scores[:top_k]]
        
        return top_docs
    
    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

# ============ 使用示例 ============
# index = DocumentSummaryIndex(llm_client=your_llm, embedding_model=your_embedder)
#
# # 入库
# index.add_document("doc_001", "这是一篇关于...的长文档内容...")
# index.add_document("doc_002", "这是另一篇关于...的文档...")
#
# # 检索
# results = index.search("用户的问题是什么？")
# for doc in results:
#     print(f"匹配文档: {doc['id']}，摘要: {doc['summary']}")
```

### 2.4 优缺点

| 优点 | 缺点 |
|------|------|
| 检索速度快（摘要短） | 多一次 LLM 调用（生成摘要） |
| 能匹配全局主题 | 摘要质量决定检索上限 |
| 返回内容完整 | 不适合细粒度检索 |

---

## 🧠 三、HyDE（假设性文档嵌入）

### 3.1 它解决什么问题

**痛点**：用户提问的方式跟文档的写法不一样。

```
用户问: "iPhone 15 发热严重吗？"         ← 口语化
文档写: "A17 Pro 芯片在持续高负载下..."   ← 书面化
                                      ← 语义差距大！embedding 匹配差
```

### 3.2 核心思想

> **先让 LLM 根据问题生成一个"假设答案"（假文档），再用这个假文档去向量库检索。**

```
用户问题: "iPhone 15 发热严重吗？"
     ↓
LLM 生成假设答案: "iPhone 15 用户反映发热问题主要集中在..."
     ↓
用这个"假设答案"去索引里做向量检索（而不是用原始问题）
     ↓
找到真正的相关文档 → 生成最终答案
```

**为什么这样有用？**

因为"假设答案"在风格上更接近文档的写法，embedding 相似度更高。

### 3.3 代码实现

```python
class HyDE:
    """
    Hypothetical Document Embeddings
    先假设一个答案，再用这个答案去检索（而不是用问题去检索）
    """
    
    def __init__(self, vector_db, llm_client):
        self.vector_db = vector_db
        self.llm = llm_client
    
    def retrieve(self, query: str, top_k: int = 5):
        # Step 1: 生成假设文档
        hypothetical_doc = self._generate_hypothetical_doc(query)
        print(f"📝 假设文档: {hypothetical_doc[:100]}...")
        
        # Step 2: 用假设文档（而不是原始问题）做检索
        results = self.vector_db.similarity_search(hypothetical_doc, k=top_k)
        
        return results
    
    def _generate_hypothetical_doc(self, query: str) -> str:
        """根据问题生成一个假设的回答文档"""
        prompt = f"""请根据以下问题，写一段假设性的回答。风格要像一篇技术文档或百科条目。
        不需要考虑回答是否正确，重点是写得像一篇真实的文档内容。

        问题：{query}
        
        假设性回答："""
        
        return self.llm.chat(prompt)
    
    def query(self, user_question: str) -> str:
        """完整的 HyDE 流程：假设 → 检索 → 回答"""
        # ① 生成假设文档
        hypo_doc = self._generate_hypothetical_doc(user_question)
        
        # ② 用假设文档检索真实文档
        real_docs = self.vector_db.similarity_search(hypo_doc, k=5)
        
        # ③ 用检索到的真实文档回答
        context = "\n\n".join([d.page_content for d in real_docs])
        final_prompt = f"""基于以下参考内容回答问题：

参考内容：
{context}

问题：{user_question}

回答："""
        
        return self.llm.chat(final_prompt)

# ============ 使用示例 ============
# hyde = HyDE(vector_db=your_vector_db, llm_client=your_llm)
# # 用户问题很口语化，但文档很书面化
# answer = hyde.query("苹果手机发热厉害不？")
# print(answer)
```

### 3.4 HyDE 与 Fusion RAG 的区别

> **Fusion RAG**：生成多个"问法"去检索 → **重在问题多样化**  
> **HyDE**：生成一个"假设答案"去检索 → **重在缩小语义鸿沟**

| 对比 | Fusion RAG | HyDE |
|------|-----------|------|
| 生成什么 | 多个相似问题 | 一个假设答案 |
| 检索依据 | 用多个问题分别检索 | 用假设答案向量检索 |
| 核心目的 | 提高召回率 | 缩短语义差距 |
| 适用场景 | 用户问题太简略 | 用户问题和文档风格不匹配 |

---

## 🔀 四、混合检索（Hybrid Search）

### 4.1 它解决什么问题

**痛点**：纯向量检索（语义搜索）在精确匹配上很弱。

```
向量检索擅长：
  "苹果公司的创始人是谁？" → 匹配到 "Steve Jobs 创立了 Apple"  ✅ （语义对得上）

向量检索不擅长：
  "帮我查 A100-80-SXM 的规格" → 匹配不到 "A100-80-SXM"  ❌ （精确术语）
  "邮箱是 user@example.com"   → 匹配不到 user@example.com  ❌ （精确字符串）
```

**换句话说**：向量检索看"意思"，关键词检索（BM25）看"字面"。

### 4.2 核心思想

```
混合检索 = 向量检索（语义） + 关键词检索（字面）
               ↓                    ↓
           得到分数1             得到分数2
               ↓                    ↓
         ┌─────┴───── 加权融合 ─────┴─────┐
         ↓                                 ↓
     最终结果                     (两个都强才是真的强)
```

### 4.3 BM25 是什么（简单理解）

BM25 是关键词检索的"黄金标准"。它的核心：

1. **词频（TF）**：一个词在文档中出现越多 → 分数越高（但有上限，防止刷词）
2. **逆文档频率（IDF）**：一个词在很多文档中都出现 → 分数越低（比如 "的"、"是" 这种词）
3. **文档长度归一化**：长文档中的关键词含金量更高

```
BM25 大概的逻辑（不必背公式，理解含义即可）：

分数 ≈ 词频 × 逆文档频率 × 长度归一化

"苹果" 在文档 A 中出现 5 次，在整个库中只有 10% 的文档包含这个词
→ 分数高 ✅

"的" 在文档 A 中出现 50 次，在整个库中 100% 的文档包含
→ 分数低 ❌（说明这是个没用的词）
```

### 4.4 代码实现

```python
from typing import List, Tuple
import numpy as np
import math
from collections import Counter

class HybridRetriever:
    """
    混合检索：向量检索（语义）+ BM25（关键词）加权融合
    """
    
    def __init__(self, vector_db, embedding_model, alpha=0.5):
        """
        alpha: 向量检索的权重（0~1）
               alpha=1  → 纯向量检索
               alpha=0  → 纯 BM25 关键词检索
               alpha=0.5 → 各占一半
        """
        self.vector_db = vector_db
        self.embed = embedding_model
        self.alpha = alpha
        
        # BM25 需要的统计信息
        self.documents = []                 # 所有文档列表
        self.avg_doc_length = 0             # 平均文档长度
        self.doc_freq = Counter()           # 每个词在多少个文档中出现
        self.total_docs = 0                 # 文档总数
    
    def add_documents(self, docs: List[str]):
        """添加文档，同时构建 BM25 统计"""
        self.documents = docs
        self.total_docs = len(docs)
        
        # 统计每个词在多少个文档中出现
        for doc in docs:
            words = set(doc.lower().split())
            for word in words:
                self.doc_freq[word] += 1
        
        # 计算平均文档长度
        total_length = sum(len(doc.split()) for doc in docs)
        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 1
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """混合检索：向量得分 + BM25 得分的加权融合"""
        
        # 1️⃣ 向量检索得分
        vec_scores = self._vector_search(query)
        
        # 2️⃣ BM25 关键词得分
        bm25_scores = self._bm25_search(query)
        
        # 3️⃣ 归一化（把分数映射到 0~1 范围）
        vec_scores = self._normalize(vec_scores)
        bm25_scores = self._normalize(bm25_scores)
        
        # 4️⃣ 加权融合
        final_scores = []
        for i in range(self.total_docs):
            combined = self.alpha * vec_scores[i] + (1 - self.alpha) * bm25_scores[i]
            final_scores.append((self.documents[i], combined))
        
        # 5️⃣ 按融合分数排序，取 Top-K
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:top_k]
    
    def _vector_search(self, query: str) -> List[float]:
        """向量检索（实际调用 vector_db，这里用简化模拟）"""
        # 实际项目中这里调用 vector_db.similarity_search_with_scores(query)
        # 简化实现：返回每个文档的相似度分数
        query_embedding = self.embed.embed(query)
        doc_embeddings = [self.embed.embed(doc) for doc in self.documents]
        
        scores = []
        for doc_emb in doc_embeddings:
            sim = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb) + 1e-10
            )
            scores.append(sim)
        return scores
    
    def _bm25_search(self, query: str) -> List[float]:
        """BM25 关键词检索"""
        k1 = 1.5   # 词频饱和度参数
        b = 0.75   # 长度归一化参数
        
        query_words = query.lower().split()
        scores = []
        
        for doc in self.documents:
            doc_words = doc.lower().split()
            doc_length = len(doc_words)
            word_counts = Counter(doc_words)
            
            score = 0.0
            for word in query_words:
                if word not in self.doc_freq:
                    continue
                
                tf = word_counts.get(word, 0)
                n = self.doc_freq[word]
                
                # BM25 核心公式
                idf = math.log((self.total_docs - n + 0.5) / (n + 0.5) + 1)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_length / self.avg_doc_length)
                score += idf * numerator / denominator
            
            scores.append(score)
        
        return scores
    
    def _normalize(self, scores: List[float]) -> List[float]:
        """Min-Max 归一化"""
        if not scores:
            return scores
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]

# ============ 使用示例 ============
# retriever = HybridRetriever(
#     vector_db=your_vector_db,
#     embedding_model=your_embedder,
#     alpha=0.6        # 语义检索占 60%，关键词占 40%
# )
# 
# results = retriever.search("A100-80-SXM 规格参数")
# for doc, score in results:
#     print(f"分数: {score:.3f} | 文档: {doc[:50]}...")
```

### 4.5 什么时候调 alpha

| alpha 值 | 含义 | 适合场景 |
|----------|------|---------|
| 1.0 | 纯语义检索 | 知识问答、开放域问题 |
| 0.7 | 语义为主，关键词为辅 | 通用场景（推荐默认） |
| 0.5 | 均衡 | 混合类型数据 |
| 0.3 | 关键词为主，语义为辅 | 代码搜索、精确术语搜索 |
| 0.0 | 纯关键词 BM25 | 正则匹配、精确查找 |

---

## 🔄 五、查询转换（Query Translation）

### 5.1 它解决什么问题

用户的问题往往不够好，需要做"翻译"。

### 5.2 三种常用转换

#### ① 查询重写（Query Rewriting）

最简单也是最常用的方法：

```python
def query_rewrite(raw_question: str, llm_client) -> str:
    """把模糊问题的改写得更清晰"""
    prompt = f"""请把以下问题改写得更清晰、更适合文档检索。
要求：
- 补充缺失的上下文
- 使用更精确的术语
- 保持原意不变

原始问题：{raw_question}
改写后："""
    
    rewritten = llm_client.chat(prompt)
    return rewritten

# 例子
# 输入: "那个新出的手机怎么样？"
# 输出: "2024 年新发布的 iPhone 16 Pro 综合性能、续航和拍照表现如何？"
```

#### ② 查询分解（Query Decomposition）

把复杂问题拆成多个子问题：

```python
def query_decomposition(complex_question: str, llm_client) -> List[str]:
    """把复杂问题拆成多个简单的子问题"""
    prompt = f"""请将以下复杂问题拆解成多个独立的子问题。
每个子问题应该：
- 只问一个具体方面
- 可以直接检索
- 用短句表达

复杂问题：{complex_question}

子问题列表（每行一个）："""
    
    result = llm_client.chat(prompt)
    sub_questions = [
        q.strip().lstrip("0123456789.、- ")
        for q in result.strip().split("\n")
        if q.strip()
    ]
    return sub_questions

# ============ 使用示例 ============
# 输入: "我应该买 MacBook Air 还是 MacBook Pro？"
# 输出: [
#   "MacBook Air M3 的性能参数和续航",
#   "MacBook Pro M3 的性能参数和续航",
#   "MacBook Air 和 MacBook Pro 的价格差异",
#   "日常办公和编程对电脑配置的要求"
# ]
```

#### ③ Step-Back Prompting（后退一步）

> **思想**：问具体问题之前，先问一个"更大"的问题，获取背景知识。

```python
def step_back_prompting(specific_question: str, llm_client) -> str:
    """先问一个更抽象的问题获取背景，再回答具体问题"""
    
    # Step 1: 生成"后退一步"的问题
    step_back_prompt = f"""请根据以下具体问题，生成一个更抽象、更通用的背景问题。
这个问题应该从更高的维度来理解该话题。

具体问题：{specific_question}
抽象背景问题："""
    
    background_question = llm_client.chat(step_back_prompt)
    
    # Step 2: 用背景问题和原始问题分别检索
    # (实际项目中两个都会去检索)
    
    return background_question

# 例子
# 具体问题: "Python 的 sort() 和 sorted() 哪个更快？"
# 后退一步: "Python 中原地排序和返回新列表两种方式的性能差异原因是什么？"
```

### 5.3 三种转换的对比

| 方式 | 效果 | 额外 LLM 调用 | 适用场景 |
|------|------|---------------|---------|
| 重写 | 1 个问题 → 1 个更好的问题 | 1 次 | 用户问得模糊 |
| 分解 | 1 个问题 → N 个子问题 | 1 次 | 复杂多维度问题 |
| Step-Back | 1 个问题 → 2 个问题 | 1 次 | 需要背景知识的专业问题 |

---

## 🌲 六、RAPTOR：递归文档树

### 6.1 它解决什么问题

**痛点**：普通的向量检索只能搜到"一段"内容，找不到"整篇文章"或"章节"级别的信息。

```
朴素的 chunk 检索：
  用户问 "Transformer 架构的核心思想是什么"
  → 可能只搜到某一段关于 self-attention 的描述
  → 但没搜到文章开头对整体架构的概述
  → 丢失了宏观信息！
```

### 6.2 核心思想

把文档建造成一棵**多层次树**：

```
                     [全局摘要]                          ← 第 3 层（最高层）
                    /          \
          [章节摘要 A]        [章节摘要 B]                ← 第 2 层
          /     |     \        /     \
    [段1]  [段2]  [段3]  [段4]  [段5]                   ← 第 1 层（原始 chunk）
```

**检索时**：从顶层开始，逐层往下找最相关的节点。

### 6.3 代码实现

```python
class RAPTOR:
    """
    递归文档树（Recursive Abstractive Processing Tree Of Organization）
    把文档建成层次化的树，支持分层检索
    """
    
    def __init__(self, llm_client, embedding_model):
        self.llm = llm_client
        self.embed = embedding_model
        self.tree = []  # 树的每一层: [[node1, node2, ...], [layer2_nodes], ...]
    
    def build_tree(self, chunks: List[str], max_cluster_size: int = 5):
        """
        从底向上构建文档树
        
        参数:
            chunks: 文档分块列表（最底层节点）
            max_cluster_size: 每层最多多少节点合并成一个摘要
        """
        print("🌱 开始构建文档树...")
        
        current_layer = [{"content": c, "level": 0} for c in chunks]
        self.tree = [current_layer]
        
        # 逐层向上抽象，直到只剩一个节点
        while len(current_layer) > 1:
            next_layer = []
            
            # 对当前层进行分组，每组生成一个摘要
            for i in range(0, len(current_layer), max_cluster_size):
                group = current_layer[i:i + max_cluster_size]
                group_texts = [node["content"] for node in group]
                
                # 生成这组的摘要
                summary = self._summarize_group(group_texts)
                
                # 新节点包含：摘要 + 子节点引用
                new_node = {
                    "content": summary,
                    "level": current_layer[0]["level"] + 1,
                    "children": group,
                    "embedding": self.embed.embed(summary)
                }
                next_layer.append(new_node)
            
            current_layer = next_layer
            self.tree.append(current_layer)
            print(f"  → 第 {len(self.tree)} 层: {len(current_layer)} 个节点")
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """
        分层检索：从顶层往下找最相关的节点
        """
        query_embedding = self.embed.embed(query)
        results = []
        
        # 从顶层开始检索
        for layer_idx, layer in enumerate(reversed(self.tree)):
            # 计算当前层所有节点与查询的相似度
            scores = []
            for node in layer:
                if "embedding" in node:
                    sim = self._cosine_similarity(query_embedding, node["embedding"])
                    scores.append((sim, node))
            
            # 取当前层最相关的节点
            scores.sort(key=lambda x: x[0], reverse=True)
            top_nodes = scores[:top_k]
            
            for sim, node in top_nodes:
                if layer_idx == 0:
                    # 最顶层：直接加摘要
                    results.append(node["content"])
                else:
                    # 非顶层：递归取出所有子节点的原始内容
                    results.extend(self._collect_leaf_content(node))
        
        # 去重
        seen = set()
        unique_results = []
        for r in results:
            if r not in seen:
                seen.add(r)
                unique_results.append(r)
        
        return unique_results[:top_k]
    
    def _summarize_group(self, texts: List[str]) -> str:
        """对一组文本生成摘要"""
        combined = "\n\n".join(texts)
        prompt = f"""请对以下内容进行概括摘要，保留核心信息：

{combined}

摘要："""
        return self.llm.chat(prompt)
    
    def _collect_leaf_content(self, node) -> List[str]:
        """递归收集叶子节点的原始内容"""
        if "children" not in node:
            return [node["content"]]
        
        contents = []
        for child in node["children"]:
            contents.extend(self._collect_leaf_content(child))
        return contents
    
    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

# ============ 使用示例 ============
# raptor = RAPTOR(llm_client=your_llm, embedding_model=your_embedder)
#
# # 准备文档分块
# chunks = [
#     "Transformer 的核心是 self-attention 机制...",
#     "Multi-head attention 将多个 attention 结果拼接...",
#     "Position encoding 为模型提供位置信息...",
#     "Feed-forward network 对每个位置的输出做非线性变换...",
#     "Layer normalization 帮助稳定训练过程...",
# ]
#
# # 建树
# raptor.build_tree(chunks)
#
# # 检索
# results = raptor.search("Transformer 的核心机制是什么？")
# for r in results:
#     print(r[:100] + "...")
```

---

## ✂️ 七、高级分块策略

### 7.1 分块为什么重要

**分割线的一句话**：

> **分块做得好不好，直接决定 RAG 的上限。** 检索结果再优化，如果原始分块就是错的，一切都白搭。

### 7.2 三种主流分块策略

#### ① 固定大小分块（Fixed-size Chunking）

```python
def fixed_size_chunking(text: str, chunk_size: int = 512, overlap: int = 50):
    """
    最简单的方法：按固定长度切块，加重叠
    overlap 是为了防止跨段信息丢失
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # 跳过时保留重叠
    
    return chunks
```

**优缺点**：
- ✅ 简单、快、可控
- ❌ 可能把完整句子/段落从中切断

#### ② 语义分块（Semantic Chunking）

```python
def semantic_chunking(text: str, llm_client) -> List[str]:
    """
    根据语义自然边界来分块
    比如：按段落、按换行、按主题变化
    """
    # 方法 1：简单规则（按段落）
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    if len(paragraphs) > 1:
        return paragraphs
    
    # 方法 2：用 LLM 识别主题边界
    prompt = f"""请根据语义将以下文本分成多个段落。
每个段落应该是一个完整的主题。
用 --- 分隔不同的段落。

文本：
{text}

分段结果："""
    
    result = llm_client.chat(prompt)
    chunks = [c.strip() for c in result.split("---") if c.strip()]
    return chunks
```

**优缺点**：
- ✅ 保留语义完整性
- ❌ 慢（需要 LLM 调用）、块大小不均

#### ③ 代理式分块（Agentic Chunking）

```python
def agentic_chunking(text: str, llm_client) -> List[Dict]:
    """
    用一个"分块代理"来智能决定怎么分
    不仅分块，还会标注块的类型和用途
    """
    prompt = f"""你是一个文档分块专家。请分析以下文本，按主题将其分成多个块。
对每个块，请给出：
1. 块内容
2. 块类型（概述/概念/示例/代码/结论）
3. 关键实体（人名、技术名词、日期等）

使用 JSON 格式输出。

文本：
{text}

输出格式：
[
  {{"content": "...", "type": "概述", "entities": ["..."]}},
  ...
]"""
    
    result = llm_client.chat(prompt)
    # 解析 JSON（简化处理）
    import json
    try:
        chunks = json.loads(result)
        return chunks
    except:
        # 解析失败时回退到简单分段
        return [{"content": text, "type": "unknown", "entities": []}]
```

**优缺点**：
- ✅ 最智能、结果最合理
- ❌ 最慢、最贵（多次 LLM 调用）

### 7.3 分块策略选择指南

```
资源充足？        是 → 语义分块或代理式分块
                  ↓ 否
需要实时处理？    是 → 固定大小分块
                  ↓ 否
数据是自然段落？  是 → 按段落分块
                  ↓
              试试混合策略：
              先用固定大小粗分，再用语义判断合并
```

---

## 🛠️ 八、完整可运行的综合示例

下面把本章的技术融合成一个完整的 `AdvancedRAGEngine`：

```python
from typing import List, Dict, Optional
import numpy as np

class AdvancedRAGEngine:
    """
    综合高级 RAG 引擎
    整合了：HyDE + 混合检索 + 查询重写 + 文档树 + 摘要索引
    
    使用方式：
    1. 添加文档
    2. 提问
    3. 系统自动选择最优策略
    """
    
    def __init__(self, vector_db, llm_client, embedding_model):
        self.vector_db = vector_db
        self.llm = llm_client
        self.embed = embedding_model
        
        # 策略开关（可配置）
        self.use_hyde = True           # 是否使用假设文档嵌入
        self.use_rewrite = True        # 是否重写问题
        self.use_hybrid = True         # 是否使用混合检索
        self.use_reflection = True     # 是否做自我反思
    
    def query(self, user_input: str) -> Dict:
        """完整的查询流程"""
        
        # ============ 阶段一：查询优化 ============
        
        # 1️⃣ 问题重写（让问题更清晰）
        if self.use_rewrite:
            query = self._rewrite_query(user_input)
            print(f"📝 重写后: {query}")
        else:
            query = user_input
        
        # 2️⃣ HyDE 生成假设文档（缩小语义差距）
        if self.use_hyde:
            hypo_doc = self._generate_hypothetical(query)
            search_query = hypo_doc
            print(f"🧠 假设文档前缀: {hypo_doc[:60]}...")
        else:
            search_query = query
        
        # ============ 阶段二：混合检索 ============
        
        if self.use_hybrid:
            # 向量检索 + 关键词检索加权融合
            retrieved = self._hybrid_search(search_query, top_k=10)
        else:
            # 纯向量检索
            retrieved = self.vector_db.similarity_search(search_query, k=10)
        
        print(f"🔍 检索到 {len(retrieved)} 条结果")
        
        # ============ 阶段三：文档过滤 ============
        
        # 3️⃣ 自我反思过滤
        if self.use_reflection:
            filtered = self._self_reflect(query, retrieved)
            print(f"✅ 反思过滤后保留 {len(filtered)} 条")
        else:
            filtered = retrieved[:5]
        
        # ============ 阶段四：生成回答 ============
        
        context = self._format_context(filtered)
        answer = self._generate_answer(query, context)
        
        return {
            "answer": answer,
            "retrieved_count": len(retrieved),
            "filtered_count": len(filtered),
            "rewritten_query": query if self.use_rewrite else None
        }
    
    def _rewrite_query(self, raw: str) -> str:
        prompt = f"请将以下问题改写得更清晰、更适合检索：\n{raw}"
        return self.llm.chat(prompt)
    
    def _generate_hypothetical(self, query: str) -> str:
        prompt = f"根据问题写一段假设性的技术文档内容作为检索依据：\n{query}"
        return self.llm.chat(prompt)
    
    def _hybrid_search(self, query: str, top_k: int):
        """简化版混合检索（实际中用 HybridRetriever 类）"""
        return self.vector_db.similarity_search(query, k=top_k)
    
    def _self_reflect(self, query: str, docs: List) -> List:
        """对每个文档做相关性评估"""
        valid = []
        for doc in docs:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            prompt = f"""问题：{query}
文档：{content[:300]}

这个文档对回答问题有帮助吗？只回答 yes 或 no："""
            judgment = self.llm.chat(prompt).strip().lower()
            if "yes" in judgment:
                valid.append(doc)
        return valid if valid else docs[:2]
    
    def _format_context(self, docs: List) -> str:
        return "\n\n---\n\n".join([
            d.page_content if hasattr(d, 'page_content') else str(d)
            for d in docs
        ])
    
    def _generate_answer(self, query: str, context: str) -> str:
        prompt = f"""基于以下参考内容回答问题：

参考内容：
{context}

问题：{query}

请给出详细、准确的回答："""
        return self.llm.chat(prompt)

# ============ 使用示例 ============
# engine = AdvancedRAGEngine(
#     vector_db=your_vector_db,
#     llm_client=your_llm,
#     embedding_model=your_embedder
# )
#
# # 提问
# result = engine.query("iPhone 16 的 A18 芯片性能提升大吗？")
# print(result["answer"])
```

---

## 📊 九、本章技术对比总览

| 技术 | 解决的问题 | 核心代价 | 优先使用时机 |
|------|-----------|---------|------------|
| **文档摘要索引** | 整篇文档主题检索 | 多一次 LLM 摘要 | 有长文档需要全局检索 |
| **HyDE** | 问题-文档语义差距 | 多一次 LLM 假设生成 | 用户口语化，文档书面化 |
| **混合检索** | 精确术语匹配差 | 维护两套索引 | 代码/型号/专业术语多 |
| **查询重写** | 用户提问模糊 | 一次 LLM 改写 | 通用场景（推荐默认） |
| **查询分解** | 复杂问题有多个方面 | 多次检索 | 对比类/多维度问题 |
| **Step-Back** | 需要背景知识支持 | 一次 LLM 抽象 | 专业领域复杂问题 |
| **RAPTOR 树** | 缺失全局/层次信息 | N 次 LLM 摘要建树 | 长文档、教科书类数据 |
| **高级分块** | 分块不合理导致检索差 | 更复杂的预处理 | 数据质量差时优先检查 |

---

## 🧩 十、常见组合模式

实际项目中，这些技术**不是孤立使用**的，而是根据场景组合：

```
场景 A：通用 QA 系统
  查询重写 → 混合检索 → 重排序 → 回答
  （最轻量的组合，适合大多数场景）

场景 B：专业文档问答（论文、技术手册）
  查询重写 + Step-Back → HyDE 检索 → RAPTOR 树检索 → 融合结果 → 回答
  （重武器组合，适合知识密集型场景）

场景 C：对话式客服
  查询重写 → 混合检索 → 自我反思过滤 → 回答
  （重视回答质量的组合）

场景 D：多文档研究报告
  查询分解 → 每个子问题做 重写+检索 → 融合所有结果 → 生成报告
  （覆盖全面的组合）
```

---

## ⚠️ 十一、避坑指南

### 11.1 常见错误

```python
# ❌ 错误 1：忽视分块质量
chunks = text.split(".")  # 按句号分！太碎了！
# ✅ 正确：按段落或语义分块
chunks = [p for p in text.split("\n\n") if p.strip()]

# ❌ 错误 2：不分青红皂白全用 HyDE
# （简单问题不需要 HyDE，反而会引入噪音）
# ✅ 正确：根据问题复杂度决定
if is_complex_question(query):
    use_hyde(query)
else:
    direct_retrieve(query)

# ❌ 错误 3：alpha 用默认值不改
# （技术文档占比高的场景，关键词权重应该更高）
# ✅ 正确：根据数据特点调参
if lots_of_technical_terms:
    alpha = 0.3  # 关键词为主
else:
    alpha = 0.7  # 语义为主
```

### 11.2 调试 checklist

```
当 RAG 效果不好时，按这个顺序排查：

1. ☐ 分块合理吗？  → 打印几个 chunk 看看有没有切断关键信息
2. ☐ 检索到了吗？  → 打印检索到的文档标题/摘要
3. ☐ 排序对吗？    → 看看 Top-1 是不是最相关的
4. ☐ 上下文太长？  → 预算一下 token，超长会被截断
5. ☐ 提示词合理？  → 换个提示词写法试试
```

---

## 📝 本章总结

### 知识地图

```
Advanced RAG 技术实现
│
├── ① 文档摘要索引（检索前 → 优化索引）
├── ② HyDE（检索前 → 缩小语义差距）
├── ③ 混合检索（检索中 → 语义+关键词互补）
├── ④ 查询转换（检索前 → 优化问题）
│   ├── 重写
│   ├── 分解
│   └── Step-Back
├── ⑤ RAPTOR 树（检索前 → 层次化组织）
├── ⑥ 高级分块（最基础的优化）
└── ⑦ 综合引擎（全部整合）
```

### 核心公式

> **高级 RAG 效果 ≈ 分块质量 × 检索策略 × 后处理质量**
> 三个因子是乘法关系 —— 任何一个拉胯，整体效果都打折。

### 关键一句话

> **所有的技术都是为了缩小"用户意图"和"文档内容"之间的差距。**  
> 不管是重写问题、生成假设文档、还是建树，最终目标都一样：让检索更准。

