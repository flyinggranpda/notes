# 🚀 RAG 进阶：从基础到高级架构

> 🎯 用费曼学习法讲清楚高级 RAG 的每一种架构思路  
> 💻 注重代码实践，每个概念都有可运行的示例  
> 📖 适合人群：已了解 RAG 基础的初学者

---

## 📌 一、为什么需要"高级 RAG"？

### 1.1 朴素 RAG 的问题

先回顾一下基础 RAG 的流程：

```
用户提问 → 向量检索 → 检索到文档 → 拼接提示词 → 大模型回答
```

这个流程有几个明显的**痛点**：

| 痛点 | 打个比方 | 后果 |
|------|---------|------|
| **检索质量不稳定** | 你去搜资料，搜出来的可能是错的 | 模型基于错误信息回答 → 错误答案 |
| **用户提问不精准** | 用户问得模糊，搜出来的也模糊 | "这个东西怎么样？" → 搜不到关键信息 |
| **检索结果太多** | 搜出来 10 条，其中 5 条是废话 | 浪费 token，干扰模型判断 |
| **知识更新慢** | 数据库里的数据是上周的 | 无法回答最新的问题 |

### 1.2 高级 RAG 的解决思路

> **核心思想：在检索"前"和检索"后"各加一道优化工序。**

```
朴素 RAG：  问题 → 检索 → 拼接 → 回答
高级 RAG：  问题 → [前索引优化] → 检索 → [后索引优化] → 拼接 → 回答
                    ↑                         ↑
              优化问题质量              优化检索结果质量
```

---

## 🔧 二、前索引 vs 后索引（核心概念）

### 2.1 前索引（Pre-retrieval）—— 检索之前做优化

**目标**：让问题更清晰、检索更精准。

```python
# 前索引的核心思路：在检索之前，对用户的原始问题进行加工

原始问题："这个东西怎么样？"       ← 太模糊
      ↓
优化后问题："iPhone 16 的续航表现如何？"  ← 更精准
```

**常见的前索引手段**：

| 手段 | 做什么 | 代码思路 |
|------|--------|---------|
| **问题重写** | 把模糊问题改得更清晰 | `llm.rewrite(question)` |
| **问题扩展** | 生成多个相似问题，分别检索 | `llm.generate_similar(question, n=3)` |
| **假设性问题** | 先生成假设答案，再基于答案检索 | `llm.hypothetical_answer(q) → 检索` |

### 2.2 后索引（Post-retrieval）—— 检索之后做优化

**目标**：对检索到的文档进行排序、过滤、压缩。

```python
# 后索引的核心思路：检索到一堆文档后，去粗取精

检索到的文档：[doc1, doc2, doc3, doc4, doc5]
      ↓
排序+过滤后：[doc3, doc1]    ← 只留下最相关的 2 条
```

**常见的后索引手段**：

| 手段 | 做什么 | 代码思路 |
|------|--------|---------|
| **重排序** | 用更精确的模型重新打分排序 | `reranker.rerank(query, docs)` |
| **过滤** | 去掉不相关的文档 | `docs = [d for d in docs if is_relevant(query, d)]` |
| **压缩** | 对长文档做摘要，减少 token | `llm.summarize(doc)` |

### 2.3 一句话记住

> **前索引 ≈ 优化问题，后索引 ≈ 优化答案的原材料。**

---

## 🏗️ 三、五种高级 RAG 架构详解

### 3.1 T-RAG（Tree RAG）—— 树状检索

**用大白话说**：  
把问题拆成树状结构，每个分支继续检索，像"刨根问底"一样层层深入。

```
                    ┌─ 子问题 A ── 检索结果 A
      ┌─ 问题 1 ──┤
      │           └─ 子问题 B ── 检索结果 B
问题 ─┤
      │           ┌─ 子问题 C ── 检索结果 C
      └─ 问题 2 ──┤
                  └─ 子问题 D ── 检索结果 D
```

**适用场景**：复杂问题需要多角度分析。

```python
# T-RAG 的核心思路（伪代码）
def t_rag(query, vector_db, llm):
    # 1. 把问题拆解成树状子问题
    sub_questions = llm.decompose_to_tree(query)
    
    # 2. 对每个子问题进行检索
    all_docs = []
    for sub_q in sub_questions:
        docs = vector_db.search(sub_q)
        all_docs.extend(docs)
    
    # 3. 整合所有检索结果，生成最终答案
    context = merge_results(all_docs)
    answer = llm.answer(query, context)
    return answer
```

---

### 3.2 C-RAG（Corrective RAG）—— 自我纠正

**用大白话说**：  
检索到文档后，先用一个"评估模型"检查这些文档的质量。
- ✅ 文档好 → 直接用
- ❌ 文档差 → 重新去互联网搜索
- 🤷 不确定 → 两个都用

```
                    ┌── 评估通过 → 直接返回检索结果
                    │
检索到文档 ──→ [评估模型]
                    │
                    ├── 评估不通过 → 重新搜索（搜互联网）
                    │
                    └── 不确定 → 检索结果 + 搜索结果的融合
```

```python
# C-RAG 的核心实现
def c_rag(query, vector_db, web_search, llm):
    # 1. 从向量数据库检索
    retrieved_docs = vector_db.search(query)
    
    # 2. 用评估模型检查检索质量
    evaluation = evaluate_retrieval(query, retrieved_docs)
    
    if evaluation == "correct":
        # ✅ 直接使用检索结果
        context = retrieved_docs
        
    elif evaluation == "incorrect":
        # ❌ 检索失败，去互联网搜索
        web_results = web_search(query)
        context = web_results
        
    else:  # "ambiguous"
        # 🤷 不确定，两个都用
        web_results = web_search(query)
        context = retrieved_docs + web_results
    
    # 3. 生成最终答案
    return llm.answer(query, context)

def evaluate_retrieval(query, docs):
    """
    评估检索结果与问题的相关性
    返回: "correct" / "incorrect" / "ambiguous"
    """
    # 这里可以用一个小模型判断
    # 或者简单根据相似度分数来判断
    scores = [compute_similarity(query, doc) for doc in docs]
    avg_score = sum(scores) / len(scores)
    
    if avg_score > 0.8:
        return "correct"
    elif avg_score < 0.3:
        return "incorrect"
    else:
        return "ambiguous"
```

---

### 3.3 Self-RAG（自我反思 RAG）—— 自问自答

**用大白话说**：  
检索到文档后，不是直接用它，而是让大模型自己判断：
"我用这个文档回答，我的答案靠谱吗？"

```
① 检索到文档 doc1、doc2、doc3
      ↓
② 对每个文档分别问大模型："用这个文档回答对不对？"
      ├── doc1 → 大模型："这个可以 ✅"
      ├── doc2 → 大模型："这个不行 ❌"
      └── doc3 → 大模型："这个可以 ✅"
      ↓
③ 只保留评估通过的文档，拼接后生成最终答案
```

```python
# Self-RAG 的核心实现
def self_rag(query, vector_db, llm):
    # 1. 检索
    retrieved_docs = vector_db.search(query, top_k=5)
    
    # 2. 对每个文档进行"自我反思"
    valid_docs = []
    for doc in retrieved_docs:
        # 问大模型：用这个文档来回答问题，靠谱吗？
        reflection_prompt = f"""
        问题：{query}
        参考文档：{doc}
        
        请判断：用这个文档来回答以上问题，是否相关且可靠？
        只回答 "yes" 或 "no"。
        """
        judgment = llm.generate(reflection_prompt)
        
        if judgment.strip().lower() == "yes":
            valid_docs.append(doc)
    
    # 3. 如果都不行，重新检索
    if not valid_docs:
        # 调整检索策略，重试
        valid_docs = retry_retrieval(query, vector_db)
    
    # 4. 用筛选后的文档生成答案
    context = "\n".join(valid_docs)
    return llm.answer(query, context)
```

#### ⚙️ 3.3.1 retry_retrieval 如何重试

这段代码有个关键问题：`valid_docs` 为空时调用的 `retry_retrieval` 该怎么重试？

直觉上可能觉得"换个关键词再搜一次"，但 Self-RAG 的设计意图是**感知到失败后，改变检索策略本身**，而不是在同一条路上再走一遍。

##### 策略 1：用 LLM 提炼关键词再搜

```python
def retry_retrieval(query, vector_db):
    """
    核心思想：既然文档全被筛掉了，说明原问题的检索方向不对。
    让 LLM 帮我们提炼更精准的关键词，换一个"角度"搜索。
    """
    # 让 LLM 分析为什么没搜到，并生成新的检索方向
    analysis_prompt = f"""
    用户问：{query}

    这个问题的检索结果全部被判定为不相关。
    请分析可能的原因，然后从完全不同的角度生成 3 组关键词用于重新检索。

    每组关键词应覆盖不同的角度：
    - 角度 1：技术术语 / 专业名词
    - 角度 2：同义改写 / 口语化表达
    - 角度 3：上级概念 / 更广泛的抽象层

    输出格式：
    角度 1: [关键词]
    角度 2: [关键词]
    角度 3: [关键词]
    """
    new_keywords = llm.generate(analysis_prompt)

    # 用新关键词分别检索
    all_docs = []
    for kw_set in new_keywords:
        docs = vector_db.search(kw_set, top_k=5)
        all_docs.extend(docs)

    return deduplicate(all_docs)
```

**举例**：

| 原始问题 | 第一次检索 → 全被筛掉 | 重试生成的新关键词 |
|---------|---------------------|------------------|
| "那个新基金怎么样" | 搜"新基金 怎么样" → 匹配的都是废话 ❌ | 角度1: "2024年三季报 新发基金 净值表现"<br>角度2: "最近成立的基金 收益如何"<br>角度3: "公募基金 新产品 业绩排行" |
| "手机发热怎么解决" | 搜"手机 发热 解决" → 泛泛的散热原理 ❌ | 角度1: "A17 Pro 芯片 温控 降频策略"<br>角度2: "iPhone 发烫 怎么处理"<br>角度3: "手机散热技术 液冷 VC均温板" |

##### 策略 2：Step-Back 后退一步搜

先搜更宽泛的背景知识，再基于背景知识确定关键词搜具体问题。

```python
def retry_retrieval_step_back(query, vector_db, llm):
    # Step 1: 生成"后退一步"的背景问题
    step_back_prompt = f"""
    原始问题：{query}

    这个问题太具体了，检索不到相关资料。
    请生成一个更广泛的上层概念问题。
    例如：
    具体："iPhone 16 A18 芯片的 GPU 性能提升多少？"
    后退："2024 年苹果芯片产品线（A17 Pro → A18）的整体技术演进"
    """
    background_query = llm.generate(step_back_prompt)

    # Step 2: 用背景问题搜索
    bg_docs = vector_db.search(background_query, top_k=3)

    # Step 3: 用背景文档的内容提炼新关键词
    bg_context = "\n".join([d.content[:200] for d in bg_docs])
    refine_prompt = f"""基于以下背景知识：
    {bg_context}

    从中提取与下面问题最相关的关键术语，生成一组新的检索关键词：
    问题：{query}"""
    new_keywords = llm.generate(refine_prompt)

    # Step 4: 用新关键词重新检索
    return vector_db.search(new_keywords, top_k=5)
```

**举例**：

```
原始问题："Transformer 的 FFN 层为什么比 attention 层参数多那么多？"
                       ↓ 搜不到
后退一步："Transformer 整体架构中各模块的参数分布"
                       ↓ 搜到了 Transformer 架构文档
从中提炼关键词："feed-forward network 参数占比 2/3"、"FFN 权重矩阵 d_model d_ff"
                       ↓
重新检索成功 ✅
```

##### 策略 3：完整的优先队列重试（推荐）

一般工业落地时，会把所有策略列成一个**优先级队列**，按"代价从小到大"依次尝试：

| 优先级 | 策略 | 代价 | 预期成功率 |
|--------|------|------|-----------|
| 1 | 同义改写关键词 | 1 次 LLM | ~40% |
| 2 | 拆解为子问题分别搜 | 1 次 LLM + 多次检索 | ~30% |
| 3 | Step-Back 后退 | 2 次 LLM | ~20% |
| 4 | 扩大搜索范围（降低阈值） | 0 次 LLM | ~10% |

```python
def retry_retrieval(query, vector_db, llm=None, max_attempts=3):
    """
    完整的重试策略：依次尝试多种不同的检索方向，
    直到检索到有效的文档，或达到最大尝试次数
    """
    all_docs = []
    attempts = []

    # 策略 1：关键词提取
    if llm:
        kw_prompt = f"从以下问题中提取 3-5 个最关键的技术名词（只输出名词，逗号分隔）：\n{query}"
        keywords = llm.generate(kw_prompt)
    else:
        keywords = query  # 没有 LLM 时直接用原问题

    docs = vector_db.search(keywords, top_k=5)
    all_docs.extend(docs)
    attempts.append(("关键词提取", keywords[:40], len(docs)))
    if len(all_docs) >= 5:
        return deduplicate(all_docs)[:5]

    # 策略 2：同义改写
    if llm:
        rewrite_prompt = f"请用完全不同的措辞改写以下问题，保持相同含义：\n{query}"
        rephrased = llm.generate(rewrite_prompt)
        docs = vector_db.search(rephrased, top_k=5)
        all_docs.extend(docs)
        attempts.append(("同义改写", rephrased[:40], len(docs)))
    if len(all_docs) >= 5:
        return deduplicate(all_docs)[:5]

    # 策略 3：拆解为子问题
    if llm:
        split_prompt = f"请将以下问题拆解为 2 个更简单的子问题：\n{query}"
        sub_questions = llm.generate(split_prompt).split("\n")
        for sub_q in sub_questions[:2]:
            docs = vector_db.search(sub_q, top_k=3)
            all_docs.extend(docs)
            attempts.append(("子问题", sub_q[:40], len(docs)))
    if len(all_docs) >= 5:
        return deduplicate(all_docs)[:5]

    # 策略 4：扩大范围（降低相似度阈值）
    docs = vector_db.search(query, top_k=20)
    docs = [d for d in docs if similarity(query, d) > 0.3]  # 默认阈值通常是 0.7
    all_docs.extend(docs)
    attempts.append(("扩大范围", "top_k=20, threshold=0.3", len(docs)))

    # 打印重试日志
    print("📋 重试过程记录：")
    for name, value, count in attempts:
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {name}: {value}... → 命中 {count} 条")

    return deduplicate(all_docs)[:5] if all_docs else []
```

> **一句话总结 retry_retrieval**：一个方向走不通，换 N 个方向试。从代价最小的策略开始，依次尝试，直到搜到为止。

---

### 3.4 Fusion RAG（融合 RAG）—— 多问题检索

**用大白话说**：  
一个用户问题 → AI 帮你生成多个问法 → 每个问法分别检索 → 结果融合排序。

```python
def fusion_rag(query, vector_db, llm):
    # 1. 根据原始问题，生成多个相似问题
    similar_questions = llm.generate_similar_questions(
        original=query,
        n=3  # 生成 3 个
    )
    # 例如："iPhone 15 怎么样？"
    #  → "iPhone 15 性能如何？"
    #  → "iPhone 15 值得买吗？"  
    #  → "iPhone 15 和 14 比有什么升级？"
    
    all_questions = [query] + similar_questions
    
    # 2. 每个问题分别检索
    all_docs = []
    for q in all_questions:
        docs = vector_db.search(q, top_k=5)
        all_docs.extend(docs)
    
    # 3. 去重 + 重排序
    unique_docs = deduplicate(all_docs)
    ranked_docs = rerank(query, unique_docs)
    
    # 4. 取 Top-K 作为最终上下文
    context = ranked_docs[:3]
    return llm.answer(query, context)
```

---

### 3.5 Rewrite RAG（重写 RAG）—— 问题重写 + 奖励机制

**用大白话说**：  
用户的问题可能问得不好 → 用一个小模型专门重写问题 → 根据回答质量给重写模型"打分奖励"。

```
用户原始问题："那个东西多少钱？"
      ↓
[重写器模型] ──→ "2024 年 iPad Air 6 的官方售价是多少？"
      ↓
    检索 → 大模型回答 → 检查回答质量
                           ↓
                    好的回答 → 给重写器加分 👍
                    差的回答 → 给重写器减分 👎
```

#### 三种重写方式

```python
# ✏️ 方式 1：最简单的 —— 直接用大模型重写
def simple_rewrite(query, llm):
    """直接用大模型重写问题"""
    prompt = f"请将以下问题改写得更加清晰、完整：\n{query}"
    return llm.generate(prompt)

# ✏️ 方式 2：自定义重写器 —— 用一个小模型专门做重写
def custom_rewrite(query, small_rewrite_model):
    """用一个可训练的小模型做重写"""
    rewritten_query = small_rewrite_model.rewrite(query)
    return rewritten_query

# ✏️ 方式 3：带奖励的重写 —— 根据回答质量反馈训练
def reward_rewrite(query, rewrite_model, retriever, llm):
    """重写 + 奖励机制"""
    # 重写问题
    rewritten = rewrite_model.rewrite(query)
    
    # 用重写后的问题检索并回答
    docs = retriever.search(rewritten)
    answer = llm.answer(query, docs)
    
    # 评估回答质量（奖励信号）
    reward = evaluate_answer_quality(query, answer)
    
    # 根据奖励更新重写模型（RLHF 思路）
    rewrite_model.update(reward)
    
    return answer
```

---

## 📊 四、五种架构对比速查

| 架构 | 核心思想 | 优点 | 缺点 | 适合场景 |
|------|---------|------|------|---------|
| **T-RAG** | 树状拆解问题 | 覆盖全面，深度挖掘 | 速度慢，token 消耗大 | 复杂多维度问题 |
| **C-RAG** | 评估检索质量，纠错 | 鲁棒性强，防止坏数据 | 多一个模型调用 | 数据质量不确定的场景 |
| **Self-RAG** | 大模型自我反思 | 精准筛选，质量可控 | 慢（每个文档都要问） | 对回答精度要求高 |
| **Fusion RAG** | 多问法分别检索后融合 | 召回率高，不容易漏 | 检索次数多 | 需要全面覆盖的场景 |
| **Rewrite RAG** | 重写问题优化入口 | 提升检索质量上限 | 需要额外模型 | 用户提问质量低的场景 |

---

## 🤔 五、实际中怎么选？

### 5.1 老师说的重点

> **这些架构只是"思路"，不是"标准答案"。**  
> 在实际项目中，你完全可以**把几种架构混在一起用**。

```python
# 实际项目中可能是这样的混合体
def my_advanced_rag(query, vector_db, web_search, llm):
    # ① 用 Rewrite 的思路：先重写问题
    rewritten = rewrite_query(query, llm)
    
    # ② 用 Fusion 的思路：生成多问题检索
    questions = [rewritten] + llm.generate_similar(rewritten, n=2)
    all_docs = [vector_db.search(q) for q in questions]
    
    # ③ 用 C-RAG 的思路：评估检索质量
    valid_docs = filter_by_relevance(rewritten, all_docs)
    
    # ④ 如果结果不理想，去互联网搜
    if not valid_docs:
        valid_docs = web_search(rewritten)
    
    # ⑤ 用 Self-RAG 的思路：反思确认
    final_docs = self_reflect(rewritten, valid_docs, llm)
    
    # ⑥ 生成最终答案
    return llm.answer(rewritten, final_docs)
```

### 5.2 选型建议

| 你的情况 | 推荐架构 |
|---------|---------|
| 刚起步，资源有限 | 朴素 RAG + 简单的重写 |
| 数据质量参差不齐 | **C-RAG**（带纠错） |
| 对精度要求极高 | **Self-RAG**（自我反思） |
| 用户可能问得不好 | **Rewrite RAG**（问题重写） |
| 需要全面覆盖不遗漏 | **Fusion RAG**（多问法融合） |

---

## ⚖️ 六、RAG vs 微调（Fine-tuning）

### 6.1 一张表看懂

| 对比维度 | RAG | 微调 |
|---------|-----|------|
| **打个比方** | 考试时给你一本书翻 | 把书上的内容全背下来 |
| **数据更新** | 🔄 随时换书，几分钟搞定 | ⏳ 重新背书，要几周 |
| **成本** | 💰 低（只需要存文档） | 💰💰💰 高（需要 GPU 训练） |
| **精准度** | 🟡 取决于检索质量 | 🔴 更精准（模型记住了） |
| **数据要求** | 🟢 随便什么文档都行 | 🔴 必须高质量数据 |
| **风格控制** | ❌ 无法控制模型语气 | ✅ 可以控制回答风格 |
| **延迟** | 🟡 需要检索，稍慢 | 🟢 直接回答，更快 |

### 6.2 实际项目中的黄金搭配

> **RAG + 微调搭配使用，不是二选一！**

```python
# 企业级应用的常见做法
def enterprise_rag_app(query):
    # ① RAG 部分：处理实时更新的数据
    latest_docs = search_latest_policies(query)
    
    # ② 微调部分：模型已经记住了核心知识
    # （模型本身对领域已经很了解）
    
    # ③ 两者结合回答
    context = latest_docs + model_internal_knowledge(query)
    return llm.answer(query, context)
```

**实际案例**：

```
政务 AI 应用：
├── 微调部分：让模型理解政府用语、公文格式（一次训练，长期有效）
└── RAG  部分：最新的红头文件、政策法规（每周更新）

医疗 AI 应用：
├── 微调部分：让模型掌握医学知识体系（一次训练）
└── RAG  部分：最新的药品说明、临床指南（实时更新）
```

### 6.3 什么时候只用 RAG？

- 数据频繁更新（新闻、政策、价格）
- 预算有限，没有 GPU
- 快速验证想法（MVP 阶段）

### 6.4 什么时候只用微调？

- 需要严格控制模型行为/风格
- 低延迟要求（不需要检索）
- 数据相对稳定，不需要频繁更新

---

## 🛠️ 七、完整代码示例：一个可运行的高级 RAG

下面是 Fusion RAG + Self-RAG 的混合实现，你可以直接参考：

```python
import openai
from typing import List, Dict

class AdvancedRAG:
    """高级 RAG 实现（融合 + 自我反思）"""
    
    def __init__(self, vector_db, llm_client, reranker=None):
        self.vector_db = vector_db      # 向量数据库
        self.llm = llm_client            # 大模型客户端
        self.reranker = reranker         # 重排序模型（可选）
    
    def query(self, user_question: str) -> str:
        """完整的高级 RAG 流程"""
        # Step 1: 生成多个问法（Fusion）
        questions = self._expand_question(user_question)
        
        # Step 2: 分别检索
        all_docs = []
        for q in questions:
            docs = self.vector_db.similarity_search(q, k=5)
            all_docs.extend(docs)
        
        # Step 3: 重排序（后索引）
        if self.reranker:
            all_docs = self.reranker.rerank(user_question, all_docs)
        else:
            all_docs = all_docs[:5]  # 简单取前 5 条
        
        # Step 4: 自我反思过滤（Self-RAG）
        filtered_docs = self._self_reflect(user_question, all_docs)
        
        # Step 5: 拼接上下文，生成答案
        context = "\n\n".join([d.page_content for d in filtered_docs])
        prompt = f"""基于以下参考内容回答问题。

参考内容：
{context}

问题：{user_question}

请基于参考内容给出详细回答："""
        
        response = self.llm.chat(prompt)
        return response
    
    def _expand_question(self, question: str) -> List[str]:
        """生成多个相似问法"""
        prompt = f"""给定问题：{question}
请生成 3 个相似但不同角度的问法，
每个问法占一行，直接输出问题内容："""
        
        result = self.llm.chat(prompt)
        similar_questions = result.strip().split("\n")
        return [question] + similar_questions[:3]
    
    def _self_reflect(self, question: str, docs: List) -> List:
        """对每个文档做自我反思评估"""
        valid_docs = []
        
        for doc in docs:
            judge_prompt = f"""问题：{question}
参考文档：{doc.page_content[:500]}

这个文档与问题是否相关？只回答 yes 或 no："""
            
            judgment = self.llm.chat(judge_prompt).strip().lower()
            if "yes" in judgment:
                valid_docs.append(doc)
        
        return valid_docs if valid_docs else docs[:2]

# ============ 使用示例 ============
# rag = AdvancedRAG(
#     vector_db=your_vector_db,
#     llm_client=your_llm_client
# )
# answer = rag.query("2025 年人工智能发展趋势有哪些？")
# print(answer)
```

---

## 📝 本章总结

### 知识地图

```
高级 RAG
│
├── ① 前索引（优化问题）
│   ├── 问题重写
│   ├── 问题扩展
│   └── 假设性问题
│
├── ② 后索引（优化结果）
│   ├── 重排序
│   ├── 过滤
│   └── 压缩
│
├── ③ 五种架构思路
│   ├── T-RAG（树状）
│   ├── C-RAG（纠正）
│   ├── Self-RAG（反思）
│   ├── Fusion RAG（融合）
│   └── Rewrite RAG（重写）
│
├── ④ 选型指南
│   └── 按需组合，灵活混用
│
└── ⑤ RAG vs 微调
    ├── RAG：低成本、动态更新、适合外部知识
    └── 微调：高成本、精准可控、适合内部知识
```

### 核心公式

> **高级 RAG = 朴素 RAG + 前索引优化 + 后索引优化**

### 关键语句

> **这些高级架构只是"设计思路"，不是"金科玉律"。**  
> 实际工程中，按需组合、灵活混用才是王道。

