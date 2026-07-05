# 🧩 LangChain 框架深入（二）— 费曼学习笔记

> 课程来源：图灵 Python · 柏汌/百川老师  
> 学习日期：2025-06-04  
> 上一课：[LangChain 框架（一）](LangChain框架-学习笔记.md)

---

## 📋 目录

1. [Agent vs Function Call——手和工具的区别](#1-agent-vs-function-call手和工具的区别)
2. [Agent 实战：给大模型装个"手"](#2-agent-实战给大模型装个手)
3. [LangChain Hub——提示词模板市场](#3-langchain-hub提示词模板市场)
4. [Model I/O 深度拆解](#4-model-io-深度拆解)
5. [PromptTemplate 三种用法](#5-prompttemplate-三种用法)
6. [模型三兄弟：LLM vs ChatModel vs Embedding](#6-模型三兄弟llm-vs-chatmodel-vs-embedding)
7. [OutputParser——让模型按格式输出](#7-outputparser让模型按格式输出)
8. [数据检索完整流程](#8-数据检索完整流程)
9. [文档加载实战](#9-文档加载实战)
10. [文本分割 & 向量化 & 存储](#10-文本分割--向量化--存储)
11. [Retriever（检索器）是什么？](#11-retriever检索器是什么)
12. [本节课总结](#12-本节课总结)

---

## 1. Agent vs Function Call——手和工具的区别

**费曼说：** 这两个东西经常搞混，一句话分清：

```
Function Call（工具） = 🔧 扳手
Agent（智能体）       = 🤲 我的手
```

| 概念 | 本质 | 类比 |
|------|------|------|
| **Function Call** | 一个具体的**工具函数**（查天气、算数学、搜文档） | 工具箱里的扳手、螺丝刀 |
| **Agent** | 一个**决策系统**，由大模型驱动，决定**什么时候用什么工具** | 我的手 + 大脑 |

**工作流程：**

```
用户说："今天北京天气怎么样？"

Agent 大脑（大模型）思考：
  → "用户需要天气信息"
  → "工具箱里有 weather_tool 可以用"
  → "调用 weather_tool('北京')"

Agent 手（执行器）：
  → 调用 weather_tool 函数
  → 拿到结果 "25°C，晴"

Agent 大脑再次思考：
  → "结果拿到了，组织成自然语言回答"
  → 输出："北京今天 25°C，天气晴朗☀️"
```

**关键点：** 工具（Function Call）不会自己决定什么时候用自己，是 **Agent 用大模型来决策**调用哪个工具。

---

## 2. Agent 实战：给大模型装个"手"

### 整体流程

```
① 准备工具（检索器）→ ② 加载提示词模板 → ③ 创建大模型
→ ④ 创建 Agent → ⑤ 告诉 Agent 能用哪些工具 → ⑥ 提问
```

### 第一步：创建检索工具

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import create_retriever_tool

# 准备数据 → 分割 → 转向量 → 存数据库
loader = WebBaseLoader("https://flk.npc.gov.cn/民法典")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever = vector_store.as_retriever()

# ⭐ 创建一个检索工具——供 Agent 调用
tool = create_retriever_tool(
    retriever=retriever,
    name="民法典检索",
    description="用于搜索中华人民共和国民法典的相关信息"
)
```

> ⚠️ **工具的名字和描述非常重要！** 它们是写给**大模型看的**，大模型根据名字和描述来判断"当前该不该用这个工具"。

### 第二步：从 Hub 加载提示词模板

```python
from langchain import hub

# 从 LangChain Hub 加载现成的 Agent 提示词模板
prompt = hub.pull("hwchase17/openai-functions-agent")
# 也可以用自己的提示词
```

### 第三步：创建 Agent 并运行

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI

# 1. 创建大模型
llm = ChatOpenAI(model="qwen-turbo")

# 2. 创建 Agent（大模型 + 工具 + 提示词）
agent = create_openai_functions_agent(
    llm=llm,
    tools=[tool],              # Agent 可用的工具列表
    prompt=prompt
)

# 3. 创建 Agent 执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=[tool],
    verbose=True               # 详细模式：展示中间步骤
)

# 4. 提问——Agent 会自动判断是否使用工具
response = agent_executor.invoke({"input": "民法典中关于离婚冷静期的规定是什么？"})
print(response['output'])
```

### Verbose 模式能看到什么？

启用 `verbose=True` 后，会展示 Agent 的**思考过程**：

```
> 进入新的 AgentExecutor 链...
> 思考：用户问的是民法典中关于离婚冷静期的规定，我需要用检索工具查一下。

> 调用：民法典检索
> 输入：离婚冷静期

> 检索结果：民法典第1077条：自婚姻登记机关收到离婚登记申请之日起三十日内...

> 思考：我找到了相关法条，现在组织回答。

> 最终回答：根据《中华人民共和国民法典》第1077条规定...
```

> 💡 **Agent 的核心价值：** 不需要你硬编码"如果问题包含XX就用XX工具"，大模型会自动判断。

---

## 3. LangChain Hub——提示词模板市场

**费曼说：** 就像手机应用商店，Hub 上有别人写好的提示词模板，你直接下载用就行。

- 网址：[smith.langchain.com/hub](https://smith.langchain.com/hub)
- 用法：

```python
from langchain import hub

# 下载别人写好的提示词模板
prompt = hub.pull("hwchase17/openai-functions-agent")

# 下载后就是一个 PromptTemplate 对象，直接使用
print(prompt)
# 输出：input_variables=['agent_scratchpad', 'input', 'tools']...
```

> 常用的 Hub 模板有：`openai-functions-agent`、`zero-shot-react-description` 等。

---

## 4. Model I/O 深度拆解

**费曼说：** Model I/O 就是把"输入→处理→输出"这三个环节标准化。

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  PromptTemplate │ → │     Model    │ → │ OutputParser │
│  （输入格式化） │    │  （大模型推理） │    │  （输出格式化） │
└──────────────┘    └──────────────┘    └──────────────┘
```

为什么要拆成三块？因为**可以灵活替换**：
- 换提示词 → 改 PromptTemplate
- 换模型 → 改 Model
- 换输出格式 → 改 OutputParser

---

## 5. PromptTemplate 三种用法

### ① 字符串提示模板（最基础）

**费曼说：** 就是带大括号的 Python 字符串格式化，只不过名字叫得高端。

```python
from langchain_core.prompts import PromptTemplate

# 定义模板
prompt = PromptTemplate(
    template="你是一个{role}专家。请回答：{input}",
    input_variables=["role", "input"]
)

# 填充数据
formatted = prompt.format(role="Python", input="什么是装饰器？")
print(formatted)
# 输出：你是一个Python专家。请回答：什么是装饰器？

# 链式调用
chain = prompt | llm
result = chain.invoke({"role": "Python", "input": "什么是装饰器？"})
```

### ② 聊天提示模板（带角色）

```python
from langchain_core.prompts import ChatPromptTemplate

# 方式一：用元组指定角色
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}专家"),
    ("human", "{input}")
])

# 方式二：直接用消息对象（效果一样）
from langchain_core.messages import SystemMessage, HumanMessage

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个{role}专家"),
    HumanMessage(content="{input}")
])

# 填充变量
formatted = prompt.invoke({"role": "Python编程", "input": "什么是装饰器？"})
print(formatted)
# 输出：messages=[SystemMessage(content='你是一个Python编程专家'),
#                HumanMessage(content='什么是装饰器？')]
```

> **元组 vs 消息对象**：两种方式完全等价，一般推荐用**元组方式**（更简洁）。

#### 多个变量怎么填？

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}专家，你的名字叫{name}"),
    ("human", "{input}")
])

# 所有变量一起传
formatted = prompt.invoke({
    "role": "金融投资",
    "name": "小巴",
    "input": "什么是定投？"
})
```

### ③ Few-shot（少量提示模板）

**费曼说：** 先给模型看几个例子，告诉它"照这个格式输出"。

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 1. 定义每个例子的模板
example_template = PromptTemplate(
    template="输入: {input}\n输出: {output}",
    input_variables=["input", "output"]
)

# 2. 给例子
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "3+5", "output": "8"},
]

# 3. 创建 Few-shot 模板
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="请按照以下格式回答问题：",  # 前缀说明
    suffix="输入: {input}\n输出: ",    # 后缀：最后要填的
    input_variables=["input"]
)

# 4. 填入最终问题
final_prompt = prompt.format(input="2×5")
print(final_prompt)
# 输出：
# 请按照以下格式回答问题：
# 输入: 2+2
# 输出: 4
# 输入: 3+5
# 输出: 8
# 输入: 2×5
# 输出:           ← 让模型接下去输出

# 5. 调用大模型
chain = prompt | llm
result = chain.invoke({"input": "2×5"})
# 输出："10"（因为模型学到的模式是"输出计算结果"）
```

---

## 6. 模型三兄弟：LLM vs ChatModel vs Embedding

**费曼说：** 大模型不止一种，LangChain 给分了三种：

| 模型类型 | 输入 | 输出 | 典型代表 | 用途 |
|----------|------|------|----------|------|
| **LLM（大语言模型）** | 字符串 | 字符串 | GPT-3, DeepSeek | 文本补全、生成 |
| **ChatModel（聊天模型）** | 消息列表 | 消息对象 | GPT-4, qwen-turbo | 多轮对话 |
| **Embedding（嵌入模型）** | 文本 | 浮点数列表 | text-embedding-v1 | 文本转向量 |

### LLM vs ChatModel 的区别

```python
# LLM（大语言模型）：文本↔文本
from langchain_openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo-instruct")
response = llm.invoke("你好")
# 返回："你好！有什么可以帮助你的？"

# ChatModel（聊天模型）：消息↔消息
from langchain_openai import ChatOpenAI
chat = ChatOpenAI(model="qwen-turbo")
response = chat.invoke([HumanMessage(content="你好")])
# 返回：AIMessage(content="你好！有什么可以帮助你的？")
```

> 实际开发中**90% 用 ChatModel**，因为大多数应用都是聊天形式。

### Embedding 模型

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-v1")

# 文档向量化（传入列表）
doc_vectors = embeddings.embed_documents(["今天天气真好", "苹果很好吃"])
print(len(doc_vectors[0]))  # 1536（维度）

# 问题向量化（传入字符串）
query_vector = embeddings.embed_query("今天天气怎么样")
print(len(query_vector))    # 1536
```

> `embed_documents` 和 `embed_query` 的区别：
> - `embed_documents` → 传列表，批量转换文档
> - `embed_query` → 传字符串，转换单个问题

---

## 7. OutputParser——让模型按格式输出

**费曼说：** 大模型默认只输出文本。但你可能想要 JSON、CSV、HTML……OutputParser 就是干这个的。

### JSON 输出解析器

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# 1. 创建解析器
parser = JsonOutputParser()

# 2. 在提示词里要求 JSON 格式
prompt = PromptTemplate(
    template="""请用 JSON 格式回答以下问题：
问题：{question}
回答格式：{{"answer": "...", "confidence": 0.0-1.0}}""",
    input_variables=["question"]
)

# 3. 链式调用：模板 → 模型 → 解析器
chain = prompt | llm | parser
result = chain.invoke({"question": "Python是静态还是动态类型？"})

print(result)
# 输出：{'answer': '动态类型', 'confidence': 0.95}
# ⭐ result 直接就是 Python 字典！不用自己 json.loads()
```

### ⚠️ 常见坑：提示词和解析器要匹配

**错误示范：**

```python
parser = JsonOutputParser()

# 提示词里没要求 JSON 格式！！！
prompt = PromptTemplate(
    template="请回答：{question}",
    input_variables=["question"]
)

chain = prompt | llm | parser
result = chain.invoke({"question": "Python是什么？"})
# ❌ 报错！模型输出的是普通文本，解析器无法解析成 JSON
```

**正确做法：提示词里明确要求输出格式 + 解析器匹配。**

---

## 8. 数据检索完整流程

**费曼说：** 数据检索 = RAG 的数据管线，从原始文件到能搜索的向量数据库。

```
加载文档（PDF/Word/网页）
    ↓
文本分割（切小段）
    ↓
向量化（转数字）
    ↓
存向量数据库
    ↓
检索器（提供搜索接口）
```

---

## 9. 文档加载实战

### 加载 PDF

```bash
pip install pypdf
```

```python
from langchain_community.document_loaders import PyPDFLoader

# 本地 PDF
loader = PyPDFLoader("财务管理手册.pdf")
documents = loader.load()  # 返回 Document 列表，每页一个

print(len(documents))      # 页数
print(documents[0])        # 第一页
# Document(page_content='第一页的文本...', metadata={'source': '...', 'page': 0})

# 在线 PDF 也能加载
# loader = PyPDFLoader("https://example.com/sample.pdf")
```

### 加载 Word 文档

```bash
pip install docx2txt  # 需要科学上网
```

```python
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("报告.docx")
documents = loader.load()
```

### 加载在线网页

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://flk.npc.gov.cn/民法典")
documents = loader.load()
```

### Document 对象的结构

```python
documents[0].page_content  # 文本内容
documents[0].metadata      # 元数据：来源、页码等
# {'source': '财务管理手册.pdf', 'page': 0}
```

---

## 10. 文本分割 & 向量化 & 存储

### 递归分割（最常用）

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # 每段 200 字符
    chunk_overlap=100,   # 重叠 100 字符（保证上下文连贯）
)

chunks = text_splitter.split_documents(documents)
print(f"分割成 {len(chunks)} 个片段")
```

### 一行代码搞定：向量化 + 存储

```python
from langchain_community.vectorstores import Chroma

# ⭐ 一行代码 = 向量化每个文档 + 存入 Chroma
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-v1"),
    persist_directory="./chroma_db"  # 持久化到本地
)
```

对比手写代码（没有 LangChain 时）：

```
手写：for 文档 → 调嵌入 API → 拿向量 → 调 Chroma API → 存入
     ≈ 20-30 行代码

LangChain：Chroma.from_documents(documents, embeddings)
     ≈ 1 行代码
```

---

## 11. Retriever（检索器）是什么？

**费曼说：** 检索器 = 向量数据库的一个"搜索接口"。你给它一个问题，它返回最相关的文档。

### 基本用法

```python
# 从向量数据库创建检索器
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}  # 返回最相似的 3 条
)

# 搜索
results = retriever.invoke("离婚冷静期是多久？")
# 返回：Document 对象列表

for doc in results:
    print(doc.page_content)
    print("---")
```

### 检索器的优化思想

**费曼说：** 原始的检索是把整段文本（200字）都向量化，检索时用 200 字去比。可以优化：

```
优化思路：存两套数据
  核心向量（20字摘要）→ 存在向量数据库 → 检索速度快
  完整文档（200字原文）→ 存在文件里    → 查到后再取
```

这样检索时只比 20 个字的摘要，速度能快很多。

常见的检索器类型（后面会学）：

| 检索器类型 | 原理 |
|-----------|------|
| **相似度检索**（默认） | 直接比整个文档的向量 |
| **摘要检索** | 存摘要向量，匹配后取原文 |
| **父子检索** | 先查大块，再深入小块 |
| **混合检索** | 向量搜索 + 关键词搜索结合 |

> 💡 当前阶段先掌握基础的相似度检索，**高阶 RAG 部分**会深入不同检索器的优化。

---

## 12. 本节课总结

### 核心知识点

```
✅ Agent ≠ Function Call（手 ≠ 工具）
✅ Agent 执行流程：思考 → 选工具 → 调用 → 组织回答
✅ Model I/O 三件套：PromptTemplate → Model → OutputParser
✅ 三种提示模板：字符串 / 聊天 / Few-shot
✅ 模型三兄弟：LLM / ChatModel / Embedding
✅ OutputParser：强制模型按 JSON 等格式输出
✅ 数据检索管线：加载 → 分割 → 向量化 → 存储 → 检索器
✅ 检索器 = 向量数据库的搜索接口
```

### 代码量对比

| 操作 | 手写代码 | LangChain |
|------|----------|-----------|
| 文本转向量 + 存库 | 20-30 行 | `Chroma.from_documents()` 1 行 |
| 加载 PDF | 自己解析 | `PyPDFLoader` 3 行 |
| 创建 Agent | — | `create_openai_functions_agent()` 3 行 |

---

> **费曼复习：** 用一句话说明 Agent 和 Function Call 的区别给非技术朋友听——
> "Function Call 是工具箱里的扳手，Agent 是你的手和脑子。脑子决定什么时候用扳手，手去执行。"

---

*笔记整理于 2026-06-30 · 图灵 Python 课堂 · 基于课程 3-LangChain框架-2-2025-6-4*

