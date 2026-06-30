# 🧩 LangChain 框架 & RAG 实战 — 费曼学习笔记

> 课程来源：图灵 Python · 柏汌/百川老师  
> 学习日期：2025-05-28  
> 上节课：[RAG 基础](RAG基础-学习笔记.md)

---

## 📋 目录

1. [上节回顾：RAG 流程速览](#1-上节回顾rag-流程速览)
2. [向量数据库——为什么需要它？](#2-向量数据库为什么需要它)
3. [Chroma DB 上手——增删改查](#3-chroma-db-上手增删改查)
4. [在线平台 vs 手写代码](#4-在线平台-vs-手写代码)
5. [完整 RAG 项目实战（手写代码版）](#5-完整-rag-项目实战手写代码版)
6. [LangChain 框架——是什么？](#6-langchain-框架是什么)
7. [LangChain 核心组件](#7-langchain-核心组件)
8. [LangChain 实战：模型调用](#8-langchain-实战模型调用)
9. [LangChain 实战：提示词模板 & 输出解析](#9-langchain-实战提示词模板--输出解析)
10. [LangChain 实战：向量存储 & RAG](#10-langchain-实战向量存储--rag)
11. [总结对比表](#11-总结对比表)

---

## 1. 上节回顾：RAG 流程速览

**费曼说：** 先复习一下，RAG 就是让大模型"开卷考试"。

```
准备数据 → 文本切割 → 向量化 → 存向量数据库
用户提问 → 问题向量化 → 向量数据库搜索相似片段
→ 拼接到提示词 → 大模型根据资料回答
```

---

## 2. 向量数据库——为什么需要它？

### 普通数据库不行吗？

**费曼说：** MySQL 能查"name = '张三'"，但查不了"哪个向量跟我最像"。

- **MySQL/MongoDB：** 擅长精确匹配、条件查询
- **向量数据库：** 专门为**高维向量相似度搜索**设计

> 向量长这样：`[0.023, -0.456, 0.789, ..., 0.112]`（1536 维）
> 普通数据库算这种"哪个最像"的查询，慢得要命。

### 常见向量数据库一览

| 数据库 | 特点 | 适用场景 |
|--------|------|----------|
| **Chroma** ⭐ | 轻量级、基于内存/Python、开箱即用 | 学习/原型开发 |
| **FAISS** | Facebook 出品、纯向量搜索库 | 高性能检索 |
| **Milvus** | 分布式、万亿级、毫秒响应 | 企业大规模生产 |
| **Pinecone** | 云服务、付费、管理简单 | 不想自己运维 |
| **Qdrant** | Rust 编写、高性能 | 需要高吞吐 |

### 怎么选向量数据库？

三个考量维度：

1. **向量生成方式：** 用在线 API 还是本地模型？
2. **延迟要求：** 需要毫秒级响应吗？
3. **团队经验：** 团队用过哪个就用哪个（学习成本最低）

> 💡 **学一个等于学全部：** 向量数据库的**用法基本一致**，会了 Chroma，其他的上手很快。

---

## 3. Chroma DB 上手——增删改查

### 安装

```bash
pip install chromadb
```

开箱即用，不需要额外安装服务！

### 创建数据库

```python
import chromadb

# 方式一：基于内存（关掉就没了）
client = chromadb.Client()

# 方式二：持久化到本地（推荐）
client = chromadb.PersistentClient(path="D:/chroma_db")

# 创建/获取集合（类似 MySQL 的"表"）
# get_or_create_collection：有就获取，没有就创建
collection = client.get_or_create_collection(name="my_knowledge_base")
```

### 添加数据（Create）

```python
collection.add(
    documents=[
        "今天天气真好，适合出去玩",
        "机器学习是人工智能的一个重要分支",
        "苹果是一种非常健康的水果"
    ],                       # 📄 原始文档（检索后返回这个）
    embeddings=[
        [1.2, 2.3, 3.4],     # 🔢 文档对应的向量（实际用嵌入模型生成）
        [4.5, 5.6, 6.7],
        [7.8, 8.9, 9.0]
    ],
    ids=["doc1", "doc2", "doc3"]  # 🆔 每条数据的编号
)
```

### 查询数据（Retrieve）

```python
# 方式一：根据 ID 查询
result = collection.get(ids=["doc1"])
print(result)

# 方式二：根据条件查询（where）
result = collection.get(
    where={"document": {"$contains": "天气"}}
)

# 方式三：向量相似度搜索 ⭐（RAG 核心！）
results = collection.query(
    query_embeddings=[[1.1, 2.2, 3.3]],  # 问题的向量
    n_results=2                           # 返回最相似的 2 条
)
print(results['documents'])  # 返回最相似的文档内容
```

> 💡 **注意：** 默认查询时不返回 embedding（因为太长了），想看的话：
> ```python
> result = collection.get(ids=["doc1"], include=["embeddings", "documents", "metadatas"])
> ```

### 更新数据（Update）

```python
collection.update(
    ids=["doc1"],
    documents=["修改后的文档内容"],
    embeddings=[[9.9, 8.8, 7.7]]
)
```

### 删除数据（Delete）

```python
collection.delete(ids=["doc1"])

# 验证：获取全部数据看看
all_data = collection.get()
```

### 关键理解：存了两个东西

| 字段 | 作用 | 备注 |
|------|------|------|
| `embeddings` | 用于**向量相似度检索** | 存的是密密麻麻的浮点数 |
| `documents` | 检索命中后**返回给用户** | 存的是原始文本 |

```
用户提问 "天气怎么样" 
    → 转向量 [1.1, 2.2, 3.3]
    → 在 embeddings 里找最像的
    → 找到 doc1 的向量最接近
    → 返回 doc1 的 document："今天天气真好，适合出去玩"
    → 把这个 document 拼到提示词里，给大模型回答
```

---

## 4. 在线平台 vs 手写代码

**费曼说：** 用在线平台就像点外卖——快、省事，但你不能加自己想要的菜。手写代码就像自己做饭——麻烦，但想放什么放什么。

### 在线平台（扣子 Coze / Dify）

| 优点 | 缺点 |
|------|------|
| ✅ 上手快，低代码/无代码 | ❌ 灵活性受限 |
| ✅ 成熟的生态支持 | ❌ 成本不可控（Token 按量计费） |
| ✅ 维护成本较低 | ❌ 数据隐私风险（数据可能被拿去训练） |
| | ❌ 技术黑箱（不知道底层怎么实现的） |

### 什么时候用哪个？

| 场景 | 推荐 |
|------|------|
| 快速验证想法 | 在线平台（Dify 等） |
| 企业正式项目 | 手写代码 |
| 学习理解原理 | 先手写再对比平台 |

> 💡 **配合使用：** 遇到代码跑不通时，可以用平台跑一遍流程，看看每个步骤的输出，再回来调自己的代码。平台可以作为**调试工具**。

---

## 5. 完整 RAG 项目实战（手写代码版）

这一节我们手写一个完整的 RAG：从 PDF 读取 → 文本分割 → 向量化 → 存储 → 检索 → 回答。

### 整体架构

```
┌─────────────────────────────────────────────────────┐
│                    RAG 项目结构                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  vector_store.py  ← 向量处理器                      │
│    ├── 连接嵌入模型（文本 → 向量）                   │
│    ├── 添加数据到 Chroma DB                          │
│    └── 搜索最相似文档                                │
│                                                      │
│  chatbot.py  ← 聊天机器人                           │
│    ├── 从向量数据库检索相关文档                      │
│    ├── 拼接提示词（问题 + 检索结果）                 │
│    └── 调用大模型生成回答                            │
│                                                      │
│  main.py  ← 主流程                                  │
│    ├── 读取 PDF → 文本分割                          │
│    ├── 初始化向量数据库 & 存入数据                   │
│    └── 启动问答循环                                  │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 第一步：安装依赖

```bash
pip install chromadb dashscope PyPDF2 python-docx numpy
```

### 第二步：向量处理器（vector_store.py）

```python
# vector_store.py — 处理向量化 & 数据库操作
import chromadb
from dashscope import TextEmbedding

class VectorStore:
    """向量处理器：文本转向量 → 存 Chroma → 检索"""
    
    def __init__(self, path="./chroma_db"):
        # 创建持久化连接
        self.client = chromadb.PersistentClient(path=path)
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name="rag_knowledge"
        )
    
    def get_embedding(self, text):
        """把文本转换成向量（在线调用千问嵌入模型）"""
        response = TextEmbedding.call(
            model="text-embedding-v1",
            input=text
        )
        return response.output['embeddings'][0]['embedding']
    
    def add_documents(self, documents):
        """添加文档到向量数据库"""
        # 把每个文档转成向量
        embeddings = [self.get_embedding(doc) for doc in documents]
        
        # 存入 Chroma
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
        print(f"✅ 已添加 {len(documents)} 条数据")
    
    def search(self, query, n_results=2):
        """搜索与问题最相似的文档"""
        query_embedding = self.get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results['documents'][0]  # 返回文档列表
```

### 第三步：聊天机器人（chatbot.py）

```python
# chatbot.py — 结合检索结果 & 大模型回答
from dashscope import Generation

class RAGChatbot:
    """RAG 聊天机器人：检索 → 拼接提示词 → 大模型回答"""
    
    def __init__(self, vector_store, n_results=2):
        self.vector_store = vector_store  # 向量数据库对象
        self.n_results = n_results        # 每次检索返回的文档数
    
    def chat(self, question):
        """用户提问 → 返回回答"""
        
        # 1️⃣ 检索：在向量数据库中找相关文档
        retrieved_docs = self.vector_store.search(
            question, 
            n_results=self.n_results
        )
        
        # 2️⃣ 拼接：把检索到的文档拼成"已知信息"
        context = "\n".join(retrieved_docs)
        
        # 3️⃣ 构建提示词
        prompt_template = """你是一个专业问答助手。

已知信息：
{context}

用户问题：{question}

请根据已知信息回答用户的问题，用中文回答。"""
        
        final_prompt = prompt_template.replace("{context}", context)\
                                       .replace("{question}", question)
        
        # 4️⃣ 调用大模型回答
        response = Generation.call(
            model="qwen-turbo",
            prompt=final_prompt
        )
        
        return response.output.text
```

### 第四步：PDF 文本提取 & 分割（utils.py）

```python
# utils.py — 文档处理工具

def extract_text_from_pdf(pdf_path, max_pages=3):
    """从 PDF 提取文本（指定前几页）"""
    import PyPDF2
    
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        total_pages = len(reader.pages)
        
        for i in range(min(max_pages, total_pages)):
            page = reader.pages[i]
            page_text = page.extract_text()
            page_text = page_text.replace('\n', '').strip()
            if page_text:
                text += page_text + "\n"
    
    return text


def split_text(text, chunk_size=200, overlap=50):
    """重叠分割文本"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) >= chunk_size - overlap:
            chunks.append(chunk)
    return chunks


def split_sentences(text):
    """按句子分割（按句号、问号、感叹号）"""
    import re
    sentences = re.split(r'([。！？…])', text)
    result = []
    for i in range(0, len(sentences) - 1, 2):
        result.append(sentences[i] + sentences[i + 1])
    if len(sentences) % 2 == 1 and sentences[-1]:
        result.append(sentences[-1])
    return result
```

### 第五步：主流程（main.py）—— 全部串起来

```python
# main.py — RAG 主程序
from vector_store import VectorStore
from chatbot import RAGChatbot
from utils import extract_text_from_pdf, split_text

# 1️⃣ 读取并分割文档
print("📖 读取 PDF...")
raw_text = extract_text_from_pdf("公司员工手册.pdf", max_pages=3)

print("✂️ 分割文本...")
chunks = split_text(raw_text, chunk_size=200, overlap=50)
print(f"   分割成 {len(chunks)} 个片段")

# 2️⃣ 初始化向量数据库 & 存入数据
print("🗄️ 初始化向量数据库...")
store = VectorStore()

print("🔢 向量化并存储...")
store.add_documents(chunks)

# 3️⃣ 创建 RAG 聊天机器人
print("🤖 创建 RAG 聊天机器人...")
bot = RAGChatbot(store, n_results=2)

# 4️⃣ 开始问答
print("\n" + "="*50)
print("💬 RAG 问答系统已启动！（输入 'exit' 退出）")
print("="*50)

while True:
    question = input("\n🙋 你的问题：")
    if question.lower() == 'exit':
        break
    
    answer = bot.chat(question)
    print(f"\n🤖 回答：{answer}")
```

### 从手写代码到 LangChain 的思考

```
手写代码 ≈ 150 行 → 实现了 RAG
用 LangChain ≈ 30 行  → 实现同样的 RAG

框架的意义：把重复的代码封装好，让你关注业务逻辑。
```

---

## 6. LangChain 框架——是什么？

**费曼说：** 做前端开发有 React/Vue，做 Java 有 Spring，做爬虫有 Scrapy。做 LLM 应用开发，就用 **LangChain**。

- **LangChain = Language + Chain（语言链）**
- 读音：`/læŋ tʃeɪn/`（不是"浪琴"😂）
- 官网：[langchain.com](https://langchain.com)
- 当前版本：0.3.x（迭代极快，半年从 0.1 到 0.3）

### 为什么要用 LangChain？

对比手写 RAG 和 LangChain RAG：

| 功能 | 手写代码 | LangChain |
|------|----------|-----------|
| 连接大模型 | 自己写 HTTP 请求 | 一行代码 |
| 文本分割 | 手写 split 函数 | 内置递归分割器 |
| 向量化存储 | 调 API → 存 Chroma | 一行代码搞定 |
| 提示词模板 | 手动 replace | `PromptTemplate` |
| 检索链 | 自己编排流程 | `RetrievalQA` 链 |

---

## 7. LangChain 核心组件

```
┌──────────────────────────────────────────────────────┐
│                  LangChain 生态                       │
├──────────────────────────────────────────────────────┤
│  核心库 (langchain)                                  │
│  ├── Model I/O ── 模型调用 & 提示词模板              │
│  ├── Chains ──── 链式调用（像管道符 |）              │
│  ├── Memory ──── 记忆（短时/长时对话历史）           │
│  ├── Agents ──── 智能代理（让模型使用工具）          │
│  └── Callbacks ─ 回调（成功/失败处理）               │
│                                                      │
│  第三方集成 (langchain-community)                    │
│  ├── OpenAI / 千问 / 百度 / Google                   │
│  ├── HuggingFace / 魔搭                              │
│  └── Chroma / FAISS / Milvus                          │
│                                                      │
│  高级框架                                           │
│  ├── LangGraph ── 图状流程（节点 + 边）              │
│  ├── LangServe ── 部署成 API                         │
│  └── LangSmith ── 调试 & 评估                        │
└──────────────────────────────────────────────────────┘
```

### ① Model I/O（模型输入输出）

| 子模块 | 作用 | 类比 |
|--------|------|------|
| **LLM / ChatModel** | 连接各种大模型 | 插头适配器 |
| **PromptTemplate** | 提示词模板 | 填空题模板 |
| **OutputParser** | 控制输出格式（JSON/CSV/HTML） | 格式化工具 |

### ② Chains（链）

**费曼说：** 就像 Linux 的管道符 `|`：

```bash
# Linux 管道：前一个命令的输出 → 下一个命令的输入
cat file.txt | grep "keyword" | wc -l

# LangChain 链：前一个组件的输出 → 下一个组件的输入
prompt_template | model | output_parser
```

### ③ Memory（记忆）

大模型本身没有记忆，你问"我叫张三"，下一个问题它就忘了。Memory 就是帮它记住对话历史。

- **短时记忆：** 存当前会话的聊天记录
- **长时记忆：** 存到数据库，下次还能用

### ④ Agents（代理）

**费曼说：** 模型是"大脑"，Agent 给大脑加上"手"。

- 大脑：能思考、能回答问题
- 手：能调用工具（查天气、查数据库、发邮件、执行代码）

```
用户问 "今天北京天气"
  → 大脑思考：我需要查天气工具
  → 手（Agent）调用天气 API
  → 大脑组织回答："北京今天 25°C，晴"
```

### ⑤ Callbacks（回调）

处理成功/失败后的后续操作，类似异步通知。

---

## 8. LangChain 实战：模型调用

### 安装

```bash
pip install langchain langchain-openai langchain-community
```

> ⚠️ `langchain-openai` 和直接 `pip install openai` 是**不同的库**。
> `langchain-openai` 是 LangChain 对 OpenAI 的封装。

### 调用大模型

```python
# 方式一：直接调用（和之前差不多）
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key="你的API-KEY",
    model="qwen-turbo",          # 也可以是 gpt-4, gpt-3.5-turbo 等
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = llm.invoke("你好，请介绍一下你自己")
print(response.content)          # 直接拿到回答文本
# 额外信息：response.response_metadata 包含 token 用量等
```

### 多轮对话

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="你是一个个人助理，你的名字叫小助手"),
    HumanMessage(content="我的名字叫张三"),
    AIMessage(content="你好张三！我是小助手，有什么可以帮你的？"),
    HumanMessage(content="我叫什么名字？"),  # 测试记忆
]

response = llm.invoke(messages)
print(response.content)
# 输出：你叫张三！
```

> 💡 LangChain 把角色封装成了消息对象：
> - `SystemMessage` → 系统角色
> - `HumanMessage` → 用户
> - `AIMessage` → AI 助手

---

## 9. LangChain 实战：提示词模板 & 输出解析

### 提示词模板（PromptTemplate）

**费曼说：** 就是把提示词里的变量用 `{}` 占位，需要时填充。

```python
from langchain_core.prompts import ChatPromptTemplate

# 1. 定义模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个世界级的{role}专家"),
    ("human", "{input}")
])

# 2. 填充变量 → 得到完整的提示词
formatted = prompt.invoke({
    "role": "Python编程",
    "input": "请解释什么是装饰器"
})

print(formatted)
# 输出：messages=[SystemMessage(content='你是一个世界级的Python编程专家'),
#                HumanMessage(content='请解释什么是装饰器')]

# 3. 直接链式调用（模板 → 模型）
chain = prompt | llm
response = chain.invoke({
    "role": "Python编程",
    "input": "请解释什么是装饰器"
})
print(response.content)
```

### 输出解析器（OutputParser）

**费曼说：** 默认大模型返回的是文本。如果你想要 JSON、CSV 或者 HTML 格式，用输出解析器。

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# 1. 定义模板 + 输出解析器
parser = JsonOutputParser()

prompt = PromptTemplate(
    template="""请用 JSON 格式回答：
问题：{question}
回答格式：{{"answer": "你的回答", "confidence": 0.0-1.0}}""",
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 2. 链式调用
chain = prompt | llm | parser
result = chain.invoke({"question": "Python 是静态类型还是动态类型？"})

print(result)
# 输出：{"answer": "动态类型", "confidence": 0.95}
# 直接就是一个 Python 字典！不用自己解析 JSON 字符串了
```

---

## 10. LangChain 实战：向量存储 & RAG

这里是最精彩的部分——用 LangChain **一行代码**搞定之前几十行的向量存储操作！

### 向量存储（一行代码！）

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # 嵌入模型

# 1️⃣ 加载文档（支持网页、PDF、本地文件）
loader = WebBaseLoader("https://www.example.com/民法典")
documents = loader.load()  # 返回 Document 对象列表

# 2️⃣ 递归分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"分割成 {len(chunks)} 个片段")

# 3️⃣ ✨ 向量化 + 存储（核心就这一行！）
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-v1")  # 嵌入模型
)
# 上面这一行 = 遍历每个文档 → 调嵌入 API → 存 FAISS
# 在之前手写代码里，这至少需要 20-30 行
```

### LangChain RAG（完整代码 ~30 行）

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

# 1️⃣ 准备数据（假设已有 documents 列表）
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 2️⃣ 创建向量存储
vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings())

# 3️⃣ 创建检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # 每次返回 3 条

# 4️⃣ 创建大模型
llm = ChatOpenAI(model="qwen-turbo")

# 5️⃣ 创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业问答助手。根据以下已知信息回答用户问题。\n\n{context}"),
    ("human", "{question}")
])

# 6️⃣ ✨ 组装 RAG 链（核心就这一行！）
qa_chain = RetrievalQA.from_llm(
    llm=llm,
    retriever=retriever,
    prompt=prompt
)

# 7️⃣ 开始提问
response = qa_chain.invoke({"query": "成年人的监护职责是什么？"})
print(response['result'])
```

### 对比：手写 vs LangChain

```
┌──────────────────────────────────────────────┐
│  手写 RAG ≈ 150 行                            │
│  ├── VectorStore 类（连接、嵌入、增删改查）   │
│  ├── RAGChatbot 类（检索、拼接、调用模型）    │
│  ├── main.py（PDF 读取、分割、组装）          │
│  └── 自定义分割函数                           │
├──────────────────────────────────────────────┤
│  LangChain RAG ≈ 30 行                        │
│  ├── RecursiveCharacterTextSplitter 分割      │
│  ├── FAISS.from_documents 向量化+存储         │
│  ├── RetrievalQA.from_llm 组装 RAG 链         │
│  └── invoke 提问                              │
└──────────────────────────────────────────────┘
```

### 推荐实践路径

```
① 先手写 RAG（理解原理）→ ② 再用 LangChain（提效）
```

> 💡 建议把**手写版本**的每一步搞清楚，再用 LangChain 简化。这样出问题时，你知道底层在干什么。

---

## 11. 总结对比表

### 本节课知识点

| 主题 | 核心要点 | 代码量 |
|------|----------|--------|
| **向量数据库** | Chroma 增删改查、持久化存储 | 少量 |
| **手写 RAG** | PDF → 分割 → 向量化 → 检索 → 回答 | ~150 行 |
| **LangChain 模型调用** | `ChatOpenAI.invoke()` | 1 行 |
| **LangChain 提示词模板** | `ChatPromptTemplate` + 管道符 | 3 行 |
| **LangChain 输出解析** | `JsonOutputParser` 格式化输出 | 2 行 |
| **LangChain 向量存储** | `FAISS.from_documents()` | 1 行 |
| **LangChain RAG** | `RetrievalQA.from_llm()` | ~30 行 |

### LangChain 核心概念一句话

| 概念 | 一句话 |
|------|--------|
| **Model I/O** | 统一接口调各种模型 + 模板化提示词 + 格式化输出 |
| **Chain** | 像管道符一样串联多个组件 |
| **Memory** | 让大模型记住对话历史 |
| **Agent** | 给大模型装上"手"，让它能用工具 |
| **Retriever** | 从向量数据库搜索相关文档 |
| **Callback** | 成功/失败后的处理回调 |

### 下节课预告

- Agent（智能代理）详解
- LangGraph（图状流程）
- 更复杂的 RAG 优化

---

> **费曼学习法复习：** 假装你要给一个刚学 Python 的朋友讲这节课的内容——
> 1. "为什么要有向量数据库？"（普通数据库算不了"最像"）
> 2. "Chroma 怎么用？"（增删改查，像操作 Excel 一样简单）
> 3. "LangChain 是干嘛的？"（让 LLM 开发像搭积木一样方便）
> 4. "手写 RAG 和 LangChain RAG 有什么区别？"（150 行 vs 30 行）

---

*笔记整理于 2026-06-30 · 图灵 Python 课堂 · 基于课程 2-LangChain框架-2025-5-28*
