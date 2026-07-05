# 🧩 LangChain 框架深入（三）— 费曼学习笔记

> 课程来源：图灵 Python · 柏汌/百川老师  
> 学习日期：2025-06-08 ~ 2025-06-11  
> 上一课：[LangChain 框架深入（二）](LangChain框架深入(二)-学习笔记.md)

---

## 📋 目录

1. [Chain（链）——为什么都用 `invoke`？](#1-chain链为什么都用-invoke)
2. [LCEL——LangChain 表达式语言（管道符的秘密）](#2-lcel-langchain-表达式语言管道符的秘密)
3. [Runnable——一切皆可"运行"](#3-runnable一切皆可运行)
4. [复杂案例：并行 + 条件判断的旅游助手](#4-复杂案例并行--条件判断的旅游助手)
5. [Agent 进阶——Self-Reflection & Self-Ask](#5-agent-进阶self-reflection--self-ask)
6. [Tools（工具）详解](#6-tools工具详解)
7. [自定义工具——把你的函数变成 Tool](#7-自定义工具把你的函数变成-tool)
8. [Memory——让大模型记住你是谁](#8-memory让大模型记住你是谁)
9. [全课程总结 & RAG 进阶预告](#9-全课程总结--rag-进阶预告)

---

## 1. Chain（链）——为什么都用 `invoke`？

**费曼说：** 在前面的课程中，你有没有发现一个现象——**不管什么操作，都是调用 `.invoke()`**？

```python
llm.invoke("你好")                    # 模型调用
prompt.invoke({"role": "专家"})       # 提示词模板
chain.invoke({"input": "问题"})       # 链调用
retriever.invoke("查询内容")          # 检索器
agent_executor.invoke({"input": "..."})  # 代理
```

**这不是巧合！** LangChain 设计了一个核心接口叫 **Runnable**，所有组件都继承它，统一用 `.invoke()` 来调用。

### 三个调用阶段

```
没有链之前：          提示词 → 手动传 → 大模型
                                              |
大模型链（LLMChain）： 提示词 + 大模型 → 封装成链 → invoke
                                              |
LCEL 管道符：          prompt | model | parser → 优雅的 |
```

---

## 2. LCEL（LangChain 表达式语言）——管道符的秘密

**费曼说：** LCEL 就是 LangChain 的管道操作符 `|`，类似 Linux 的 `|`：

```bash
# Linux 管道：前一个输出 → 后一个输入
cat file.txt | grep "hello" | wc -l
```

```python
# LCEL 管道：前一个组件的输出 → 后一个组件的输入
chain = prompt | model | parser
result = chain.invoke({"question": "你是谁？"})
```

### 管道符执行流程

```
用户输入 {"question": "讲个冰淇淋的笑话"}
         │
         ▼
┌─────────────────┐
│  PromptTemplate  │  ← 把变量填入模板
│  "讲个关于{question}的笑话"            │
│  输出: "讲个关于冰淇淋的笑话"          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ChatOpenAI      │  ← 调用大模型
│  输出: "为什么冰淇淋不打架？..."       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  OutputParser    │  ← 格式化输出
│  输出: {"joke": "..."}                │
└────────┬────────┘
         │
         ▼
      最终结果
```

---

## 3. Runnable——一切皆可"运行"

### 什么是 Runnable？

**费曼说：** Runnable 是一个**统一接口**，只要一个类"实现了 Runnable"，它就能：
1. 用 `|` 管道符连接
2. 调用 `.invoke()` 执行
3. 支持批量 `.batch()`、异步 `.ainvoke()` 等

```python
from langchain_core.runnables import Runnable

# 所有核心组件都继承自 Runnable：
# PromptTemplate → BasePromptTemplate → RunnableSerializable → Runnable
# ChatOpenAI     → BaseChatModel    → BaseLanguageModel  → Runnable
# OutputParser   → BaseOutputParser → Runnable
```

### 查看继承链

```python
# 看看 PromptTemplate 的继承链
print(PromptTemplate.__mro__)
# 输出：(..., RunnableSerializable, Runnable, ...)
# 最顶层就是 Runnable！
```

### 只要实现 Runnable，就能用管道符

```python
from langchain_core.runnables import RunnableLambda

# 把自己的函数变成 Runnable！
def my_func(x):
    return f"处理结果: {x}"

runnable_func = RunnableLambda(my_func)

# 现在它就能用管道符了
chain = prompt | model | runnable_func
result = chain.invoke({"input": "测试"})
```

### Runnable 的核心方法

| 方法 | 作用 | 类比 |
|------|------|------|
| `.invoke(input)` | 单次调用（最常用） | 普通函数调用 |
| `.batch(inputs)` | 批量调用（传列表） | 同时处理多个输入 |
| `.stream(input)` | 流式输出 | 一个字一个字输出 |
| `.ainvoke(input)` | 异步调用 | 不阻塞主线程 |

### 批量调用示例

```python
prompt = PromptTemplate(
    template="{name}喜欢吃什么？",
    input_variables=["name"]
)
chain = prompt | llm

# 批量处理多个输入
results = chain.batch([
    {"name": "猪八戒"},
    {"name": "孙悟空"},
    {"name": "唐僧"}
])

for r in results:
    print(r.content)
# 猪八戒喜欢吃人参果
# 孙悟空喜欢吃仙桃
# 唐僧喜欢吃素斋
```

---

## 4. 复杂案例：并行 + 条件判断的旅游助手

**费曼说：** 这个案例把链的威力发挥到了极致——**并行执行 + 条件分支 + 多模型串联**。

### 功能需求

用户提问"今天故宫的天气怎么样？"，系统：
1. 从问题中提取**地点**和**查询类型**（天气/景点）
2. **并行查询：** 查天气（走搜索工具）+ 查景点介绍（走本地知识库）
3. **条件判断：** 如果用户只问景点，就不查天气
4. **汇总回答：** 综合天气和景点信息，给出旅游建议

### 架构图

```
用户输入: "今天故宫的天气怎么样？"
         │
         ▼
┌─────────────────────────────────────┐
│  第一步：提取信息（提示词 + 大模型）  │
│  输出: {"location": "故宫", "type": "weather"}  │
└────────────────┬────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│  第二步：条件分支（RunnableBranch）   │
│                                      │
│  如果 type 包含 "weather":           │
│    ┌─────────────────────────────┐   │
│    │  并行执行：                  │   │
│    │  ① 搜索天气（搜索工具）      │   │
│    │  ② 检索景点（向量数据库）    │   │
│    └─────────────────────────────┘   │
│                                      │
│  否则（只问景点）：                   │
│    ┌─────────────────────────────┐   │
│    │  只检索景点介绍              │   │
│    └─────────────────────────────┘   │
└────────────────┬──────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  第三步：生成建议（大模型 + 汇总）  │
│  输出: 故宫今天25°C，适合游览...    │
└─────────────────────────────────────┘
```

### 完整代码实现

```python
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# ========== 1. 初始化 ==========
llm = ChatOpenAI(model="qwen-turbo")
search_tool = TavilySearchResults(max_results=1, api_key="your_key")

# 向量数据库（已存好景点数据）
vector_store = Chroma(embedding_function=OpenAIEmbeddings())
retriever = vector_store.as_retriever()

# ========== 2. 第一步：从问题提取信息 ==========
extract_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个旅游助手。从用户问题中提取地点和查询类型。
以 JSON 格式返回: {{"location": "地点", "type": "天气/景点/行程"}}"""),
    ("human", "{input}")
])

extract_chain = extract_prompt | llm | JsonOutputParser()

# ========== 3. 并行（天气 + 景点） ==========
# 天气查询工具
def search_weather(x):
    location = x["location"]
    result = search_tool.invoke(f"{location}今天天气")
    return result

weather_tool = RunnableLambda(search_weather)

# 景点检索工具
def search_attraction(x):
    location = x["location"]
    docs = retriever.invoke(location)
    return docs[0].page_content if docs else "暂无景点信息"

attraction_tool = RunnableLambda(search_attraction)

# 并行执行
parallel_tasks = RunnableParallel(
    weather=weather_tool,
    attraction=attraction_tool,
    location=lambda x: x["location"]  # 直接传递地点
)

# ========== 4. 第二步：条件分支 ==========
branch = RunnableBranch(
    # 条件1：如果 type 包含 "weather"
    (lambda x: "天气" in x.get("type", ""), parallel_tasks),
    # 默认：只查景点
    lambda x: {
        "location": x["location"],
        "attraction": retriever.invoke(x["location"])[0].page_content
    }
)

# ========== 5. 第三步：生成最终建议 ==========
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的旅游顾问，请结合景点信息和天气给出建议。"),
    ("human", """地点: {location}
景点信息: {attraction}
天气情况: {weather}

请给出游览建议。""")
])

final_chain = final_prompt | llm

# ========== 6. 串联整个流程 ==========
full_chain = extract_chain | branch | final_chain

# ========== 7. 提问 ==========
result = full_chain.invoke({"input": "今天故宫的天气怎么样？"})
print(result.content)
# 输出：故宫今天温度21-33°C，适合户外活动。建议穿轻便衣物，注意防晒...
```

### 关键组件说明

| 组件 | 作用 |
|------|------|
| `RunnableLambda` | 把普通函数变成 Runnable，就能用管道符 |
| `RunnableParallel` | 并行执行多个任务，互不依赖 |
| `RunnableBranch` | 条件分支，类似 if/else |
| `lambda x: ...` | 匿名函数，提取/转换数据 |

> 💡 **设计思路：** 每个`|`就是一个处理步骤，数据像流水线一样流经每个组件。复杂的业务逻辑 = 拆成一个个小 Runnable + 用管道符组合。

---

## 5. Agent 进阶——Self-Reflection & Self-Ask

**费曼说：** 基础 Agent 只是"调用工具就完事了"。进阶 Agent 会**自我反思、自我纠错**。

### Agent 的四个进阶能力

| 能力 | 说明 | 类比 |
|------|------|------|
| **反思（Reflection）** | 执行前先自我检查 | 三思而后行 |
| **纠错（Self-Correct）** | 发现错了主动重试 | 发现拿错工具换一个 |
| **迭代学习（Iterative）** | 每次改进一点点 | 越用越聪明 |
| **可解释性（Explainable）** | 展示思考过程 | 深度思考模式 |

### 自省代理（Self-Reflection Agent）

```python
from langchain.agents import create_self_ask_with_search_agent
from langchain.agents import AgentExecutor
from langchain import hub
from langchain_community.tools.google_search import GoogleSearchResults

# 1. 创建搜索工具（谷歌搜索，每月 100 次免费）
search_tool = GoogleSearchResults(api_key="your_google_api_key")

# 2. 加载提示词模板
prompt = hub.pull("hwchase17/self-ask-with-search")

# 3. 创建自省代理
agent = create_self_ask_with_search_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)

# 4. 创建执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True,          # 显示反思过程
    handle_parsing_errors=True  # 自动纠错
)

# 5. 提问
result = agent_executor.invoke({"input": "iPhone 16 的售价是多少？"})
print(result['output'])
```

### Verbose 模式看到的反思过程

```
> 进入 AgentExecutor 链...
> 思考：用户想知道 iPhone 16 的价格，我需要搜索一下。

> 动作：调用 Google Search
> 输入：iPhone 16 price 2025

> 观察：搜索到结果...（反思）这个结果全面吗？需要进一步确认...

> 思考：结果显示了基本价格，让我再查一下具体配置对应的价格。

> 动作：调用 Google Search
> 输入：iPhone 16 各版本价格对比

> 观察：获得了更详细的信息...

> 思考：现在我有了足够的信息来回答。

> 最终回答：iPhone 16 起售价为 $799...
```

### Self-Ask 代理

**费曼说：** 它会自己问自己问题——从答案里再挖问题，层层深入。

```python
from langchain.agents import create_self_ask_with_search_agent

# 和上面类似，但使用不同的 agent type
agent = create_self_ask_with_search_agent(
    llm=llm,
    tools=[search_tool],
    prompt=hub.pull("hwchase17/self-ask-with-search")
)
```

> ⚠️ **实际经验：** LangChain 内置的这些自省/自问代理**效果有限**，真正的 Agent 开发在第三阶段会深入讲。现在先知道有这个概念。

### Agent vs Chain 的本质区别

| | **Chain（链）** | **Agent（代理）** |
|------|----------------|-------------------|
| **控制者** | 开发者（你） | 大模型自己 |
| **执行顺序** | 硬编码，固定顺序 | 大模型动态决定 |
| **工具选择** | 开发者指定 | 大模型根据描述选择 |
| **灵活性** | 低，但稳定 | 高，但可能失控 |
| **适用场景** | 流程固定的任务 | 需要决策的任务 |

---

## 6. Tools（工具）详解

**费曼说：** Agent 有了"手"（决策能力），还需要"工具"（实际操作能力）。工具就是 Agent 能操作的函数/API。

### 工具的三个核心属性

```python
tool.name        # 工具名字 → 大模型用它来识别
tool.description # 工具描述 → 大模型判断"该不该用这个"
tool.args        # 工具参数 → 大模型决定传什么值
```

**工具的名字和描述是写给大模型看的！** 写得太模糊，大模型会选错工具。

### 查看工具属性

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

print(f"名字: {wiki.name}")        # "wikipedia"
print(f"简介: {wiki.description}")  # "当需要回答关于人物、地点..."
print(f"参数: {wiki.args}")        # {'query': {'title': '查询', 'type': 'string'}}
```

### 内置工具一览

| 工具 | 来源 | 是否需要梯子 | 免费额度 |
|------|------|-------------|---------|
| **TavilySearch** | `tavily` | ✅ 需要 | 1000次/月 |
| **GoogleSearch** | `google-api-python-client` | ✅ 需要 | 100次/月 |
| **Wikipedia** | `wikipedia` | ✅ 需要 | 无限 |
| **DuckDuckGo** | `duckduckgo_search` | ✅ 需要 | 无限 |

---

## 7. 自定义工具——把你的函数变成 Tool

### 方式一：用 `@tool` 装饰器（推荐）

```python
from langchain.tools import tool

@tool
def add_numbers(a: int, b: int) -> int:
    """计算两个数字的和"""
    return a + b

# 查看工具属性
print(add_numbers.name)        # "add_numbers"
print(add_numbers.description) # "计算两个数字的和"
print(add_numbers.args)        # {'a': ..., 'b': ...}

# 调用方式（变成 Tool 后需要通过 .run()）
result = add_numbers.run({"a": 3, "b": 5})
print(result)  # 8
```

### 方式二：用 `Tool.from_function()`（更灵活）

```python
from langchain.tools import Tool

def query_order(order_id: str) -> str:
    """查询订单状态"""
    data = {"1024": "已发货", "1025": "配送中"}
    return data.get(order_id, "未找到订单")

order_tool = Tool.from_function(
    name="订单查询",
    func=query_order,
    description="根据订单ID查询订单状态",
    args_schema=None  # 也可用 pydantic 定义参数结构
)
```

### 方式三：用 `StructuredTool`（带参数校验）

```python
from langchain.tools import StructuredTool

def refund_policy(company_name: str) -> str:
    """查询公司退款策略"""
    if "公司" not in company_name:
        company_name += "公司"  # 自动纠正参数
    return f"{company_name}的退款策略：7天无理由退货"

refund_tool = StructuredTool.from_function(
    name="退款查询",
    func=refund_policy,
    description="查询某个公司的退款策略，需要传入公司名称"
)
```

### 自定义工具的重要经验

**大模型传参可能不准！** 比如你定义参数叫 `company_name`，大模型可能只传 `"tom"` 而不是 `"tom公司"`。

```python
# 解决方案：在工具函数里做容错
def refund_policy(company_name: str) -> str:
    if "公司" not in company_name:
        company_name += "公司"  # 容错处理
    return f"{company_name}的退款策略：..."
```

> 💡 **工具命名规范很重要！** 名字要精确、描述要清晰，不然大模型会"误会"。

### 把多个工具组装给 Agent

```python
# 准备多个工具
tools = [
    order_tool,     # 订单查询
    refund_tool,    # 退款查询
    search_tool,    # 网络搜索
]

# 创建 Agent
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# 提问——大模型会自动选择该用哪个工具
result = agent_executor.invoke({
    "input": "请问订单1024的状态是什么？"
})
# 大模型判断 → 应该用 order_tool → 提取ID=1024 → 查询 → 回答
```

---

## 8. Memory——让大模型记住你是谁

**费曼说：** 大模型本身没有记忆——你问"我叫张三"，下一句它就不记得了。Memory 就是帮它记住上下文。

### 原理：就做了一件事

```
没有 Memory：
  用户提问 → 大模型回答（每次都是独立的）

有 Memory：
  用户提问 → 先读取历史记录 → 拼到提示词里 → 大模型回答
  → 把回答存到历史记录里
```

### 简单粗暴的实现（不用框架）

```python
# Memory 的本质就是——把历史对话存下来，下次提问时拼到提示词里！
history = []

# 第一轮
history.append(HumanMessage(content="我叫张三"))
response1 = llm.invoke(history)
history.append(AIMessage(content=response1.content))

# 第二轮——带着历史记录一起问
history.append(HumanMessage(content="我叫什么名字？"))
response2 = llm.invoke(history)
print(response2.content)  # 你叫张三！
```

### 用 LangChain Memory

#### 1. 创建消息存储

```python
from langchain_community.chat_message_histories import ChatMessageHistory

history = ChatMessageHistory()
print(history.messages)  # []

# 添加消息
history.add_user_message("你好")
history.add_ai_message("你好！今天过得怎么样？")
history.add_user_message("我是百川")
history.add_ai_message("你好百川！有什么可以帮你的？")

print(history.messages)
# [HumanMessage('你好'), AIMessage('你好！'), HumanMessage('我是百川'), AIMessage('你好百川！')]
```

#### 2. 使用 `RunnableWithMessageHistory`（新版推荐）

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# 创建提示词模板（注意有个占位符 {history}）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手。"),
    ("placeholder", "{history}"),   # ← 占位符，会被历史消息替换
    ("human", "{input}")
])

# 基础链
chain = prompt | llm

# 消息历史存储（用字典模拟，实际用数据库）
store = {}

def get_session_history(session_id: str):
    """获取某个用户的历史记录"""
    if session_id not in store:
        store[session_id] = []
    return store[session_id]

# 包装带记忆的链
chain_with_memory = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",    # 用户输入对应的 key
    history_messages_key="history" # 历史消息对应的 key（要跟 placeholder 一致）
)

# 第一轮对话
result1 = chain_with_memory.invoke(
    {"input": "你好，我是张三"},
    config={"configurable": {"session_id": "user_001"}}
)
print(result1.content)

# 第二轮——大模型记得上面说的话
result2 = chain_with_memory.invoke(
    {"input": "我叫什么名字？"},
    config={"configurable": {"session_id": "user_001"}}
)
print(result2.content)  # 你叫张三！
```

#### 3. 持久化存储（存到 JSON 文件）

```python
import json

def save_history(session_id: str, history: list, filepath="./history.json"):
    """把聊天记录存到 JSON 文件"""
    # 加载已有数据
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    except:
        all_data = {}
    
    # 把消息对象转成字典
    all_data[session_id] = [
        {"type": type(msg).__name__, "content": msg.content}
        for msg in history
    ]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

def load_history(session_id: str, filepath="./history.json"):
    """从 JSON 文件恢复聊天记录"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    except:
        return []
    
    records = all_data.get(session_id, [])
    history = []
    for r in records:
        if r["type"] == "HumanMessage":
            history.append(HumanMessage(content=r["content"]))
        elif r["type"] == "AIMessage":
            history.append(AIMessage(content=r["content"]))
    return history
```

### Memory 的注意事项

| 问题 | 说明 |
|------|------|
| **Token 消耗** | 历史消息越多，Token 消耗越大 |
| **对话长度** | 实际中一般只保留最近 N 轮（比如 5 轮） |
| **Session ID** | 用用户 ID 区分不同用户的记忆 |
| **持久化** | 内存存储关了就丢，生产环境用 Redis/数据库 |

---

## 9. 全课程总结 & RAG 进阶预告

### LangChain 框架全系列总结

```
┌─────────────────────────────────────────────────────────────┐
│                   LangChain 知识体系                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  第一课：认识 LangChain                                      │
│  ├── Model I/O（模型调用、提示词模板、输出解析）             │
│  ├── 向量存储（Chroma/FAISS，一行代码存库）                  │
│  └── RetrievalQA（RAG 链，30 行 vs 手写 150 行）             │
│                                                              │
│  第二课：深入核心组件                                        │
│  ├── Agent vs Function Call（手 vs 工具）                    │
│  ├── PromptTemplate 三种用法                                 │
│  ├── 模型三兄弟（LLM / ChatModel / Embedding）               │
│  └── 数据检索管线（加载→分割→向量化→存储→检索器）            │
│                                                              │
│  第三课：链 & Agent 实战                                     │
│  ├── LCEL 表达式语言（管道符 | 的秘密）                      │
│  ├── Runnable 接口（一切皆可 invoke）                         │
│  ├── RunnableParallel（并行执行）                             │
│  ├── RunnableBranch（条件分支）                               │
│  └── Tools 详解（内置工具 + 自定义工具）                      │
│                                                              │
│  第四课：高级 Agent & Memory                                 │
│  ├── 自省代理（Self-Reflection）                              │
│  ├── Memory（记忆原理 + 持久化）                              │
│  └── 全系列知识体系回顾                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

> **费曼复习：** 用一句话说清楚这四个核心概念——
> - **Runnable**：所有组件都有的"统一接口"，所以都能用 `invoke`
> - **LCEL**：用 `|` 把 Runnable 串起来的"管道语言"
> - **Agent**：大模型自己决定"用什么工具、按什么顺序"
> - **Memory**：每次把历史对话拼到提示词里，假装大模型有记忆

---

*笔记整理于 2026-06-30 · 图灵 Python 课堂 · 基于课程 4/5-LangChain框架-3/4-2025-6-8/11*

