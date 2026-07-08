# Agent 智能体开发（第二节课）— 初见老师

## 1. 课程元信息

- **课程主题**：Function Calling 实战 + Agent 核心认知框架
- **主讲老师**：初见老师
- **适用阶段**：进阶（需先完成第1节 Agent 智能体课程）
- **前置知识**：
  - Agent 基础概念（记忆、工具使用、落地难原因）
  - Function Calling 基础概念与核心流程
  - Python 基础 + Pandas 基础
  - HTTP 请求基础知识
- **时长**：约 2 小时 20 分钟

---

## 2. 核心概念图谱

### 2.1 Function Calling 实战概念

| 中文术语 | 英文术语 | 通俗解释 |
|---------|---------|---------|
| 动宾结构命名 | Verb-Object Naming | 函数命名规范，如 `calculate_salary_statistics`（计算_薪资_统计） |
| 结构化参数 | Structured Parameters | 用 JSON Schema 清晰定义每个参数的名称、类型、约束 |
| 函数描述规范 | Function Description Spec | 函数描述必须包含：功能说明、参数约束、返回值规范 |
| 函数库 | Function Registry | 以字典形式存储函数名到函数对象的映射，用于手动调用 |
| 意图识别 | Intent Recognition | 大模型理解用户自然语言请求，判断是否需要调用函数 |
| 工具角色消息 | Tool Role Message | 将函数返回值包装为 role="tool" 的消息格式回传大模型 |
| 手动函数调用 | Manual Function Invocation | 大模型仅输出调用意图，实际执行由外部代码完成 |

### 2.2 Agent 认知框架概念

| 中文术语 | 英文术语 | 通俗解释 |
|---------|---------|---------|
| 规划与执行 | Plan & Execute | Agent 先将任务拆解为步骤计划，再逐步执行并评估结果 |
| 自我询问 | Self-Ask | Agent 在回答问题前，先向自己提出多个子问题并逐一回答 |
| 思考与反思 | Think & Reflect | 模拟人类复杂决策过程，执行后反思并调整 |
| 反应式智能体 | ReAct Agent | 通过"思考→行动→观察→再思考"循环完成任务 |
| 中间步骤 | Intermediate Steps | 记录 Agent 历史思考与行动轨迹，供后续步骤参考 |
| 重试机制 | Retry Mechanism | Agent 执行失败时自动重试的配置 |
| 超时机制 | Timeout Mechanism | 限制 Agent 单次执行的最长时间 |
| 意图识别前置 | Pre-Intent Recognition | 在 ReAct 循环前先识别用户意图，决定使用哪个工具 |

### 2.3 模型与工具概念

| 中文术语 | 英文术语 | 通俗解释 |
|---------|---------|---------|
| 国产大模型 | Domestic LLM | 智谱 ChatGLM、DeepSeek、千问、文心一言、星火等均支持 function calling |
| 硅基流动 | SiliconFlow | 国内模型部署平台，可查看各模型是否支持工具调用功能 |
| 谷歌搜索 API | Google Search API | 用于 Agent 实时获取网络信息的搜索工具 |
| 本地嵌入模型 | Local Embedding Model | 在本地运行的文本向量化模型，避免消耗 API token |

---

## 3. 技术原理 / 流程拆解

### 3.1 Function Calling 手动调用完整流程

```
[用户提问] 
    → [首次调用大模型（带函数定义）]
        → [大模型判断是否调用函数]
            ├─ 否 → [直接回复]
            └─ 是 → [输出 tool_calls（含函数ID/名称/参数）]
                → [从函数注册表找到对应函数]
                → [手动执行函数（外部代码）]
                → [获取函数返回值]
                → [将返回值包装为 role="tool" 消息，追加到消息列表]
                → [二次调用大模型（带工具返回结果）]
                → [大模型生成最终自然语言回复]
```

### 3.2 函数定义规范（SOP）

**Step 1 — 命名规范（动宾结构）**
```
✅ calculate_salary_statistics    (计算_薪资_统计)
✅ query_train_tickets            (查询_火车_票)
✅ get_current_date               (获取_当前_日期)
✅ query_mysql_data               (查询_MySQL_数据)
❌ calc                           (过于模糊)
❌ func1                          (无意义)
```

**Step 2 — 参数定义（结构化）**
```json
{
    "name": "calculate_salary_statistics",
    "parameters": {
        "type": "object",
        "properties": {
            "employee_data": {
                "type": "string",
                "description": "包含员工数据的JSON格式字符串"
            }
        },
        "required": ["employee_data"]
    }
}
```

**Step 3 — 描述编写（三段式）**
1. **功能说明**：函数是做什么的（如"计算薪资统计信息，包括平均值、中位值、最高值和最低值"）
2. **参数约束**：每个参数的说明、类型、是否必填
3. **返回值规范**：返回值的格式（如"以JSON格式返回薪资统计信息"）

### 3.3 Agent 认知框架分类

#### 框架一：规划与执行 (Plan & Execute)

```
[用户输入] 
    → [任务理解] → [是否需要拆分？]
        ├─ 是 → [任务分解] → [生成执行计划（步骤列表）]
        │       → [按步骤逐一执行] → [观察每步结果]
        │       → [评估是否达成目标] → [未达成则调整计划]
        │       → [达成则返回最终结果]
        └─ 否 → [直接执行]
```

**特点**：先规划再执行，类似 ReAct 但更结构化

#### 框架二：自我询问 (Self-Ask)

```
[用户问题] 
    → [模型自我提问1] → [回答1]
    → [模型自我提问2] → [回答2]
    → ...
    → [综合所有子回答] → [生成最终答案]
```

**适用场景**：多步推理、涉及多个知识点的问题（如数学计算、复合查询）

#### 框架三：ReAct（思考→行动→观察）

```
[用户输入] 
    → [思考（Thought）]：分析当前状态和下一步该做什么
    → [行动（Action）]：决定调用哪个工具及传入参数
    → [观察（Observation）]：获取工具返回结果
    → [再思考]：基于观察结果继续推理
    → ... 循环直到得出最终答案
    → [最终回答（Final Answer）]
```

### 3.4 ReAct 参数配置详解

| 参数 | 作用 | 说明 |
|------|------|------|
| `max_retries` | 最大重试次数 | 工具调用失败时自动重试 |
| `max_execution_time` | 超时时间（秒） | 超过时间强制返回 |
| `early_stopping_method` | 停止条件 | Agent 判断何时停止思考 |
| `return_intermediate_steps` | 返回中间步骤 | 调试用，可查看完整思考链 |
| `intermediate_steps` | **历史记录（必填）** | 将之前的思考/行动/观察填充到提示词中，供大模型参考 |

**`intermediate_steps` 的重要性**：
- 它的作用是将 Agent 之前的思考和行动记录全部填充到提示词中
- 方便大模型参考之前做了哪些规划，避免重复或偏离
- 没有这个参数，Agent 无法记住自己之前的思考过程
- **这是 ReAct 框架的必填参数**，缺少会直接报错

---

## 4. 案例 / 代码实战复盘

> **案例总览**：本节课通过 4 个递进式案例，从"理解 Function Calling 底层原理"到"对接真实数据源"，再到"对比 ReAct 自动调用方案"，完整覆盖了智能体调用外部工具的两种核心方式（手动 vs 自动）及其实际应用场景。

---

### 4.1 案例一：薪资统计 — 理解 Function Calling 的完整手动调用流程

- **教学目的**：**理解 Function Calling 的底层执行原理**（老师原话："第一个案例我们需要了解这个 function calling 它是如何去执行的"）
- **要回答的核心问题**：大模型说"我要调用函数"之后，到底发生了什么？
- **数据集**：模拟员工数据（姓名、年龄、薪资、部门、婚否、工作年限）
- **涉及函数**：
  1. `calculate_salary_statistics` — 计算薪资均值、中位值、最大/最小值
  2. `group_salary_by_department` — 按部门分组统计员工数量、平均薪资、平均年龄
  3. `filter_employees_by_condition` — 按条件过滤员工
  4. `calculate_experience_salary_correlation` — 计算经验与薪资相关性

**函数定义示范（展示三大命名规范）：**
```python
# 函数1：统计薪资信息 — 动宾结构命名
def calculate_salary_statistics(employee_data: str) -> str:
    """计算薪资统计信息：平均值、中位值、最高值、最低值"""
    try:
        df = pd.read_json(employee_data)  # JSON → DataFrame
        stats = {
            "mean": round(df["salary"].mean(), 2),
            "median": round(df["salary"].median(), 2),
            "max": round(df["salary"].max(), 2),
            "min": round(df["salary"].min(), 2),
            "range": round(df["salary"].max() - df["salary"].min(), 2)
        }
        return json.dumps(stats, ensure_ascii=False)
    except Exception as e:
        return f"计算错误: {str(e)}"

# 函数2：按部门分组统计
def group_salary_by_department(employee_data: str) -> str:
    """按部门统计员工数量、平均薪资、平均年龄"""
    df = pd.read_json(employee_data)
    result = df.groupby("department").agg({
        "name": "count",
        "salary": "mean",
        "age": "mean"
    }).round(2)
    return result.to_json(force_ascii=False)

# 更多函数：过滤条件、经验薪资相关性分析等...
```

**手动调用 Function Calling 完整流程（核心代码）：**
```python
import json

# 定义函数注册表（字典映射：函数名 → 函数对象）
function_registry = {
    "calculate_salary_statistics": calculate_salary_statistics,
    "group_salary_by_department": group_salary_by_department,
    # ... 更多函数
}

# 函数定义（JSON Schema格式，传给大模型）
functions = [
    {
        "name": "calculate_salary_statistics",
        "description": "计算薪资统计信息...",
        "parameters": {...}
    },
    # ... 更多函数定义
]

# ── 第一步：首次调用大模型，判断是否需要调用函数 ──
response = llm.chat(messages=messages, functions=functions)
# 大模型返回 tool_calls（如果决定调用函数的话）

if response.tool_calls:
    tool_call = response.tool_calls[0]
    func_name = tool_call.function.name      # 获取函数名称
    func_args = json.loads(tool_call.function.arguments)  # 获取参数
    
    # ── 第二步：手动执行函数（大模型不执行，由外部代码执行）──
    func = function_registry[func_name]
    result = func(**func_args)
    
    # ── 第三步：将结果包装为 tool 角色消息，追加到对话历史 ──
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result
    })
    
    # ── 第四步：第二次调用大模型，基于工具结果生成自然语言回答 ──
    final_response = llm.chat(messages=messages)
```

**输入/输出分析：**
| 模块 | 输入 | 处理 | 输出 |
|------|------|------|------|
| `calculate_salary_statistics` | JSON格式员工数据字符串 | 解析JSON → 计算均值/中位/最大/最小 | JSON格式统计结果 |
| `group_salary_by_department` | JSON格式员工数据字符串 | group by department → 聚合统计 | JSON格式分组结果 |
| 大模型（首次调用） | 用户问题 + 函数定义 | 意图识别 → 选择函数 → 输出参数 | tool_calls 对象（含函数ID/名称/参数） |
| 外部代码（手动执行） | 函数名 + 参数 | 从注册表中查找函数并执行 | 函数返回值 |
| 大模型（二次调用） | 原消息 + tool 返回结果 | 基于函数结果生成自然语言回复 | 自然语言总结 |

> **⭐ 这个案例要说明的核心道理**：
> 1. **大模型只负责"决策"不负责"执行"** — 它告诉你"该调用哪个函数、传什么参数"，但实际执行必须由外部代码完成
> 2. **这是一个两轮对话**：第一轮定策略，第二轮做总结
> 3. **这是最底层的实现方式** — Agent 框架（如 LangChain）内部就是把你的工具自动封装成这种 JSON 格式传给大模型的

---

### 4.2 案例二：爬虫查车票 — 多工具协作解决"大模型不知道的事"

- **教学目的**：**展示多工具协作场景**（老师原话："就像现在你去问大模型，你觉得它能实时帮你查车票吗？不可能"），让大模型通过组合多个函数弥补自身知识的局限性（不知道实时日期、不知道实时车票）
- **要回答的核心问题**：大模型不知道今天几号、不知道实时车票，怎么办？——给它封装工具！
- **关键创新**：任务需要的两个函数之间存在**依赖关系**（需先获取日期，再用日期查票），大模型会自动识别并按顺序调用

**工具1 — 获取当前日期：**
```python
from datetime import datetime

def get_current_date() -> str:
    """返回当前日期（年月日）"""
    return datetime.now().strftime("%Y年%m月%d日")
```

**工具2 — 查询车票（爬虫）：**
```python
import requests

def query_train_tickets(date: str, from_station: str, to_station: str) -> str:
    """根据日期、出发站、终点站查询火车票信息"""
    from_code = city_map[from_station]  # 城市名称 → 代码映射
    to_code = city_map[to_station]
    url = f"https://...{date}...{from_code}...{to_code}..."
    response = requests.get(url, headers=headers)
    return parse_ticket_data(response.json())
```

**执行流程（多工具自动协作）：**
```
用户：查询明天长沙到上海的票
                          ↓
第一轮大模型调用 → 大模型识别：我需要知道今天几号
              → 调用 get_current_date()
              → 返回 "2025年7月25日"
                          ↓
第二轮大模型调用 → 大模型计算：明天是7月26日
              → 调用 query_train_tickets("2025-07-26", "长沙", "上海")
              → 返回车票信息列表（JSON）
                          ↓
第三轮大模型调用 → 大模型总结车票信息
              → 以表格/列表形式自然语言回复给用户
```

> **⭐ 这个案例要说明的核心道理**：
> 1. **Function Calling 可以串联多个工具完成复杂任务** — 大模型会自动分析"我需要先做什么，再做什么"
> 2. **大模型的知识局限性可以用工具弥补** — 不知道日期就给日期工具，不知道实时数据就给爬虫工具
> 3. **这种"多工具自动编排"正是 Agent 思想的雏形** — 只不过这里靠的是 Function Calling 手动编排，而 Agent 是自动编排
> 4. ⚠️ 爬虫代码需要 `city.json` 城市映射文件，且频繁请求 12306 会封 IP，仅做演示用

---

### 4.3 案例三：MySQL 查询 — Function Calling 对接真实数据库

- **教学目的**：**展示 Function Calling 是国产大模型的标配能力**（老师原话："这个 function calling 会成为所有大模型的标配"），以及如何对接真实世界的数据库
- **要回答的核心问题**：大模型不懂你的数据库表结构，怎么让它帮你写 SQL？
- **数据模型**：电商场景（用户表 → 订单表 → 订单明细表 → 商品表）

**表结构参考信息（必须提供给大模型）：**
```sql
-- 用户表：user(id, name, age)
-- 商品表：product(id, name, price)
-- 订单表：order(id, user_id, order_date)
-- 订单明细表：order_detail(id, order_id, product_id, quantity)
```

**关键函数实现：**
```python
def query_mysql_data(sql: str) -> str:
    """执行MySQL查询并返回JSON格式结果"""
    connection = pymysql.connect(
        host="localhost", user="root", 
        password="...", database="..."
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            data = [dict(zip(columns, row)) for row in result]
            return json.dumps(data, ensure_ascii=False, default=str)
    finally:
        connection.close()
```

**完整交互流程：**
```
用户："张三买了哪些商品？"
                          ↓
大模型收到问题 + 表结构参考信息
                          ↓
大模型生成 SQL（参考了提供的表字段名）：
  → SELECT p.name FROM user u 
     JOIN order o ON u.id = o.user_id
     JOIN order_detail od ON o.id = od.order_id
     JOIN product p ON od.product_id = p.id
     WHERE u.name = '张三'
                          ↓
外部代码执行 SQL → 返回 [{"name": "iPhone 15 Pro"}, {"name": "iPhone 15 Pro"}]
                          ↓
大模型总结回复："张三购买了 2 部 iPhone 15 Pro。"
```

> **⭐ 这个案例要说明的核心道理**：
> 1. **Function Calling 是国产大模型的"标配能力"** — 千问、智谱、DeepSeek、文心一言、星火都支持
> 2. **大模型需要"表结构参考"才能生成正确的 SQL** — 它不知道你的数据库长什么样，必须显式提供字段信息
> 3. **大模型生成 SQL 但不执行 SQL** — 执行由外部代码完成（遵循 Function Calling 的"模型决策、代码执行"原则）
> 4. **对智谱 ChatGLM 的特别说明**：可本地部署 6B 模型，但仅支持非流式调用

---

### 4.4 案例四：ReAct Agent — 工作中更常用的工具调用方案

- **教学目的**：**对比 Function Calling（手动调用）与 ReAct（自动调用）的差异**，展示工作中更常用的方案
- **要回答的核心问题**：Function Calling 理解原理就够了，工作中怎么用更方便？
- **关键区别**：Function Calling 需要你手动写 `if tool_calls:` 逻辑，而 ReAct Agent 自动管理"思考→行动→观察"循环

**自定义工具集（共4个）：**
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI

# 工具1：数学计算（含异常字符预处理）
def calculator(expression: str) -> str:
    """数学计算工具 — 注意：需处理模型传入的异常字符"""
    lines = expression.split("\n")
    clean_expr = lines[0].strip()  # 取第一行，去除换行符导致的计算错误
    try:
        return str(eval(clean_expr))
    except:
        return f"计算错误: {clean_expr}"

# 工具2：内部知识库查询（可替换为完整RAG）
def knowledge_base(query: str) -> str:
    """查询内部知识库 — 可用于公司内部文档检索"""
    return kb_search(query)

# 工具3：网络搜索（Tavily / 谷歌搜索）
search_tool = TavilySearch()

# 工具4：维基百科查询
wiki_tool = WikipediaQueryRun()
```

**ReAct Agent 构建与配置：**
```python
# 所有工具放入列表
tools = [calculator, knowledge_base, search_tool, wiki_tool]

# 创建 ReAct Agent（使用 LangChain）
llm = ChatOpenAI(model="qwen-max", ...)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_execution_time=30,       # ⏱ 超时控制：30秒
    max_retries=3,               # 🔄 重试机制：最多3次
    early_stopping_method="generate",  # 停止条件
    return_intermediate_steps=True,     # 调试用：返回中间思考步骤
    handle_parsing_errors=True          # 解析错误处理
)

# ⚠️ intermediate_steps 是必填参数，不能省略
result = agent_executor.invoke({
    "input": "小猫爱学的休假类型是什么？",
    "intermediate_steps": []     # 将历史思考/行动记录填充到提示词中
})
```

> **`intermediate_steps` 参数的重要性**（老师特别强调）：
> 它的作用是把 Agent 之前「思考→行动→观察」的完整轨迹填充到下一轮的提示词中。这样大模型在下一轮思考时能参考"我之前已经做了什么、做到了哪一步"，避免重复或偏离。**它是 ReAct Agent 的必需品，不传会直接报错。**

**ReAct 循环演示（以"5000元预算去东京能住几晚"为例）：**
```
用户："5000人民币预算，去东京旅行5天，按当前汇率能住几晚中等酒店？"

思考1：需要查找当前人民币对日元汇率
行动1：search_tool("人民币对日元汇率2025")
观察1：汇率为 20.65 (1人民币 = 20.65日元)

思考2：需要查找东京中等酒店价格
行动2：search_tool("东京中等价位酒店平均价格")
观察2：平均价格为 15000日元/晚

思考3：计算5000人民币能换多少日元，能住几晚
行动3：calculator("5000 * 20.65 / 15000")
观察3：≈ 6.88晚

思考4：用户旅行5天，6.88晚足够覆盖，给出建议
最终回答："按当前汇率，5000元约可兑换103250日元，
           东京中等酒店约15000日元/晚，可住6晚，
           完全覆盖您5天的住宿需求。"
```

**调试经验分享 — 计算器工具的异常处理：**
```python
# 问题：大模型传入的表达式带多余换行符和特殊字符
# 例如传入 "5000 * 20.65\n\n" 导致 eval 报错 "语法无效"

# 解决方案：对表达式做预处理
lines = expression.split("\n")
clean_expr = lines[0].strip()   # ✅ 只取第一行，去除空白
result = eval(clean_expr)        # ✅ 现在可以正常计算
```

> **⭐ 这个案例要说明的核心道理**：
> 1. **ReAct 是对比 Function Calling 的"自动挡"方案** — 你不需要手动写两轮调用 + `if tool_calls` 判断，Agent 自动管理整个思考-行动-观察循环
> 2. **工作中更常用 ReAct** — 老师原话："Function Calling 工作不常用，用的多的还是 agent + tools"
> 3. **提示词至关重要** — ReAct 的 prompt 必须清晰说明可用工具和思考方式，否则 Agent 表现会大打折扣
> 4. **小模型不适用于 ReAct** — 本地小模型会把 function calling 能力裁剪掉，建议使用千问 Plus/Max 或等效的线上模型
> 5. **集成 RAG 的两种方式**：
>    - 方式一：在 ReAct 循环前加意图识别，判断是否走知识库
>    - 方式二：把 RAG 封装成一个 tool，让 Agent 自己决定何时调用（老师推荐）

---

## 5. 避坑指南（隐性知识提取）

### 5.1 Function Calling 开发陷阱

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 模型返回非法表达式 | 大模型生成计算表达式时带多余字符（如换行符 `\n`） | 对表达式做预处理：`expr.split("\n")[0].strip()` |
| 函数名语义不清 | 大模型无法确定该用哪个函数 | 采用**动宾结构**命名，如 `calculate_salary_statistics` |
| 参数描述不完整 | 大模型不知道参数类型和约束 | 三段式描述：功能 + 参数约束 + 返回值规范 |
| 大模型不输出 tool_calls | 小模型（<8B/6B/4B）蒸馏时裁剪了 function calling 能力，**虽收到函数定义但无法产出结构化调用** | 使用支持 function calling 的模型（千问 Plus/Max，或 >= 16B 本地模型） |
| 国产模型敏感词限制 | 涉及政治/国际话题时返回"包含不合适内容" | 更换模型或调整问题表述 |
| 工具返回值格式不匹配 | ReAct 策略对工具返回格式有严格要求 | 引入 MCP 统一标准 |

### 5.2 模型选择与部署

| 问题 | 说明 | 建议 |
|------|------|------|
| 千问 Top 不支持 function calling | 主打性价比，裁剪了工具调用能力 | 使用千问 Plus 或 Max |
| 千问 7plus 最近更新 | 2025年7月14日更新，基于千问3 | 使用最新版千问 7plus 或 Max |
| DeepSeek 满血版 671B 支持 | 但蒸馏小模型（6B/8B）会删除 function calling | 本地部署至少 >= 16B |
| 智谱 ChatGLM 6B 可本地部署 | 支持 function calling 的小模型之一 | 可作为本地部署的备选 |
| 硅基流动查看模型能力 | 可筛选"工具调用"功能 | 选择模型前先查是否支持 |
| 本地嵌入模型 vs API | API 消耗 token 快（一天可能消耗完免费额度） | 嵌入模型建议用本地的 |

### 5.3 爬虫与数据获取

| 问题 | 说明 | 建议 |
|------|------|------|
| 12306 反爬 | 频繁请求会封 IP | 不要频繁爬取，仅做演示用 |
| 谷歌搜索 API 注册麻烦 | 需要验证，门槛较高 | 可用 Tavily 搜索替代（每月1000次免费） |
| 城市名称映射文件 | 爬虫需要城市代码映射 | `city.json` 文件必须存在 |

### 5.4 ReAct Agent 配置陷阱

| 参数/问题 | 说明 | 解决方案 |
|-----------|------|---------|
| **`intermediate_steps` 必须传入** | 缺少会直接报错 | 初始化时传入空列表 `[]` |
| 计算工具表达式异常 | 模型传入带换行的表达式 | 用 `split("\n")[0]` 预处理 |
| 千问敏感内容过滤 | 涉及政治话题返回错误 | 更换问题或模型 |
| 工具调用失败 | 格式不匹配导致返回错误 | 使用 MCP 或自建工具时做兼容处理 |
| 提示词必须写好 | ReAct 依赖提示词引导 | 清晰说明可用工具和思考方式 |
| Token 消耗大 | ReAct 多轮思考消耗大量 token | 小案例测试即可，复杂任务用专用框架 |

### 5.5 函数定义三大规范（老师强调重点）

1. **命名规范**：动宾结构，准确体现功能意图，如 `query_train_tickets`
2. **结构化参数设计**：核心参数放前面，参数名用下划线，保持语义连贯
3. **描述必须包含三要素**：
   - 函数功能是做什么的
   - 每个参数的约束（类型、是否必填）
   - 返回值的格式规范

---

## 6. 对比与思考

### 6.1 Function Calling vs ReAct 工具调用（深入对比）

| 维度 | Function Calling | ReAct 工具调用 |
|------|-----------------|---------------|
| **本质** | 大模型自带的函数调用能力 | Agent 的一种思考-行动策略 |
| **调用方式** | 手动调用（外部代码执行） | 自动调用（Agent 执行器编排） |
| **函数定义方式** | 手动编写 JSON Schema | 通过 LangChain Tool 封装 |
| **是否依赖模型能力** | 是，必须模型支持 function calling | 可通过 prompt 引导，模型要求较低 |
| **适用场景** | 简单的单次函数调用 | 复杂的多步推理任务 |
| **错误处理** | 需手动写 try/catch | 内置重试机制、超时机制 |
| **是否常用** | 工作中不常用，更多用于理解原理 | **工作中更常用**，灵活且稳定 |

**老师原话总结：**
> "Function Calling 工作不常用，用的多的还是 agent + tools。Function Calling 只是大模型的能力，让你理解它是怎么做的。Agent 它会自动封装成这种 JSON 格式。"

### 6.2 三种 Agent 认知框架对比

| 框架 | 工作原理 | 适用场景 | 优势 | 劣势 |
|------|---------|---------|------|------|
| **规划与执行 (Plan & Execute)** | 先拆解任务→生成计划→逐步执行→观察结果→调整 | 复杂多步骤任务、旅行规划 | 结构清晰、易于调试 | 灵活性相对较低 |
| **自我询问 (Self-Ask)** | 先自我提问子问题→逐个回答→综合得到最终答案 | 多步推理、数学计算、多知识点联合查询 | 推理链清晰 | 工具兼容性差，容易报错 |
| **ReAct (思考·行动·观察)** | 思考→行动→观察→再思考的循环 | **最通用、最常用** | 灵活、稳定、适应性强 | 提示词要求高，token消耗大 |

**老师结论**：目前用的最多的是 ReAct，其次是 Plan & Execute。

### 6.3 Function Calling 国产模型支持情况

| 模型厂商 | 模型名称 | 是否支持 Function Calling | 备注 |
|---------|---------|------------------------|------|
| 阿里云 | 千问 Plus / Max | ✅ 支持 | 推荐用于 Agent 开发 |
| 阿里云 | 千问 Top | ⚠️ 部分支持（能力被裁剪） | 不推荐用于 Agent |
| 智谱 AI | ChatGLM 6B / 6B-32K | ✅ 支持 | 可本地部署，仅支持非流式 |
| DeepSeek | DeepSeek V2 / R1 | ✅ 支持（满血版） | 蒸馏小模型不支持 |
| 百度 | 文心一言 | ✅ 支持 | — |
| 科大讯飞 | 星火 | ✅ 支持 | — |
| 硅基流动 | 平台模型 | 视具体模型而定 | 可筛选"工具调用"功能查看 |

### 6.4 工具集成方式演进

```
传统方式：每个工具自定义接口 → 格式不统一 → 调用失败率高
                                      ↓
引入 MCP：统一工具输入/输出标准格式 → 不再依赖模型适配格式
                                      ↓
              使 ReAct 等策略更加稳定可靠
```

### 6.5 ReAct 与 LangGraph 的关系

- **LangChain**：作为"脚手架"，提供构建 Agent 的基本组件
- **LangGraph**：专门负责**工作流编排**，是 Agent 的核心框架
- **趋势**：LangChain 已建议使用 LangGraph 构建 Agent
- **后续课程**：多 Agent 和 LangGraph 放在一起讲，因为 LangGraph 是构建多智能体的核心框架

### 6.6 课后思考与延伸

1. **老师核心观点**：
   - Function Calling 是理解 Agent 底层原理的基础，但工作中直接使用不多
   - 实际工作中更常用 Agent + Tools（ReAct 策略）
   - ReAct 的提示词一定要写好，直接影响 Agent 表现

2. **下节课预告**：
   - 多智能体（Multi-Agent）
   - LangGraph 框架

3. **建议延伸方向**：
   - 刷面试题（场景设计题为主，问项目怎么做、用什么架构）
   - 后续微调课程（约一个月周期）
   - 四个实战项目（涉及不同技术栈和优化方法）

4. **练习建议**：
   - 将 RAG 封装为 ReAct 的 tool
   - 尝试用 Debug 模式调试 ReAct 的计算工具问题

---

## 7. 本节课思维导图

- Agent 智能体开发（第二节课）
  - 课前回顾
    - 上节课概念回顾（记忆、工具、落地难、Function Calling）
    - MCP 简介（统一工具接口标准）
    - OpenAI 推荐 JSON 格式
  - Function Calling 实战
    - 函数定义规范（老师强调的重点）
      - 命名：动宾结构（calculate_salary_statistics）
      - 参数：结构化，核心参数优先，下划线命名
      - 描述：三段式（功能 + 参数约束 + 返回值规范）
    - 案例一：薪资统计
      - 数据集：员工信息（年龄/薪资/部门/婚否/年限）
      - 多个函数：统计薪资、分组统计、条件过滤、相关性分析
      - 手动调用流程：定义函数 → 注册表 → 首次调用 → 判断 → 手动执行 → 回传 → 二次调用
    - 案例二：爬虫查车票
      - 工具1：获取当前日期
      - 工具2：查询12306车票
      - 多工具协作：日期→计算明天→查票→总结
    - 案例三：国产大模型 Function Calling
      - 智谱 ChatGLM（可本地部署 6B，仅支持非流式）
      - 硅基流动平台筛选支持工具调用的模型
      - 千问 Plus/Max 推荐，Top 不推荐
    - 案例四：MySQL 查询
      - 需提供表结构信息
      - 大模型生成 SQL → 手动执行 → 结果回传
  - Agent 核心认知框架
    - 规划与执行 (Plan & Execute)
      - 任务理解 → 拆分 → 生成计划 → 逐步执行 → 观察评估
      - 适合复杂多步骤任务
    - 自我询问 (Self-Ask)
      - 自我提问 → 子回答 → 综合答案
      - 适合多步推理/数学计算
      - 工具兼容性较差，容易报错
    - ReAct（思考·行动·观察）
      - 最常用、最稳定的策略
      - 参数配置详解
        - `intermediate_steps`：必须传入，记录历史思考轨迹
        - `max_retries`：重试次数
        - `max_execution_time`：超时时间
        - `return_intermediate_steps`：调试用
      - 自定义工具实践
        - 计算器（需处理异常字符）
        - 知识库查询（可替换为 RAG）
        - 搜索工具
        - 维基百科工具
      - Debug 调试方法
        - 设断点 → Debug 模式 → F8 单步执行
        - 观察函数传入参数 → 解决表达式异常
    - 工具调用方式对比
      - Function Calling：手动调用，单次，理解原理用
      - ReAct：自动调用，多轮，工作中常用
      - Agent 底层可用 Function Calling 或 ReAct 实现
  - 趋势与展望
    - LangChain → LangGraph 迁移（Agent 构建建议用 LangGraph）
    - LangGraph 是后续多 Agent + 项目核心
    - MCP 将统一工具调用标准
    - 下阶段：多智能体 + LangGraph + 微调 + 四个实战项目
