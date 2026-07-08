# Agent 智能体开发（第三节课）— 初见老师

## 1. 课程元信息

- **课程主题**：多智能体系统（MAS）概述 + AutoGPT / CrewAI 简介 + LangGraph 框架核心概念
- **主讲老师**：初见老师
- **适用阶段**：进阶（需先完成第1、2节 Agent 智能体课程，理解 Function Calling 和 ReAct 原理）
- **前置知识**：
  - Agent 基础概念（单智能体、ReAct、Function Calling）
  - LangChain 框架基础
  - Python 类型注解（TypedDict）
  - 基本图论概念（有向无环图 DAG）
- **时长**：约 2 小时 20 分钟

---

## 2. 核心概念图谱

### 2.1 多智能体系统（MAS）概念

| 中文术语 | 英文术语 | 通俗解释 |
|---------|---------|---------|
| 多智能体系统 | Multi-Agent System (MAS) | 由多个具有智能和自主能力的个体（Agent）组成的协作系统 |
| 单智能体 | Single Agent | 只有一个智能体独立完成任务，类似于一个人包揽全部工作 |
| 并行处理 | Parallel Processing | 多个 Agent 同时处理不同子任务，提高整体效率 |
| 分布式框架 | Distributed Framework | 每个 Agent 独立模块化，可单独开发、测试、替换 |
| 可扩展性 | Scalability | 新增或移除 Agent 不影响系统整体运行 |
| 角色扮演 | Role Playing | 每个 Agent 被分配特定角色（如开发、测试、产品经理） |
| 群组对话 | Group Chat | 多个 Agent 之间通过对话机制进行协作交流 |
| 发言规则 | Speaker Rule | 定义多个 Agent 之间的发言顺序和交互规则 |

### 2.2 AutoGPT / AutoGen 概念

| 中文术语 | 英文术语 | 通俗解释 |
|---------|---------|---------|
| 配置文件 | Config File | 定义 Agent 角色、任务、终止条件的 YAML/JSON 文件 |
| 系统角色 | System Role | 给 Agent 设定的核心身份提示词（如"你是编程助手"） |
| 用户代理 | User Agent | 模拟用户提出需求的 Agent |
| 监督代理 | Supervisor Agent | 协调多个 Agent 对话顺序的管理者角色 |
| 终止条件 | Termination Condition | 判断任务何时完成的关键字或规则 |
| 群组聊天管理器 | Group Chat Manager | 管理多 Agent 对话流程的调度器 |
| 对话轮次限制 | Max Turns | 控制最大对话次数，防止无限循环 |

### 2.3 CrewAI 概念

| 中文术语 | 英文术语 | 通俗解释 |
|---------|---------|---------|
| 团队 | Crew | 一组具有不同角色的 Agent 组成的协作团队 |
| 任务 | Task | 分配给每个 Agent 的具体工作步骤 |
| 共享记忆 | Shared Memory | 多个 Agent 之间可以访问的共同记忆空间 |
| 流程 | Flow/Process | 定义 Agent 执行任务顺序的管线 |

### 2.4 LangGraph 核心概念

| 中文术语 | 英文术语 | 通俗解释 |
|---------|---------|---------|
| 状态图 | StateGraph | LangGraph 的核心构建块，基于状态驱动的图结构 |
| 状态 | State | 贯穿整个工作流的数据容器（字典类型），节点间共享信息 |
| 节点 | Node | 图中的每个处理单元，一个节点可以是一个 Agent 或一个功能函数 |
| 边 | Edge | 连接节点的有向路径，定义执行流向 |
| 条件边 | Conditional Edge | 根据当前状态决定下一步走向哪个节点的逻辑分支 |
| 子图 | Subgraph | 节点内部可嵌套的子工作流，形成树状结构 |
| 编译 | Compile | 构建图之后执行的校验步骤，确保图结构正确 |
| 持久化 | Persistence | 将工作流状态持久化存储，支持中断恢复 |
| 归并函数 | Reducer Function | 处理多个节点返回同一状态字段时的合并逻辑 |
| 多模式状态 | Multi-Schema State | 区分输入状态、中间状态、输出状态的不同模式 |

---

## 3. 技术原理 / 流程拆解

### 3.1 多智能体 vs 单智能体

```
单智能体：
[一个 Agent] → 感知 → 规划 → 行动 → 反思 → 输出
              （一个人完成所有工作）

多智能体（MAS）：
[Agent 1: 路径规划] ─┐
[Agent 2: 障碍检测] ─┼─→ [汇总决策] → [执行]
[Agent 3: 速度监控] ─┘
  （多个专家分工协作）
```

**多智能体的三大优势：**
1. **并行处理**：多个 Agent 同时处理不同子任务，比单 Agent 串行处理效率高
2. **分布式框架**：每个 Agent 独立模块，可单独测试、替换、升级
3. **可扩展性强**：新增或移除 Agent 不影响系统整体

### 3.2 AutoGPT 多智能体架构

```
[配置文件] → 定义角色/任务/终止条件
      ↓
[初始化 Agent] → 用户代理 / 编码代理 / 审查代理 / 测试代理
      ↓
[自定义发言规则] → 定义 Agent 之间的对话顺序
      ↓
[群组聊天管理器] → 调度对话流程，控制最大轮次
      ↓
[Agent 间交互] → 需求 → 编码 → 审查 → 测试 → 完成
```

**对话控制流程（示例：软件开发场景）：**
```
用户提需求 → 开发 Agent 编码 → 审查 Agent 审核代码
    ↑                              |
    └──── 不通过（重新编码） ←──────┘
    ↓
审查通过 → 测试 Agent 测试 → 任务完成 → 返回用户
```

### 3.3 LangGraph 核心架构

```
                    ┌─────────────┐
                    │   状态(State) │  ← 贯穿全局的字典
                    │  {"question":  │
                    │   "答案": ""}  │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐      ┌────▼────┐      ┌────▼────┐
    │ 节点1   │ ───→ │ 节点2   │ ───→ │ 节点3   │
    │ (Agent1)│ 边    │ (Agent2)│ 边    │ (Agent3)│
    └─────────┘      └─────────┘      └─────────┘
         │                                │
         └──── 条件边（不满足则返回）──────┘
```

**LangGraph 构建流程（SOP）：**

**Step 1 — 定义状态（State）**
```python
from typing_extensions import TypedDict

class GraphState(TypedDict):
    question: str      # 输入问题
    answer: str        # 最终答案
    # 可以根据需要添加更多字段
```

**Step 2 — 定义节点（Node）**
每个节点是一个普通函数，接收状态字典，返回更新后的字典：
```python
def search_node(state: GraphState) -> dict:
    """搜索节点：从状态中获取问题，执行搜索"""
    question = state["question"]
    result = search(question)
    return {"search_result": result}
```

**Step 3 — 构建图（Graph）**
```python
from langgraph.graph import StateGraph

# 创建状态图
graph = StateGraph(GraphState)

# 添加节点
graph.add_node("search", search_node)
graph.add_node("generate", generate_node)

# 设置入口点
graph.set_entry_point("search")

# 添加边
graph.add_edge("search", "generate")
graph.add_edge("generate", END)  # END 是终止节点

# 编译图（校验）
app = graph.compile()
```

**Step 4 — 执行**
```python
result = app.invoke({"question": "什么是LangGraph?"})
```

### 3.4 状态的多模式构建

```
输入状态 (Input Schema)    中间状态 (State)        输出状态 (Output Schema)
   ┌──────────┐          ┌──────────────┐          ┌──────────┐
   │ question  │ ──────→ │ question      │ ──────→ │ answer   │
   └──────────┘          │ search_result│          └──────────┘
                         │ answer        │
                         └──────────────┘
```

- **输入状态**：限定用户可以传入哪些字段
- **中间状态**：所有节点共享的完整状态（含临时数据）
- **输出状态**：限定最终可以输出哪些字段（隐藏中间过程）

### 3.5 LangGraph vs LangChain

```
LangChain 线性流程（有向无环图）：
  [输入] → [节点1] → [节点2] → [节点3] → [输出]
  （不能回到之前的步骤）

LangGraph 图状流程（支持循环）：
       ┌─────────────────────┐
       │                     ↓
  [输入] → [编码] → [审查] → [测试] → [输出]
                ↑              │
                └── 不通过 ────┘
  （可以回到之前的步骤，支持条件分支）
```

| 维度 | LangChain | LangGraph |
|------|-----------|-----------|
| **本质** | 构建大模型应用的**脚手架框架** | 构建**多智能体**的高级工作流框架 |
| **工作流结构** | 线性/顺序执行（有向无环图） | 图状结构（支持循环、分支、条件） |
| **状态管理** | 无内置状态管理 | **内置强大状态管理机制** |
| **多智能体** | 不原生支持 | **原生支持多Agent协作** |
| **复杂决策流** | 有限支持 | 支持条件边、子图等复杂逻辑 |
| **持久化** | 不支持 | 支持长时间运行和中断恢复 |
| **适用场景** | 简单到中等复杂度的 AI 应用 | 复杂多步骤、多智能体场景 |
| **生态关系** | 作为 LangGraph 的**底层依赖** | 与 LangChain **无缝集成** |

---

## 4. 案例 / 代码实战复盘

### 4.1 案例一：AutoGPT 模拟软件开发团队

- **教学目的**：展示多智能体**角色分工**的基本模式，理解对话式协作流程
- **要回答的核心问题**：多个 Agent 之间如何通过对话机制协同完成开发任务？
- **团队组成**：用户代理（提需求）→ 编码代理（写代码）→ 审查代理（审代码）→ 测试代理（跑测试）

**配置文件定义（config）：**
```python
# 角色1：用户代理
user_agent_config = {
    "name": "user_agent",
    "system_message": "你是用户代理，负责提出产品需求",
    "human_input_mode": "ALWAYS"   # 是否启用人机交互
}

# 角色2：编码代理
coder_agent_config = {
    "name": "coder",
    "system_message": "你是编程助手，负责编写高质量 Python 代码",
    "max_consecutive_auto_reply": 5  # 最大连续回复次数
}

# 角色3：审查代理
reviewer_config = {
    "name": "code_reviewer", 
    "system_message": "你是代码审查专家，负责审核代码质量和安全性",
}

# 角色4：测试代理
tester_config = {
    "name": "tester",
    "system_message": "你是测试工程师，负责编写和执行测试用例",
}

# 终止条件
termination_keywords = ["测试通过", "任务完成", "SUCCESS"]
```

**自定义发言规则（核心逻辑）：**
```python
def custom_speaker_rule(messages, agents):
    """自定义的发言顺序规则"""
    if len(messages) == 0:
        # 消息为空 → 用户先提需求
        return user_agent
    
    last_message = messages[-1]
    current_agent = last_message["agent"]
    
    if current_agent == user_agent:
        # 用户提完需求 → 编码
        return coder
    elif current_agent == coder:
        # 编码完成 → 检查是否包含 Python 代码
        if "python" in last_message["content"].lower():
            return reviewer   # 有代码 → 审查
        else:
            return coder      # 无代码 → 重新编码
    elif current_agent == reviewer:
        return tester         # 审查通过 → 测试
    elif current_agent == tester:
        return None           # 测试完成 → 结束
```

**群组聊天配置：**
```python
from autogen import GroupChat, GroupChatManager

# 创建群组聊天
group_chat = GroupChat(
    agents=[user_agent, coder, reviewer, tester],  # 参与本次任务的 Agent
    messages=[],                                     # 消息列表
    max_round=10,                                    # 最大发言轮次（老师设为2仅演示）
    speaker_selection_method=custom_speaker_rule,    # 自定义发言规则
    allow_repeat_speaker=False                       # 不允许同一人连续发言
)

# 创建管理器
manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"model": "qwen-plus", "temperature": 0}
)
```

**执行结果演示：**
```
用户需求：创建一个处理 CSV 文件的 Python 函数，计算数值列的统计信息

→ 编码 Agent：生成代码（含 pandas 读取、异常处理、函数封装、注释说明）
→ 审查 Agent：检查代码质量（老师说明：2轮限制下未完全展示）
→ 测试 Agent：生成测试用例
→ 任务完成
```

> **⭐ 核心道理**：
> 1. **提示词（Prompt）决定 Agent 行为** — 每个 Agent 的表现取决于 system_message 怎么写
> 2. **自定义发言规则至关重要** — 默认的随机/轮询规则无法满足特定业务流程
> 3. **对话轮次限制防止无限循环** — `max_round` 是必备的安全措施
> 4. **小模型可能陷入死循环** — 能力差的模型可能一直生成不合规的代码，永远通不过审查

---

### 4.2 案例二：CrewAI 多智能体协作（研究写作场景）

- **教学目的**：了解 CrewAI 的 Agent + Task + Crew 模式
- **要回答的核心问题**：CrewAI 如何通过"团队 + 任务管线"的方式组织多 Agent 协作？

**核心代码：**
```python
from crewai import Agent, Task, Crew, Process

# 1. 定义 Agent
researcher = Agent(
    role="情报研究员",
    goal="搜索和收集最新的技术信息",
    backstory="你是一名专业的技术情报研究员",
    tools=[search_tool, web_scraper]  # 可使用的工具
)

analyst = Agent(
    role="数据分析师",
    goal="分析收集到的技术数据，提炼洞察",
    backstory="你是一名资深数据分析专家"
)

writer = Agent(
    role="技术写手",
    goal="将研究成果整理成清晰的技术报告",
    backstory="你是一名优秀的技术内容创作者"
)

# 2. 定义 Task（每个 Agent 对应一个 Task）
research_task = Task(
    description="搜索 AI Agent 框架的最新发展",
    agent=researcher
)

analysis_task = Task(
    description="分析搜索到的技术信息，总结关键趋势",
    agent=analyst
)

writing_task = Task(
    description="将分析结果写成技术博客",
    agent=writer
)

# 3. 创建团队（Crew）并启动
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential  # 顺序执行
)

result = crew.kickoff()  # 启动工作流
```

> **⭐ 核心道理**：
> 1. **CrewAI 底层默认使用 ReAct 策略** — 每个 Agent 内部还是思考→行动→观察 的循环
> 2. **Task 是核心组织单位** — 每个 Agent 分配一个 Task，形成顺序执行管线
> 3. **共享记忆** — 多个 Agent 之间通过 Shared Memory 协作

---

### 4.3 案例三：LangGraph 构建最简单的工作流

- **教学目的**：理解 LangGraph 最核心的 3 个概念：**State（状态）、Node（节点）、Graph（图）**
- **要回答的核心问题**：LangGraph 的最小可运行工作流怎么写？

**完整代码：**
```python
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Step 1: 定义状态（State）— 贯穿整个工作流的字典
class GraphState(TypedDict):
    question: str
    answer: str

# Step 2: 定义节点（Node）— 普通函数，接收状态返回状态
def search_node(state: GraphState) -> dict:
    """搜索节点"""
    question = state["question"]
    result = f"关于 '{question}' 的搜索结果..."
    return {"search_result": result, "question": question}

def generate_node(state: dict) -> dict:
    """生成答案节点"""
    search_result = state["search_result"]
    answer = f"基于搜索结果生成的答案: {search_result}"
    return {"answer": answer}

# Step 3: 构建图（Graph）
workflow = StateGraph(GraphState)

# 添加节点
workflow.add_node("search", search_node)
workflow.add_node("generate", generate_node)

# 设置入口点
workflow.set_entry_point("search")

# 添加边（定义流向）
workflow.add_edge("search", "generate")
workflow.add_edge("generate", END)

# 编译（校验图结构）
app = workflow.compile()

# Step 4: 执行
result = app.invoke({"question": "LangGraph是什么？"})
print(result["answer"])
```

**状态校验演示：**
```python
# ⚠️ 状态字段名必须严格一致
# 定义状态为 question，节点中写 questions → 报错！
class GraphState(TypedDict):
    question: str    # 单数

def my_node(state: GraphState):
    q = state["questions"]  # ❌ 拼写错误！运行时会报 KeyError
```

> **⭐ 核心道理**：
> 1. **State 是整个图的"血液"** — 所有节点通过 State 共享和传递信息
> 2. **字段名必须严格一致** — 定义用的字段名和调用用的字段名拼写不同会直接报错
> 3. **compile() 做图结构校验** — 确保所有节点和边连接正确
> 4. **每个节点就是一个 Agent 或功能单元** — 多智能体就是把多个节点连起来

---

### 4.4 案例四：LangGraph 多模式状态构建

- **教学目的**：理解如何区分**输入状态**、**中间状态**和**输出状态**，实现信息隔离
- **要回答的核心问题**：如何让某些中间数据（如搜索原始结果）不被下游节点看到？

```python
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# 定义输入状态（用户只能传这些字段）
class InputState(TypedDict):
    question: str

# 定义中间状态（节点间共享所有字段）
class IntermediateState(TypedDict):
    question: str
    search_result: str
    answer: str

# 定义输出状态（最终只公开这些字段）
class OutputState(TypedDict):
    answer: str

# 节点1：搜索 — 使用 InputState 和 IntermediateState
def search_node(state: InputState) -> dict:
    question = state["question"]
    return {"search_result": f"搜索结果: {question}"}

# 节点2：生成答案 — 使用 OutputState
def generate_node(state: IntermediateState) -> dict:
    return {"answer": f"最终答案: {state['search_result']}"}

# 构建图时指定输入/输出模式
workflow = StateGraph(
    input=InputState,           # 只接收 question
    state=IntermediateState,    # 内部使用完整状态
    output=OutputState          # 只输出 answer
)

workflow.add_node("search", search_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("search")
workflow.add_edge("search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# 执行 — 用户只能传入 question
result = app.invoke({"question": "什么是智能体？"})
# result 只包含 answer 字段，search_result 对外隐藏
print(result)  # {"answer": "最终答案: ..."}
```

> **⭐ 核心道理**：
> 1. **输入/输出状态类似 API 接口的请求/响应** — 限定外部可见的字段
> 2. **中间状态是"内部实现细节"** — 可包含大量临时数据但不对外暴露
> 3. **典型应用场景**：不想让用户看到搜索的原始返回，只展示最终的整理结果

---

## 5. 避坑指南（隐性知识提取）

### 5.1 多智能体开发陷阱

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| Agent 陷入死循环 | 小模型能力差，持续生成不合规内容 | 设置 `max_round` 对话轮次上限 |
| 交互不流畅 | 提示词写得不好 | 精心设计每个 Agent 的 `system_message` |
| 对话顺序混乱 | 使用默认的随机/轮询规则 | **自定义发言规则**，按业务流程定义 |
| Token 消耗巨大 | 多 Agent 对话链产生大量token | 控制 max_round，使用性价比模型 |
| 代码审查缺失 | 模型未严格按规则执行 | 在提示词中明确"必须检查是否包含 Python 代码" |
| 状态字段名不一致 | 节点函数中拼写错误 | 使用 TypedDict 类型注解，IDE 会提示 |

### 5.2 LangGraph 开发陷阱

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 状态字段名拼写错误 | question 写成 questions | 严格保持一致，利用 TypedDict 类型校验 |
| 类型不匹配 | 传入 int 但定义要求 str | LangGraph 推荐字典方式（`TypedDict`），在节点内做类型校验 |
| 输入/输出模式配置错误 | input/output/state 三者的关系没理清 | 记住：input 限定入参，state 完整状态，output 限定出参 |
| 节点函数遗忘返回字段 | 节点未返回所需状态字段 | 确保每个 node 函数返回与状态定义一致的字段 |
| 不理解子图 | 认为所有 Agent 必须在同一层级 | 节点可以嵌套包含子工作流，形成树状结构 |

### 5.3 模型选择建议

| 问题 | 说明 | 建议 |
|------|------|------|
| 多智能体 Token 消耗大 | 多个 Agent 对话产生大量交互 | 使用性价比合适的模型 |
| 小模型能力不足 | 可能陷入死循环或生成低质量内容 | 推荐千问 Plus/Max |
| 千问版本选择 | 不断更新中 | 使用最新版本（2025-07-14 更新了千问3版） |

### 5.4 AutoGPT 配置要点

- **`max_round` / `max_consecutive_auto_reply`**：必须设置，防止无限循环
- **`allow_repeat_speaker`**：设为 `False` 避免同一个人连续发言
- **`speaker_selection_method`**：建议自定义函数而非使用默认值
- **`human_input_mode`**：控制是否启用人机交互
- **`termination_condition`**：定义任务完成的关键字（如"任务完成"、"SUCCESS"）

### 5.5 框架选择建议

| 框架 | 适用场景 | 老师评价 |
|------|---------|---------|
| **LangGraph** | 复杂多智能体、企业级应用 | **核心框架**，工作中最常用，与 LangChain 无缝集成 |
| **AutoGPT/AutoGen** | 简单角色扮演、快速原型 | 了解即可，适合简单多角色对话场景 |
| **CrewAI** | 任务管线式协作 | 了解即可，底层使用 ReAct 策略 |
| **Coze (扣子)** | 低代码快速搭建 | 不适合本地部署，定制化弱 |

---

## 6. 对比与思考

### 6.1 三大多智能体框架对比

| 维度 | AutoGPT (AutoGen) | CrewAI | LangGraph |
|------|------------------|--------|-----------|
| **核心机制** | 对话式角色扮演 | 团队 + 任务管线 | 状态图驱动的工作流 |
| **角色定义** | 通过 `system_message` 配置 | 通过 `role` / `goal` / `backstory` 定义 | 每个节点 = 一个 Agent |
| **协作方式** | 群组对话 + 自定义发言规则 | 顺序任务执行 | 图结构 + 条件边 |
| **状态管理** | 弱（依赖对话历史） | 共享 Memory | **强（内置 State Graph）** |
| **循环支持** | 通过对话轮次间接实现 | 不支持 | **原生支持（条件边）** |
| **持久化** | 不支持 | 不支持 | **支持（中断恢复）** |
| **学习曲线** | 中等 | 简单 | **陡峭（但最强大）** |
| **企业适用度** | 一般 | 一般 | **推荐（与 LangChain 生态集成）** |

### 6.2 单智能体 vs 多智能体

| 维度 | 单智能体 | 多智能体 |
|------|---------|---------|
| 复杂度 | 简单直接 | 复杂但功能强大 |
| 协作 | 靠自己完成所有工作 | 多个专家分工协作 |
| 并行性 | 串行处理 | 可并行处理 |
| 模块化 | 紧耦合 | 松耦合（可独立替换） |
| 资源消耗 | 低 | 高（多模型调用） |
| 适用场景 | 简单任务、单一领域 | 复杂任务、多领域知识 |

### 6.3 LangGraph 的核心设计理念

**为什么 LangGraph 采用图结构而非线性结构？**

```
现实世界的业务流程是"图"而不是"线"：

软件开发流程：
  需求 → 编码 → 测试 ──→ 发布
              ↑        │
              └── 不通过 ┘
              （需要循环）

文档处理流程：
  文档 → 分类 ─→ 摘要 → 翻译 → 输出
            └→ 关键词提取 → 索引
              （需要分支）
```

**LangGraph 的五大架构模式：**
1. **网络架构** — 多个 Agent 自由交互
2. **领导架构** — 一个主管 Agent 调度多个子 Agent
3. **监管+工具架构** — 主管 Agent 将子 Agent 作为工具调用
4. **层级架构** — 多层级的树状组织
5. **自定义架构** — 完全自由连接的复杂图结构

### 6.4 三阶段课程全景

```
第一阶段：基础篇
  → Python / LLM / RAG / LangChain 基础

第二阶段：Agent 基础（当前阶段）
  → 第1节：Agent 概念 + 记忆 + 工具 + React
  → 第2节：Function Calling 实战 + 认知框架
  → 第3节（本节课）：多智能体 + LangGraph 核心
  → 后续：LangGraph 深入（子图/边/持久化） → Coze / Dify

第三阶段：进阶
  → MCP（模型上下文协议）
  → 微调（LoRA / 量化）
  → 四个实战项目
```

### 6.5 课后思考与延伸

1. **老师明确的下节课预告**：
   - LangGraph 剩余内容：子图（Subgraph）、COMMAND 命令、边（Edge）、流式输出、持久化、人机交互
   - Coze（扣子）— 低代码平台简介

2. **预习建议**：
   - 子图（Subgraph）
   - 发送线（Send）
   - COMMAND 命令（承上启下的关键）
   - 流式输出
   - 持久化

3. **面试导向建议**：
   - 多智能体项目经验比手写代码更重要
   - 面试常问：项目怎么做的、用什么架构、遇到什么问题、怎么优化

---

## 7. 本节课思维导图

- Agent 智能体开发（第三节课）
  - 课前回顾
    - Function Calling → 模型函数调用能力
    - Agent 认知框架 → ReAct 最常用
  - 多智能体系统（MAS）
    - 定义：多个自主 Agent 协作完成复杂任务
    - 三大优势
      - 并行处理（提高效率）
      - 分布式框架（降低复杂度，独立替换）
      - 可扩展性强（增删不影响整体）
    - 应用场景
      - 制造业/工业控制（质检/调度/能耗）
      - 智能交通（车辆/红绿灯协同）
      - 多 Agent 问答 + RAG
  - AutoGPT (AutoGen)
    - 核心概念
      - 配置文件定义角色/任务/终止条件
      - 群组对话（GroupChat）
      - 自定义发言规则（Speaker Rule）
      - 对话轮次限制（Max Round）
    - 软件开发团队案例
      - 4 个角色：用户 → 编码 → 审查 → 测试
      - 条件判断流程（是否包含 Python 代码 → 决定下一步）
      - ⚠️ 必须自定义发言规则，默认规则不可用
      - ⚠️ 提示词决定 Agent 表现
  - CrewAI
    - 核心组件：Agent / Task / Crew / Process
    - Agent + Task — 对关系
    - 底层默认使用 ReAct 策略
    - 支持共享 Memory
  - LangGraph 框架（重点）
    - 什么是 LangGraph
      - LangChain 团队开发的**多智能体框架**
      - 基于**状态图（StateGraph）**
      - 与 LangChain 无缝集成
    - 五大核心功能
      - 支持循环流（回到之前步骤）
      - 状态管理（节点间共享信息）
      - 多参与者支持（Agent 交互）
      - 条件分支（根据状态决定走向）
      - 持久化（中断恢复）
    - LangGraph vs LangChain
      - LangChain：线性流程（有向无环图），适合简单到中等的 AI 应用
      - LangGraph：图状流程（支持循环/分支），适合复杂多智能体
      - 两者**不是替代关系**，LangGraph 依赖 LangChain 组件
    - 三大核心概念
      - **State（状态）**：TypedDict 定义的字典，贯穿整个工作流
      - **Node（节点）**：普通函数，接收状态返回部分状态
      - **Graph（图）**：add_node → set_entry_point → add_edge → compile
    - 状态多模式构建
      - Input State：限定输入字段
      - Intermediate State：内部完整状态
      - Output State：限定输出字段（隐藏中间数据）
    - 构建流程（SOP）
      - Step 1: 定义 State（TypedDict）
      - Step 2: 定义 Node 函数
      - Step 3: 构建 Graph（add_node → 边 → compile）
      - Step 4: invoke 执行
    - 五大架构模式
      - 网络架构 / 领导架构 / 监管+工具 / 层级架构 / 自定义架构
  - 后续课程预告
    - LangGraph 剩余内容（子图/COMMAND/边/持久化）
    - Coze（扣子）低代码平台
    - Dify 本地部署
    - MCP + 微调（四阶段）
    - 四个实战项目：RAG / Agent / 预警系统 / MCP+微调+RAG
