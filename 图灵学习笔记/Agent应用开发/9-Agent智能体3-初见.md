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

#### 多智能体系统 Multi-Agent System (MAS)

**是什么**：多个具备智能和自主能力的个体（Agent）组成的协作系统。每个 Agent 拥有独立的感知、推理和行动能力，它们通过通信和协调来完成单个 Agent 无法独立完成的复杂任务。

**如何运作**：MAS 的核心机制是"分工 + 协调"。每个 Agent 负责一个子领域（如数据采集、分析、决策），通过消息传递或共享记忆进行信息交换。协调机制有两种模式：（1）**中心化协调**— 一个"主管"Agent 统一调度；（2）**去中心化协调**— Agent 之间通过投票或协商达成共识。MAS 的决策流程通常是：任务分解 → 分配子任务 → 各 Agent 独立执行 → 结果汇总 → 综合决策。

**怎么使用**：在实际开发中，你不需要从零实现 MAS 通信机制。框架层面，AutoGen 提供了 GroupChat（群组对话模式），CrewAI 提供了 Crew（团队模式），LangGraph 提供了 StateGraph（有状态图模式）。使用时，先确定任务可以拆分为哪些独立子任务，然后为每个子任务设计一个 Agent（定义角色/工具/目标），最后通过框架的协调机制将它们串联起来。典型场景包括：多 Agent 问答系统、软件开发团队模拟、智能交通调度等。

---

#### 单智能体 Single Agent

**是什么**：只有一个智能体独立完成整个任务，它包揽感知、规划、工具调用、生成回复的全部工作。这是最简单的 Agent 形态，也是本节之前两种课程中一直使用的模式。

**如何运作**：单 Agent 遵循"输入 → 思考 → 行动 → 观察 → 输出"的循环。它接收用户问题后，自行决定调用哪些工具（如搜索、计算器），处理工具返回的结果，最终生成回答。整个过程是串行的，没有其他 Agent 参与协作。

**怎么使用**：适用于任务边界清晰、不需要多领域专家协作的场景。例如：单一文档问答、简单的代码生成、信息检索。直接在 LangChain/LlamaIndex 中定义一个 Agent 并配置工具即可。

---

#### 并行处理 Parallel Processing

**是什么**：多个 Agent 同时处理不同的子任务，各自独立地进行推理和工具调用，互不阻塞。

**如何运作**：在 MAS 中，当任务被分解为多个独立子任务后，这些子任务被分配给不同的 Agent 同时执行。框架层使用异步调度或线程池机制来管理并发。关键约束是：子任务之间不能有数据依赖（否则必须串行）。

**怎么使用**：在 LangGraph 中，可以通过"扇出（fan-out）"模式实现：一个前置节点将状态拆分后路由到多个并行节点。在 AutoGen 中，群组聊天天然支持多个 Agent 轮流发言（实际是顺序的而非并行的，严格意义上的并行需要借助异步框架）。并行处理的核心价值是降低延迟——如果 3 个 Agent 各需 2 秒，串行 6 秒，并行只需 2 秒。

---

#### 分布式框架 Distributed Framework

**是什么**：每个 Agent 是独立运行的模块，可以在不同进程、不同机器上部署，通过标准通信协议（如 HTTP/gRPC）交换信息。

**如何运作**：每个 Agent 是一个独立服务，拥有自己的模型实例、工具集和记忆存储。Agent 之间不直接共享内存，而是通过网络消息进行协作。这使得系统可以水平扩展（给某个角色加更多实例）而不影响其他组件。

**怎么使用**：AutoGen 的 Core API 支持事件驱动的分布式 Agent，Agent 通过 Topic（主题）发布和订阅消息。LangGraph 支持将子图部署为独立服务。实践上，分布式框架的优势是：可用不同语言实现不同 Agent（Python 做推理 Agent、Go 做高性能数据处理 Agent）、独立升级/替换单个 Agent 不影响整体。

---

#### 可扩展性 Scalability

**是什么**：新增或移除 Agent 不影响系统整体运行的能力。这是 MAS 相比单 Agent 的核心优势。

**如何运作**：取决于协调机制的设计。如果使用去中心化协调，新 Agent 只需注册自己的能力和通信地址即可加入系统；如果使用中心化协调，需要更新主管 Agent 的调度策略（例如重新生成"可用 Agent 列表"）。

**怎么使用**：设计时应避免硬编码 Agent 依赖关系——使用注册中心（如简单的字典映射或 Redis）来管理 Agent 列表；通过一致的消息协议（如定义统一的 Message Schema）确保新 Agent 能理解交互流程。在 CrewAI 中，扩展就是在 Crew 的 agents 列表中添加新成员；在 LangGraph 中，扩展就是在图中 add_node。

---

#### 角色扮演 Role Playing

**是什么**：每个 Agent 被分配一个特定角色（如开发工程师、测试工程师、产品经理），角色定义了 Agent 的行为边界、说话风格和决策偏好。

**如何运作**：角色的核心承载就是 System Prompt。开发 Agent 的 System Prompt 会强调编码规范和测试习惯，审查 Agent 的 Prompt 则强调代码安全性和最佳实践。当同一个任务被不同角色的 Agent 处理时，它们会从不同视角分析问题，产生互补的输出。

**怎么使用**：定义一个 Agent 时，角色由三个要素构成：（1）**角色名称**（Role）— 如"资深代码审查专家"；（2）**目标**（Goal）— 如"确保代码质量和安全"；（3）**背景故事**（Backstory）— 如"你有 10 年后端开发经验"。在 CrewAI 中这三者是 Agent 的必填参数。在 AutoGen 中通过 `system_message` 体现。

---

#### 群组对话 Group Chat

**是什么**：多个 Agent 加入同一个"聊天室"，轮流发言，共享同一个消息线程。这是 AutoGen 实现多 Agent 协作的核心模式。

**如何运作**：群组聊天中有一个 GroupChat Manager（群聊管理器），它持有消息历史并决定下一个发言者。Manager 收到一条消息后，通过发言者选择算法（如轮询、LLM 选择、自定义规则）决定谁接着发言，然后将发言权交给该 Agent。消息会被广播给所有参与者，因此后发言的 Agent 能看到完整上下文。流程：用户发布任务 → Manager 选第一个 Agent → Agent 回复 → Manager 再选下一个 → 直到终止条件触发。

**怎么使用**：在 AutoGen 中创建 `GroupChat(agents=[...], messages=[], max_round=N, speaker_selection_method=custom_func)`，然后通过 `GroupChatManager` 启动。群组对话适用于软件开发团队模拟、多人会议讨论等需要来回对话的场景。需要注意设置 `max_round` 防止无限循环。Microsoft 的 AutoGen 官方文档建议为每个 Agent 提供有意义的 `name` 和 `description`，这些信息会被 Manager 的 LLM 选择器用来决定下一位发言者。<sup>[[1]](#ref-autogen-groupchat)</sup>

---

#### 发言规则 Speaker Rule

**是什么**：决定群组对话中多个 Agent 的发言顺序和交互规则。这是群组对话能否顺畅进行的关键因素。

**如何运作**：AutoGen 提供了几种内置规则：`auto`（LLM 基于上下文选择）、`random`（随机选）、`round_robin`（轮流）、`manual`（人工指定）。但这些内置规则在复杂业务场景中往往不够用——例如，你要求"编码 Agent 发言后必须由审查 Agent 接话"就无法通过轮询实现，需要自定义规则。自定义规则是一个 Python 函数，它接收最后发言者和群组对象，返回下一个发言者。

**怎么使用**：定义一个 `custom_speaker_selection_func(last_speaker: Agent, groupchat: GroupChat) -> Union[Agent, str, None]` 函数，在其中编写业务逻辑：检查最后消息的内容（如是否包含代码）、判断当前状态（如"编码阶段/审查阶段"），返回对应的下一个 Agent。如果返回 `None`，对话结束。在 GroupChat 中通过 `speaker_selection_method=custom_func` 传入。AutoGen 也支持基于状态的发言者选择（StateFlow 模式），发言规则实际上等同于有限状态机中的状态转移。<sup>[[2]](#ref-autogen-stateflow)</sup>

---

### 2.2 AutoGPT / AutoGen 概念

#### 配置文件 Config File

**是什么**：定义 Agent 角色、任务、终止条件等参数的 YAML/JSON/代码文件。配置文件是多 Agent 系统的"剧本"，决定了系统启动后的行为。

**如何运作**：启动时，框架读取配置文件创建 Agent 实例，将配置中的 `system_message` 注入到每个 Agent 的提示词中，按配置设定终止条件。配置内容通常包含：Agent 名称、模型选择、系统消息、温度参数、最大回复次数、启用的工具列表等。

**怎么使用**：在 AutoGen 中，配置直接在代码中以字典形式定义（如案例一所示）。在 AutoGen Studio（低代码界面）中，配置通过可视化界面生成 JSON 文件。配置项的精心设计直接决定了系统质量——`system_message` 写得太泛会导致 Agent 行为不可控，写得太具体会导致缺乏灵活性。

---

#### 系统角色 System Role / System Message

**是什么**：给 Agent 设定的核心身份提示词，是决定 Agent 行为和能力的"人格设定"。它相当于一条永久性的 System Prompt，在每条对话前注入。

**如何运作**：每次调用模型时，`system_message` 总是放在消息列表的最开头。模型依据这个系统级提示来理解自己的身份、任务边界和行为规则。一个写得好的系统角色应该包含：身份声明、行为规范、输出格式要求、边界条件。

**怎么使用**：
```
system_message = "你是 Python 编程专家，负责编写高质量、可维护的代码。" +
"要求：1) 必须包含异常处理；2) 必须添加类型注解；3) 必须写注释说明核心逻辑"
```
老师强调：**提示词决定 Agent 表现** — 同样的模型，不同的 System Message 可以产生天壤之别的行为。

---

#### 用户代理 User Agent

**是什么**：在多 Agent 系统中模拟用户提出需求的 Agent。它不是真实的人，而是又一个由 LLM 驱动的 Agent，但它的任务是"扮演用户"。

**如何运作**：用户代理的 System Message 类似"你是一个产品经理，需要向开发团队提出需求"。它的发言内容不来自真实键盘输入，而是由模型根据角色设定自动生成。这使得全自动的多 Agent 模拟成为可能——不需要人工参与即可完成一个完整的协作流程。

**怎么使用**：设置 `human_input_mode="NEVER"` 让 Agent 自动发言，或设置为 `"ALWAYS"` 等待人工输入。在演示场景中，用户代理通常用来自动化测试多 Agent 系统的交互流程。

---

#### 监督代理 Supervisor Agent

**是什么**：协调多个 Agent 对话顺序的管理者角色。在群组聊天中，GroupChat Manager 本质上就是一个监督代理——但它只负责"调度"而非"执行"。

**如何运作**：监督代理不属于业务 Agent 列表（不参与具体任务执行），它独立运行一个 LLM 调用或规则引擎，持续观察群组对话的当前状态，然后决定：下一位发言者是谁、当前进展是否符合预期、是否需要终止对话。在层级架构（Hierarchical）中，监督代理还负责拆解任务和分配子任务。

**怎么使用**：在 AutoGen 中由 `GroupChatManager` 自动承担监督职责。在 CrewAI 的 Hierarchical Process 中，需要配置 `manager_llm` 来指定管理 Agent 使用的模型（通常比执行 Agent 更强）。在 LangGraph 中，监督角色通过"条件边"的条件函数来实现——本质上就是在每个决策点手动编码调度逻辑。

---

#### 终止条件 Termination Condition

**是什么**：判断多 Agent 任务何时完成的规则或关键字。没有终止条件，多 Agent 系统可能无限对话下去。

**如何运作**：每次一个 Agent 发言后，Manager 检查最新消息是否满足终止条件。如果满足，Manager 停止发出新的发言请求，GroupChat 结束并返回消息历史。终止条件可以是：（1）**关键字匹配** — 消息中包含"任务完成"、"SUCCESS"；（2）**最大轮次** — 达到 `max_round` 上限；（3）**超时** — 超过时间限制；（4）**自定义函数** — 任意复杂的判断逻辑。

**怎么使用**：AutoGen 中通过 `GroupChat(max_round=N)` 设置轮次限制。在 `custom_speaker_selection_func` 中返回 `None` 也可以终止对话。老师强调：**`max_round` 是必备的安全措施**，即使有关键字终止条件也要设置轮次上限作为兜底。

---

#### 群组聊天管理器 Group Chat Manager

**是什么**：负责管理多 Agent 对话全生命周期的调度器。它是群组聊天的"大脑"。

**如何运作**：GroupChatManager 持有群组配置（Agent 列表、发言规则、终止条件）和消息历史。它的工作循环如下：
1. 收到新消息 → 2. 调用发言者选择算法确定下一个发言者 → 3. 请求该 Agent 生成回复 → 4. 将回复广播给所有 Agent → 5. 检查终止条件 → 6. 未满足则回到步骤 2

**怎么使用**：
```python
manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"model": "qwen-plus", "temperature": 0}
)
```
Manager 本身也需要配置一个 LLM（因为 `auto` 模式的选择器需要模型来判断谁应该发言）。

---

#### 对话轮次限制 Max Turns

**是什么**：限制整个群组对话或单个 Agent 的最大发言次数，防止系统陷入无限循环。

**如何运作**：对话轮次计数器在每次 Agent 发言后递增。在 AutoGen 中，`GroupChat.max_round` 控制全局总发言次数，`Agent.max_consecutive_auto_reply` 控制单个 Agent 的连续自动回复次数。当计数器达到上限时，Manager 强制结束对话。

**怎么使用**：老师经验：`max_round=10` 是一个常用起始值，对于简单的角色扮演（如 4 个 Agent 的软件开发流程），2-3 轮已经能完成一个周期。设置太大会浪费 Token，太小会导致任务执行不完整。

---

### 2.3 CrewAI 概念

#### 团队 Crew

**是什么**：一组具有不同角色的 Agent 组成的协作团队。Crew 是 CrewAI 的一等公民——系统以 Crew 为单位组织执行。

**如何运作**：Crew 将 Agent 列表和 Task 列表绑定在一起，指定执行流程（Process），然后通过 `crew.kickoff()` 一次性启动全部工作。Crew 内部自动管理：任务分配、工具共享、记忆同步。CrewAI 支持两种执行模式：（1）**顺序（Sequential）** — 任务按列表顺序依次执行，前一个 Task 的输出作为后一个的上下文；（2）**层级（Hierarchical）** — 引入一个 Manager Agent 动态分配任务。<sup>[[3]](#ref-crewai-process)</sup>

**怎么使用**：
```python
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential  # 或 Process.hierarchical
)
result = crew.kickoff()
```
启动后，CrewAI 会按照 Process 定义的策略自动调度每个 Agent 执行对应的 Task。Crew 还可以配置 `memory=True`（启用共享记忆）、`planning=True`（启用执行前自动规划）等高级选项。

---

#### 任务 Task

**是什么**：分配给每个 Agent 的具体工作步骤。Task 是 CrewAI 的最小执行单位，它绑定了具体的 Agent 和输出要求。

**如何运作**：Task 包含描述（description）、预期输出格式（expected_output）和关联 Agent。在顺序执行中，前一个 Task 的输出自动作为后一个 Task 的上下文。在层级执行中，Manager Agent 将 Task 动态分配给最合适的 Agent。

**怎么使用**：
```python
research_task = Task(
    description="搜索和收集 AI Agent 框架的最新发展",
    expected_output="一份包含 5 个框架的详细对比列表",
    agent=researcher
)
```
Task 还可以配置 `context`（从其他 Task 获取额外上下文）、`callback`（完成回调）、`human_input`（是否等待人工确认）等参数。CrewAI v1.0+ 还支持 Pydantic Output，即通过 Pydantic 模型约束 Task 的输出结构。

---

#### 共享记忆 Shared Memory

**是什么**：多个 Agent 之间可以共同访问的记忆空间，支持跨任务、跨会话的信息复用。CrewAI 实现了三级记忆架构。

**如何运作**：CrewAI 的记忆系统分为四个层次：<sup>[[4]](#ref-crewai-memory)</sup>
- **短期记忆（Short-Term Memory）** ：使用 ChromaDB 向量存储，保存当前对话上下文，支持快速检索最近交互信息
- **长期记忆（Long-Term Memory）** ：通过 SQLite3 持久化存储跨会话的重要信息与任务结果
- **实体记忆（Entity Memory）** ：追踪对话中出现的关键实体（人物、地点、概念）及其关系
- **上下文记忆（Contextual Memory）** ：维护上下文窗口的智能管理

当 Agent 需要信息时，先查短期记忆（当前会话），再查长期记忆（历史会话），最后查实体记忆（跨会话的实体关系）。这解决了传统对话系统中"上下文信息丢失"的核心痛点。

**怎么使用**：
```python
crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,  # 一键开启三级记忆
    embedder={"provider": "ollama", "config": {"model": "mxbai-embed-large"}}
)
```
启用后，Agent 会自动使用记忆来增强回复的相关性。

---

#### 流程 Flow / Process

**是什么**：定义 Agent 执行 Task 顺序的管线策略。CrewAI 支持两种 Process：顺序（Sequential）和层级（Hierarchical）。

**如何运作**：
- **顺序执行**：Task 按列表顺序执行，Agent 一对一绑定到 Task，前一个输出自动作为后一个的上下文。简单直接，适合管线清晰的场景。
- **层级执行**：需指定 `manager_llm`，CrewAI 自动创建一个 Manager Agent，它将 Task 列表作为参考，根据执行情况动态分配和调整工作。Manager 拥有审批权和再分配权，可以要求某个 Agent 重做不合格的任务。<sup>[[5]](#ref-crewai-hierarchical)</sup>

**怎么使用**：
```python
# 顺序模式
Crew(..., process=Process.sequential)

# 层级模式（需要配置 manager_llm）
Crew(..., process=Process.hierarchical,
     manager_llm=OpenAI(model="gpt-4"))
```
老师说明：CrewAI 的每个 Agent 底层默认使用 ReAct 策略（思考→行动→观察循环），Process 决定了 Agent 之间的协作顺序，而非 Agent 内部的推理方式。

---

### 2.4 LangGraph 核心概念

#### 状态图 StateGraph

**是什么**：LangGraph 的核心构建块，一种基于状态驱动的工作流图结构。它借鉴了有限状态机（FSM）的思想——图中的每个节点代表状态或动作，边代表状态转移条件。

**如何运作**：StateGraph 接收一个状态定义（TypedDict），然后让你添加节点和边。运行时，状态字典从入口节点开始，沿着边流动到各个节点，每个节点读取状态、执行逻辑、更新状态，然后根据当前状态值或边定义决定下一步流向。关键区别：传统的 DAG 工作流一旦节点执行完就自动进入下一个节点，而 StateGraph 的"条件边"可以根据状态值动态决定下一步走向——这使得循环和分支成为可能。

**怎么使用**：
```python
from langgraph.graph import StateGraph, START, END

# 1. 定义状态
class State(TypedDict):
    messages: list[str]

# 2. 构建图
graph = StateGraph(State)
graph.add_node("node1", node1_func)
graph.add_edge(START, "node1")
graph.add_edge("node1", END)

# 3. 编译
app = graph.compile()

# 4. 执行
result = app.invoke({"messages": ["你好"]})
```
StateGraph 是 LangGraph 所有功能的基础——节点、边、条件边、子图都在它之上构建。

---

#### 状态 State

**是什么**：贯穿整个工作流的数据容器，通常是 TypedDict 类型。State 是图的"血液"——所有节点通过它共享和传递信息。

**如何运作**：每个节点函数接收当前 State 作为参数，处理后返回一个**部分更新**（partial update）字典。LangGraph 运行时将这些部分更新合并到全局 State 中。合并规则由 Reducer 定义——默认是"后写覆盖"（last-write-wins），但可以通过 Annotated 自定义合并逻辑（如列表拼接）。State 的生命周期：`用户输入 → 初始化 State → 节点1 → 节点2 → ... → 最终 State → 输出`。

**怎么使用**：
```python
# 定义状态结构
class GraphState(TypedDict):
    question: str
    answer: str
    messages: Annotated[list, operator.add]  # 带 Reducer 的字段
```

**典型模式对比**：
| 状态字段 | Reducer | 效果 |
|---------|---------|------|
| `question: str` | 默认（覆盖） | 每次都替换 |
| `messages: Annotated[list, add]` | `operator.add` | 追加而不是替换 |

---

#### 节点 Node

**是什么**：图中的每个处理单元，本质就是**一个普通的 Python 函数**。节点可以是 Agent 调用、工具函数、API 请求、条件判断逻辑等任何操作。

**如何运作**：节点函数的签名是 `node_func(state: State) -> dict`——输入当前完整状态，返回一个字典（包含要更新的字段）。LangGraph 运行时会自动将返回的字典 merge 回全局状态。节点不直接操作全局状态，只声明"我要更新哪些字段"。

**怎么使用**：将 Agent 封装成节点是最常见的用法：
```python
def agent_node(state: GraphState) -> dict:
    """Agent 节点：调用大模型"""
    response = llm.invoke(state["question"])
    return {"answer": response.content, "messages": [response]}
```
将一个 Agent 部署为一个节点，多个节点组成图就是"多智能体"。

---

#### 边 Edge

**是什么**：连接节点的有向路径，定义执行流向。边分为两种：普通边（无条件，执行完上一个自动走）和条件边（根据状态动态选择）。

**如何运作**：
- **普通边（add_edge）**：`node_A → node_B`，节点 A 执行完后**自动**进入节点 B。编译时会校验路径完整性（确保所有节点都可到达且不遗漏出口）。
- **条件边（add_conditional_edges）**：节点 A 执行完后，运行一个**路由函数**，根据路由函数的返回值决定下一步走向哪个节点。路由函数的输入是当前状态，输出是下一个节点的名称（字符串）。

**怎么使用**：
```python
# 普通边
graph.add_edge("search", "generate")

# 条件边
graph.add_conditional_edges(
    "review",
    router_func,  # 接收 state，返回 "pass" 或 "fail"
    {"pass": "deploy", "fail": "rework"}
)
```
边是 LangGraph 的"控制流"——它决定了多 Agent 的工作流程是串行、循环还是分支。

---

#### 条件边 Conditional Edge

**是什么**：根据当前状态的某些字段值，动态决定下一步执行走向哪个节点的逻辑分支。这是 LangGraph 支持循环和复杂工作流的核心机制。

**如何运作**：条件边由三部分构成：（1）**源节点** — 条件判断在哪个节点之后触发；（2）**路由函数** — 接收当前 State，返回目标节点名称的字符串；（3）**路径映射** — 字典，将路由函数的返回值映射到具体的节点名。路由函数可以任意复杂：检查消息内容、检查字段值、让 LLM 做决策（LLM as a Router）。

**怎么使用**：
```python
def router(state: GraphState) -> str:
    """根据审查结果路由"""
    if "通过" in state["review_result"]:
        return "pass"
    else:
        return "fail"

workflow.add_conditional_edges(
    "review_node",
    router,
    {"pass": "deploy_node", "fail": "rework_node"}
)
```
用条件边实现的循环：`rework_node → review_node` 形成闭环，直到审查通过才走向 `deploy_node`。

---

#### 子图 Subgraph

**是什么**：节点内部可嵌套的子工作流。它是"图中的图"——一个 Node 内部可以是一个完整的 StateGraph。

**如何运作**：创建一个独立的 StateGraph，编译后作为另一个图的节点使用。子图拥有自己独立的状态空间、节点集和边——但它可以访问父图传入的状态。子图执行完后将结果返回到父图的 State 中。这实现了层级化的工作流设计。

**怎么使用**：如一个"文档处理"父图，其"质量检查"节点可能是一个子图，子图内部包含"格式检查 → 内容审核 → 图片校验"三个子节点。子图可以是父图节点、也可以包含父图的节点。

**为什么需要子图**：复杂工作流需要层级抽象。如果没有子图，所有节点都在同一层级，几十个节点连在一起会变成"面条式架构"。子图让开发者可以将复杂逻辑封装为独立模块，每个子图可独立测试和复用。

---

#### 编译 Compile

**是什么**：构建图之后执行的校验步骤。不是 Python 语言的编译（不会生成字节码），而是**图结构校验**：确保所有节点都有入口和出口、边的目标节点都存在、没有孤立节点。

**如何运作**：调用 `graph.compile()` 时，LangGraph 运行时遍历整个图结构进行校验：检查所有 `add_node` 的节点是否都被边连接、检查 `set_entry_point` 和 `add_edge` 中的目标节点是否存在、检查条件边的路径映射中的目标节点是否存在。校验通过后返回一个 `CompiledGraph` 对象（即 `app`），这个对象才具备 `invoke` 方法。

**怎么使用**：
```python
# compile 前可以继续加节点和边
app = workflow.compile()  # 这步之后就不能再修改图结构了
result = app.invoke(...)  # 只能 invoke，不能 add_node
```
常见的编译错误：`KeyError: "Node 'xyz' not found"` — 边引用了不存在的节点名。

---

#### 持久化 Persistence

**是什么**：将工作流状态持久化存储到外部存储（如 SQLite/Postgres），支持中断恢复和长时间运行的工作流。

**如何运作**：LangGraph 的持久化通过 Checkpointer 实现。工作流执行的每一步（每个节点完成后），Checkpointer 将当前 State 保存到存储中。如果进程崩溃或被中断，可以从最近的 Checkpoint 恢复。这依赖于 Checkpointer 的配置和 `app.invoke()` 或 `app.astream()` 中传入的 `thread_id` 参数。

**怎么使用**：
```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 配置持久化
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
app = workflow.compile(checkpointer=checkpointer)

# 执行（自动保存断点）
result = app.invoke(
    {"question": "多步骤任务"},
    config={"configurable": {"thread_id": "my_task_001"}}
)
```
持久化的价值：多 Agent 系统可能运行很久（如数据采集→清洗→分析→报告，可能数小时），中途中断不需要重头开始。也支持人在回路（Human-in-the-Loop）——让工作流在某个节点暂停等待人工审核，审核通过后再恢复执行。

---

#### 归并函数 Reducer Function

**是什么**：处理多个节点并行更新同一状态字段时的合并逻辑。如果没有 Reducer，并行节点中后完成的会覆盖先完成的。

**如何运作**：当图中有并行执行的分支（多个节点同时更新 State 的同一个 key），LangGraph 需要知道如何合并这些更新。默认行为是"后写覆盖"（last-write-wins），但这对于需要累积的字段（如消息列表、分数累加）是有问题的。Reducer 通过 `Annotated` 类型注解声明在 State 定义的字段上：`field: Annotated[type, reducer_func]`。LangGraph 内置了 `operator.add`（列表拼接）和 `add_messages`（消息去重合并），也支持自定义合并函数。<sup>[[6]](#ref-langgraph-reducer)</sup>

**怎么使用**：
```python
from typing import Annotated, TypedDict
import operator

class State(TypedDict):
    # 普通字段：后写覆盖
    answer: str
    # 带 Reducer 的字段：列表拼接
    messages: Annotated[list, operator.add]
    # 内置消息 Reducer
    messages: Annotated[list, add_messages]
```

**实战案例**：如果一个"搜索"节点拆分为 3 个并行搜索 Agent，每个 Agent 返回搜索结果到同一字段，用 `operator.add` 可以让结果全部追加到列表中，而不是只有最后一个节点生效。如果不加 Reducer 而使用并行执行，LangGraph 会抛出 `INVALID_CONCURRENT_GRAPH_UPDATE` 错误。<sup>[[7]](#ref-langgraph-concurrent)</sup>

---

#### 多模式状态 Multi-Schema State

**是什么**：为同一张图定义三种不同的状态 Schema：输入状态（用户传什么）、中间状态（内部共享什么）、输出状态（最终返回什么）。实现信息隔离。

**如何运作**：StateGraph 构建时通过 `input` / `state` / `output` 三个参数分别指定不同的 TypedDict：
- **Input Schema**：限定用户只能传哪些字段（如 `question: str`）
- **State Schema**：内部完整状态，所有节点共享（如含 `question` + `search_result` + `answer`）
- **Output Schema**：限定最终输出只暴露哪些字段（如只暴露 `answer`）

运行时，LangGraph 在入口处校验用户输入是否符合 Input Schema，执行完毕后只输出 Output Schema 中定义的字段。<sup>[[8]](#ref-langgraph-multischema)</sup>

**怎么使用**：
```python
workflow = StateGraph(
    input=InputState,           # 用户只能传 question
    state=IntermediateState,    # 内部用完整状态
    output=OutputState          # 只输出 answer
)
```
**典型场景**：你不想让用户看到系统内部的 `search_raw_html`、`debug_logs` 等中间状态。多模式状态让 LangGraph 在输出时自动过滤掉这些字段，类似 REST API 中请求体和响应体使用不同的 Schema。

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

---

### 参考资料

- <a id="ref-autogen-groupchat"></a>[[1]] Microsoft AutoGen — Group Chat 设计模式，GroupChat Manager 通过发言者选择算法（轮询/LLM/自定义）管理 Agent 对话顺序。参考：https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/selector-group-chat.html
- <a id="ref-autogen-stateflow"></a>[[2]] AutoGen StateFlow — 基于状态机的自定义发言者选择，将 GroupChat 发言规则建模为有限状态机。参考：https://microsoft.github.io/autogen/0.2/blog/2024/02/29/StateFlow/
- <a id="ref-crewai-process"></a>[[3]] CrewAI Process — 顺序执行（Sequential）和层级执行（Hierarchical）两种任务分配策略。参考：https://docs.crewai.com/
- <a id="ref-crewai-memory"></a>[[4]] CrewAI Memory — 三级记忆架构：短期记忆（ChromaDB）、长期记忆（SQLite3）、实体记忆（RAG 实体追踪）。参考：https://docs.crewai.com/core-concepts/Memory/
- <a id="ref-crewai-hierarchical"></a>[[5]] CrewAI Hierarchical Process — Manager Agent 自动创建、动态分配任务、审批和重做机制。参考：https://docs.crewai.com/core-concepts/Process/
- <a id="ref-langgraph-reducer"></a>[[6]] LangGraph Reducer Guide — 通过 Annotated 类型注解定义状态字段的合并逻辑，内置 operator.add 和 add_messages。参考：https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
- <a id="ref-langgraph-concurrent"></a>[[7]] LangGraph INVALID_CONCURRENT_GRAPH_UPDATE — 并行节点更新同一字段时，无 Reducer 会报错。参考：https://docs.langchain.com/oss/javascript/langgraph/INVALID_CONCURRENT_GRAPH_UPDATE
- <a id="ref-langgraph-multischema"></a>[[8]] LangGraph Multi-Schema State — input/state/output 三种 Schema 实现信息隔离。参考：https://langchain-ai.github.io/langgraph/concepts/low_level/#schema
