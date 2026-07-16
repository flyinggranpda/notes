# LangGraph State Schema

## 三个核心组件

LangGraph 将智能体工作流建模为图，通过三个关键组件定义智能体行为：

### 1. State（状态）
一种**共享数据结构**，用来表示数据当前的快照。LangGraph 内部的数据管理就存储在这个 state 之中。state 里面可以存任何类型（int、str、字典、列表等），是**很灵活的操作**。

### 2. Node（节点）
进行**编码的函数**——把函数封装成一个节点。它可以接收当前状态作为外部输入，同时返回更新后的状态，与状态进行交互（提取数据、返回数据）。

### 3. Edge（边）
根据当前状态确定**下一个执行 node 的函数**。边可以是直连（硬性规定走哪条路），也可以是**条件边**（根据条件决定走向）。灵活度很高，相比 LangChain 来说适用度更高。

通过组合节点和边可以创造复杂的循环工作流。

---

## StateGraph 类

`StateGraph` 是主要使用的图类，在注册图的时候必须使用它：

```python
from langgraph.graph import StateGraph

builder = StateGraph(OverallState)
# 传入所需的主状态信息
```

## State 的重要性

定义图的第一步就是定义图的 state。进行项目编写设计时，**第一步不要想实现什么功能，要想中间会保存什么样的数据**。因为项目做一半不好改，state 相当于一种**地基**的存在。

- 代码体量上 state 的体量最小（十几二十行）
- 节点可能几百行甚至上千行
- state 相当于一种"灵魂"般的存在

State 由 **schema** 与 **reducer 函数**组成。reducer 函数决定了如何对状态进行更新。State schema 将作为图中所有 node 和 edge 的输入模式。

---

## Input/Output Schema — 输入输出隔离

LangGraph 中，**state schema**、**input schema**、**output schema** 三个概念用于管理不同的状态：

- **State schema** — 总括，包含 input 和 output 的所有内容
- **Input schema** — 输入状态，限定调用时可传入的数据
- **Output schema** — 输出状态，限定返回的数据

这样能做到**隔离**：避免下个节点访问到某些数据。例如输入信息从 input 传入，从 output 提取，实现限制。

![状态流转示意]
START → 接收 input schema → 通过 state schema 处理 → 通过 output schema 输出

- State schema：**完整内部状态**（全局状态空间），所有节点都可以访问
- Input：接受什么输入，是它的**子集**
- Output：选回返回什么输出，也是它的**子集**
- input/output 都是可选参数

### 代码示例：输入输出隔离

```python
from typing_extensions import TypedDict

class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class OverallState(InputState, OutputState):
    # 继承 InputState 和 OutputState 的所有字段
    pass
```

上面的 `OverallState` 等价于：

```python
class OverallState(TypedDict):
    question: str
    answer: str
```

### 节点处理逻辑

```python
def answer_node(state: OverallState) -> OutputState:
    # 从 state 中承接 question
    question = state["question"]
    # 处理逻辑
    if question.lower() == "bye":
        answer = "再见"
    else:
        answer = "你好"
    # 返回字典，把 answer 赋值为处理后的值
    result = {"question": question, "answer": answer}
    print(f"result: {result}")
    return result  # 写入 state，覆盖或追加
```

### 图的构建与调用

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(OverallState)

# 传入 OverallState 后，它会被解析：把其中每个字段拆出来
# 图中存储的不是类名，而是字段名
builder.add_node("answer_node", answer_node)
builder.add_edge(START, "answer_node")
builder.add_edge("answer_node", END)

graph = builder.compile()

# 还可以指定 input_schema / output_schema 进行严格限定
# 调用时状态必须存在于 input_schema 中，输出状态必须处于 output_schema 中
```

```python
# 调用：传入 question（在 input_schema 中，可以传）
result = graph.invoke({"question": "你好"})
print(f"图调用结果: {result}")
# 输出只包含 answer（output_schema 中只有 answer，question 被隔离）
```

> 在 node 中输出的是完整的两个状态信息（question + answer），但由于 output_schema 只声明了 answer，最终 result 只输出 answer。这就是**隔离**的处理。

---

## 私有状态传递机制

### 状态传递的核心逻辑

图中状态存储的是**字段名**而非**类名**。两个类即使类名不同，只要**字段名相同**，就可以在状态之间传递数据。

```python
class NodeOneOutput(TypedDict):
    private_data: str

class NodeTwoInput(TypedDict):
    private_data: str
```

这两个类的 `private_data` 字段相同，因此 node_one 产出的数据可以传递到 node_two。

### 临时状态机制

```python
class OverallState(TypedDict):
    a: str

class NodeOneOutput(TypedDict):
    private_data: str

class NodeTwoInput(TypedDict):
    private_data: str
```

```python
def node_one(state: OverallState) -> NodeOneOutput:
    return {"private_data": "some data"}

def node_two(state: NodeTwoInput) -> OverallState:
    return {"a": state["private_data"]}
```

处理流程：
1. `node_one` 从 `OverallState` 中传入，返回 `NodeOneOutput`（包含 `private_data`）
2. 但 `private_data` 不在全局状态 `OverallState` 中注册过
3. LangGraph 检测到节点返回了未注册的状态，会将其在 **RAM 中进行临时存储**
4. 直到下一个节点（`node_two`）对其进行**消费**后，临时状态被**销毁**，值被提取
5. 如果该状态在 `OverallState` 中注册了，则会一直存在，不会被消费销毁

**总结：**
- **已注册的状态** → 持久存在，不被销毁
- **未注册的私有状态（临时状态）** → 等待被消费，消费后销毁以释放内存
- 内部的底层处理（暂存、销毁）都是 LangGraph 底层自动实现的

> 费曼说：可以把 state 理解成一个数据仓库。注册在主 state 里的字段就像放在货架上的商品，一直存在。节点临时产生的私有字段就像快递中转站的包裹——等下一个节点签收之后就销毁，不占地方。

---

## 关键概念理解

| 组件 | 角色 |
|------|------|
| **State** | 数据存储的依据，定义有什么数据 |
| **Node** | 逻辑处理，每个节点是一个功能 |
| **Edge** | 连接不同节点，实现工作流（普通边 vs 条件边） |

LangGraph 更像是一种**工作流框架**，而不仅仅是一个简单的处理工具。

### State 的特点
- LangGraph 接收主状态后，先解析为其中的字段（如 input、output）
- 存储的是**字段内容**而非**状态类本身**
- 是一种对信息管理封装很好的框架
- 用户只需关心有什么状态，赋值、处理、传递等底层操作由 LangGraph 自动管理
