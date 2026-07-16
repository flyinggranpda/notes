# LangGraph 消息列表状态追加（add_messages）

## Reducer 的作用

Reducer 是**如何更新状态中的信息**的机制。LangGraph 的状态更新不仅限于默认的覆盖操作，还可以进行：

- **追加**（列表尾部添加）
- **数值累加**（1 → 2 → 3 → 4）
- **自定义操作**（如开平方、复杂逻辑）

LangGraph 不仅为大模型服务，它更主要的是一个**工作流框架**，大模型只是其中的一个组件。

---

## 常用 Reducer 函数

### 1. 默认覆盖

不指定任何 reducer 时，操作就是覆盖：

- 如果没有值，传进来赋值
- 如果有值，新值覆盖旧值

```python
class OverallState(TypedDict):
    foo: int
    bar: list
```

```python
# node_one：foo = 2（覆盖原来的 1）
# node_two：bar = "bye"（覆盖，不会追加）
```

```python
result = graph.invoke({"foo": 1, "bar": "hi"})
```

可以看到，即使 `bar` 是列表类型，不指定 reducer 时也只是简单覆盖，不会追加。

### 2. add_messages — 消息列表追加

专门用于处理**消息列表**（HumanMessage、AIMessage 等），实现**多轮对话的记忆功能**。

#### 状态定义

```python
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class OverallState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

`Annotated` 表示对 `messages` 变量附加一个处理原则——使用 `add_messages` 这个 reducer 来进行后续的状态处理。

#### 节点定义

```python
def node_one(state: OverallState) -> OverallState:
    return {"messages": [AIMessage(content="hello from node one")]}

def node_two(state: OverallState) -> OverallState:
    return {"messages": [AIMessage(content="hello from node two")]}
```

#### 图的构建与并行执行

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(OverallState)
builder.add_node("node_one", node_one)
builder.add_node("node_two", node_two)

# 并行执行：START 同时连接到两个节点
builder.add_edge(START, "node_one")
builder.add_edge(START, "node_two")
builder.add_edge("node_one", END)
builder.add_edge("node_two", END)

graph = builder.compile()
```

> **并行执行顺序**：当两个节点同时到达时，谁先注册谁先执行。这里 `node_one` 先注册，所以先执行。

#### 运行结果

```python
result = graph.invoke({"messages": []})
```

`messages` 不会覆盖，而是通过 `add_messages` reducer 不断在后面追加：

```
[AIMessage(content="hello from node one"), AIMessage(content="hello from node two")]
```

### 3. operator.add — 列表追加 / 数值累加

用于列表尾部追加或数值的累加操作。

```python
from operator import add

class OverallState(TypedDict):
    counter: Annotated[int, add]
    log: Annotated[list, add]
```

### 4. operator.mul — 数值相乘

```python
from operator import mul

class OverallState(TypedDict):
    value: Annotated[int, mul]
```

> ⚠️ 底层设计有 bug：在最开始会乘个零，导致结果总是零。如果要做乘法，应使用自定义 reducer。

---

## 总结

| Reducer | 用途 | 说明 |
|---------|------|------|
| **默认（覆盖）** | 任意类型 | 无值赋值，有值覆盖 |
| **add_messages** | 消息列表 | 追加 HumanMessage/AIMessage 等对话消息 |
| **operator.add** | 列表 / 数值 | 列表追加、数值累加 |
| **operator.mul** | 数值 | ⚠️ 底层有 bug，推荐用自定义 reducer |
| **自定义 reducer** | 任意 | 自由定义更新逻辑 |
