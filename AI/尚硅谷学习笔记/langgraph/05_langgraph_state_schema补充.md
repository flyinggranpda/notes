# LangGraph State Schema（补充）

## 示例一：基础工作流

### 状态定义

```python
from typing_extensions import TypedDict

class OverallState(TypedDict):
    user_input: str
    user_output: str
```

### 节点实现

```python
def node_one(state: OverallState) -> OverallState:
    # 从状态中提取 user_input
    using_input = state["user_input"]
    # 构造字典：将 user_input 的值赋给 user_output
    result = {"user_output": using_input, "user_input": using_input}
    return result  # 返回给状态进行赋值

def node_two(state: OverallState) -> OverallState:
    # 从状态中提取 user_output
    output = state["user_output"]
    print(output)
    return state
```

### 工作流流程

```
State: {user_input, user_output}

START → node_one → node_two → END
```

1. 外部调用 `invoke` 传入 `user_input`
2. `node_one` 从 state 中提取 `user_input`，处理后写回 `user_output`
3. `node_two` 提取 `user_output` 进行输出

### 节点和边的注册

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(OverallState)

# 注册节点
builder.add_node("node_one", node_one)  # 名称可自定义，不一定要和函数名相同
builder.add_node("node_two", node_two)

# 注册边
builder.add_edge(START, "node_one")
builder.add_edge("node_one", "node_two")
builder.add_edge("node_two", END)

graph = builder.compile()

# 调用
result = graph.invoke({"user_input": "hello"})
```

> 注意：状态不好变，但图很好改变。只要把边的连接方式一变，改造就完成了。

---

## 示例二：输入输出隔离

### 三个状态类

```python
class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class OverallState(InputState, OutputState):
    pass  # 同时包含 question 和 answer
```

`OverallState` 包含 `InputState` 和 `OutputState`，三者是**包含关系**。

- **InputState**：只能输入，不可被改写
- **OutputState**：只可以被输出
- 通过 `input_schema` / `output_schema` 实现隔离

### 隔离的实现

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(
    OverallState,
    input_schema=InputState,    # 限定输入
    output_schema=OutputState   # 限定输出
)
```

在构造图时指定 `input_schema` 和 `output_schema`，内部处理就会做隔离：

```python
def answer_node(state: OverallState) -> OutputState:
    # 从 InputState 中提取 question
    question = state["question"]
    # 处理逻辑
    if question.lower() == "bye":
        answer = "再见"
    else:
        answer = "你好"
    # 返回值中 answer 属于 OutputState，question 不属于
    result = {"answer": answer, "question": question}
    print(f"result: {result}")
    return result
```

输出只包含 `answer`（属于 OutputState），`question` 不会被输出。

> 这种隔离模式适用于需要严格数据隔离的场景，但不是每回都会用。

---

## 示例三：私有状态传递

### 状态定义

```python
class OverallState(TypedDict):
    a: str

class NodeOneOutput(TypedDict):
    private_data: str

class NodeTwoInput(TypedDict):
    private_data: str  # 字段名和 NodeOneOutput 相同
```

`NodeOneOutput` 和 `NodeTwoInput` 类名不同，但字段名 `private_data` 相同。

### LangGraph 的状态管理机制

LangGraph 的状态管理是**基于字段名而不是类名**：

- 构造器中传入 `OverallState`，内部会把字段拆解出来存储
- 存储的是**字段而非类**
- 不在乎类叫什么名字，最在乎的是**字段到底有没有**

### 私有状态的暂存与消费

```python
builder = StateGraph(OverallState)  # 只注册了 OverallState
builder.add_node("node_one", node_one)
builder.add_node("node_two", node_two)
```

```python
def node_one(state: OverallState) -> NodeOneOutput:
    # 直接返回 private_data（不在 OverallState 中）
    return {"private_data": "node one 设置"}

def node_two(state: NodeTwoInput) -> OverallState:
    # 直接从字段名 private_data 承接
    return {"a": state["private_data"]}
```

当 `node_one` 返回的 `private_data` 不在 `OverallState` 中注册时：

1. LangGraph 内部检测到没有这个值
2. 把它**暂存到 RAM（内存）** 中，不做长期持久化
3. 直到下一次消费 `private_data` 时（`node_two`），把值提取出来
4. 同时**释放**这个私有的临时状态

### 类型注解的作用

```python
def node_one(state: OverallState) -> NodeOneOutput:
    # -> NodeOneOutput 只是一个书写规范
```

类型注解是**书写规范**，用于限定编写者的处理。即使去掉也可以运行，但加上可以：
- 规范代码格式
- 协作时方便他人
- 错误时及时反馈

### 私有状态的使用场景

当某个状态**不想要任何外部介入，也不需要通过中间变量输出，传递后即销毁**时，可以使用私有状态传递。
