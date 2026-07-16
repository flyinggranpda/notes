# LangGraph 状态处理聊天机器人示例

## 综合案例：聊天机器人状态管理

展示如何通过**多种 reducer 组合**来实现一个聊天机器人的状态管理。

### 状态定义

```python
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from operator import add

class OverallState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # 自定义消息追加
    text: Annotated[list[str], add]                           # 列表追加
    score: Annotated[float, add]                              # 数值累加
```

三种不同的 reducer：
1. `messages` — 使用 `add_messages` 进行对话消息叠加
2. `text` — 使用 `operator.add` 进行字符串列表追加
3. `score` — 使用 `operator.add` 进行分数累加

### 节点定义

```python
def process_user_message(state: OverallState) -> OverallState:
    # 提取最新的一条消息
    last_message = state["messages"][-1]
    # 返回处理结果
    return {
        "messages": [last_message],      # 追加消息
        "text": [last_message.content],   # 追加文本内容
        "score": 1.0                      # 累加分数
    }

def add_sentiment_tag(state: OverallState) -> OverallState:
    return {
        "text": ["positive"],   # 追加情感标签
        "score": 0.5            # 追加分数
    }
```

### 图的构建

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(OverallState)
builder.add_node("process_user_message", process_user_message)  # 先注册
builder.add_node("add_sentiment_tag", add_sentiment_tag)        # 后注册

builder.add_edge(START, "process_user_message")
builder.add_edge(START, "add_sentiment_tag")
builder.add_edge("process_user_message", END)
builder.add_edge("add_sentiment_tag", END)

graph = builder.compile()
```

> 先注册的节点先执行（`process_user_message` 先执行）。

### 调用与结果

```python
result = graph.invoke({
    "messages": [HumanMessage(content="hello, how are you?")],
    "text": ["greeting"],
    "score": 0.0
})
```

**处理流程：**

1. **`process_user_message`**（先执行）
   - 提取最后一条消息 `"hello, how are you?"`
   - `messages` → 追加（`add_messages`）
   - `text` → 追加 `"hello, how are you?"` → `["greeting", "hello, how are you?"]`
   - `score` → 累加 1.0 → `0.0 + 1.0 = 1.0`

2. **`add_sentiment_tag`**（后执行）
   - `text` → 追加 `"positive"` → `["greeting", "hello, how are you?", "positive"]`
   - `score` → 累加 0.5 → `1.0 + 0.5 = 1.5`

## 综合总结

这个示例展示了 LangGraph 状态管理的**综合能力**：

| 字段 | Reducer | 行为 |
|------|---------|------|
| `messages` | `add_messages` | 对话消息叠加 |
| `text` | `operator.add` | 字符串列表追加 |
| `score` | `operator.add` | 数值累加 |

可以通过不同方式组合使用：
- **默认覆盖** — 单纯填写空白
- **内置 Reducer** — `add`、`add_messages` 等
- **自定义函数** — 实现任意复杂处理
