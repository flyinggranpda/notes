# LangGraph 列表状态追加（operator.add）

## operator.add 列表追加

`operator.add` 可以将元素**追加到现有元素中**，支持多种数据类型：

- **列表追加**
- **字符串连接**
- **数值相加**

### 状态定义

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class OverallState(TypedDict):
    data: Annotated[list[int], add]  # 使用 add 作为 reducer，表示列表追加
```

### 节点定义

```python
def producer_one(state: OverallState) -> OverallState:
    return {"data": [1, 2]}

def producer_two(state: OverallState) -> OverallState:
    return {"data": [3, 4]}
```

### 图的构建

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(OverallState)
builder.add_node("producer_one", producer_one)
builder.add_node("producer_two", producer_two)

# 并行执行
builder.add_edge(START, "producer_one")
builder.add_edge(START, "producer_two")
builder.add_edge("producer_one", END)
builder.add_edge("producer_two", END)

graph = builder.compile()
```

### 运行结果

```python
result = graph.invoke({"data": [0]})
# 结果: data = [0, 1, 2, 3, 4]
```

流程：
1. 调用时传入 `data = [0]`
2. `producer_one`（先注册，先执行）→ 追加 `[1, 2]` → `[0, 1, 2]`
3. `producer_two` → 追加 `[3, 4]` → `[0, 1, 2, 3, 4]`

由于指定了 `operator.add`，值不会被覆盖，而是不断追加。

## operator.add 字符串连接

同样使用 `operator.add`，会自动识别数据类型进行字符串连接：

```python
class OverallState(TypedDict):
    text: Annotated[str, add]
```

```python
def node_one(state: OverallState) -> OverallState:
    return {"text": " hello"}

def node_two(state: OverallState) -> OverallState:
    return {"text": " world"}
```

```python
result = graph.invoke({"text": "say"})
# 结果: text = "say hello world"
```

与列表追加完全相同的模式，只是数据类型从 `list` 换成了 `str`。

## operator.add 数值累加

```python
class OverallState(TypedDict):
    value: Annotated[int, add]
```

```python
# node_one: return {"value": 5}
# node_two: return {"value": 3}
```

```python
result = graph.invoke({"value": 10})
# 结果: value = 10 + 5 + 3 = 18
```

## operator.mul — 数值相乘

```python
class OverallState(TypedDict):
    value: Annotated[float, mul]

def multiply_node(state: OverallState) -> OverallState:
    return {"value": 5.0}
```

```python
result = graph.invoke({"value": 5.0})
# 预期: 25.0，实际: 0.0
```

> ⚠️ **缺陷说明**：官方设计上存在缺陷。在执行初始阶段、第一个节点处理前会默认调用 reducer，默认值是 0。初始乘 0 之后，无论怎么乘结果都是 0。目前不会使用这个，后续版本可能会修复。

如果需要进行乘法操作，应该使用**自定义 reducer**。

## 总结

`operator.add` 是一个很方便的一键式组件，LangGraph 封装得很好。它会**根据数据类型自动判断**采用什么处理原则。

| 数据类型 | 操作 | 示例 |
|---------|------|------|
| `list` | 列表追加 | `[1] + [2, 3] = [1, 2, 3]` |
| `str` | 字符串连接 | `"hello" + " world" = "hello world"` |
| `int / float` | 数值相加 | `10 + 5 + 3 = 18` |
| `mul`（乘法） | ⚠️ 有缺陷 | 初始乘零导致结果恒为 0 |
