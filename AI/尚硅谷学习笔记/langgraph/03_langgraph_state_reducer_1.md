# LangGraph State Reducer（一）

## Reducer 的作用

Schema 定义了**状态如何存储**，而 **Reducer** 定义了**状态如何更新和处理**。

在前面的例子中，更新状态就是直接赋值（覆盖）。而 Reducer 可以让我们进行更丰富的操作：

- **追加**（列表、消息）
- **数学计算**（累加、相乘）
- **自定义处理逻辑**

## Reducer 默认行为

不指定 reducer 时，默认使用**覆盖更新**。

## 常用 Reducer 函数

### 1. 默认覆盖（不指定 reducer）

```python
# 不指定 reducer 时，直接覆盖
state["value"] = new_value  # 新值覆盖旧值
```

流程示例：
1. `START` → 传入状态 `{"foo": "bar"}`
2. `node_one` → 传入 `{"foo": 2}`  → 覆盖为 2
3. 再传入 `{"foo": "bye"}` → 覆盖为 "bye"

### 2. `add_message` — 消息追加

用于消息列表（如 ChatMessage、AIMessage、HumanMessage 等）的追加。

### 3. `operator.add` — 列表追加 / 数值累加

```python
from operator import add
from typing import Annotated

class OverallState(TypedDict):
    output: Annotated[int, add]  # 指定 add 作为 reducer
```

会根据数据类型自动判断操作方式。例如状态中有 `output: int`，多个节点都进行加一操作：

```python
# 节点 1：return {"output": 1}
# 节点 2：return {"output": 1}
# 节点 3：return {"output": 1}
# 结果：output = 1 → 2 → 3（累加，每次加 1，不会覆盖）
```

每个节点返回 1，由于指定了 `add` reducer，值会依次累加（1 → 2 → 3 → ...），而不是每次覆盖成 1。

### 4. `operator.mul` — 数值相乘

与 `add` 类似，但执行乘法操作：

```python
# 节点 1：return {"value": 2}
# 节点 2：return {"value": 3}
# 结果：value = 2 * 3 = 6
```

> ⚠️ 底层设计有一些缺陷，通常不会在生产中使用。

### 5. 自定义 Reducer

可以根据自己的设计，定义任意状态更新逻辑，自由度很高。

---

## 总结

| Reducer 类型 | 说明 | 使用场景 |
|-------------|------|---------|
| **默认（覆盖）** | 不指定 reducer，新值直接覆盖旧值 | 简单赋值 |
| **add_message** | 消息列表追加 | 对话消息 |
| **operator.add** | 数值累加 / 列表合并 | 计数器、日志收集 |
| **operator.mul** | 数值相乘 | （少用） |
| **自定义 reducer** | 自由定义更新逻辑 | 复杂业务处理 |
