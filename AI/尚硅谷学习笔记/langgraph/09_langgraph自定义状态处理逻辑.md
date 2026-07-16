# LangGraph 自定义状态处理逻辑（Custom Reducer）

## 为什么需要自定义 Reducer

当内置的 reducer（默认覆盖、`add`、`add_messages` 等）无法满足需求时，可以自定义 reducer。例如：

- 需要对字典进行合并更新
- 修复 `operator.mul` 的零值 bug
- 进行开平方、开根号等复杂运算
- 任何 LangGraph 原生未提供的操作

## 自定义 Reducer 的规范

### 函数签名要求

```python
def custom_reducer(current_value, new_value) -> output:
    # 处理逻辑
    return result
```

- **第一个参数**：当前值（原值）
- **第二个参数**：新值（要更新的值）
- **返回值**：处理后的结果

> **注意**：参数名不限定，是按**位置**匹配的（第一个是当前值，第二个是新值）。返回值的数据类型必须和当前值保持一致。

### 示例：字典合并

```python
from typing import Annotated
from typing_extensions import TypedDict

# 自定义 reducer：合并字典
def custom_reducer(current_value: dict, new_value: dict) -> dict:
    result = current_value.copy()
    result.update(new_value)  # 合并字典
    return result

class OverallState(TypedDict):
    meta_data: Annotated[dict, custom_reducer]  # 传入自定义 reducer
```

```python
def process_node(state: OverallState) -> OverallState:
    # 返回的 meta_data 会通过 custom_reducer 合并到当前状态中
    return {"meta_data": {"temp": 25, "region": "shanghai"}}
```

```python
builder = StateGraph(OverallState)
builder.add_node("process_node", process_node)
builder.add_edge(START, "process_node")
builder.add_edge("process_node", END)
graph = builder.compile()

graph.invoke({"meta_data": {"user_id": "abc", "session": "1"}})
# meta_data 最终会合并：{"user_id": "abc", "session": "1", "temp": 25, "region": "shanghai"}
```

## 自定义乘法 Reducer

解决 `operator.mul` 初始乘零的缺陷：

```python
# 全局变量：追踪是否是第一次调用
is_first_call = True

def custom_multiply_reducer(current_value: float, new_value: float) -> float:
    global is_first_call
    print(f"被调用：当前值={current_value}，新值={new_value}")
    
    if is_first_call:
        is_first_call = False
        return new_value  # 第一次调用直接返回新值，避免乘零
    else:
        return current_value * new_value  # 后续正常相乘
```

```python
class OverallState(TypedDict):
    value: Annotated[float, custom_multiply_reducer]
```

```python
def multiply_node(state: OverallState) -> OverallState:
    return {"value": 2.0}
```

```python
# 传入 5.0，经过 multiply_node（乘 2），结果为 10.0
result = graph.invoke({"value": 5.0})
# 而不是 0.0
```

### 自定义连续乘法

```python
# 节点 1：乘 2
# 节点 2：乘 2
# 节点 3：乘 0
```

通过自定义 reducer，可以完全控制状态的更新方式——**覆盖、追加、相乘、任何自定义逻辑**都可以实现。

## 自定义 Reducer 的灵活性

| 场景 | 内置方案 | 自定义方案 |
|------|---------|-----------|
| 字典合并 | 不支持 | 自定义 `update` 合并 |
| 乘法 | `operator.mul`（有缺陷） | 自定义乘法，跳过初始零值 |
| 开平方 / 开根号 | 不支持 | 自定义数学运算 |
| 复杂业务逻辑 | 不支持 | 任意自定义代码 |

> 费曼说：自定义 Reducer 就像是给了你一把万能钥匙——内置 Reducer 打不开的锁，自己造一把钥匙就行。只要符合"两个参数进、一个同类型出"的格式，你想怎么处理都行。
