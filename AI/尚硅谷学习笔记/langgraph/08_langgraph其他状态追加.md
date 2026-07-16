# LangGraph 其他状态追加

## operator.add 的多类型支持

`operator.add` 可以自动识别要追加的数据类型——底层源码考虑了多种数据类型，使用起来很方便。

![类型支持]
- **列表追加** → `Annotated[list[int], add]`
- **字符串连接** → `Annotated[str, add]`
- **数值相加** → `Annotated[int, add]`

### 示例对比

| 数据类型 | 初始值 | 节点 1 返回值 | 节点 2 返回值 | 最终结果 |
|---------|--------|--------------|--------------|---------|
| `list[int]` | `[0]` | `[1, 2]` | `[3, 4]` | `[0, 1, 2, 3, 4]` |
| `str` | `"say"` | `" hello"` | `" world"` | `"say hello world"` |
| `int` | `10` | `5` | `3` | `18` |

三种操作是完全相同的模式，只是**数据类型换了**。

## operator.mul — 数值相乘（有缺陷）

### 状态定义

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import mul

class OverallState(TypedDict):
    value: Annotated[float, mul]
```

### 缺陷说明

```python
# 初始传入 5.0，预期经过乘法节点后变成 25.0
result = graph.invoke({"value": 5.0})
# 实际结果: 0.0
```

**原因**：在执行初始阶段、第一个节点处理前，LangGraph 会默认调用 reducer。`mul` 的默认值是 0，初始调用时值就已经变成了 0（`5.0 * 0 = 0`），之后无论怎么乘结果都是 0。

### 替代方案

使用**自定义 reducer** 来实现乘法或更复杂的数学运算。

## 使用方式总结

`operator.add` 的使用方式非常统一：

1. 在状态定义中，用 `Annotated[类型, add]` 修饰字段
2. 节点正常返回该字段的值
3. LangGraph 自动按数据类型执行追加操作

```python
class OverallState(TypedDict):
    # 列表追加
    items: Annotated[list[str], add]
    # 字符串连接
    text: Annotated[str, add]
    # 数值累加
    count: Annotated[int, add]
```
