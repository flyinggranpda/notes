# LangGraph 节点等待机制

## 问题场景

当一个节点有**多个前序节点**时，可能会因为图结构的拓扑顺序导致该节点被**多次执行**。

```
图结构示意：

    A
   / \
  B   B2    ← 两个分支
   \ /
    C
    |
    D
```

实际运行路径：
1. A → B → C → D（第一次）
2. A → B2 → C → D（第二次）

在这个结构中，如果 `B` 和 `B2` 是串行关系，`D` 节点实际上会被执行**两次**——因为它无法感知到还有另一个前序节点未完成。

## 等待机制（wait_for）

### `wait_for` 参数

在注册节点时，设置 `wait_for=True`，表示该节点会等待所有上游数据流到达后再执行。

```python
builder.add_node("D", d_node, wait_for=True)  # 开启等待机制
```

### 工作流程

开启等待后，节点的行为变为：

1. 节点监控**是否有数据将会流向它**
2. 如果检测到上游有**多个节点**将流向它，它会等待所有数据流到达
3. 所有前序节点完成后，再进行汇总处理

```
开启等待后的流程：

A → B ─┐
       ├→ 等待 → C → D（只执行一次）
A → B2 ┘
```

`B` 和 `B2` 都完成后，才会进入 `C`，最后 `D` 只执行一次。

### 如果不开启等待

如果不开启等待，图结构相当于变成两条独立的路径：

```
不开启等待的实际流程：

A → B → C → D（第一次运行 D）
A → B2 → C → D（第二次运行 D）
```

**最显著的特征：D 输出了两遍。**

## Pregel 算法与超步（Superstep）

LangGraph 底层使用 Google 的 **Pregel 算法**，它是一种图结构的逻辑处理算法。

**超步（Superstep）** 的概念类似于回合制游戏：
- 每个超步中，所有活跃节点各自执行一步
- 不同路径上的节点可能不是同一时刻触发，但在超步的概念上属于**同时触发**
- 当多个节点同时操作同一数据时，LangGraph 会在超步边界进行同步等待

```
超步 1: A 执行
超步 2: B 和 B2 同时执行（属于同一超步）
超步 3: C 执行（等待 B 和 B2 都完成）
超步 4: D 执行
```

## 最佳实践

**当进行汇合节点时，最好把等待机制开启。** 原因：

1. **状态不易变**：图的状态一开始就要定好
2. **图容易变**：后续添加功能、修改边连接方式时，可能会多出节点或路径
3. **避免隐蔽的 Bug**：如果是涉及数据操作的关键节点，执行两遍可能导致数据错误

> 费曼说：多路数据汇合就像十字路口的红绿灯——不开等待就是没有红绿灯，所有车各走各的，D 节点会被反复碾压。开了等待就像装了红绿灯，等各个方向的车都到齐了再统一放行。

### 代码示例

```python
from langgraph.graph import StateGraph, START, END

class OverallState(TypedDict):
    aggregate: list[str]

def node_a(state: OverallState) -> OverallState:
    print("A 添加")
    return {"aggregate": ["A"]}

def node_b(state: OverallState) -> OverallState:
    print("B 添加")
    return {"aggregate": ["B"]}

def node_b2(state: OverallState) -> OverallState:
    print("B2 添加")
    return {"aggregate": ["B2"]}

def node_c(state: OverallState) -> OverallState:
    print("C 添加")
    return {"aggregate": ["C"]}

def node_d(state: OverallState) -> OverallState:
    print("D 添加")
    return {"aggregate": ["D"]}

builder = StateGraph(OverallState)
builder.add_node("A", node_a)
builder.add_node("B", node_b)
builder.add_node("B2", node_b2)
builder.add_node("C", node_c)
builder.add_node("D", node_d, wait_for=True)  # 开启等待

builder.add_edge(START, "A")
builder.add_edge("A", "B")
builder.add_edge("A", "B2")
builder.add_edge("B", "C")
builder.add_edge("B2", "C")
builder.add_edge("C", "D")
builder.add_edge("D", END)

graph = builder.compile()
result = graph.invoke({"aggregate": []})
# 输出: A, B, B2, C, D（D 只执行一次）
```

```
不开启等待时输出: A, B, C, D, B2, C, D（D 执行两次）
开启等待后输出:   A, B, B2, C, D（D 执行一次）
```

> 注意：`B` 先注册因此先输出，`B2` 后注册后输出，但在超步概念上它们属于同步处理。
