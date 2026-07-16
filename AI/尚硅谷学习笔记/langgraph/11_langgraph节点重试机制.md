# LangGraph 节点重试机制

## 节点（Node）回顾

在 LangGraph 中，**节点就是 Python 函数**。它从状态中提取信息，处理后再返回给状态。定义节点后，使用 `add_node` 方法添加到图中。

### add_node 的用法

```python
builder.add_node("node_name", node_function)
# 第一个参数：映射名（图中使用的名称）
# 第二个参数：函数本身
```

如果不传映射名，系统会分配一个与函数名相同的默认名称。但按命名规范，一般要传映射名。

## Config 参数

`config` 是一个 `RunnableConfig` 对象，在多轮对话等场景中很重要：

- 在网页对话中（如 ChatGPT），每次对话根据**线程 ID** 从记忆中取出历史
- LangGraph 支持**人机交互**——运行到一半时打断，用户输入新值后继续运行
- 打断后需要把**线程 ID** 传进来，从缓存中重新导入记忆、对话历史、用户输入，从断点继续运行

```python
def my_node(state: OverallState, config: RunnableConfig):
    # config 包含线程 ID 等信息
    pass
```

## START 和 END 节点

- **START** — 特殊节点，表示图的开始。调用 `invoke`/`stream` 时，值从 START 传入
- **END** — 特殊节点，表示图的终止。到 END 表示工作流结束

## 节点缓存（Node Cache）

### 为什么需要缓存

当节点处理高资源消耗且重复性的操作时，可以开启缓存来节省计算资源。

### 缓存策略设置

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint import MemorySaver

# 在 add_node 时设置缓存策略
builder.add_node(
    "compute_node",
    compute_function,
    cache_policy={"ttl": 3}  # 生存时间 3 秒，3 秒后缓存消失
)
```

```python
# 编译时指定缓存存储方式
graph = builder.compile(checkpointer=MemorySaver())
# 支持多种存储：内存、Redis、MongoDB、PostgreSQL 等
```

### 缓存的工作流程

```python
# 第一次调用：传入 s=5 → 节点计算 → 输出 result=10
result1 = graph.invoke({"s": 5})
# 输出: result=10 (未命中缓存)

# 第二次调用：传入 s=5（相同值）→ 直接返回缓存结果，不再计算
result2 = graph.invoke({"s": 5})
# 输出: result=10, metadata: cache=True (命中缓存)
```

> 注意：3 秒后缓存过期，再次调用会重新计算。

## 节点重试策略（Retry Policy）

### 为什么需要重试

在调 API 或查询数据库时，可能遇到网络异常或服务限流：
- 大模型服务商 API 调用频率过高时返回 500 错误
- 等待一秒钟再连就能连上

### 默认重试机制

```python
from langgraph.graph import StateGraph, START, END

# 在 add_node 时设置重试策略
builder.add_node(
    "api_node",
    api_function,
    retry_policy={
        "max_attempts": 3,  # 最大重试次数
    }
)
```

默认情况下，遇到以下异常时会重试：
- 网络错误
- 连接异常

而明显的代码错误（如 ValueError）不会重试——重试多少次都没用，应该去改代码。

### 模拟不稳定的 API

```python
attempt_counter = 0

def unstable_api_call(state: OverallState) -> OverallState:
    global attempt_counter
    attempt_counter += 1
    print(f"第 {attempt_counter} 次尝试")
    
    if attempt_counter < 3:
        raise Exception("模拟 API 调用失败")  # 前两次抛异常
    
    # 第三次成功
    result = state["x"] * 2
    return {"result": result}
```

### 自定义重试策略

当默认重试不满足需求时（如内部定义的特殊异常），可以自定义重试判断逻辑：

```python
def custom_retry_on(exception: Exception) -> bool:
    """自定义重试判断函数"""
    error_msg = str(exception)
    if "模拟 API 调用失败" in error_msg:
        print(f"捕获到异常: {error_msg}")
        return True   # 返回 True 表示要重试
    return False      # 返回 False 表示不重试，直接报错
```

```python
builder.add_node(
    "api_node",
    unstable_api_call,
    retry_policy={
        "max_attempts": 5,
        "retry_on": custom_retry_on  # 自定义判断逻辑
    }
)
```

### 不会重试的异常类型

```python
def bad_function(state: OverallState) -> OverallState:
    raise ValueError("值错误，再怎么重试也没用")
```

`ValueError` 这类代码错误，即使设置了重试策略也不会重试——因为重试多少次都是同样的错误。

---

## 总结

| 节点特性 | 说明 | 使用场景 |
|---------|------|---------|
| **Config** | 线程 ID / 上下文信息 | 多轮对话、人机交互恢复 |
| **缓存** | 相同输入跳过计算直接返回结果 | 高频重复操作、节省资源 |
| **默认重试** | 网络错误时自动重试 | API 调用、数据库连接 |
| **自定义重试** | 自定义判断是否重试 | 特殊异常处理 |
| **不重试异常** | 代码错误直接报错 | ValueError 等明显错误 |
