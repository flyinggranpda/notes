# LangGraph 介绍

## 基本概念与流程

这个 lang graph 的大概是什么意思？我已经给大家说了，现在先来操作一下，把这个环境创建一下。很简单，大家用 Anaconda，用那个 commander 去创一下，就是用这条命令，先是 create，把 Python 3.12 创一下，把它激活一下，用那个 pip 把这个包安上，然后确认一下。

一会儿我会用那个 PyCharm 来演示开发。大家要是有用别的也可以自己去随便，反正是能用一个编译器就行，然后把刚配的环境给它接入进来。

接下来看，光讲概念其实没啥用，讲大家也记不清楚。咱们看例子，写个例子，就像他这个简单示例一样，我一边写一边给大家说。

首先把这个环境选成刚才配置的 LG graph，在写这个 LangGraph 的时候，它分三步骤：**节点、变化、状态**。这是很重要的。使用 LangGraph 的时候一定要时刻记住，这件事儿是最基础的事情。把这三个东西都定义出来，才能进行一个图的编译以及它的运行。

## 状态（State）定义

```python
from typing import TypedDict

# 状态定义
class OverallState(TypedDict):
    # 里面传出我们使用的时候一些状态值
    using_input: str  # 示例：using put
```

首先先把这个状态写出来，它一般是个类（class），一般写的是 `overall state`。里面传的参数是 `type dict`，这就是它的一个状态。里面它要去传出一些我们使用的时候的状态值。这其实就是在一个类里面放了一些你规定的值就 OK 了。

## 节点（Node）定义

节点其实很简单，它的概念就是**能进行任何逻辑操作的函数**就是节点。把函数定义出来之后，再给它注册为节点就 OK 了。但是可能它实现的逻辑或者你要完成的功能可能会比较复杂。

```python
def node_1(state: OverallState) -> OverallState:
    # 从状态中提取数据
    using_input = state["using_input"]
    # 处理逻辑
    print(f"input: {using_input}")
    # 返回更新后的状态
    return {"using_input": using_input}
```

写个最简单节点，定义一个函数叫 `node_1`。其中要写一个 `state` 括号，把 `state` 放进去。还要指定它输出到哪个状态之中，写一个箭头，再把这个 `overall state` 放进去 OK，然后里面可以写逻辑了。比如把 `using_input` 提出来，从 `state` 中获取数据。从 state 里面去抓取数据的话，一定要用这种方式。

```python
return state
```

`State` 就是这样，`return state` 然后把 `using_input` 给提出来。把处理完的结果返回到这个状态之中。看文档是怎么做的——定义了一个 `message state`，把 message 返回来。我们其实也差不多，只不过用字符串的形式来反馈出来。

## 图（Graph）的构建

### 创建构造器

注册好节点之后，就可以去注册这个图。

```python
from langgraph.graph import StateGraph

# 创建图的构造器
builder = StateGraph(OverallState)
# 这就是一个图的构造出来了
```

先写一个构造器，graph = `StateGraph(OverallState)`，把这个状态传进来。这就是一个图的构造出来了。

### 注册节点

接下来注册节点。

```python
builder.add_node("node_1", node_1)
```

第一个是这个 node 在它图里的名称，不一定非得跟这个函数重名。第二个是把这个原函数放进来。

### 注册边（Edge）

节点注册好了，接下来注册边。边就是连接节点的一个小组件。

```python
# 添加边
builder.add_edge(START, "node_1")  # 从 start 到 node_1
builder.add_edge("node_1", END)    # 从 node_1 到 end
```

添加边的时候，首先从 `START` 开始注册。这也是 LangGraph 里面的一个很重要的组件叫 `START`，`END` 表示图结束，`START` 表示图开始。

### 编译与调用

图注册好了，编译出来。

```python
graph = builder.compile()
```

编译图之后就可以进行调用图。调用用 `invoke`，里面传一个参数，那个参数就是状态中的 `using_input`，在调用的时候给状态进行赋值。

```python
graph.invoke({"using_input": "hello"})
```

里面传一个字典，把 `using_input` 传进去。这就是一个最简单、最普通的 LangGraph 的图处理流程：先注册一个状态，在一个节点中把这个状态提出来，进行输出，完成逻辑处理功能，然后把图注册好，把边添加上，编译进行调用。

## 课程章节概览

回到文档来看，下一部分是我们要讲的重点。有几个章节：

### 第三章：Graph API（重点）
一些比较基础的内容——状态、边、节点，这三个最重要的部分是在第三章里面。还有 `send`（并行发送处理）、`commander`（走向同时进行赋值）、`runtime context`（运行实时上下文）、可视化等。这部分是比较重要比较基础的部分，要细讲。

### 第四章：高级特性（重要）
在进行项目编写的时候几乎都会用到这些处理。比如：

- **持久化**
- **流处理**（很重要）— 就像大模型对话网页（如 DeepSeek）打字机效果一样输出，提示正在使用什么工具、正在干嘛，把目前正在处理的事情反馈出来
- **中断** — 人机交互功能。LangGraph 运行一半时停住，需要人为输入或其他外部来源的输入才能继续运行
- **时间旅行**（状态回溯）— 运行时可以进行状态的回复，避免错误以及进行追溯操作
- **子图** — 图也可以作为一个节点的注册，进行层层嵌套，完成比较复杂的功能

### 第五章：Functional API
它其实不是很重要，一般不会用到这种 Functional API 的特性。它通过一种标准的函数构造思想来去 grab LangGraph 的处理，底层也是用的 Google 的 Pregel 架构。写起来跟正常写函数差不多。
