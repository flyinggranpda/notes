# 第四章 · SDD+Harness 实战启动 —— 可执行复刻笔记

> 范围：本项目笔记对应课程《项目需求分析与项目工程初始化（SDD+Harness 实战启动）》第四章（4-1 ~ 4-15）。
> 目标：跟随本笔记可完整复刻出 `my-first-project` 全栈项目骨架（Harness + openSpec + superPower 三层理念）。
> 阅读方式：严格按 Step 序号执行；每个 Step 即一个技术决策点或可复制命令。

---

## 0. 本章地图

| 节次 | 主题 | 核心产出 |
|---|---|---|
| 4-1 | 认知转变 | 人的五大优势、AI 驱动开发模式 |
| 4-2 | 结构化文档体系 | 五层文档体系 + 四原则 |
| 4-3 | 三件套能力介绍 | Harness/openSpec/superPower + opsx 四步 |
| 4-4 | 目录结构与知识工程 | 一行 Prompt 生成项目骨架 |
| 4-5 | 初体验 | /opsx 四步演示 + 产出物清单 |
| 4-6 | Spec 编写实战 | 高质量 Spec（Todo 模块） |
| 4-7 | Spec Review | 万能 Review Prompt + 三档处理 |
| 4-8 | AIWorkSpace | git submodule 多仓库管理 |
| 4-9 | 架构选型 | AI 辅助选型 4 步 + ADR |
| 4-10 | 前端工程初始化 | 前端选型落地 |
| 4-11 | 后端工程初始化 | 后端选型落地 + 三大关注点 |
| 4-12 | 数据库选型与设计 | MySQL + Flyway + 双 ID 模式 |
| 4-13 | 阶段性 Review | 整体架构：一父两子 + 三层治理 |
| 4-14 | 关键经验与 FAQ | 五条踩坑经验 + BFF + 六 FAQ |
| 4-15 | 本章小结 | 质量公式 + 课后任务 |

---

## Part 1 · 认知与方法论（4-1 ~ 4-5）

### 4-1 从传统开发到 AI 驱动开发

**Step 1 对比两种开发模式的量化差异**

| 维度 | 传统模式 | AI 驱动模式 |
|---|---|---|
| 参与人数 | 6+ 人（PM/UI/前后端/QA/运维） | 1 人 + AI |
| 交付周期 | 21–38 工作日 | 1–2 天 |
| 协作方式 | 串行流水线 | 并行协作 AI |
| 信息损耗 | 每次跨角色传递衰减（脑中 100% → 手上 60%） | 规范驱动零衰减 |
| 文档作用 | 人—人沟通媒介 | 给 AI 的精确指令集 |

**Step 2 记住信息衰减的经典案例**
- 产品经理想："根据用户浏览历史推荐"
- PRD 写成："根据用户行为推荐" → 设计理解"点击行为" → 开发理解"搜索关键词"
- 最终上线："关键词搜索建议" ← 需求已完全变形

**Step 3 人的五大不可替代优势（决定 AI 协作分工）**

| # | 能力 | 职责 |
|---|---|---|
| 1 | 问题定义 | 决定 AI 做什么 |
| 2 | 上下文构建 | 让 AI 理解环境 |
| 3 | 结果验证 | 判断 AI 做对没 |
| 4 | 技术决策 | 选择 AI 怎么做 |
| 5 | 成本控制 | 控制 AI 花多少 |

**Step 4 核心技术认知**
- 公式：`AI 输出质量 = f(输入质量)`；不是 AI 做不了，是 AI 做什么都需要你先做对。
- 解决三件套：Harness（护栏）+ openSpec（规范）+ superPower（增强）。

---

### 4-2 基于 Harness 构建结构化文档体系

**Step 1 识别"模糊愿望"的危害**
- 一句话需求："帮我做一个后台管理系统，要有用户管理、权限管理、数据看板。"
- 其中至少包含 4 类未定义项：用户范围/是否 SSO、RBAC vs ABAC/权限粒度、看板数据/实时性/图表库、React vs Vue/TS/REST vs GraphQL。
- 结论：你给的是模糊愿望，AI 需要的是精确规范。

**Step 2 结构化文档三特征**
1. 机器可读：固定字段格式，AI 能直接解析执行（WHEN 用户点击登录按钮，THEN 调用 API /api/login，验证成功跳转首页）。
2. 格式统一：同类型文档同模板、字段名全局一致（API Spec 标准模板）。
3. 相互引用：文档间形成知识网络（UI Spec 引用 API Spec，实体字段全局对齐）。

**Step 3 为什么 AI 比人更需要结构化文档**

| 维度 | 人类协作 | AI 协作 |
|---|---|---|
| 沟通方式 | 可意会、脑补、追问 | 只看文字，不猜测不推断 |
| 上下文 | 共同经历和默契 | 每次对话从零开始 |
| 模糊容忍 | "差不多就行"可接受 | 模糊 = 随机输出 |
| 纠错 | 当场追问澄清 | 生成完才知道理解错了 |
| 质量保证 | 经验 + 直觉 | 依赖明确验收标准 |

**Step 4 五层结构化文档体系**

| 层级 | 回答的问题 | 内容 | 类比 |
|---|---|---|---|
| 1. Product Spec | 做什么 | 产品目标、功能清单、MVP 范围 | 设计概念图 |
| 2. Design Doc | 怎么做 | 技术架构、选型决策、模块拆分、ADR | 结构图 |
| 3. API Spec | 怎么连 | 前后端接口契约（字段/类型/错误码/WHEN-THEN） | 管线图 |
| 4. UI Spec | 长什么样 | 页面结构、交互流程、组件状态、布局 | 装修图 |
| 5. Exec Plan | 先做什么 | 任务拆分、优先级、依赖关系、Sprint 范围 | 施工排期表 |

**Step 5 四原则（写文档时的硬约束）**
1. 精确度 > 完整度（宁可少写，不可写模糊）
2. 可验证性 > 描述性（每个要求有验收标准）
3. 分层引用 > 全量加载（按需引用，别把五层文档全塞给 AI）
4. 文档与代码同步演进

---

### 4-3 Harness + openSpec + superPower 能力介绍

**Step 1 遇到问题找谁：两层结构**
- Harness 护栏层（管"AI 不能做什么"，红线）：

| 问题 | 解决文件 | 作用 |
|---|---|---|
| 代码风格不统一 | `.cursorrules` | 写死命名规范、缩进、禁用 API |
| AI 忽视项目背景 | `AGENTS.md` | 顶层导航，AI 每次工作前先读 |
| 模板格式乱 | `.harness/templates/` | 标准化文档模板库 |
| AI 犯过的错重犯 | `knowledge/ai-error-log.md` | 错误日志，只增不删 |
| 不知道项目约定 | `knowledge/conventions.md` | 命名/架构/规范约定 |

- openSpec 规范层（管"AI 应该做什么"，蓝图，位于 docs/）：

| 问题 | 目录 |
|---|---|
| 不知道产品做什么 | `docs/product-specs/` |
| 架构决策没记录 | `docs/design-docs/`（ADR） |
| 前后端接口对不上 | `docs/api-specs/` |
| UI 做出来不像稿 | `docs/ui-specs/` |
| 开发顺序混乱 | `docs/exec-plans/` |

**Step 2 /opsx 指令链路（需求驱动开发四步）**

```
/opsx:explore → /opsx:propose → /opsx:apply → /opsx:archive
```

| Step | 动作 | 产出 | 人是否介入 |
|---|---|---|---|
| explore | 扫描 docs/+knowledge/，定位已有/缺失/过期 | 信息版图报告 | 否 |
| propose | 基于 explore 生成实施计划（新建/更新哪些 Spec、任务拆分、风险预判） | 方案 | 是（审核后才执行） |
| apply | 按方案创建 Spec + 代码，受 Harness 约束，superPower 自动触发 | 代码 + 文档 | 是（验收） |
| archive | 踩坑写 error-log、经验写 lessons-learned、更新 exec-plans | 项目记忆 | 否 |

**Step 3 superPower 自动触发条件表（正常自动、异常手动）**

| superPower | 触发时机 | 手动调用 |
|---|---|---|
| 上下文管理 | /opsx:apply 执行时，按需注入 Spec | `/context @file` |
| 代码生成增强 | 生成代码时，套用 templates/ 模板 | `/template @file` |
| 迭代修复 | 测试/构建失败，错误分析→修复→验证，最多 3 轮 | `/debug` |
| 经验沉淀 | /opsx:archive 时，写入 knowledge/ | `/log @file` |
| Agent 编排 | 任务范围过大，自动拆分 scope，每次改一个模块 | `/scope dir/` |
| Prompt 模板 | 遇到常见场景，匹配 prompt-library.md | `/lib @file` |

**Step 4 记忆口诀**
- 推进需求走 /opsx 主线；遇到问题查速查表；superPower 正常自动、异常手动。

---

### 4-4 项目目录结构（一行 Prompt 生成骨架）

**Step 1 在 Qoder 输入初始化 Prompt（原样复制）**

```
帮我创建一个 my-first-project 项目，并基于 harness+openspec+superpower 的理念来初始化 AI coding 全栈开发的项目目录。
```

- 模式选择：Plan 模式（推荐，AI 先出计划、用户确认后执行）。
- 执行说明：先不跑插件市场，按目录结构理念创建骨架文件。
- 预期：2 分钟内生成骨架；过程中 TypeScript 报错是预期内的（pnpm install 后消失）。

**Step 2 目标目录结构（三大理念的物理映射）**

```
my-first-project/
├── AGENTS.md                    # Harness — AI Agent 顶层指令
├── .qoder/rules/                # Harness — 编码规则
│   ├── global.md  frontend.md  backend.md
├── openspec/                    # OpenSpec — 规格驱动开发
│   ├── config.yaml
│   ├── specs/   { frontend | backend | database }
│   └── changes/ { TEMPLATE.md | archive/ }
├── docs/                        # Superpowers — 设计文档
│   ├── architecture.md
│   └── workflows.md  (8阶段开发流程)
├── plans/                       # Superpowers — 实现计划
│   └── TEMPLATE.md  (2-5分钟粒度)
└── README.md  package.json  .gitignore  .env.example
```

**Step 3 三大理念职责（三个"先行"）**

| 理念 | 解决什么问题 | 对应目录 | 类比 | 先行原则 |
|---|---|---|---|---|
| Harness | AI 怎么写代码 | AGENTS.md + .qoder/rules/ | 员工手册/交通规则 | 规则先行（先定义行为边界） |
| OpenSpec | AI 写什么代码 | openspec/specs/ + changes/ | 施工蓝图 | 规格先行（先把需求写成文件） |
| Superpowers | 做对了没有 | docs/workflows.md + plans/ + tests/ | 里程碑+质检单 | 验证先行（先写测试定义成功标准） |

**Step 4 Harness 层两条关键约束**
- AGENTS.md：顶层总纲，AI 第一个读的文件；控制在 100 行内，只放导航信息。
- .qoder/rules/ 分领域规则：global.md（camelCase/conventional commit/import 分组）、frontend.md（FC 写法/Tailwind 优先/Zustand）、backend.md（controller→service→repository/DTO 验证）。

**Step 5 Superpowers 层 8 阶段工作流**

| 阶段 | 名称 | 做什么 |
|---|---|---|
| 1 | Brainstorm | 与 AI 头脑风暴，明确问题空间 |
| 2 | Spec | 想法沉淀为 openspec 规格文档 |
| 3 | Plan | spec 拆解为 2–5 分钟粒度实现计划 |
| 4 | TDD ★ | 先写测试，定义"什么叫做完了" |
| 5 | Implement | AI 按 plan+test 生成实现代码 |
| 6 | Verify | 跑测试、Lint、类型检查 |
| 7 | Review | 人工 review AI 产出 |
| 8 | Commit | 通过后提交，关联 spec 变更 |

**Step 6 OpenSpec 层：需求的唯一真相**
- specs/ = 真相仓库：backend/spec.md → API 端点+输入输出+错误码+业务规则；AI 每次开工先读 spec。
- changes/ = 变更管理：不能直接改 specs/，先在 changes/ 写提案（改什么+为什么+影响哪些模块），通过后合并到 specs/。
- 变更四件套：proposal.md（我要干啥）→ design.md（怎么干）→ tasks.md（拆几步、咋验收）→ archive/（完成后归档）。

**Step 7 知识工程硬规则（写进 rules 的铁律）**
- YAGNI：只写当前 spec 要的，不预留"将来可能"的接口。
- DRY：第三次出现才抽象（一次照写、二次容忍、三次提炼）。
- TDD：RED → GREEN → REFACTOR，先写会失败的测试，亲眼看它失败。

---

### 4-5 初体验：/opsx 四步演示

**Step 1 /opsx:explore（探索）**
- 输入：`/opsx:explore 创建AI全栈应用 my-first-project，含用户登录（邮箱+密码）、AI多轮对话（流式输出）、知识库RAG检索；技术栈 Next.js 14 + Spring Boot 3.2`
- 结果：workspace 为空 → 结论"全新项目，需从头搭建"。

**Step 2 /opsx:propose（方案 + 人审批）**
- AI 输出方案：①创建 .harness/ + docs/ + knowledge/；②生成 3 份 API Spec + 技术 ADR；③前端 Next.js 骨架 + 后端四层架构；④V1+V2 Migration 脚本。
- AI 标注风险：流式输出需 SSE/WebSocket；RAG 需 embedding 模型选型。
- 人的审批（propose ≠ apply）：
  - 修改 1："向量存储先用 MySQL 全文搜索做 MVP，不引入额外向量数据库"
  - 修改 2："前端登录页先做最简版本，不需要第三方 OAuth"
- 确认后进入 apply。

**Step 3 /opsx:apply（执行 + superPower 自动介入）**
- 产出目录：.harness/（rules.yml + templates/）、docs/（product-specs/api-specs×3/design-docs/exec-plans）、knowledge/、frontend/（Next.js 14 骨架）、backend/（Spring Boot 四层）、AGENTS.md（27 行顶层导航）、.cursorrules、README.md。
- superPower 自动触发：上下文管理（注入 product-specs）、代码生成增强（套用 Controller 模板：统一 ApiResponse<T> + 异常处理）、Agent 编排（自动拆分：先建目录→写文档→生代码）。
- 人的介入示例：AI 问"要 Remember Me 吗？" → 答"MVP 不做"（10 秒）。
- 耗时：30 秒（传统方式 1–2 天）。

**Step 4 /opsx:archive（归档沉淀）**
- 自动写入：lessons-learned.md（"先写 product-specs 再写 api-specs 效率最高；Spring AI 版本建议锁定 1.0.0-M6"）、conventions.md（"全局响应 ApiResponse<T> / 全局异常 @RestControllerAdvice / 禁止 Controller 直接调 ChatModel"）、exec-plans/sprint-1.md 标记完成。

**Step 5 产出物清单（4 条指令的结果）**
1. 全栈项目骨架：Next.js + Spring Boot + Migration 脚本，完整可运行。
2. openSpec 文档：product-specs/、api-specs/×3、design-docs/ ADR。
3. Harness 护栏：AGENTS.md、.cursorrules、.harness/templates/。
4. 项目记忆：conventions.md、lessons-learned.md。

**Step 6 apply / archive 步骤详解（执行期实际发生了什么）**

*apply（Step 3）——把确认过的方案真正落地，四步中唯一产出代码的阶段，实际做 5 件事：*

| 动作 | 产出 | 说明 |
|---|---|---|
| 建目录结构 | .harness/、docs/、knowledge/、frontend/、backend/ | 先搭骨架再填内容 |
| 生成 Spec 文档 | 3 份 api-specs/（auth/chat/knowledge）、design-docs/tech-selection-adr.md、exec-plans/sprint-1.md | 与 propose 方案一一对应 |
| 生成代码骨架 | frontend/（Next.js 14）、backend/（Spring Boot 四层 controller→service→repository） | 可运行但不含业务细节 |
| 写护栏文件 | AGENTS.md（27 行顶层导航）、.cursorrules（命名+禁止 API 约束） | AI 以后每次干活先读 |
| 初始化项目记忆 | knowledge/ 的 conventions/lessons-learned 占位 | 供 archive 阶段填充 |

superPower 自动介入的 3 个典型场景（无需手动调用）：

| superPower | 介入点 | 效果 |
|---|---|---|
| 上下文管理 | 写代码时自动注入 product-specs | 代码始终对齐产品规范 |
| 代码生成增强 | 写 Controller 时自动套用 templates/ 模板 | 统一 ApiResponse<T> + 全局异常处理 |
| Agent 编排 | 自动拆"先建目录→写文档→生代码"并限 scope | 每步只改一个模块，不一把梭 |

人的介入点：执行中 AI 抛决策问题（如"要 Remember Me 吗？"→ 答"MVP 不做"，约 10 秒）。结果：约 30 秒完成（传统 1–2 天）。

*archive（Step 4）——经验沉淀，不产生新功能，但保证下次迭代质量，实际写 3 个文件：*

| 文件 | 写什么 | 为什么 |
|---|---|---|
| knowledge/lessons-learned.md | 教训（"先写 product-specs 再写 api-specs 效率最高；Spring AI 版本建议锁定 1.0.0-M6"） | AI 下次遇到类似场景直接跳过坑 |
| knowledge/conventions.md | 约定（ApiResponse<T> / @RestControllerAdvice / 禁止 Controller 直接调 ChatModel） | 形成项目规范，AI 自动遵守 |
| exec-plans/sprint-1.md | 标记"项目初始化"已完成 | 迭代计划状态可追踪 |

**Step 7 完整执行案例（my-first-project）**

*场景*：AI 对话应用，含用户登录（邮箱+密码）、AI 多轮对话（流式）、知识库 RAG 检索。技术栈 Next.js 14 + Spring Boot 3.2。

- Step 1 explore：输入 `/opsx:explore 创建AI全栈应用 my-first-project，含用户登录（邮箱+密码）、AI多轮对话（流式输出）、知识库RAG检索。技术栈：Next.js 14 + Spring Boot 3.2` → AI 回复"workspace 为空，全新项目，需从头搭建"。
- Step 2 propose：AI 输出 4 步方案（建目录/3 份 API Spec+ADR/前后端骨架/Migration 脚本）并标注风险（流式需 SSE/WebSocket；RAG 需 embedding 选型）→ 人修改 2 处（"向量存储先用 MySQL 全文搜索做 MVP"；"登录页先做最简版本，不要第三方 OAuth"）→ 确认进入 apply。
- Step 3 apply（30 秒）：顺序 = 建目录 → 写 auth/chat/knowledge 三份 API Spec → 写 tech-selection-adr.md → 生成 frontend 骨架 → 生成 backend 四层骨架 → 写 AGENTS.md（27 行）+ .cursorrules。superPower 介入：写 auth Controller 自动套统一模板；自动限 scope。人机对话："要 Remember Me 吗？"→"MVP 不做"。
- Step 4 archive：AI 自动写入 lessons-learned.md（含 Spring AI 锁 1.0.0-M6）、conventions.md（三条约定）、exec-plans/sprint-1.md 标记完成。
- Step 5 验收核对：①前端+后端+Migration 可运行；②3 份 API Spec+ADR 齐全；③AGENTS.md+.cursorrules+templates 到位；④项目记忆已写入。

---

## Part 2 · Spec 工程（4-6 ~ 4-7）

### 4-6 Spec 编写实战（以 Todo 模块为例）

**Step 1 破除误区**
- Spec = 你给 AI 的"编程语言"（人写 Spec → AI 执行，替代人写 Java/Python → 机器执行）。
- 人是主编不是作者：AI 负责吐字数铺细节，你负责审、删、补、定调。

**Step 2 Spec 打磨核心循环（2–3 轮收敛）**

```
① 人给 Prompt（框架要求：边界/场景/字段/验收）
   ↓
② AI 生成初稿（几秒出 800 字，但异常路径仅 30–40%）
   ↓
③ 人审三把放大镜（边界审 / 场景审 / 数据结构审）
   ↓
④ 人给精确指令（具体 + 可执行 + 有理由，如改 7–8 个修改点）
   ↓
⑤ AI 输出 v2（边界完整、场景全、字段带约束）
```

**Step 3 三把放大镜的审查点**
1. 边界审："不包含"够清楚吗？标签分类做不做？子任务嵌套做不做？批量删除做不做？
2. 场景审：WHEN/THEN 异常漏多少？标题为空/超长？删别人 Todo（越权）？重复点完成（幂等）？
3. 数据结构审：title 长度限制？priority 枚举值？dueDate 应有但漏了？（抽象 string = 后端要崩）

**Step 4 喂回 AI 的指令三要求**
- 具体：不说"加一些异常"，说"添加更新不存在的 ID 时返回 404"。
- 可执行：AI 能直接照做。
- 有理由：防越权/性能要求。

**Step 5 Todo Spec 初始 Prompt（可复制）**

```
为 Todo 待办清单模块生成 Spec 文档，结构必须包含：
1. 模块边界（包含/不包含）
2. 核心场景（用 WHEN/THEN/AND 格式枚举正常+异常路径）
3. 数据结构（请求/响应字段，含类型与约束）
4. 验收标准（可勾选）
附加要求：Todo 只支持单用户私有，不做协作；字段命名前后端统一用驼峰法；异常路径必须覆盖。
```

**Step 6 Todo 模块关键决策点（AI 生成 + 人审改后的最终态）**
- 模块边界（不包含）：多用户协作、认证授权（不引入 Spring Security）、数据库持久化（仅内存）、分页/排序/搜索/过滤、软删除/审计字段、批量操作、附件/子任务/提醒、WebSocket/SSE、国际化。
- 字段约束：title 为 string，trim 后长度 [10,200]（UTF-16 计数）；tags 最多 10 个、单个 [1,20]、后端保序去重；dueDate 为 YYYY-MM-DD；priority 严格枚举 "LOW"|"MEDIUM"|"HIGH" 默认 "MEDIUM"；id 由后端生成 UUID v4 不可变。
- 核心场景示例：GIVEN 存在 id='X' 且 completed='false'；WHEN PUT /api/todos/X body={"completed":true}；THEN 200 且 completed=true 且 title 保持原值。
- 排序规则：按 priority 降序（HIGH>MEDIUM>LOW）→ dueDate 升序（null 最后）→ createdAt 升序。
- 人审补充的遗漏：更新不存在的 ID 返回 404；title 不足 10 字符返回错误。

**Step 7 五条打磨心法**
1. 一个模块一份 Spec（别混 Todo 和 User 注册）。
2. 异常清单必带（空值/超长/越权/并发/重复/超时至少过一遍）。
3. 数字必给单位/范围（"1 秒内响应"而非"快"）。
4. 枚举必列全（status 有几个值列几个，别留"等等"）。
5. 不确定标 [待确认]（AI 拍脑袋是大忌）。

**Step 8 人不可替代的三件事**
1. 设定边界（什么不做、什么是 MVP = 商业判断）。
2. 找漏补缺（异常路径、安全场景 = 工程经验）。
3. 拍板定调（多方案选哪个、何时定稿 = 决策权）。

---

### 4-7 Spec Review 机制（自动化文档审查）

**Step 1 理解为什么必须让 AI 再审一遍**
- 事故案例：Spec 人工审三遍 → AI 生成代码上线 → 用户改 URL 里的 id 删掉别人数据（越权漏洞）。
- 根因：写和审是同一人，思维盲区一致。
- AI 审的价值：①不疲劳（第一道审稿人）；②知识面广（读过 OWASP 与海量案例）。

**Step 2 万能 Review Prompt（原样复制）**

```
请审查这份 Spec，重点找出以下问题：
1. 安全风险——越权、信息泄露、未做限流的接口
2. 数据问题——字段类型/长度/索引是否合理
3. 逻辑漏洞——"没说清楚"的灰色地带
4. 异常路径——空值/超长/并发/重复点击是否覆盖
5. 验收标准——ACCEPTANCE 是否可量化、可测试
每条问题给出：在 Spec 哪一行 + 什么问题 + 怎么改
```

**Step 3 AI 审查输出的分级（S/A/B）示例**
- S（必须改）：越权（删除只校验 Todo 存在未校验 ownerId → WHERE userId=current + 加 ACCEPTANCE）；无限流条款；priority 枚举表与请求体类型不一致。
- A（强烈建议）：VARCHAR(255) 下中文最多 33 字；"重复完成"未定义时间窗口（→5 秒内重复算同一次）。
- B（可选打磨）：ACCEPTANCE 未量化（→"创建后 200ms 内出现"）。

**Step 4 三档处理方式**
1. 档①直接改 Spec：无争议项（越权/字段长度/命名）立刻改，再核对。
2. 档②补到 ADR：Spec 里说不清楚（软删何时硬删）→ docs/adr/ 单独写决策记录。
3. 档③放 backlog：这版不修但标启动条件（"DAU 超 XX 时启动"）。

**Step 5 修复成本账（Review 阶段是上线的 1/1000 代价）**

| 阶段 | 修复动作 | 相对成本 |
|---|---|---|
| Spec Review | 改一行 Spec，30 秒 | 1× |
| 编码阶段 | 改代码+改测试，30 分钟 | 60× |
| 联调阶段 | 前后端+测试+部署，2 小时 | 240× |
| 上线后发现 | 修复+回滚+复盘，1–3 天 | 1000× |

---

## Part 3 · 项目搭建（4-8 ~ 4-12）

### 4-8 AIWorkSpace：git submodule 多仓库管理

**Step 1 痛点确认**
- 联调冲突：前端引用 API 文档 v1，后端已更新 v2——两边都没错，缺一个"看到全部"的地方。
- AI 上下文是仓库级的：前端仓看不到后端 Spec，后端仓看不到前端字段。

**Step 2 git submodule 本质**
- 总仓库只挂指针（指向子仓库特定 commit），不复制代码；子仓库独立维护、独立提交、独立打版本。

**Step 3 升级命令（在 Qoder 输入）**

```
把当前项目升级为 AIWorkSpace。在项目根目录下，把已有的 frontend 仓库和 backend 仓库作为 git submodule 引入。引入后更新 AGENTS.md，补上对 frontend/ 和 backend/ 的说明。
```

- 前置条件：准备 3 个远程仓库 URL（父项目 + backend + frontend）；Qoder 会在本地 git init → commit → push → `git submodule add <url>`。
- 实际执行的命令：
```bash
git submodule add <前端仓库URL> frontend
git submodule add <后端仓库URL> backend
# 更新 AGENTS.md，补 frontend/ 和 backend/ 说明
```

**Step 4 处理遗留事项**
1. 删除备份目录：若 `backend.bak` 等备份目录与仓库一致，手动删除。
2. HTTPS 推送失败：Qoder 会转 SSH 方式，可忽略。
3. 空目录清理：src/ 空目录是 OpenSpec init 痕迹（git 不跟踪空目录），`rmdir src` 清理。

**Step 5 验证 + 新人拉取**
- 验证：出现 `.gitmodules`，frontend/ 和 backend/ 指向对应远程仓库。
- 新人一条命令拉全套：
```bash
git clone <my-first-project仓库> --recursive
# 忘加 --recursive 补救：
git submodule update --init --recursive
```

**Step 6 AIWorkSpace 四大核心价值**
1. AI 能看到全部（总控仓库层做跨前后端一致性检查：字段对齐、接口契约）。
2. 专注不被干扰（进子仓库开发时 AI 只看当前模块）。
3. 独立打版本（前端发版不影响后端，回滚不互相阻塞）。
4. 新人开箱即用（一行命令拉全套）。

---

### 4-9 如何利用 AI 做前后端架构选型

**Step 1 为什么 AI 时代仍需选型**
- 训练数据量差异：主流框架生成质量远高于小众框架（React 相关仓库是 Vue 的 3–4 倍、Angular 的 5 倍以上；Spring Boot 是 Express 的 2–3 倍）。
- 框架约定强度：约定越强 AI 越不易错（Next.js App Router 文件系统路由 vs Vue Router 分离式路由）。
- 选型成本：做到一半换框架 = 重做（选型 1 小时 vs 重做 1 周）。

**Step 2 AI 辅助选型 4 步法**
1. 写结构化 Prompt 对比方案（必须含你的约束条件：团队技能/部署环境）。
2. 让 2–3 个不同 AI 模型各自分析，看结论是否一致。
3. 人做最终决策，写成 ADR（Architecture Decision Record）。
4. 把 ADR 存入规范，后续 AI 开发参考。

**Step 3 选型 Prompt 模板（可复制，关键是"我的情况"约束段）**

```
我要为 my-first-project 做前后端技术选型。
我的情况：
- 团队3人，全员熟悉React，Vue只用过2.x
- 项目周期8周，必须上线MVP
- AI辅助开发为主，要选AI生成质量最高的方案
请对比 Next.js 14 vs Nuxt 3（前端）、Spring Boot 3 vs NestJS（后端），
从AI生成质量/约定强度/生态/部署/学习成本五维度打分，给明确结论。
```

- 不给约束 → AI 回答"两个都好各有优劣"（正确但无用）；给了约束 → 可执行结论。
- 技巧：不要一次采纳，追问补充团队情况（如"如果团队 Spring Boot 经验充足且是线上长期迭代项目，怎么选？"），结论可能完全翻转。

**Step 4 短期 vs 长期选型原则**
- 短期（MVP）：看开发速度（Next.js 全栈得分高）。
- 长期（3 年后）：看工程性（Spring Boot 分布式事务、复杂工作流）。

**Step 5 选型结果写入规范文档（要素清单）**
- 场景前提（生命周期/团队基线/部署形态/协作模式）。
- 技术选型约束（决策维度优先级：可持续迭代能力 > 生态成熟度 > 框架约定强度 > 团队学习成本 > AI 生成质量 > 部署敏捷度）。
- 锁定栈 + 禁止动作 + 何时重新评估。

---

### 4-10 前端工程初始化：框架选型落地

**Step 1 前端框架 AI Coding 适配度横评**

| 框架 | AI 训练数据量 | 约定强度 | AI 生成质量 | 结论 |
|---|---|---|---|---|
| React + Next.js | 最多（百万级） | 文件路由=强 | 最高 | ✅ 第一选择 |
| Vue + Nuxt 3 | 中等（React 的 1/3） | 路由分离=中等 | 良好 | 偶尔混淆写法 |
| Angular | 较少 | Module=最强 | 一般 | 版本 2–17 混杂 |
| Svelte/Solid | 极少 | — | 差 | 常出伪代码 |

**Step 2 UI 组件库选型：可控性 > 开箱即用**
- shadcn/ui：代码在本地（copy 进项目），AI 能直接改源码，Tailwind 原子类不冲突 → 唯一推荐。
- Ant Design：npm 黑箱，改内部样式需 override 一堆 CSS → AI Coding 不友好。
- MUI：sx prop 复杂，AI 易写出不一致配置 → 调试成本高。

**Step 3 选型落地（锁定栈）**
- `Next.js 14 + App Router + TypeScript + Tailwind CSS + shadcn/ui + Zustand`。

**Step 4 面试口径（可直接复用）**
- 为什么选 React 不选 Vue？→ AI 训练数据量大 3–4 倍，复杂场景生成质量更高。
- 为什么用 Next.js？→ App Router 文件系统路由对 AI 天然友好，无需维护路由配置。
- UI 为什么用 shadcn？→ 代码在本地完全可控，AI 能直接改源码，不是 npm 黑箱。

---

### 4-11 后端 Spring AI 工程初始化

**Step 1 后端框架 AI Coding 适配度横评**

| 框架 | 训练数据量 | 分层约束 | AI 生成质量 | 大模型集成 |
|---|---|---|---|---|
| Spring Boot (Java) | 最多（碾压级） | Controller/Service/Repo 强制 | 最高 | Spring AI 原生 |
| NestJS (TS) | 中等 | 装饰器+DI | 良好 | 需手动封装 HTTP |
| Express (JS) | 多但散 | 几乎无约束 | 风格不统一 | 完全手动 |
| FastAPI (Python) | 中等 | 路由+Pydantic | 良好 | 适合 ML 不适合 Web |

**Step 2 后端选型三大特殊关注点（前端没有的）**
1. **AI 调用封装**：5–6 个模块都调大模型，各自直接调 = 改一处改 6 处 → 封装统一 `AiService`（限流/计费/日志一处搞定）。
2. **错误处理边界**：AI 调用会超时（30s+）、限流（RPM/TPM）、返回格式异常 → 必须 `@RestControllerAdvice` 全局捕获 + 统一错误码 + 降级策略。
3. **配置管理**：dev 用 DeepSeek（便宜）、prod 用 GPT-4o（强）；API Key 禁止硬编码 → 多环境 Profile 隔离，模型切换改配置不改代码。

**Step 3 选型落地（锁定栈）**
- `Spring Boot 3 + Spring AI + Java 17（或 21）+ Maven`。

**Step 4 面试口径**
- 为什么选 Spring Boot？→ 训练数据碾压 + 企业级成熟 + Spring AI 原生集成大模型。
- AI 调用怎么封装？→ 统一 AiService，单点修改全模块生效。
- 异常处理怎么做？→ @RestControllerAdvice 全局捕获 + 统一错误码 + AI 超时降级。
- 配置管理怎么做？→ 多环境 Profile 隔离 + Secret 注入 + 模型切换改配置不改代码。

---

### 4-12 Vibe coding 如何进行数据库选型与设计

**Step 1 数据库选型：选 MySQL**
- 理由：AI 训练数据量最大（SQL 生成质量最高）、生态最成熟、团队招人方便、Spring Data JPA 支持最成熟。
- 核心理念：Vibe Coding 不是探索新技术，而是用 AI 把成熟技术用到极致。

**Step 2 迁移工具：选 Flyway（弃 Liquibase）**
- Liquibase：XML/YAML 多一层抽象，AI 生成的 XML 人难懂、排查困难。
- Flyway：纯 SQL 文件，AI 生成 DDL 直接当 Migration，人可读、可复制 SQL 直接执行。
- 铁律：人看得懂比什么都重要（AI 生成的最终要人验收）。

**Step 3 Schema 设计底线（表结构规则，必须一开始写死）**
1. 字段约束不能太松（VARCHAR(255) 无限制 = 埋雷，AI 会生成无校验代码 → 脏数据）。
2. 主键策略必须统一（混用自增/雪花/UUID → 联表类型不匹配，JOIN 查不出数据）。
3. 公共字段必须强制添加（非"讨论后决定"，是"没得商量"）。
- 代价对比：前后端改一行重新部署即可；表结构改一次 = 停服迁移百万行数据，代价差 10 倍以上。

**Step 4 主键策略：双 ID 模式（推荐）**
- 内部主键：`BIGINT AUTO_INCREMENT`，永远不出库、不进 URL、不进 API 响应。
- 对外暴露 ID：单独一列 `public_id`（CHAR(26) ULID 或 BIGINT 雪花），UNIQUE 索引；所有 REST 路径、前端跳转、日志关联用它。
- 理由：自增在 InnoDB 聚簇索引几乎顺序写（性能秒杀 UUID）；同时避免暴露 /conversations/123 防爬虫遍历、防用户改 URL 越权、防竞对算 DAU。
- 一句话：禁止 UUID 做主键，禁止在 URL 暴露内部自增 ID。

**Step 5 公共字段（BaseEntity）**

```sql
id          BIGINT AUTO_INCREMENT PRIMARY KEY
public_id   CHAR(26) NOT NULL UNIQUE      -- ULID/雪花，对外用
created_at  DATETIME(3) NOT NULL          -- 毫秒精度（用 DATETIME 不用 TIMESTAMP：TIMESTAMP 上限 2038 且有时区转换）
updated_at  DATETIME(3) NOT NULL ON UPDATE
created_by  BIGINT NULL                   -- user.id，系统操作时为 NULL
updated_by  BIGINT NULL
```

**Step 6 索引策略**
- 反直觉原则：索引不是越多越好（拖慢写入）；不要拍脑袋提前加索引，等查询变慢再加。
- 最左前缀法则：等值过滤列在前，范围/排序列在后。
- AI 对话应用预设索引：conversations 建 `INDEX (user_id, deleted_at, updated_at DESC)`；messages 建 `INDEX (conversation_id, created_at)`（content 不加索引）；ai_usage_log 建 `INDEX (user_id, created_at)` 与 `INDEX (created_at)`。

**Step 7 Qoder 三轮对话法（数据库设计实战）**
- 第一轮：让 AI 定表设计规范（主键策略/公共字段/字段约束原则/索引策略）——先定规范，不要直接生成 SQL。
- 第二轮：先要 ER 图确认实体关系（一对一/一对多/多对多）→ 再输出建表 SQL（users/conversations/messages）。技巧：关系错了，字段全错。
- 第三轮：Flyway 规范 + `V1__init_schema.sql` + 规范追加到 backend.md。

**Step 8 Migration 铁律**
1. 已发布的文件绝不改内容。
2. 每文件单一职责（新建表/加字段/改索引分开）。
3. 版本号只增不减。

**Step 9 面试口径**
- 为什么选 MySQL 不选 PG？→ AI 训练数据量大 + JPA 支持成熟 + 团队经验匹配。
- 主键策略？→ BIGINT 自增/雪花 ID，全项目统一。
- 索引怎么加？→ 跟着查询走，WHERE 用什么字段就索引什么。
- Migration 怎么管理？→ Flyway + 版本递增 + 单一职责 + 已发布不可改。
- 公共字段？→ id / created_at / updated_at / deleted——不讨论，必须有。

---

## Part 4 · 复盘与总结（4-13 ~ 4-15）

### 4-13 阶段性 Review：项目整体架构

**Step 1 仓库拓扑：一父两子**

| 角色 | 路径 | 内容 |
|---|---|---|
| 工作区父仓 | `.` | Harness + Spec + submodule 指针（不放业务代码） |
| Backend submodule | `backend/` | Spring Boot 3.3.5，端口 8080 |
| Frontend submodule | `frontend/` | Next.js 16 + React 19，端口 3000 |

- 父仓核心原则：不放业务代码，只追踪两个子仓的 commit SHA；业务和治理解耦，各自独立演进。

**Step 2 技术选型（长期项目锁定栈）**

| 层 | 选型 | 决定性理由 |
|---|---|---|
| 前端 | React 19 + Next.js 16 + TS | SSR + Vercel 原生 + 薄 BFF |
| 后端 | Spring Boot 3.3.x + JDK 17 | 分层约定强 + 生态完备 + 团队经验 |
| 接口契约 | OpenAPI / springdoc-openapi | 跨语言类型对齐 |
| 数据库 | MySQL（注：课件 4-9 规范示例为 Postgres，4-12/4-13 为 MySQL，此处以 4-12/4-13 为准） | 生态成熟 |

- 禁止动作（除非走 propose 推翻）：后端切 Next.js API Routes / Server Actions；前端切 Vue/Nuxt/Svelte；仅以"AI 生成质量更高"为由推翻已锁定栈。

**Step 3 前后端分工：前厅 vs 后厨**
- 动线：用户浏览器 → Next.js（薄 BFF/SSR） → Spring Boot（业务后端） → DB。
- BFF 允许：SSR 预取、多接口聚合、缓存读。
- BFF 禁止：在 `app/api/**/route.ts` 写业务、Server Actions 搞鉴权事务。
- 验收硬约束：`frontend/lib/backend.ts` 首行必须 `import 'server-only'`。

**Step 4 三层治理**

| 层 | 目录 | 管什么 | 类比 |
|---|---|---|---|
| Harness | .qoder/ | AI 不能干啥（硬规则、斜杠命令、子 agent） | 缰绳 |
| OpenSpec | openspec/ | AI 按啥顺序干（先写 Spec，人类签字，AI 再动手） | 流程 |
| Superpowers | .qoder/skills/ | AI 用啥姿势干（可组合、自动触发的方法论技能） | 技能树 |

- Harness 三条底线（always on）：YAGNI、DRY（第三次出现才抽象）、TDD（RED→GREEN→REFACTOR）。

**Step 5 openspec/ 目录结构**

```
openspec/
├── project.md                     ← 项目级 Spec（AI 进仓库第一份要读的）
├── specs/http-server/spec.md      ← 已固化能力
├── changes/<change-name>/         ← 进行中变更
│   ├── proposal.md                我要干啥
│   ├── design.md                  怎么干
│   └── tasks.md                   拆几步、咋验收
└── changes/archive/               ← 已归档变更（按日期命名）
```

**Step 6 .qoder/ 目录结构（AI 的中央厨房）**

```
.qoder/
├── rules/       ← always on 硬规则（coding-conventions.md、spec-driven-workflow.md）
├── commands/    ← 斜杠命令（propose.md、apply.md、archive.md）
├── skills/      ← Superpowers 技能（brainstorming/、test-driven-development/…）
└── agents/      ← 自定义子 agent
```

---

### 4-14 总结实战中的关键经验与常见问题

**Step 1 五条踩坑经验**

| # | 经验 | 核心结论 | 记忆 |
|---|---|---|---|
| 1 | Spec 投入产出比 | 30 分钟好 Spec 省 3 小时反复改代码 | 越赶时间越要写 Spec |
| 2 | AGENTS.md 短而准 | 100 行内，只放导航信息 | 太长 = AI 不看 |
| 3 | 约束要写数字 | "30 行以内"有效，"写规范"无效 | 数字胜过形容词 |
| 4 | 每次只做一件事 | 范围明确 = 可控 | 小步快跑，每步确认 |
| 5 | AI 犯错先查输入 | 90% 是输入问题（Spec 清楚吗/约束覆盖吗/上下文够吗） | 提升输入 = 提升输出 |

**Step 2 BFF（Backend For Frontend）决策点**
- 定位：浏览器 ↔ BFF ↔ Spring Boot 的中间层。
- 解决的问题：接口聚合（5 接口 → 1 接口）、字段裁剪（50 字段 → 3 字段，体积瘦身 80%）、SSR + SEO、后端字段变更兜底。
- 边界：只是助理不是老板——写业务、动数据库绝对不允许越界。

**Step 3 六个高频 FAQ**

| 问题 | 一句话解法 |
|---|---|
| AI 跳步直接动 src/ | spec-driven-workflow 拦截，回退到 brainstorming 重来 |
| 子仓改完忘了升级父仓指针 | 子仓 push 后回父仓 `git add backend && commit "bump"` |
| AI 跳过 RED 直接写实现 | rules 锁死 TDD，先写失败测试，亲眼看它失败再 GREEN |
| 前后端 API 类型对不上 | 引同一份 OpenAPI schema，`openapi-typescript` 自动生成 TS 类型 |
| AI 想在 Next.js 写业务 | BFF 边界禁止 route.ts/Server Actions 写业务，锁 `import 'server-only'` |
| OpenSpec change 名以数字开头报错 | change name 必须字母开头，日期前缀放在字母后面 |

---

### 4-15 本章小结与 SDD+Harness 流程复盘

**Step 1 核心公式（乘法关系，任一为零则总质量为零）**

```
AI 全栈开发质量 = Spec 精确度 × 护栏完备度 × 人工审核密度
```

| 因子 | 决定什么 | 为 0 时后果 | 对应节 |
|---|---|---|---|
| Spec 精确度 | AI 知道做什么 | AI 自由发挥 | 4-2、4-6 |
| 护栏完备度 | AI 不会做什么 | AI 跑偏 | 4-3、4-7、4-10 |
| 人工审核密度 | 错误多快被发现 | 无人把关 | 4-7、4-14 |

**Step 2 学完本章的交付物清单**
- ✅ 完整项目骨架（Harness + OpenSpec + Superpowers 目录结构）
- ✅ 结构化文档体系（AGENTS.md + rules + specs 目录）
- ✅ 高质量 Spec（至少一份含边界/场景/数据结构的 Spec）
- ✅ 前后端初始化（前端 Next.js + 后端 Spring Boot）
- ✅ DB Migration 就绪（Flyway + V1__init_schema.sql）
- ✅ 技术选型 ADR + 规范文件

**Step 3 课后自测任务（复刻验证）**
- 用本章方法从零创建"代办事项 APP"，验收标准：
  1. 一份完整的 harness 目录结构；
  2. 至少写一个功能 Spec；
  3. 前后端初始化完毕；
  4. 将选型结论记录到规范文件；
  5. 记录过程中犯的错误。

---

## 附录 A · 复刻速查（命令与 Prompt 汇总）

**初始化项目骨架**
```
帮我创建一个 my-first-project 项目，并基于 harness+openspec+superpower 的理念来初始化 AI coding 全栈开发的项目目录。
```

**/opsx 四步主线**
```
/opsx:explore <需求描述>
/opsx:propose
/opsx:apply
/opsx:archive
```

**升级为 AIWorkSpace**
```
把当前项目升级为 AIWorkSpace。在项目根目录下，把已有的 frontend 仓库和 backend 仓库作为 git submodule 引入。引入后更新 AGENTS.md，补上对 frontend/ 和 backend/ 的说明。
```

**技术选型（含约束段）**
```
我要为 my-first-project 做前后端技术选型。
我的情况：<团队/周期/部署/AI辅助开发为主>
请对比 <A vs B>，从AI生成质量/约定强度/生态/部署/学习成本五维度打分，给明确结论。
```

**Spec 编写初始 Prompt**
```
为 <模块> 生成 Spec 文档，结构必须包含：1.模块边界 2.核心场景(WHEN/THEN/AND) 3.数据结构(类型与约束) 4.验收标准(可勾选)。
```

**Spec Review 万能 Prompt**
```
请审查这份 Spec，重点找出：1.安全风险 2.数据问题 3.逻辑漏洞 4.异常路径 5.验收标准。每条问题给出：在哪一行+什么问题+怎么改。
```

**新人拉取全套环境**
```bash
git clone <父仓URL> --recursive
git submodule update --init --recursive   # 忘加 --recursive 时补救
```

**Flyway Migration 三铁律**
1. 已发布文件绝不改内容 2. 每文件单一职责 3. 版本号只增不减
