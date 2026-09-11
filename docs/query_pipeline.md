# Query 处理流程：拆分 / 改写 / 落域（orchestrator_plan vs juagent）

> 对比两个电视语音助手编排器工程对用户 query 的处理链路，重点三块：**拆分(split)**、**改写(rewrite)**、**落域(domain routing)**，并标注 **LLM 调用点**。
>
> - `orchestrator_plan`：精简独立版，只做编排，单意图的"工具+参数"下沉给外部 **compare 服务(8084)**。
> - `juagent`：完整版，编排 + 域执行一体，`domains/` 进程内直调。

---

## 1. 整体定位差异

| | orchestrator_plan | juagent |
|---|---|---|
| 角色 | 只做**编排**（拆分/改写/落域/状态机），单意图工具+参数交外部 compare(8084) | 编排 + **域执行一体**，`domains/` 进程内直调，零网络 |
| 域数量 | 7 域（vod/children/education/music/audio/fan_agent_qa/device） | **8 域**（多一个 **sports 体育**），共 45 工具，schema JSON 数据驱动 |
| 落域实现 | 亮屏交给 compare，本工程只管息屏关键词规则 | 亮屏也自己落域：`domains/core/pipeline.py` 规则 + LLM 分类器兜底 |

---

## 2. 拆分 (split) —— 两者一致

核心在 `planner.plan_query()`，按**有无历史**分两条路：

- **无历史 → 纯确定性规则**（`split_into_subqueries`）：
  - 有连接词只在连接词处切：`然后|接着|顺便|顺带|同时|以及|也|再|又`
  - 无连接词用逗号兜底切
  - 守卫：程度副词（"音量再大一点"的"再"不切）、单意图复合句（台词/引号反查/自述句即使有逗号也不切）
  - 特殊：天气+媒资意图拆成 [查天气(并行), 搜片(依赖)]
- **有历史 → 唯一走 LLM**（`_PLAN_PROMPT`）输出结构化 JSON 计划，约束"每 query 单域（媒资/设备/问答）、不丢目标、只有实体依赖上一步才 depends_on_previous"

**串/并行**：`depends=False` 并行下发，`depends=True` 串行（`next_batch` 全发 / `_next_run` 依赖一次一个）。依赖由 LLM 标（`_DEPENDENCY_PROMPT`），再用 `demote_fake_dependencies` 把误标的并列意图翻回并行。

---

## 3. 改写 (rewrite) —— 两者一致，"规则优先，LLM 兜底"

四类改写函数，按状态机顺序调用：

1. **跨轮记忆改写** `rewrite_with_memory`：先确定性——`resolve_answer_role`（"该导演"→上轮答案人名）、序号消解（"第N部"→《片名N》）；残留指代才上 LLM（`_MEMORY_REWRITE_PROMPT`）；已含《实体》则跳过 LLM
2. **多轮 dialog 改写** `rewrite_with_dialogs`："同上/再来一遍"复用历史 query（纯规则）
3. **条件累积继承** `inherit_conditions`：同设备跨轮把上轮检索条件（category/tag/area/language/grade）merge 进本轮，LLM 合并 + 确定性兜底防丢词
4. **依赖意图改写** `rewrite_dependent`（续跑时）：序号→片名(规则) → 天气上下文 → 开放排序意图("评分最高/最新"→结果第一条片名) → 残留指代 LLM(`_REWRITE_PROMPT`)

核心思想：**LLM 只在若干"窄缝"出现**，每个前面都有规则闸门，LLM 失败回退规则。

---

## 4. 落域 (domain routing) —— 主要差异点

**orchestrator_plan**（`tv_domain_for`）：
- **息屏(tvMode=6)**：纯关键词正则分诊，优先级 设备控制 → 影视片段 → 儿歌推荐 → 疑问句 → 教育 → 歌手/歌曲 → 影视/有声剧 → 兜底 audio
- **亮屏(tvMode=0)**：`return ""` **不干预**，落域交外部 compare（注释写明：亮屏强制 domain 曾导致大幅掉分）
- 具体工具名由 compare 返回，本工程只传 domain 分诊 + 少量 `force_tool`

**juagent**（两级）：
- **息屏**：同上 `tv_domain_for` 纯规则
- **亮屏/自动**：自己在 `domains/core/pipeline.py:route_domains` 落域——**先约 560 行高精度词法锚点规则**（唱/K歌→音乐；播控整句→设备；年级/教材→教育；"X的导演是谁"→泛知识；动画→少儿…），全不命中才落 **LLM 分类器**（`DOMAIN_SYSTEM` + few-shot）输出域名 JSON 数组
- **域内选工具**：`select_tool` 同样"词法高置信 + LLM 兜底"（`TOOL_SYSTEM` 写死易混工具边界如 vod_search vs vod_search_all），参数由 `PARAM_SYSTEM` 抽取

juagent `DOMAIN_GUIDE` 几条硬规则：动漫/动画一律归少儿；出现"听/收听/有声"即使题材像影视也归有声；询问事实/属性/人物即使提到电影演员也优先归泛知识（"无间道导演是谁"）。

---

## 5. 流程图（🔴 = 调用 LLM）

以 juagent 完整版为主线，orchestrator_plan 差异见 L8–L10 备注。

```mermaid
flowchart TD
    A["POST /slowAgent/poc_id<br/>query + memory + toolHistory + tvMode"] --> B{有历史 memory?}

    %% ---------- 改写阶段（首轮 _first）----------
    B -->|有| R1["rewrite_with_memory<br/>规则:该导演→人名/第N部→片名"]
    R1 --> R1Q{残留指代?且无《实体》}
    R1Q -->|是| L1(["🔴 LLM: _MEMORY_REWRITE_PROMPT"])
    R1Q -->|否/已自包含| R2
    L1 --> R2["rewrite_with_dialogs<br/>规则:同上/再来一遍"]
    R2 --> R3["inherit_conditions 条件累积继承"]
    R3 --> R3Q{需合并上轮条件?}
    R3Q -->|是| L2(["🔴 LLM: _CONDITION_INHERIT_PROMPT"])
    R3Q -->|否| P
    L2 --> P

    %% ---------- 拆分阶段 ----------
    B -->|无| S1["split_into_subqueries<br/>规则:连接词/逗号切 + 守卫"]
    S1 --> S1Q{切出多段?}
    S1Q -->|多段| L3(["🔴 LLM: _DEPENDENCY_PROMPT 判依赖"])
    S1Q -->|单段| P
    L3 --> P

    P{plan_query 汇总 有历史?}
    P -->|有历史| L4(["🔴 LLM: _PLAN_PROMPT 拆结构化计划"])
    P -->|无历史| PLAN["intents[] + depends"]
    L4 --> PLAN

    PLAN --> DM["demote_fake_dependencies<br/>规则:误标依赖翻回并行"]
    DM --> BATCH{next_batch 串/并行调度}

    %% ---------- 执行 + 落域 ----------
    BATCH --> DEP{依赖意图?(续跑 _continue)}
    DEP -->|是| RD["rewrite_dependent<br/>规则:第N部→片名/已含实体跳过"]
    RD --> RDQ{开放排序/残留指代/天气?}
    RDQ -->|开放排序| L5(["🔴 LLM: _OPEN_ORDINAL_PROMPT"])
    RDQ -->|残留指代| L6(["🔴 LLM: _REWRITE_PROMPT"])
    RDQ -->|天气| L7(["🔴 LLM: _WEATHER_CONTEXT_PROMPT"])
    RDQ -->|已自包含| ROUTE
    L5 --> ROUTE
    L6 --> ROUTE
    L7 --> ROUTE
    DEP -->|否| ROUTE

    ROUTE{落域 tv_domain_for tvMode?}
    ROUTE -->|息屏 tv=6| KW["纯关键词正则分诊 → 域"]
    ROUTE -->|亮屏 tv=0| AUTO["domains/pipeline.route_domains"]

    AUTO --> AUTOR["先 ~560行词法锚点规则"]
    AUTOR --> AUTORQ{规则命中?}
    AUTORQ -->|命中| DOM["确定域"]
    AUTORQ -->|全不中| L8(["🔴 LLM: DOMAIN_SYSTEM 分类器 + few-shot"])
    L8 --> DOM
    KW --> DOM

    %% ---------- 域内选工具 ----------
    DOM --> TOOL{select_tool 候选>1?}
    TOOL -->|单候选/词法高置信| PARAM
    TOOL -->|仍不确定| L9(["🔴 LLM: TOOL_SYSTEM 工具选择"])
    L9 --> PARAM
    PARAM["fill_params 抽参数"] --> PARAMQ{需LLM抽参?}
    PARAMQ -->|是| L10(["🔴 LLM: PARAM_SYSTEM"])
    PARAMQ -->|否| OUT
    L10 --> OUT
    OUT["组装 steps 返回"]
```

---

## 6. LLM 调用点清单

| # | 环节 | 触发条件 | Prompt | 两工程差异 |
|---|------|---------|--------|-----------|
| L1 | 记忆改写 | 有历史且残留指代、未自包含 | `_MEMORY_REWRITE_PROMPT` | 相同 |
| L2 | 条件继承合并 | 同设备跨轮需合并检索条件 | `_CONDITION_INHERIT_PROMPT` | 相同 |
| L3 | 依赖判定 | 无历史但切出多段 | `_DEPENDENCY_PROMPT` | 相同 |
| L4 | 整句拆计划 | 有历史 | `_PLAN_PROMPT` | 相同 |
| L5 | 开放排序改写 | 依赖意图含"评分最高/最新" | `_OPEN_ORDINAL_PROMPT` | 相同 |
| L6 | 依赖改写 | 依赖意图残留指代 | `_REWRITE_PROMPT` | 相同 |
| L7 | 天气上下文改写 | 天气依赖意图 | `_WEATHER_CONTEXT_PROMPT` | 相同 |
| L8 | **域分类器** | 亮屏且词法规则全不命中 | `DOMAIN_SYSTEM` + few-shot | **仅 juagent**；orchestrator_plan 甩给 compare |
| L9 | **工具选择** | 域内多候选、词法不确定 | `TOOL_SYSTEM` | **仅 juagent**；orchestrator_plan 由 compare 决定 |
| L10 | **参数抽取** | 需从 query 抽结构化参数 | `PARAM_SYSTEM` | **仅 juagent**；orchestrator_plan 由 compare 决定 |

---

## 7. 两个关键规律

1. **"规则闸门 + LLM 兜底"**：每个 LLM 调用前都有规则先处理（消解指代、切分、词法落域），只有规则搞不定才调 LLM，且 LLM 失败还会回退规则。实际 LLM 调用次数远少于分支数。

2. **两工程 LLM 覆盖范围不同**：
   - **编排层 LLM（L1–L7）两者完全相同** —— 拆分、改写、依赖的窄缝。
   - **落域/选工具/抽参（L8–L10）只有 juagent 有** —— `orchestrator_plan` 把这三步整个下沉给外部 compare(8084)，亮屏不落域（`tv_domain_for` 亮屏直接返回空）。

---

## 8. 关键代码位置

### orchestrator_plan
- 入口：`server.py` `Server.do_POST` → `_serve_slow`/`_serve_slow_impl`
- 状态机：`runtime.py` `TraceStateMachine.tick()` → `_first()` / `_continue()`
- 规划：`planner.py` `plan_query()`、`split_into_subqueries()`、`check_dependency()`
- 改写：`rewrite_with_memory()`、`rewrite_with_dialogs()`、`inherit_conditions()`、`rewrite_dependent()`
- 落域：`tv_domain_for()`；双域并行 `server.py` `_serve_dual_domain`
- 配置：`config.py`（COMPARE_URL 默认 127.0.0.1:8084、MAX_PLAN_STEPS=6 等）

### juagent
- 入口：`orchestrator/server.py` `Server._serve_slow`
- 状态机：`orchestrator/runtime.py` `TraceStateMachine.tick()` → `_first()` / `_continue()`
- 规划/改写：`orchestrator/planner.py`（同名函数）
- 落域：`orchestrator/planner.py` `tv_domain_for()` + `domains/core/pipeline.py` `route_domains()`、`select_tool()`、`fill_params()`
- 域定义：`domains/core/registry.py` 从 `schema/*.json` 加载（8 域 45 工具）
- 路由 prompt：`domains/core/prompts.py`（`DOMAIN_SYSTEM`/`DOMAIN_GUIDE`/`TOOL_SYSTEM`/`PARAM_SYSTEM`）


---

# 附：目标架构 —— LLM 主干 + 规则补丁（单调用/tick）

> 设计原则（与上面"规则优先"的现状相反）：
> **LLM 是主干，规则只是补丁，LLM 必须兜底。** 删掉任何规则补丁，LLM 主干仍能独立跑完；规则只在 LLM 解析失败或明显跑偏时覆盖/兜底。

## A1. 核心不变量：一个 tick = 且仅 = 一次 LLM 调用

状态机 `TraceStateMachine.tick()` 本就是"一 tick 推进一次"，且**规划只做一次**：首帧算完整 plan 存入 `_state[trace_id]`，后续 tick 不重新规划，只按 cursor 取下一批。据此把现在散落的 10 个 LLM 调用点，收敛成**一个统一调用**，每 tick 触发一次：

- **T0（首帧，toolHistory 为空）**：输入 `query + shortMem + dialogData + toolHistory([]) + tvMode` → 输出**完整 plan（含当前批次）**。一次吃掉原 L1(记忆改写)+L2(条件继承)+L3(依赖)+L4(拆分)+L8(落域)。
- **T1+（续跑，toolHistory 带真实结果）**：输入 `T0 存的 plan + 上下文 + toolHistory(真实结果) + cursor` → 输出**当前 step 的自包含 query**。一次吃掉原 L5/L6/L7(依赖改写)。

**每轮总调用次数** = tick 数：
- 全并行（无依赖）：1 tick = **1 次 LLM**。
- 依赖链深度 k：k 个 tick = **k 次 LLM**（每 tick 仍是 1 次）。依赖链那 k 次省不掉——后一步参数依赖前一步真实结果，必须等执行边界。

相比"结构化 selector + 确定性 resolver 消除 T1+ LLM"的方案，本方案 T1+ 仍调一次 LLM，但**能看到真实 toolHistory**，保留 ReAct 的自适应（结果为空/意外时可换策略），同时把该 tick 内原本 3~7 个碎调用压成 1 个。

## A2. 简化后的输出契约（压 token = 降 response time）

输出 token 串行生成，是最直接的延迟杠杆。原则：**只输出 LLM 必须决策的内容；可由 runtime 确定性派生的一律不生成。**

**去掉的冗余**：
- `now`（当前批次）——由 runtime 从依赖关系 `next_batch` 派生，不让 LLM 复制一遍。
- `id:"s1"`——按数组下标由 runtime 补 `s{i}`。
- `plan.intents` 双层包裹——拍平成顶层数组。
- `depends_on_previous:false` 默认值——并行步直接省略该字段。
- 长字段名——`query→q`、`domain→d`、`depends_on_previous→dep`。

**T0 输出**（顶层数组，短键）：
```json
[{"q":"搜刘德华的电影","d":"vod"},
 {"q":"第三部的导演是谁","d":"qa","dep":1}]
```
- `q` = 改写后自包含的子 query
- `d` = 域（vod/audio/children/education/sports/music/device/qa）
- `dep` = 依赖标记，**只在依赖时出现**，值=所依赖步的序号（并行步不写）
- id 由 runtime 按下标补 `s1/s2`；当前批次由 runtime 依 `dep` 派生

**T1+ 输出**（单值场景直接纯文本，最省 token）：
```
查询《暗战》的导演是谁
```
- 不套 JSON（省括号和键）；domain 已在 T0 存好，无需再出
- 复用现有 `_clean_llm_text` 剥 thinking/引号壳

**token 对比**（单条 intent 结构开销）：
- 原：`{"id":"s1","query":"...","domain":"vod","depends_on_previous":false}` ≈ 键+标点 55 字符
- 新：`{"q":"...","d":"vod"}` ≈ 18 字符

叠加去掉 `now` 整批复制 + `plan/intents` 包裹，**结构性 token 降到约 1/3**；T1+ 转纯文本再省一截。

## A3. schema 之外更大的延迟杠杆（prompt 里必须钉死）

- **禁止 thinking 前缀**（baseline 模型偶发，用 1~2 个 few-shot 强化）
- **禁止 ```json 代码块包裹**（省几十 token）
- **约束 q 长度**，别让模型把"搜刘德华的电影"扩写成长句

## A4. 可选的进一步压缩（取舍留档）

1. **连 `d`(domain) 都不让 LLM 出**：信任下游域层自己的路由（亮屏本有分类器），T0 只吐 `q`+`dep`。更省 token，但放弃"LLM 一把定域"的主干性，域层要再判一次。
2. **`dep` 布尔 vs 序号**：只支持"依赖上一步"用布尔（出现即 true）即可；要支持"依赖非紧邻的第 N 步"才需序号。当前模型只有 `depends_on_previous`，布尔够用，序号是余量。

## A5. 需要定的一个设计点

T1+ 时 LLM 是**只解析当前步**（plan 结构锁定，只填当前 step 的 query）还是**允许改 plan**（看结果后增删后续步）：
- 只解析当前步：可控、plan 不漂移、prompt 简单 —— **建议先做这个**。
- 允许改 plan：更开放的 ReAct，但要处理 plan 变更与 cursor 一致性。

## A6. 风险提示

- **极短键（`q`/`d`）对模型可靠性略有影响**（模型在长描述性键上训练更多），用 1~2 个 few-shot 锚格式可抵消，实测风险低。
- **落域精度**：代码注释与 `0910_*域评测明细.csv` 记录过纯 LLM 落域大幅掉分（音乐 -36.5% 等）。翻成 LLM 主干后，关键词规则要保留为**高优先级覆盖补丁**，并用 `eval_bench` A/B 量化掉分。
