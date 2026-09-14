# hcAgent：多轮 + 单轮串行多意图 支持方案

> 范围：**仅 hcAgent 侧**（engine + 评测驱动 + 用例集）。
> 目标场景（用户确认）：**两者组合打通** —— 同一次对话既有「跨轮多轮会话上下文」，单轮内又有「串行依赖的多意图」。
>
> ## 状态：方案 A 已落地并真机验证（2026-09-14）
> - **改了** `app/prompts.py` MT_REWRITE_PROMPT：
>   1. 「禁止压扁多意图 + 用 `;` 分隔 + 保留串行依赖指代」约束；
>   2. 「人物属性指代——优先还原人名」：问导演/妻子/主演属性时把「该导演/他」还原成候选里的人名，**不要带（普通话版）等修饰词片名**。
> - **真机验证**（相同 device 连发多请求）：
  - 「第三部导演是谁？他导过的电影」→ plan1 fan_knowledge 导演 + plan2 vod_search 该导演电影（params.director=拜伦·霍华德/杰拉德·布什）。两意图执行、串行依赖正确。
  - 「这导演老婆叫啥？演过啥电影」→ 修复后 mt_query=「杰拉德·布什老婆叫啥名字；杰拉德·布什演过啥电影」，step1 正确 fan_knowledge 妻子、step2 vod actor=杰拉德·布什。链式「导演→妻子→妻子电影」打通。（修复前 step1 误拆成 vod_search_all _普通话）
> - **回归**：`tests/test_multiturn_serial.py` 3 个 mock 用例（首轮只放 s1 + toolHistory 续跑发 s2 + 人名指代还原）通过；既有 mock 测试除 `test_parallel`（multiintent 确定性拆分导致旧断言 t0_times==1 失效，与本次无关）外全绿。
> - 验证要点：engine 串行靠 toolHistory 里的 `id:sN` 由 `_executed_indexes` 辨识推进；评测侧 `runtime_executed._run_runtime` 的 `_plan_steps_dedup` 丢弃 `id` 且 payload 无 tool_history——若要评测侧也驱动串行，需补（见 §1 L2）。

---

## 0. 现状：哪一步已经是通的，哪一步堵死

以代码事实为准（非假设）：

### 引擎侧已具备（`app/engine.py`）
- **单轮串行**：T0 计划支持 `dep` 字段（`parse_plan` → `Plan`），`_SM._forward` → `next_batch(plan, executed)` 按依赖图推进，T1 `_continue` 用 `req.data.toolHistory` 改写依赖步为自包含 query。
- **跨请求续跑**：`TraceStateMachine._state[device_id]` 是进程内 TTL 缓存（`PLAN_TRACE_TTL`），`build_response` 以 **device_id**（非 trace_id）为 tick 键，同 device 连续请求能命中并续跑。
- **多轮 merge 不吞串行**：`build_response` 分支（行 747-756）——
  - `history`（toolHistory）非空 → `mt_query = query`，交 `_SM._continue` 用 toolHistory 改串行步，**不走** `_mt_rewrite`；
  - 无 history 且无任何上下文（`_has_mt_history` 假）→ 原句直通，不喂 LLM 改写；
  - 仅无 history 但有上下文 → `_mt_rewrite` 合并多轮。
  - **结论：engine 已经把「多轮上下文」和「单轮串行 toolHistory」两条线做了互斥分支，不会互相污染。** 这一层基本不用改。

### 唯一堵死的缺口：评测侧不驱动串行（`runtime_execute.py` + 三个评测脚本）

上游契约（`docs/runtime_execute_upstream.md` payload）：
```json
{ "feature_code", "device_id", "retext", "tv_mode", "enable_slow", "client_sid", "debug" }
```
**没有 `toolHistory` / `history` 字段**。`_run_runtime` 每次发**一个** HTTP SSE 请求，取 `steps[0]` 就结束。

- `run_sse_bench.py` / `run_sse_7domain_eval`：单意图单请求。
- `run_sse_multiintent.py`：200 例 e80803 全**并行**内容+设备，串行依赖候选 = **0**。
- `run_sse_multiturn.py`：30 会话 × 5 轮，**只测单意图继承跨轮上下文**，单轮内无串行；已有素材串行依赖候选 = **0**。

→ 所以「单轮串行 + 多轮」在当前评测链路里从数据到驱动**都不存在**。engine 有马力，评测没上跑轮。

---

## 1. 要动的 3 层

### L1 用例/数据（缺口最硬的）
新增一条含「多轮上下文 + 单轮串行」的 gold 用例集。现有两个数据集串行依赖都是 0，必须补。

示例（串行多意图，第 2 步依赖第 1 步结果）：
```csv
session, turn, device_id, client_sid, retext, gold_tools, gold_params, expect_steps
s1, 1, dv1, sid1, "想看的刘德华电影", "vod_keyword_search", "刘德华", 1
s1, 2, dv1, sid1, "搜一下刘德华的电影，第三个的导演是谁",
     "vod_keyword_search;fan_knowledge_agent", "刘德华;nil", 2    ← 单轮 2 步串行，且依赖首轮上下文
```
- `gold_tools` 以 `;` 分隔表示串行顺序，`gold_params` 对应；`expect_steps=2` 断言这轮要跑出 2 个串行工具步。
- 放 `benchmark/cases/`（如 `sheet_serial_multiturn.csv`），沿用 device_id/client_sid 复用约定。

### L2 评测驱动（核心代码缺口）
给 `_run_runtime` 加**回传 toolHistory** 能力，并新增「串行续跑」驱动器：

1. **payload 扩展**：`_run_runtime(..., tool_history=None)` —— 非空时把已执行步序列化进请求体
   ```json
   { ..., "tool_history": [ {"toolName": "vod_search", "parameters": {...}, "result": "..."} ] }
   ```
   （上游若只认 `history`/别的键名，则收敛成上游实际读的字段名，见 §2 待确认项。）

2. **新增 `run_sse_serial_multiturn.py`**：对每条用例，
   - 读取每个 session turn 的 `expect_steps`；
   - 当 `expect_steps > 1`：进入串行续跑循环 ——
     ```
     executed = []
     while len(all_steps) < expect_steps and not stop:
         steps, stop = _run_runtime(retext, device_id, client_sid,
                                    tool_history=executed)   # 回传已执行步
         executed += steps
     ```
   - 首轮（turn 奉 `turn>1`，同一 session）继续发**同一 device_id + client_sid**，让远端 engine 走多轮上下文分支；单轮内循环则靠 tool_history 走 `_continue` 串行分支。
   - 结果断言走 engine 现有的「tool+param / select」判分，串行第二张表顺序用 `next_batch` 顺序核对。

### L3 可选防线：确定性 native 断点
若远端 engine 续跑在真实环境被 TTL/无状态问题打断，可加一条**引擎侧兜底**：`build_response` 在 `_mt_rewrite` 前，先看 tool_history 是否已推进到某一串行步（已执行索引集非空且尚有待执行 `dep` 步），确保无论上游走哪条分支，单轮内串行都优先于多轮合并。**这层是否要动取决于 L2 用真实远端跑通的置信度，默认先不动，L2 测不通则升这里。**

---

## 2. 待确认（2 个遮盖点，需在实现前找答案）

1. **远端 engine 续跑 state 的真实生命周期**：`_SM._state[device_id]` 在绑定的当前进程内 TTL 300s。评测连发两个请求打到同一 runtime 时，是否能命中同一进程实例？若 runtime 是 load balancing / 无状态多副本，串行续跑依赖就是要打穿的关键点 —— 必须先确认单条 circuit 内 series 是否保持同一实例。
2. **tool_history 回传字段的真实名 **：上游 engine `_continue` 直接读 `req.data.toolHistory`（见 build_response 行 737）。`_run_runtime` 需要把它映射到上游 HttpRequest data 的 `toolHistory` —— 需要知道 Proto 字段名 / 序列化键名。若上游不认自己引擎的键，就需在请求构造侧对齐。

## 3. 落地顺序（建议）

1. 先用现有 `sheet0821_multiturn.csv` 里**只能找到的串行候选**（哪怕只有 1 个）打一条 `_run_runtime` 修复后的 log，确认在同一 device_id + client_sid 下，带 tool_history 的续跑请求能产出第 2 步 —— **验证 L2 通路**（先做，不写一堆新用例）。
2. 扩展新素材集 `sheet_serial_multiturn.csv`，覆盖 3 类：
   a. 纯多轮上下文（话插依赖，串行 step=1）；
   b. 纯单轮串行（step≥2，无跨轮上下文）；
   c. **组合：多轮上下文里这一轮同时有单轮串行（链条 2 步）** —— 这是「两者组合打通」的正交验法。
3. 接入 `run_sse_serial_multiturn.py` 判段，跑全量、出 tool+param/select 报告。
4. （只有在 TTL/无状态上真的踩坑时才动 engine `_mt_rewrite` 顺序的兜底。）

> 范围墙：本方案只在 hcAgent 侧。`runtime_execute_upstream.md` 是上游契约的提炼，不修改上游服务端；所有 tool_history 支持若上游本就不读该字段，则必须在请求构造侧符合上游真实 schema（确认项 2）。

## 5. 交付判段（DoD）
- [ ] 新增 gold 集含「组合式」`a/b/c` 三类，串行依赖用例 ≥10 条。
- [ ] `run_sse_serial_multiturn.py` 可对带 `expect_steps>1` 的 turn 自动串行续跑，不再只取 `steps[0]`。
- [ ] 组合式 c 类在图率 tool+param ≥ 上一基线（多轮基线 / multiturn 55%→75%，串行 step 数按顺序核）—— 至少证明不比「只跑单步」差。
- [ ] 保留原生 `run_sse_multiturn.py` / `run_sse_multiintent.py` 后缀，不改动已回归基线的既有评测。