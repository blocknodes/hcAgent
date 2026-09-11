# 目标架构原型（LLM 主干 + 单 tick 单调用）

> 在 hcAgent（原 FastAPI mock）上落地 `docs/query_pipeline.md` 附录的【目标架构】：
> **LLM 是主干，规则只是补丁，LLM 必须兜底。** 删掉任何规则补丁，LLM 主干仍能独立跑完；
> 规则只在 LLM 解析失败或明显跑偏时覆盖/兜底。
>
> 现状（`app/mock.py`）与目标本质相反（规则优先、零 LLM）。本原型翻转为 LLM 优先，
> 并把散落的 10 个调用点收敛成**统一调用/每 tick**。

## 代码位置

| 文件 | 作用 |
|---|---|
| `app/config.py` | LLM 网关/模型/开关（读 `JU_*`；默认 juagent 网关） |
| `app/llm.py` | `chat()` + `parse_json()` + 可注入 fake（对齐 juagent/llm.py） |
| `app/prompts.py` | `T0_PLAN_PROMPT` / `T1_STEP_PROMPT`（短键架构） |
| `app/engine.py` | `TraceStateMachine` T0/T1+、`build_response`、规则补丁、落域、mock 工具 |
| `app/main.py` | 请求入口；默认 LLM 主干，`JU_USE_MOCK=1` 退回规则 mock |
| `tests/test_llm_engine.py` | 注入假 LLM 的 LLM 主干用例（离线） |
| `tests/test_slow_agent.py` | 规则 mock 回归用例 |

## 核心不变量：一个 tick ＝ 恰好一次 LLM

- **T0（首轮，toolHistory 空）**：一次 LLM 出整份计划。输入 `query`，输出顶层数组短键
  `[{"q","d","dep"?}]`。**一次吃掉**原 记忆改写 + 条件继承 + 依赖判定 + 拆分 + 落域。
  `id` 和 `now(当前批次)` 由 runtime 依 `dep` 派生，不让 LLM 复制（target A2 压 token）。
- **T1+（续跑，带真实 toolHistory）**：只解析当前待执行 step 的自包含 query。plan 结构
  锁定不漂移（A5 首选方案）。能看到真实工具结果 → 保留 ReAct 自适应（结果空/意外可换策略）。
- **失败兜底**：任一 LLM 失败/解析失败 → 回退规则补丁（确定性拆步 `_rule_plan` + `domain_for`
  落域 + `_mock_tool`）。故"删顺序规则补丁，LLM 主干仍能跑"成立。

## 批次派生（next_batch）

- `dep_on` 存**1-based 所依赖计划序号**（`dep":1`=依赖第 1 条 `s1`）。
- 无依赖条目并行全发；依赖条目需其依赖项已在 toolHistory 里执行过才放行（严格串行）。
- `stop` 判定：`remaining` 是否为空（是否还有未下发的计划条目）。

## 规则补丁（保留的"规则层"）

| 补丁 | 逻辑 | 触发 |
|---|---|---|
| 卡通双域 | 动漫/动画片/卡通 → children+vod 并行，`parallel=True` | 首轮无历史 |
| 息屏落域 `domain_for(tv_mode=6)` | 设备→儿歌→歌→有声→教育→体育→答疑→兜底 | tvMode=6 |
| 指代消解 `_rewrite_dependent` | 第N部→《结果第N》；`该导`→title | 规则优先，残留指代才 T1 LLM |

## 真实网关（`../juagent/run.sh`）

```
JU_LLM_API_BASE=http://10.19.96.219:4003/v1
JU_LLM_MODEL=baseline
```
原型默认走该网关。可用 `JU_LLM_*`/`JU_*`/`HC_LLM_*` 覆盖。真实网关首次请求较慢。

## 测试

```bash
# 规则 mock 回归
python -m pytest tests/test_slow_agent.py -q        # 9 passed

# LLM 主干（假 LLM，离线）
python -m pytest tests/test_llm_engine.py -q        # 7 passed

# 全部
python -m pytest tests/ -q                           # 16 passed
```

LLM 主干用例覆盖：M1 单意图自包含、M2 并行、M2 依赖串行（首批只放 `s1`，toolHistory 带回实体后
二步放 `s2` + `dependsOn:[s1]`）、M3 跨轮（第N→实体，规则不改才触发 LLM）、LLM 失败回退规则。
每用例断言 `T0 恰 1 次`/`T1 恰 1 次`，从而校验"1 tick=1 次调用"不变量。

## 尝跑（真 LLM）

```bash
JU_LLM_API_BASE=http://10.19.96.219:4003/v1 JU_LLM_MODEL=baseline bash run.sh
curl -s localhost:8082/slowAgent/poc_1 -X POST \
  -H 'content-type: application/json' \
  -d '{"traceId":"t1","deviceId":"d1","data":{"query":"搜刘德华的电影，第三个的导演是谁","tvMode":"0","debug":true}}'
```

## 取舍 / 后续

- 极短键 `q/d/dep` 可能对模型可靠性略降（A6），用 1~2 个 few-shot 锚格式可抵消。
- 后端工具仍 mock（`_mock_tool`）；真实域执行/选工具/抽参待接入 juagent `domains/`。
- T1+ 目前"只解析当前步"（A5 首选）；"允许改 plan"留档为后续放开 ReAct。