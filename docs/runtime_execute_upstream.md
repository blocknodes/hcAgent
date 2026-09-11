# hcAgent 上游 runtime 接口（runtime_execute.py 提炼）

> 从 `runtime_execute.py` 提炼其直连的上游服务接口。它不经过 hcAgent 自身，而是
> 直连**远端 runtime**（SSE 流式接口），真实执行工具并回读结果。本文档是给对接 /
> 复现 / 调试用的接口契约速查。
>
> 目标端：`POST http://10.18.210.7:31392/api/runtime/execute`（SSE）。

---

## 1. 一句话

`runtime_execute.py` = 一键跑通完整 ReAct 多步链路的调试/验收脚本：

```
python3 runtime_execute.py "刘德华的电影"
python3 runtime_execute.py --device-id <id> "帮我查下无间道的导演是谁，然后再搜下他的片子"
python3 runtime_execute.py --device-random "俄罗斯队，顺便声音调节"
python3 runtime_execute.py --json "query"            # 完整 JSON 输出
```

双层输出：`TOOL 帧`（真实工具执行的 llmSemantic 工具名 + 参数 + 结果）与
开 `debug:true` 时的 `DEBUG_MAP 帧`（rpc.SlowAgentApi 的计划，给出**数据集命名的
真工具名序列** vod_search / fan_knowledge_agent ……）。

---

## 2. 请求接口（上游 HTTP POST）

| 项 | 值 |
|---|---|
| URL | `http://10.18.210.7:31392/api/runtime/execute` |
| Method | `POST` |
| Content-Type | `application/json` |
| 传输 | SSE（Server-Sent Events，`data:` 帧流） |
| 单次连接超时 | `SSE_TIMEOUT = 90s` |

### 请求体 payload

```json
{
  "feature_code": "861003009000014000000712",   // 默认 DEFAULT_FEATURE_CODE
  "device_id":    "86100300900001400000071212345678", // 默认 DEFAULT_DEVICE_ID；--device-random 生成随机 id
  "retext":       "刘德华的电影",                  // 用户原始请求（必填）
  "tv_mode":      "0",                           // 0=亮屏 6=息屏；默认 "0"
  "enable_slow":  true,                          // 走慢任务（慢端规划）
  "client_sid":   "<可选>",                      // 多轮会话标记（client-sid）
  "debug":        true                           // 默认开；开时返回 DEBUG_MAP 计划帧
}
```

默认值：`DEFAULT_FEATURE_CODE`、`DEFAULT_DEVICE_ID`（见脚本顶部常量）。

---

## 3. 响应：SSE 帧流

每一行 `data:` 后跟一个 JSON 对象即为 `frame`。脚本按 `frame.header.messageType`
分发，主要帧类：

| messageType | 作用 | 脚本消费 |
|---|---|---|
| `FIRST_PACKAGE` | 首包握手 | 忽略跳过 |
| `TOOL` | **一次真实工具执行**（工具名/意图 + 参数 + 结果摘要） | `_summarize_tool_frame` 聚合 |
| `DEBUG_MAP` | rpc.SlowAgentApi 计划（真工具名序列） | `_summarize_debug_trace` 提取 steps |
| `TEXT`（其它） | 流式文本（泛问答工具的回答；无边界分段） | 拼接 `texts` |
| —（end） | `frame.header.status == 2` | 置 `stop = True` |

### `header` 结构（脚本用到的字段）

```json
{ "messageType": "TOOL", "status": 0 }   // status==2 → 流结束（stop）
```

---

## 4. TOOL 帧：真实工具执行结果

`frame.data.application_data.data` 是工具执行的载荷。字段见下（脚本 `_summarize_tool_frame`）：

```jsonc
{
  "data": {
    "llmSemantic": {                    // 工具域/意图 → 工具名
      "intent": "vod_keyword_search | fan_knowledge_agent ..."
    },
    "memoryContent": "{\"memoryQuery\":\"...\",\"query\":\"...\"}",  // 字符串，JSON 序列化
    "ttscontent": "给用户的自然语言结果",
    "memoryData": [ {"subsort": "...", "data": [媒资...]} ],        // 记忆搜索结果候选媒资
    "vagueTtsContent": "模糊搜索定位文案",
    "data": { "content": { "searchResultList": [ {"total": N, "data": [媒资...]} ] }, "showText": "…" }
  }
}
```

**脚本提取规则**：
- `tool` ← `llmSemantic.intent`（助手兜底 `domain`；都没有 `"unknown-tool"`）；
- `params` ← `memoryContent` 解 JSON 后的 `memoryQuery` / `query`；
- `candidates` ← 优先 `memoryData`（打平 `[{data:[...]}]`）；缺则取
  `data.content.searchResultList`（模糊搜索候选）；截前 20 条；
- `tts` ← `ttscontent`，缺则 `vagueTtsContent` → `data.showText`，各截 120 字符。

### `TOOL` 帧与 `DEBUG_MAP` 计划的关系（脚本 `_align_tool_names`）
以 **DEBUG_MAP 的 plan 为权威序列**（慢端规划的`真工具名` vod_search、结构化 query 参数），
再按顺序就近补 TOOL 帧的 `tts` / `candidates` / `frame_tool`。plan 之外的 TOOL 帧不再展示，
避免 `llmSemantic` 帧与 plan 重复描述同一执行流。

---

## 5. DEBUG_MAP 帧：rpc.SlowAgentApi 计划

`frame.data.application_data.debug.traces[]` 里 phase==`rpc.SlowAgentApi` 的 trace：

```jsonc
// trace
{ "phase": "rpc.SlowAgentApi",
  "response": "…双引号转义的 JSON 字符串…" }   // 需两次 json.loads 解开
```

解开后：
```jsonc
{ "code": 200, "message": "...", "data": {
    "steps": [ { "id": "s1", "toolName": "vod_search",
                 "retext": "搜索科幻电影", "parameters": {...}, ... } ]
}}
```

- `_summarize_debug_trace`：双重 `json.loads` 解开 `response`，取 `data`。
- `_plan_steps_dedup`：拍平所有 plan 帧的 `steps`，以 `toolName|retext|parameters` 去重
  （multi-intent 里同一查询可能被 rpc 调多次）。
- `_align_tool_names`：plan step 各生成一条展示记录，再按序补 TOOL 帧的 tts/候选。

---

## 6. 流式 TEXT 帧 → 分步回答（`_split_plan_stream`）

TEXT 帧把所有步骤的回答连在一起、无边界。脚本用 plan 里 `fan_knowledge_agent`
步骤的 `retext` 做锚点切分：

1. 取 retext 去掉疑问词（是谁/叫什么/导演…）后剩下的**实体片段**；
2. 在拼接的 assistant 文本里找该片段出现位置作为锚点；
3. 相邻锚点切段，归属到对应 `fan_knowledge_agent` 步骤下（`{plan_step_index: answer}`）；
4. 只有一条 fan 回答时整段给它；找不到锚点则给最后一个。

---

## 7. CLI 参数（运行时用，非上游契约之外）

| 参数 | 行为 |
|---|---|
| `retext` | 用户原始请求（必填） |
| `--feature-code` | 覆盖 feature_code |
| `--device-id` | 固定 deviceId（多轮会话复用） |
| `--device-random` | 随机 deviceId（每次独立，避免 sessions/多样性复用） |
| `--client-sid` | 可选会话 sid |
| `--tv-mode` | 0=亮屏 6=息屏（默认 0） |
| `--expect-steps N` | 期望工具步数，不满足则退出码非 0（回归断言） |
| `--no-debug` | 关闭 debug（默认开，为了拿 DEBUG_MAP 计划真工具名） |
| `--json` | 完整 JSON 输出 |

**退出码**：`0` 正常；`1` 期望步数不符；`2` 缺 retext。

---

## 8. 关键点速查

- 上游是**远端 runtime 的 SSE 慢任务接口**，非 curl 一次性 JSON：要逐 `data:` 帧订阅。
- 真工具名（vod_search 等数据集命名）在 **DEBUG_MAP**，不是每一次 TOOL 帧
  （老帧是 `llmSemantic.intent`，两者在同一执行流的两种描述，依赖对齐而非混拼）。
- `stop=true` 由 `header.status==2` 收尾；多步依赖靠多次 TOOL/plan 推进，无独立结束帧。
- 媒资候选字段（围栏侧）`mediaTitle` / `director` / `category` / `childCategory` /
  `doubanRate` / `pubdate` / `mediaId` / `episodeTitle` / `summary` —— 展示时脚本只捞这 9 个。