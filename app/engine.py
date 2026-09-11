"""目标架构核心：纯 LLM 编排 —— 一个 tick ＝ 恰一次 LLM，编排层零规则。

对齐 docs/query_pipeline.md 附录 A2/A5：
  - T0(首轮，无 toolHistory)：一次 LLM 出整份计划 + 当前批。短键数组
    [{"q","d","tool","dep"?}] —— 拆分/改写/落域/选工具 全由 LLM 决策。
    id 与 now(当前批) 由 runtime 依 dep 派生（A2，非规则，是契约派生）。
  - T1+(续跑，带真实 toolHistory)：一次 LLM 把当前待执行 step 改写成自包含 query，
    保留 ReAct 自适应（A5 首选：plan 锁定不漂移）。
  - 无规则补丁：不拆连接词、不落域词法、不关键词换工具、不做规则指代替换。
    LLM 失败/输出非法 → 降级为单条自包含意图(透传原 query，非启发式)，不注入语言规则。

对外主入口 build_response(req)。同一 traceId 状态在进程内按 TTL 缓存。
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import config, hctools, llm
from .prompts import T0_PLAN_PROMPT, T1_STEP_PROMPT

# 域名闭合集（供 LLM 输出校验）。空串=未定，由 LLM 必须给出；为保鲁棒给一个中性默认。
_DOMAINS = {"vod", "children", "education", "music", "audio", "sports", "device", "qa"}


@dataclass
class Intent:
    query: str
    domain: str
    tool: str = "execute"
    index: int = 0          # 1-start 计划序号
    depends: bool = False
    dep_on: int = 0        # 所依赖条的 1-start 计划序号（0=无依赖）


@dataclass
class Plan:
    intents: list[Intent] = field(default_factory=list)

    def total(self) -> int:
        return len(self.intents)


# ---------------------------------------------------------------------------
# T0 短键计划解析（纯解析，不含语言判断）
# ---------------------------------------------------------------------------
def parse_plan(value: Any) -> Plan:
    """把 T0 顶层数组（[{"q","d","tool","dep"?}]）解析成 Plan；非法 → 空 Plan。

    容忍 baseline 常见怪输出：
      - 两层包裹 {"plan":{"intents":[...]}}
      - 裸 dict 含单个 q → 当单意图
      - 短键 (q/d) 与长键 (query/domain/depends_on_previous) 兼容
    """
    if isinstance(value, dict):
        v = value.get("plan") if isinstance(value.get("plan"), dict) else value
        v = v.get("intents") if isinstance(v, dict) else v
        if isinstance(v, list):
            value = v
        elif "q" in value or "query" in value:
            value = [value]
        else:
            return Plan()
    if not isinstance(value, list):
        return Plan()
    intents: list[Intent] = []
    for i, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        q = item.get("q") or item.get("query")
        if not isinstance(q, str) or not q.strip():
            continue
        dom = (item.get("d") or item.get("domain") or "").strip()
        tool = (item.get("tool") or item.get("toolName") or "").strip()
        d = item.get("dep") or item.get("depends_on_previous")
        dep_on = 0  # 1-start；0=无依赖
        if isinstance(d, int) and d > 0:
            dep_on = d
        elif isinstance(d, bool) and d and i > 1:
            dep_on = i - 1
        intents.append(Intent(
            query=q.strip(),
            domain=dom if dom in _DOMAINS else ("" if not dom else dom),
            tool=tool or "execute",
            depends=dep_on > 0, dep_on=dep_on, index=i,
        ))
    return Plan(intents=intents)


def _extract_bare_objects(text: str) -> list[dict]:
    """把多个裸 JSON 对象串（{"q":..},{"q":..}）切成列表；失败 → []。

    这是对 baseline 常犯"数组写丢方括号"的宽容解析，不是业务规则。
    """
    import json as _json
    try:
        decoder = _json.JSONDecoder()
        items: list[dict] = []
        i, n = 0, len(text)
        while i < n:
            while i < n and text[i] not in "{[":
                i += 1
            if i >= n:
                break
            try:
                obj, end = decoder.raw_decode(text, i)
            except _json.JSONDecodeError:
                break
            if isinstance(obj, dict):
                items.append(obj)
            elif isinstance(obj, list):
                items.extend(obj)
                return items
            i = end
        return items
    except Exception:  # noqa: BLE001
        return []


def _clean_llm_text(text: str) -> str:
    """剥 baseline 的 thinking 前缀/引号壳；整段思考 → 原始串（非启发式，只解 shell）。"""
    t = (text or "").strip()
    low = t.lower()
    if low in {"thinking", "思考"}:
        return ""
    if low.startswith("thinking") or low.startswith("思考"):
        parts = t.split("\n\n")
        candidate = parts[-1].strip()
        if candidate and candidate.lower() not in {"thinking", "思考"}:
            t = candidate
        else:
            return ""
    if t.startswith('"') and t.endswith('"'):
        t = t[1:-1]
    elif len(t) >= 2 and t[0] in "「『" and t[-1] in "」』":
        t = t[1:-1]
    return t.strip()


# ---------------------------------------------------------------------------
# 批次派生（契约派生，非语言规则：由 dep 序号确定当前可执行批）
# ---------------------------------------------------------------------------
def _executed_indexes(history: list[dict[str, Any]] | None) -> set[int]:
    out: set[int] = set()
    for item in history or []:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if isinstance(idx, int) and idx > 0:
            out.add(idx)
            continue
        i = item.get("id") or item.get("stepId")
        if isinstance(i, str) and i.startswith("s"):
            tail = i[1:]
            if tail.isdigit():
                out.add(int(tail))
    return out


def next_batch(plan: Plan, executed: set[int]) -> list[Intent]:
    """下一批：无依赖全发(并行)；依赖意图需其依赖项已执行(串行)。"""
    out: list[Intent] = []
    for it in plan.intents:
        if it.index in executed:
            continue
        if it.depends and it.dep_on not in executed:
            continue
        out.append(it)
        if len(out) >= config.MAX_PLAN_STEPS:
            break
    return out


def _tool_result_text(history: list[dict[str, Any]] | None) -> str:
    """把上一步工具结果压缩成给 T1 的参考文本（保留实体）。仅 text bookkeeping。"""
    texts: list[str] = []
    for item in reversed(history or []):
        if not isinstance(item, dict):
            continue
        for k in ("result", "answer", "tts"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                texts.append(v.strip())
            elif isinstance(v, (list, dict)):
                texts.append(str(v))
    # 简化为一份，避免超长
    return texts[0] if texts else ""


# ---------------------------------------------------------------------------
# TraceStateMachine：T0 / T1+（纯 LLM）
# ---------------------------------------------------------------------------
class TraceStateMachine:
    def __init__(self, ttl: float = config.PLAN_TRACE_TTL):
        self._lock = threading.Lock()
        self._state: dict[str, dict[str, Any]] = {}
        self._ttl = ttl

    async def tick(self, trace_id: str, query: str, history: list[dict[str, Any]] | None) -> tuple[list[Intent], bool]:
        entry = self._load(trace_id)
        if entry is None:
            return await self._first(trace_id, query, history)
        return await self._continue(trace_id, history)

    # ---- 首轮：T0（一次 LLM）----
    async def _first(self, trace_id: str, query: str, history) -> tuple[list[Intent], bool]:
        plan = await self._plan(query)
        if not plan.intents:
            # LLM 没出计划（空/纯聊天/失败）→ 兜底：单一自包含意图（非语言规则）
            return [Intent(query=query, domain="", tool="execute", index=1)], True
        executed = _executed_indexes(history)
        batch = next_batch(plan, executed)
        if not batch:
            batch = [plan.intents[0]]
        stop = len(executed) + len(batch) >= plan.total()
        if not stop:
            self._store(trace_id, plan)
        return batch, stop

    async def _plan(self, query: str) -> Plan:
        message = await llm.chat([{"role": "system", "content": T0_PLAN_PROMPT},
                            {"role": "user", "content": f"用户请求：{query}"}],
                           model=config.MODEL)
        if "__error__" in message:
            return Plan()
        text = llm.text_of(message)
        return self._resolve_plan(text)

    def _resolve_plan(self, text: str) -> Plan:
        parsed = llm.parse_json(text)
        # 形态 3：parse_json 只取到一个裸对象，原文还有别的 → 切片
        if isinstance(parsed, dict) and ("q" in parsed or "query" in parsed):
            items = _extract_bare_objects(text)
            if len(items) > 1:
                return parse_plan(items)
            return parse_plan(parsed)
        return parse_plan(parsed)

    # ---- 续跑：T1+（纯 LLM 改写当前 step 的自包含 query）----
    async def _continue(self, trace_id: str, history) -> tuple[list[Intent], bool]:
        plan: Plan = self._state[trace_id]["plan"]
        executed = _executed_indexes(history)
        batch = next_batch(plan, executed)
        if not batch:
            self._clear(trace_id)
            return [], True
        tool_result = _tool_result_text(history)
        # 批内依赖改写的多个意图相互独立，并发执行（真并发收益点）
        async def resolve(it: Intent) -> Intent:
            if it.depends:
                q = await self._rewrite_step(it.query, tool_result)
            else:
                q = it.query
            return Intent(query=q, domain=it.domain, tool=it.tool,
                          index=it.index, depends=it.depends, dep_on=it.dep_on)

        resolved = list(await asyncio.gather(*[resolve(it) for it in batch]))
        cursor = max((i.index for i in resolved), default=0)
        remaining = [i for i in plan.intents if i.index > cursor]
        stop = not remaining
        if stop:
            self._clear(trace_id)
        else:
            self._store(trace_id, plan)
        return resolved, stop

    async def _rewrite_step(self, q: str, tool_result: str) -> str:
        user = f"待执行意图:{q}\n" + (f"工具结果:{tool_result}" if tool_result else "")
        message = await llm.chat([{"role": "system", "content": T1_STEP_PROMPT},
                            {"role": "user", "content": user}],
                           model=config.MODEL)
        if "__error__" in message:
            return q
        out = _clean_llm_text(llm.text_of(message))
        return out if out else q

    # -- 状态 --
    def _store(self, trace_id: str, plan: Plan) -> None:
        with self._lock:
            self._state[trace_id] = {"plan": plan, "ts": time.time()}

    def _load(self, trace_id: str) -> dict[str, Any] | None:
        if not trace_id:
            return None
        now = time.time()
        with self._lock:
            st = self._state.get(trace_id)
            if st is not None and now - st["ts"] > self._ttl:
                self._state.pop(trace_id, None)
                return None
            return st

    def _clear(self, trace_id: str) -> None:
        with self._lock:
            self._state.pop(trace_id, None)


# ---------------------------------------------------------------------------
# hcTools 参数解析（不再 mock：把 (query, domain) 交给 hcTools，拿最终 tool+params）
# ---------------------------------------------------------------------------
# hcAgent 内部域 → hcTools domain_key。当前只启用了 vod（其余暂不接）。
_HCTOOLS_DOMAINS = {"vod": "vod"}


async def _hctools_params(it: Intent) -> tuple[str, dict[str, Any]]:
    """对启用域调用 hcTools 拿最终参数；未启用/失败返回 ("", {})，沿用 mock 参数。"""
    domain_key = _HCTOOLS_DOMAINS.get(it.domain)
    if not domain_key:
        return "", {}
    try:
        return await hctools.predict(it.query, domain_key)
    except Exception:  # noqa: BLE001 hcTools 不可用不阻断编排
        return "", {}


# ---------------------------------------------------------------------------
# 响应组装
# ---------------------------------------------------------------------------
async def _build_steps(batch: list[Intent]) -> list[dict[str, Any]]:
    # 批内各 step 的 hcTools 调用相互独立，并发执行
    results = await asyncio.gather(*[_hctools_params(it) for it in batch])
    steps: list[dict[str, Any]] = []
    for it, (tool, params) in zip(batch, results):
        steps.append({
            "id": f"s{it.index}",
            "toolName": tool or it.tool or "execute",
            "parameters": params or {"query": it.query},
            "plan": None,
            "dependsOn": [f"s{it.dep_on}"] if it.depends else [],
            "retext": it.query,
        })
    return steps


_SM = TraceStateMachine()


async def build_response(req) -> dict[str, Any]:
    if req.data is None:
        return _error("missing data")
    query = (req.data.query or "").strip()
    if not query:
        return _error("empty query")
    trace_id = req.traceId or ""
    device_id = req.deviceId or ""
    history = req.data.toolHistory or []

    batch, stop = await _SM.tick(trace_id, query, history)
    steps = await _build_steps(batch)
    body = {
        "code": 200, "message": "success", "traceId": trace_id, "deviceId": device_id,
        "data": {
            "planId": f"plan_{uuid.uuid4().hex[:4]}",
            "schemaVersion": "1.0", "planType": "execute", "planConfidence": 0.9,
            "steps": steps,
        },
        "stop": stop,
    }
    if not steps and stop:
        body["data"]["final"] = True
    if req.data.debug is True:
        body["data"]["debug"] = {
            "planId": body["data"]["planId"],
            "intents": [{"index": i.index, "query": i.query, "domain": i.domain,
                         "tool": i.tool, "depends_on_previous": i.depends} for i in batch],
        }
    return body


def _error(message: str) -> dict:
    return {"code": 400, "message": message, "data": {"steps": []}, "stop": True}


def reset() -> None:
    """测试用：清空状态机。"""
    with _SM._lock:
        _SM._state.clear()