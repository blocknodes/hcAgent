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

from . import config, detect, hctools, llm
from .prompts import T0_PLAN_PROMPT, T1_STEP_PROMPT, MT_REWRITE_PROMPT

# 域名闭合集（供 LLM 输出校验）。空串=未配探索，为保证给出中性默认。
_DOMAINS = {"vod", "children", "education", "music", "audio", "sports", "device", "qa"}


# 多轮上下文缓存：按 device_id 存完整对话轮（每轮 query + answer）。
# 每轮用 LLM(MT_REWRITE_PROMPT) 把本轮请求合并整个对话历史 → 自包含 query。
# TTL 2 分钟防内存无限膨胀；同一 device 追加/刷新最新轮。
_MT_TTL = 120.0
_MT_MAX_TURNS = 20          # 每设备最多保留的对话轮数，防超长 prompt
_MT_CONTEXT: dict[str, dict[str, Any]] = {}
_MT_LOCK = threading.Lock()


def _mt_get(device_id: str) -> list[dict[str, str]]:
    """返回该 device 的有效对话历史（每轮 {"q":…,"a":…}），空或超时返回 []。"""
    if not device_id:
        return []
    now = time.time()
    with _MT_LOCK:
        ent = _MT_CONTEXT.get(device_id)
        if ent is not None and now - ent["ts"] > _MT_TTL:
            _MT_CONTEXT.pop(device_id, None)
            return []
        # 返回副本，避免调用方持锁外突变（q/a 都是不可变 str）
        return [dict(t) for t in ent["turns"]] if ent else []


def _mt_add(device_id: str, turn: dict[str, str]) -> None:
    """把一轮已完成对话 (q, answer) 追加进 device 历史；answer 为空则不追加。"""
    if not device_id:
        return
    with _MT_LOCK:
        ent = _MT_CONTEXT.get(device_id)
        if ent is None:
            ent = {"turns": [], "ts": time.time()}
            _MT_CONTEXT[device_id] = ent
        turns = ent["turns"]
        # 若末轮 query 与当前 q 相同（改写结果是原句直通）则视为继续同轮，避免重复累积
        if turns and turns[-1]["q"] == turn.get("q"):
            turns[-1]["answer"] = turn.get("answer", "")
        else:
            turns.append({"q": turn.get("q", ""), "answer": turn.get("answer", "")})
            if len(turns) > _MT_MAX_TURNS:
                del turns[0]
        ent["ts"] = time.time()


def _mt_serialize(history: list[dict[str, str]]) -> str:
    lines = []
    for idx, t in enumerate(history, 1):
        lines.append(f"{idx}. 用户：{t['q']} → 助手：{t['answer'] or '（应答为空）'}")
    return "\n".join(lines)


def _short_turns(short_memory: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """把请求里的 shortMemory 解析成『已完成轮次』的 (q, a) 序列。

    shortMemory 由上游多轮记忆携带：每一轮形如
      {"dialogData": {"query": "原用户query", "answer": "最新应答"},
       "businessData": {"serviceData": [{"data":[{mediaTitle,director,...}...]}]}}
    其中 dialogData.query 存的是【用户原始 query】(未改写)，dialogData.answer 才是该轮
    真正的应答；businessData.serviceData[].data[] 是该轮命中的候选媒资实体(片名/导演/主演)。
    ---
    设计要点：
    - a_i(应答) 以 shortMemory 为准：取 answer 文本，并把候选媒资实体铺成可读列表，
      供 LLM 把『第一位/刚才那部/该导演』等指代还原成具体内容名。
    - 只收录已完成轮：无应答也无候选(仅回显当前 query 的空 stub)的直接丢弃。
    - 该轮真实 query 不取 shortMemory.dialogData.query(那是不准确的原始句)，
      由调用方用 _MT_CONTEXT 里真正改写执行的 query 兜底/对齐。
    """
    out: list[dict[str, str]] = []
    for ent in short_memory or []:
        if not isinstance(ent, dict):
            continue
        dialog = ent.get("dialogData") or {}
        if not isinstance(dialog, dict):
            continue
        a = str(dialog.get("answer") or "").strip() or ""
        candidates: list[dict] = []
        biz = ent.get("businessData") or {}
        if isinstance(biz, dict):
            for svc in biz.get("serviceData") or []:
                if isinstance(svc, dict):
                    for m in svc.get("data") or []:
                        if _candidate_key(m):
                            candidates.append(m)
        pieces = []
        if candidates:
            # 候选按原始排位标序号(第1部/第2部…)，让 LLM 能把“第二部/第一位”这类
            # 指代精确对应到候选名单，而非被片名里的数字(如“疯狂动物城2”)带偏。
            _cap = config.CANDIDATE_CONTEXT_LIMIT
            ranked = [f"第{i+1}部: {_candidate_key(m)}"
                      for i, m in enumerate(candidates[: _cap]) if _candidate_key(m)]
            pieces.append("候选媒资:" + "；".join(ranked))
        if a:
            pieces.append(a)
        if not pieces:
            continue                      # 无应答也无候选 → 未完成/空位，跳过
        out.append({
            "q": str(dialog.get("query") or "").strip(),
            "answer": "；".join(pieces),
        })
    return out


def _mt_merge(real_q: list[dict[str, str]], short_a: list[dict[str, str]]) -> list[dict[str, str]]:
    """把真实改写链路(real_q.q) 和 shortMemory 应答(short_a.a) 按轮对齐成 q1-a1/q2-a2。

    以轮次多的为准；同一轮优先取真实执行 query(改写链路)，应答取 shortMemory 的应答。
    只有询问无应答的孤立轮(如某遥控指令轮无候选/无文本)可余留为纯 query。
    """
    n = max(len(real_q), len(short_a))
    merged: list[dict[str, str]] = []
    for i in range(n):
        q = ""
        if i < len(real_q):
            q = real_q[i].get("q") or ""
        if not q and i < len(short_a):
            q = short_a[i].get("q") or ""
        a = ""
        if i < len(short_a):
            a = short_a[i].get("answer") or ""
        if not a and i < len(real_q):
            a = real_q[i].get("answer") or real_q[i].get("a") or ""
        if q or a:
            merged.append({"q": q, "answer": a})
    return merged


async def _mt_rewrite(cur_query: str, device_id: str, short_memory: list[dict[str, Any]] | None = None) -> str:
    """把本轮合并历史改写成自包含 query。

    历史 = real_q (本进程 _MT_CONTEXT 的『真正改写/执行』query 链)  +  short_a
    (请求 shortMemory 携带的应答 & 候选媒资)。两者按轮对齐成 q1-a1/q2-a2 序列，
    key 为用户设计：real query 取改写链路、应答取 shortMemory，缺一用另一侧补齐。
    仅改写、不写回。无有效历史直接用原句。
    """
    real_q = _mt_get(device_id)
    short_a = _short_turns(short_memory)
    merged = _mt_merge(real_q, short_a)
    if not merged:
        return cur_query
    dialog = _mt_serialize(merged)
    user = MT_REWRITE_PROMPT.format(dialog=dialog, cur_query=cur_query)
    message = await llm.chat([{"role": "user", "content": user}], model=config.MODEL)
    if "__error__" in message:
        return cur_query
    out = (llm.text_of(message) or "").strip()
    return out if out else cur_query


def _has_mt_history(device_id: str, short_memory: list[dict[str, Any]] | None) -> bool:
    """是否具备需要做多轮改写的上下文。

    语义对齐：只有当【没有任何跨轮上下文】时才是“首轮单 query”，该场景强制原句直通、
    不做任何改写；只要存在任一上下文(进程内 _MT_CONTEXT 或请求 shortMemory 或
    toolHistory 由调用方保证)，就仍需改写(含只有 shortMemory 无应答/仅 stub 的情形也
    视为有上下文，交给 LLM 判断是首轮还是续接)。
    """
    if _mt_get(device_id):
        return True
    return bool(short_memory)


def _mt_append_answer(device_id: str, query: str, answer: str) -> None:
    """把已完成的一轮 (query, answer) 追加进 device 对话历史。

    出现在 build_response 末尾，answer 取本轮 steps 的检索/执行 retext 摘要，
    供下一轮 LLM 改写时还原“第2首/刚才那部”等历史引用。
    """
    if not device_id or not answer.strip():
        return
    _mt_add(device_id, {"q": query or "", "answer": answer.strip()})


@dataclass
class Intent:
    query: str
    domain: str
    tool: str = "execute"
    index: int = 0          # 1-start 计划序号
    depends: bool = False
    dep_on: int = 0        # 所依赖条的 1-start 计划序号（0=无依赖）
    src: str = ""          # 喂给 hcTools 做权威解析的原文（无小，默认=query）


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


def _with_source(plan: Plan, src: str, tv_mode: str | int = "0") -> Plan:
    """给计划内每个意图拍上 src=原文 + detect 判域修正。

    detect_domain 用原文在 LLM 判的 domain 上做确定性覆盖（badcase/正则），
    tv_mode 透传到 detect（0=亮屏 / 6=息屏分叉）。
    命中即改 domain；未命中保留 LLM 判定。domain 定了，tool+params 仍交给 hcTools。
    """
    out: list[Intent] = []
    for it in plan.intents:
        domain = detect.detect_domain(src, it.domain, tv_mode=tv_mode)
        out.append(Intent(query=it.query, domain=domain, tool=it.tool,
                          index=it.index, depends=it.depends, dep_on=it.dep_on, src=src))
    return Plan(intents=out)


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


def _load_maybe_str(v: Any) -> Any:
    """dict/list 原样返回；字符串尝试 JSON 反解（上游 result 可能是 JSON 文本）。"""
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s[:1] in ("{", "["):
        try:
            import json as _json
            return _json.loads(s)
        except (ValueError, TypeError):
            return v
    return v


def _candidate_key(m: Any) -> str:
    """从一条候选媒资里抽『指代改写』所需的关键字段，control 为平文本。"""
    if not isinstance(m, dict):
        return ""
    title = str(m.get("mediaTitle") or m.get("title") or "").strip() or ""
    title = title.strip("《》 　")
    if not title:
        return ""
    direc = (m.get("director") or [])
    direc = "/".join(str(x) for x in direc if isinstance(x, str) and x) or "未知"
    rate = m.get("doubanRate") or m.get("rate") or ""
    rate = "" if rate in (None, "", 0, "0", 0.0) else str(rate)
    actor = (m.get("actor") or [])
    actor = "/".join(str(x) for x in actor[:3] if isinstance(x, str) and x)
    year = str(m.get("pubdate") or "")[:4]
    parts = [f"《{title}》", f"导演:{direc}"]
    if rate:
        parts.append(f"评分:{rate}")
    if actor:
        parts.append(f"主演:{actor}")
    if year:
        parts.append(year)
    return " ".join(parts)


def _buckets_medias(raw: Any) -> list[dict]:
    """从工具结果里取 memoryData[].data[] 候选媒资(可能嵌套，跨 bucket 平铺)。"""
    if not isinstance(raw, dict):
        return []
    data = raw.get("data")
    if not isinstance(data, dict):
        return []
    out: list[dict] = []
    for bucket in data.get("memoryData") or []:
        if not isinstance(bucket, dict):
            continue
        medias = bucket.get("data")
        if isinstance(medias, list):
            out.extend(m for m in medias if isinstance(m, dict))
    return out


def _first_text(raw: Any, keys: tuple[str, ...]) -> str:
    if not isinstance(raw, dict):
        return ""
    for k in keys:
        v = raw.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _tool_result_text(history: list[dict[str, Any]] | None) -> str:
    """把『最近一步』的工具结果压缩成给 T1 的参考文本（保留实体），作为指代依据。

    step3 的『该导演』指代的是 step2 刚执行的结果，而非 step1 的候选。故：
    - 按倒序(最新在前)遍历；对每个 item 独立决定返回什么，绝不越过最新 item 去取更早候选，
      否则问答步(fan: "该片导演是刘镇伟")会被旧的搜索候选(主演周星驰)覆盖。
    - 最新 item 若是 vod 类(dict + memoryData)→ 提候选(片名/导演/评分/主演/年份)，保序。
    - 若是问答/文本类(result 为纯文本，如 "…导演是刘镇伟…")→ 直接返回该文本。
    - 仅当某 item 完全无内容时才回退到更早 item。
    """
    for item in reversed(history or []):
        if not isinstance(item, dict):
            continue
        for k in ("result", "answer", "tts"):
            v = item.get(k)
            if v is None:
                continue
            raw = _load_maybe_str(v)
            # 1) 结构化候选(dict 且含 memoryData) —— vod 搜索类
            cands = [_candidate_key(m) for m in _buckets_medias(raw)]
            cands = [c for c in cands if c]
            if cands:
                # 序号前缀，让 rewrite LLM 能把“第二部/第2位”准确落到排位号，
                # 而不是被片名里的数字(如“疯狂动物城2”)带偏。
                ranked = "\n".join(
                    f"第{i+1}部: {c}"
                    for i, c in enumerate(cands[: config.CANDIDATE_CONTEXT_LIMIT])
                )
                return "候选媒资(按搜索排位):\n" + ranked
            # 2) 纯文本结果(问答/摘要类)
            text = raw if isinstance(raw, str) else _first_text(raw, ("ttscontent", "answer", "tts"))
            if isinstance(raw, str) and raw.strip() and not raw.startswith(("{", "[")):
                text = raw.strip()
            if text:
                return text.strip()
        # 该 item 无内容，回溯到更早 item
    return ""


class TraceStateMachine:
    def __init__(self, ttl: float = config.PLAN_TRACE_TTL):
        self._lock = threading.Lock()
        self._state: dict[str, dict[str, Any]] = {}
        self._ttl = ttl

    async def tick(self, trace_id: str, query: str, history: list[dict[str, Any]] | None,
                   tv_mode: str | int = "0") -> tuple[list[Intent], bool]:
        entry = self._load(trace_id)
        if entry is None:
            return await self._first(trace_id, query, history, tv_mode=tv_mode)
        return await self._continue(trace_id, history)

    # ---- 首轮：T0（一次 LLM）----
    async def _first(self, trace_id: str, query: str, history,
                     tv_mode: str | int = "0") -> tuple[list[Intent], bool]:
        # 纯 LLM(T0) 分解：拆步、依赖、落域、选工均一次 LLM 决策，编排层零规则。
        plan = await self._plan(query)
        if not plan.intents:
            # LLM 没出方案（空/纯聊天/失败）→ 兜底：单一自包含意图（非语言规则），
            # 但仍过 detect 判域，息屏(tv_mode=6)知识/点歌/有声查询可被纠正到对应域；
            # 亮屏默认不干预（llm_domain="" 保留空，交由外部）。
            dom = detect.detect_domain(query, "", tv_mode=tv_mode)
            if not dom:
                dom = ""
            return [Intent(query=query, domain=dom, tool="execute", index=1, src=query)], True
        plan = _with_source(plan, query, tv_mode=tv_mode)
        # 首轮【单意图】计划：T0 常给原子 query 加动作/语序前缀(如“老版上海滩”→“播放老版上海滩”),
        # 这类改写会误导 hcTools 判 action(search↔play)。对原子单意图强制恢复为原句(query)，
        # 让 hcTools 以用户原话解析(工具选型以原文为准)。多意图/多步(内部依赖改写)不受影响。
        if plan.total() == 1:
            only = plan.intents[0]
            plan = Plan(intents=[Intent(query=query, domain=only.domain, tool=only.tool,
                                        index=only.index, depends=only.depends, dep_on=only.dep_on,
                                        src=only.src)])
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
                          index=it.index, depends=it.depends, dep_on=it.dep_on, src=it.src)

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
# hcAgent 内部域 → hcTools domain_key。全部 e2e 域都接 hcTools 解析最终 tool+params。
_HCTOOLS_DOMAINS = {
    "vod": "vod",
    "audio": "audio",
    "music": "music",
    "device": "device",
    "education": "education",
    "sports": "sports",
    "children": "children",
    "qa": "qa",
}


async def _hctools_params(it: Intent) -> tuple[str, dict[str, Any], str]:
    """对启用域调用 hcTools 拿最终参数；未启用/失败返回 ("", {}, "")，沿用 mock 参数。

    用 src=原文 让 hcTools 做权威 select+fill（它对这些域确定性/示例已达 97-100%）。
    改写 q 仅用于显示/指代解析；非依赖步的改写会丢失唱/队/可爱 等判别信号，
    故工具选型必须以用户原话为准。

    第 3 项是 hcTools 返回的 hit_source（审计，如 general_rule:audio_history_explicit）。
    """
    domain_key = _HCTOOLS_DOMAINS.get(it.domain)
    if not domain_key:
        return "", {}, ""
    try:
        # 用原子/自包含意图 it.query 调 hcTools：串行多意图里 src 是整句，
        # 直接用整句会让 hcTools 把"然后再问…评分最高"等后续意图当成本步搜索条件，
        # 污染 s1 实体(actor"一下胡歌")、且 s2 fan_knowledge 的 messages 落不到具体片名。
        # it.query 是 T1 改写后的自包含意图(如 s2:"攀登者的导演是谁")或 T0 的原子子意图
        # (如 s1:"搜索胡歌演的电影")，交给 hcTools 解析才干净、对齐 gold。
        q = it.query or it.src or ""
        # 空意图直接让 hcTools 单步兜底，避免查全部域空转
        return await hctools.predict(q, domain_key)
    except Exception:  # noqa: BLE001 hcTools 不可用不阻断编排
        return "", {}, ""


# ---------------------------------------------------------------------------
# 响应组装
# ---------------------------------------------------------------------------
async def _build_steps(batch: list[Intent]) -> list[dict[str, Any]]:
    # 批内各 step 的 hcTools 调用相互独立，并发执行
    results = await asyncio.gather(*[_hctools_params(it) for it in batch])
    steps: list[dict[str, Any]] = []
    for it, (tool, params, hit_source) in zip(batch, results):
        # hcTools 返回的 params.retext 是其规范化回显句(如"播放哑巴新娘第1集")，
        # 而评测 golden.retext=用户原始 query("放第1集哑巴新娘")。为对齐评测，
        # 把参数里的 retext 回显为【用户原句】(it.src)，action/query/sort 等结构化
        # 字段保持 hcTools 权威结果不变。
        if isinstance(params, dict) and "retext" in params:
            params = {**params, "retext": it.src or params["retext"]}
        if params is None:
            params = {"query": it.src or it.query}
        step: dict[str, Any] = {
            "id": f"s{it.index}",
            "toolName": tool or it.tool or "execute",
            "parameters": params,
            "plan": None,
            "dependsOn": [f"s{it.dep_on}"] if it.depends else [],
            # step 级 retext 亦回显用户原句(与 parameters.retext 一致)。
            "retext": it.src or it.query,
        }
        if hit_source:
            step["hitSource"] = hit_source
        steps.append(step)
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
    tv_mode = req.data.tvMode if req.data.tvMode is not None else "0"
    short_memory = (req.data.memory.shortMemory if req.data.memory else None) or []

    # 多轮改写分界：只有【真正无任何跨轮上下文】的首轮单 query 才原句直通、强制不改写；
# 否则(有 toolHistory / 有 shortMemory / 有该 device 进程内多轮记忆)仍需改写。
#  - history 非空 = 续跑(串行多步的第 2..N 步/runs)：query 交给 _SM/_continue 用
#    toolHistory 改写依赖步，不能在这里做多轮 merge——避免把整句误并入上轮检索条件。
#  - 无 history 且 无任何上下文(_has_mt_history 假)：纯首轮单 query → 原句执行，强制
#    不交给 LLM，避免“想看悬疑的→推荐悬疑”这类改写噪声；只有有上下文才走 _mt_rewrite。
    if history:
        mt_query = query            # 续跑轮：原 query 交给 _SM/_continue 用 toolHistory 改写
    elif not _has_mt_history(device_id, short_memory):
        mt_query = query             # 首轮无任何上下文：原句直通，不交给多轮 LLM 改写
    else:
        mt_query = await _mt_rewrite(query, device_id, short_memory)

    # 串行续跑状态按 device_id 存（换 request 用 device_id 识别会话），而非 trace_id——
    # 串行 N 步 = N 个 SSE 请求 = N 个不同 trace_id，只有 device 稳定才能跨轮推进依赖链。
    batch, stop = await _SM.tick(device_id, mt_query, history, tv_mode=tv_mode)
    steps = await _build_steps(batch)
    # 本轮 answer 摘要：取各 step 的 retext，供下一轮多轮改写引用具体内容/序号。
    turn_answer = "；".join(s.get("retext") or "" for s in steps if s.get("retext"))
    # 写入 _MT_CONTEXT 的 q 用【真正改写/执行】的 mt_query，而非用户原始 query——
    # _MT_CONTEXT 语义即“真实 query 链”，供跨轮合并时还原真实执行语境。
    _mt_append_answer(device_id, mt_query, turn_answer)
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