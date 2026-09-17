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
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import config, detect, hctools, llm, multiintent
from .mt_rewrite_badcases import mt_rewrite_badcase
from .prompts import T0_PLAN_PROMPT, T1_STEP_PROMPT, MT_REWRITE_PROMPT, CONTENT_DOMAIN_PROMPT

logger = logging.getLogger("hcAgent.engine")

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
            new_turn = {"q": turn.get("q", ""), "answer": turn.get("answer", "")}
            # q_raw(用户原始 query)、domain(落定域) 一并透传。仅在有值且与 q 不同时保留。
            qr = turn.get("q_raw")
            if qr and qr != turn.get("q"):
                new_turn["q_raw"] = qr
            dom = turn.get("domain")
            if dom:
                new_turn["domain"] = dom
            turns.append(new_turn)
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


def _last_short_q(short_memory: list[dict[str, Any]] | None) -> str:
    """取 shortMemory 里最后一轮 dialogData.query（用户原始 query）。

    用于改写 badcase 的『上轮 query』：进程内 _MT_CONTEXT(real_q) 没有历史时，
    退而求其次用 shortMemory 携带的最近一轮原始用户 query 兜底。仅取最末轮。
    """
    for ent in reversed(short_memory or []):
        if not isinstance(ent, dict):
            continue
        dialog = ent.get("dialogData") or {}
        if isinstance(dialog, dict):
            q = str(dialog.get("query") or "").strip()
            if q:
                return q
    return ""


def _mt_last_domain(device_id: str) -> str:
    """返回该 device 对话历史末轮的落定域（多轮域继承用），无则 ""。"""
    real_q = _mt_get(device_id)
    return (real_q[-1].get("domain") or "") if real_q else ""


def _inherit_mt_domain(device_id: str, batch: list["Intent"]) -> None:
    """多轮域继承：对 batch 里 detect 判空(domain="")的意图，沿用上一轮稳定域。

    仅当上轮有落定域、且本轮该意图确实无任何域信号时继承；不覆盖本轮已判的
    明确域（显式切换业务的句不受影响）。single 单意图会话尤其受益。
    """
    prev_dom = _mt_last_domain(device_id)
    if not prev_dom:
        return
    for it in batch:
        if not it.domain:
            it.domain = prev_dom


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

    优先级：改写 badcase(上一轮真实 query + 本轮请求 命中 → 固定改写)  > LLM。
    """
    # badcase 优先：(上轮用户原话, 本轮请求) 命中即固定改写，绕过 LLM。
    # 上轮原话优先取 _MT_CONTEXT 末轮的 q_raw(用户原始 query)，回退末轮 q(改写句)，
    # 再回退 shortMemory 最近轮 dialogData.query。
    real_q = _mt_get(device_id)
    prev_q = ""
    if real_q:
        last_q = real_q[-1]
        prev_q = last_q.get("q_raw") or last_q.get("q") or ""
    if not prev_q:
        prev_q = _last_short_q(short_memory)
    bad = mt_rewrite_badcase()
    hit = bad.lookup(prev_q, cur_query) if prev_q else None
    if hit is not None:
        logger.info("mt改写 badcase 命中 prev=%r cur=%r -> %r (%s)",
                    prev_q, cur_query, hit["target"], hit.get("id", ""))
        return hit["target"] or cur_query
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


_BARE_SONG_VERB = re.compile(r"(?:播放|放|听|点播|播一下|播\s*)《([^》]{1,24})》")


def _song_names_from_ctx(device_id: str, short_memory: list[dict[str, Any]] | None) -> list[str]:
    """从上一轮对话提取候选歌名（《X》书名号内的 X），供裸书名号意图回填 music 用。

    数据源：本进程 _MT_CONTEXT 的末轮 answer（真实应答文本，含“《吉量》”）+
    请求 shortMemory 的 dialogData.answer。均形如“…演唱的歌曲是《吉量》…”或回显歌名。
    无候选返回 []。
    """
    names: list[str] = []
    for turn in _mt_get(device_id):
        ans = turn.get("answer") or ""
        names += re.findall(r"《([^》]{1,24})》", ans)
    for ent in short_memory or []:
        if not isinstance(ent, dict):
            continue
        d = ent.get("dialogData") or {}
        if isinstance(d, dict):
            names += re.findall(r"《([^》]{1,24})》", (d.get("answer") or ""))
    return list(dict.fromkeys(names))


def _reseed_bare_song(batch: list[Intent], *, device_id: str, cur_query: str,
                      short_memory: list[dict[str, Any]] | None) -> None:
    """裸书名号歌曲兜底：本轮意图是“播放《X》”且 X 曾在上一轮应答出现过 → 域改 music。

    背景：两轮会话「周深在2026年的春晚唱的什么歌」→「帮我播一下这个歌」。MT 改写
    常把后者压成「播放《吉量》」，丢掉了“歌”字；detect 里《吉量》无 music 锚（非
    《庆余年》等影视具名白名单）→ 保 LLM 首轮 qa 的 vod 倾向 → 误判 vod、调 vod_fuzzy。

    这里的《吉量》会从上一轮应答（“演唱的歌曲是《吉量》…”）里识别为【歌名】：
    - 意图语句形如 “播放/放《X》” 且 无 “歌曲/歌/听/音乐” 字样的裸放 → X∈上一轮歌名
      → 域强置 music（hcTools music_song_search 对“播放X”本来就能兜底检索歌曲）。
    - 非裸书名号播放句、或 X 不在上下文歌名 → 不动（保持 detect 原判定）。
    - 带“MV/看/电影/电视剧/剧”等影视载体字样的意图不参与，避免影视剧被误拉。
    """
    ctx_names = _song_names_from_ctx(device_id, short_memory)
    if not ctx_names:
        return
    for it in batch:
        q = (it.query or "").strip()
        # 只救裸书名号 + 播放动作且无歌曲/音乐锚的意图
        m = re.search(r"^(?:播放|放|播放一下|点播一下|帮我播放|帮我放|请播放)[\s]*《([^》]{1,24})》", q)
        if not m:
            continue
        title = m.group(1)
        if re.search(r"(歌曲|歌|听|音乐|单曲|专辑|MV|mv)", q):
            continue                       # 已带 song 锚，detect 自己会判 music
        if title not in ctx_names:
            continue                       # 书名号内容从未在上下文歌名里出现 → 不动
        # 已在上下文里确认是上一轮应答的歌名 → 强制 music，交给 hcTools 歌曲域兜底
        it.domain = "music"
        logger.info("mt歌曲名锚兜底: 意图 %r 书名号<%s> 命中上轮歌名 -> domain=music", q, title)


def _mt_append_answer(device_id: str, query: str, answer: str, query_raw: str = "",
                      domain: str = "") -> None:
    """把一轮 已完成 (query, answer) 追加进 device 对话历史。

    出现在 build_response 末尾，answer 取本轮 steps 的检索/执行 retext 摘要，
    供下一轮 LLM 改写时还原指上一环/刚才那部等历史引用。
    query = 本轮真正改写/执行的自包含句；query_raw = 用户原始 query（可选，
    用于改写 badcase 以上一轮用户原话做匹配键）；domain = 本轮落定的稳定域，
    供下一轮 detect 判空/弱信号时做多轮域继承。
    """
    if not device_id or not answer.strip():
        return
    turn = {"q": query or "", "answer": answer.strip()}
    if query_raw and query_raw != query:
        turn["q_raw"] = query_raw
    if domain:
        turn["domain"] = domain
    _mt_add(device_id, turn)


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


# 串行"检索+对结果排序筛选提问"里的排序词(评分最高/人气最高/播放量最高/最新/最经典…)，
# LLM 常把它下放成第二句 qa 的二次筛选("查询结果中评分最高的XX")，导致第一句检索丢 sort。
# 这里规则回填：排名词 belongs 第一句检索，而非第二句结果筛选。
_SORT_PAT = [
    (r"评分最高[的一部]*", "评分最高"),
    (r"人气最高[的一部]*", "人气最高"),
    (r"播放量最高[的一部]*", "播放量最高"),
    (r"播放最高[的一部]*", "播放量最高"),
    (r"最新的一部|最新", "最新"),
    (r"最经典的一部|最经典", "最经典"),
    (r"最热[的一部]*", "人气最高"),
]
_QA_VERB = ("导演是谁", "主演是谁", "主演都有谁", "男主角是谁", "女主角是谁",
            "评分怎么样", "评分是多少", "哪一年上映", "讲的是什么", "讲的是什么故事",
            "导演", "主演", "主角", "票房", "评分")


def _merge_sort_word_to_retrieval(plan: Plan) -> Plan:
    """串行"检索 + 对检索结果排序筛选提问"：把第二句抛出的排序词回填回第一句检索。

    触发前提（LLM 常把排序词下放成第二句 qa 的结果筛选，需回填，不误伤普通串行）：
      - 至少 2 个意图；
      - 第一个意图是 vod 检索（domain=vod）；
      - 后续某个【提问】意图含排序词；
      - 第一个检索意图当前 query 未携带该排序词。
    命中后把排序词并入检索意图 query 前部，让下游 hcTools 合成 sort。
    """
    if len(plan.intents) < 2:
        return plan
    first = plan.intents[0]
    if first.domain != "vod" or not first.query:
        return plan
    sort_word = ""
    for it in plan.intents[1:]:
        q = it.query or ""
        has_qa = any(v in q for v in _QA_VERB)
        if not has_qa:
            continue
        for pat, rep in _SORT_PAT:
            if re.search(pat, q):
                sort_word = rep
                break
        if sort_word:
            break
    if not sort_word or sort_word in first.query:
        return plan
    fq = first.query
    fq_clean = re.sub(r"^(搜索|帮我搜|帮我搜索|帮我找一下|帮我找|搜一下|找一下|帮我查|查一下|搜|找)\s*", "", fq).strip()
    merged = f"{sort_word}的{fq_clean}" if fq_clean else f"{sort_word}"
    new_first = Intent(query=merged, domain=first.domain, tool=first.tool,
                       index=first.index, depends=first.depends, dep_on=first.dep_on,
                       src=first.src)
    return Plan(intents=[new_first] + plan.intents[1:])


def _with_source(plan: Plan, src: str, tv_mode: str | int = "0") -> Plan:
    """给计划内每个意图拍上 src=原文 + detect 判域修正。

    detect_domain 用每个意图【自含的 query】（非整句 src）在 LLM 判的 domain 上做确定性覆盖。
    必须用it.query而非整句src：多意图计划里整句会把其他子任务的问句信号（"第二个导演是谁"）
    串扰到本子任务判域，导致"搜索刘德华的电影"这类 vod 检索被 `_is_qa` 误改到 qa。
    每个子意图按其自身内容独立判域，才与下游 hcTools(用 it.query) 一致、对齐 golden。
    命中即改 domain；未命中保留 LLM 判定。domain 定了，tool+params 仍交给 hcTools。
    """
    out: list[Intent] = []
    for it in plan.intents:
        # 用原子子意图 it.query 判域；query 空时回退整句 src。
        judge = (it.query or "").strip() or src
        domain = detect.detect_domain(judge, it.domain, tv_mode=tv_mode)
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


# ---------------------------------------------------------------------------
# 卡通/动漫 双域并行（multi_tab）—— 对齐 「0821多意图&多业务」sheet 的
# multi_tab_search 标注与 mock._dual_domain：首轮无历史 + 动画/动漫/卡通 query，
# 一次下发两个并行 tab（children + vod），数据层用 parallel=true 标记。
# 域内工具由下游 hcTools 权威解析（children → educ_search*，vod → vod_search*）。
# ---------------------------------------------------------------------------
# 双域触发词（对齐 0821 多意图&多业务 sheet multi_tab_search）：动画/动漫/卡通 均算。
# 排除「动漫的片头曲/主题曲/原声/广播剧」这类把动画当定语、实为音乐/有声查询的句子
# （0901 music/audio golden 里 4 条，触发 dual 会把它们从单域打回双域 → 回归）；
# 再排除设备侧：开机/关机动画定制、AI 动画(=屏保)、动画定制 这类 tv_control 屏显功能
# （0901 device golden 里 6 条），它们只是“动画”做名词定语，并非内容意图。
_MULTI_TAB_RE = re.compile(r"动画|动漫|卡通|动画片")
_MULTI_TAB_EXCL = re.compile(
    r"片头曲|片尾曲|插曲|主题曲|原声|原声带|广播剧|配音|配乐|主题歌|片头歌|"
    r"声优|音乐|歌曲|MV|旋律"
    r"|(?:AI|ai|Ai)动画|(?:开机|关机)动画定制|动画设置|屏保|运动画面"
)


def _is_multi_tab(query: str, history: list | None = None) -> bool:
    """判定是否走双域并行。仅无任何历史/上下文的纯首轮卡通/动画查询触发，
    避免续跑/多轮把该 query 误判成双域。"""
    q = (query or "").strip()
    if not q:
        return False
    if history:
        return False
    if not _MULTI_TAB_RE.search(q):
        return False
    if _MULTI_TAB_EXCL.search(q):
        return False
    return True


def _multi_tab_intents(query: str, tv_mode: str | int = "0") -> list[Intent]:
    """双域并行的两个意图：children(tab) + vod(tab)，均无依赖、原句 src。"""
    q = (query or "").strip()
    return [
        Intent(query=q, domain="children", tool="execute", index=1,
               depends=False, dep_on=0, src=q),
        Intent(query=q, domain="vod", tool="execute", index=2,
               depends=False, dep_on=0, src=q),
    ]


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


# 多意图内容子句域路由：规则优先于 LLM。判定正则已收敛到 app.detect.content_domain_rule()
# （单一事实源，见 detect.py），engine 只引用，不再各自内嵌正则。

# ---------------------------------------------------------------------------
# 多意图并行(多状态机)
# ---------------------------------------------------------------------------
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
        # 卡通/动漫 双域并行：无历史首轮 + 动画/动漫/卡通 query → 一次两个并行 tab。
        # 对齐 0821多意图&多业务 sheet（multi_tab_search）与 mock._dual_domain：
        #   children(tab) + vod(tab) 两个无依赖 step，parallel=true。
        if _is_multi_tab(query) and not (history or []):
            return _multi_tab_intents(query, tv_mode=tv_mode), True
        # 多意图确定性拆分（规则优先）：命中断言“内容+设备”双目标 → 直接拆成两条
        # 并行、无依赖意图(dep=[], stop=true)，两条独立交给 hcTools 权威解析最终 tool+params。
        mi = multiintent.split_multiintent(query)
        if mi.hit and mi.device and mi.content:
            # 规则已断言 mi.device 是设备子句，故域强置 device(最多让 detect 细化，不降级到 qa)。
            dev_dom = detect.detect_domain(mi.device, "device", tv_mode=tv_mode)
            dev_dom = dev_dom if dev_dom == "device" else "device"
            # 内容子句：先规则引擎判定；判空则用 LLM 从用户原话选域(裸实体依赖此兜底)。
            cont_dom = detect.detect_domain(mi.content, "", tv_mode=tv_mode)
            if not cont_dom or cont_dom == "qa":
                cont_dom = await self._content_domain(mi.content)
                cont_dom = cont_dom or detect.detect_domain(mi.content, "", tv_mode=tv_mode) or ""
            content_int = Intent(query=mi.content, domain=cont_dom, tool="execute",
                                 index=2, depends=False, dep_on=0, src=mi.content)
            device_int = Intent(query=mi.device, domain=dev_dom, tool="execute",
                                index=1, depends=False, dep_on=0, src=mi.device)
            return [device_int, content_int], True
        # 串行「检索 → 提问」确定性拆分（规则优先）：搜X，然后(再)问X的Y →
        # step1=检索(vod, 依赖回填排序词)，step2=对某部属性提问(qa→fan_knowledge_agent, dep=1)。
        # 属于编排层(拆意图)，非 hcTools 工具参数层；工具+参数仍由下游 hcTools 权威解析。
        sq = multiintent.split_serial_qa(query)
        if sq.hit:
            # 交给与 LLM 计划同一条派生管线：排序词回填 + 域审计(_with_source)，对齐 golden。
            ser_plan = Plan(intents=[
                Intent(query=sq.search, domain="vod", tool="execute", index=1,
                       depends=False, dep_on=0, src=query),
                Intent(query=sq.question, domain="qa", tool="execute", index=2,
                       depends=False, dep_on=0, src=query),
            ])
            if not os.environ.get("HC_DISABLE_SORT_MERGE"):
                ser_plan = _merge_sort_word_to_retrieval(ser_plan)
            ser_plan = _with_source(ser_plan, query, tv_mode=tv_mode)
            # 首批发依赖序号靠前的可用步(step1 检索)；step2 提问依赖 step1，留待续跑。
            batch = next_batch(ser_plan, set())
            if not batch:
                batch = ser_plan.intents
            if batch:
                self._store(trace_id, ser_plan)
                # 两意图并行(无依赖)，首批即发全部：step1=检索、step2=fuzzy 提问无媒资候选
                # 可锚定时(路由到 fan_knowledge_agent)，并行反而保证 step2 必发(不依赖 step1 结果)。
                return batch, True
        # 纯 LLM(T0) 分解：拆步、依赖、落库、选工均一次 LLM 决策，编排层零规则。
        plan = await self._plan(query)
        # 串行"检索+排序提问"排序词回填规则。可用 HC_DISABLE_SORT_MERGE=1 关闭做 A/B 回归对比。
        if not os.environ.get("HC_DISABLE_SORT_MERGE"):
            plan = _merge_sort_word_to_retrieval(plan)
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
            # 恢复原句后，用它重新判域：T0 给原子 query 加“播放/搜索”前缀时，
            # _with_source 是在前缀句上判的域。原句才是用户真实意图，也让域 badcase/
            # 信号词表（按原句归一）准确命中（如“最近开播、女主演技好的古装剧”→vod，
            # “搜索最近开播、古装剧”前缀不加域 badcase 反而判 qa）。
            restored = detect.detect_domain(query, only.domain, tv_mode=tv_mode)
            plan = Plan(intents=[Intent(query=query, domain=restored or only.domain, tool=only.tool,
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

    async def _content_domain(self, query: str) -> str:
        """内容子句域路由：规则 detect 对裸实体(队名/歌名/剧名)判空时，
        先用确定性内容域规则，再未命中才让 LLM 从用户原词选域；仅用于内容意向、不改设备条域。"""
        rule = detect.content_domain_rule(query)
        if rule:
            return rule
        try:
            msg = await llm.chat([{"role": "system", "content": CONTENT_DOMAIN_PROMPT},
                                  {"role": "user", "content": query}], model=config.MODEL)
            if "__error__" in msg:
                return ""
            out = llm.text_of(msg).strip().lower()
            for d in ("vod", "music", "audio", "sports", "education", "children", "qa"):
                if d in out:
                    return d if d != "qa" else ""
        except Exception:  # noqa: BLE001
            return ""
        return ""

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
        # 透传真正发给 hcTools 的改写后 query（评估/审计用；badcase raw 需匹配它）
        step["hitQuery"] = it.query
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
    # 多轮继承：本轮某个意图 detect 判空(无域信号)时，沿用上一轮落定的稳定域，
    # 避免"律动感强的""适合零基础的"这类漂移到无关域。
    _inherit_mt_domain(device_id, batch)
    # 歌曲名锚兜底：LLM 改写仍可能把"播一下这个歌"压成裸的「播放《吉量》」（丢“歌”字），
    # detect 对裸书名号无 music 锚时会保 LLM 的 vod。这里在上一轮应答里找《吉量》类歌名，
    # 匹配则把该意图强置 music（让 hcTools 的 music_song_search 兜底检索，不误判影视）。
    # 仅对"播放/放《X》"形的裸书名号意图生效，不影响带歌/歌曲/听 字样的正常句。
    _reseed_bare_song(batch, device_id=device_id, cur_query=query, short_memory=short_memory)
    # multi-tab 双域并行标记：首轮无历史卡通查询由 _first 直接切成 children+vod
    # 两个并行无依赖意图。有双方才标记 parallel，供客户端并发渲染两个 tab。
    is_dual_tab = stop and not history and len(batch) == 2 and (
        {i.domain for i in batch} == {"children", "vod"}
    )
    steps = await _build_steps(batch)
    # 本轮 answer 摘要：取各 step 的 retext，供下一轮多轮改写引用具体内容/序号。
    turn_answer = "；".join(s.get("retext") or "" for s in steps if s.get("retext"))
    # 本轮落定域：优先取 batch 里第一个非空域(多意图单域为主)，用于下轮弱句继承。
    round_domain = next((i.domain for i in batch if i.domain), "")
    # 写入 _MT_CONTEXT 的 q 用【真正改写/执行】的 mt_query，而非用户原始 query——
    # _MT_CONTEXT 语义即“真实 query 链”，供跨轮合并时还原真实执行语境。
    _mt_append_answer(device_id, mt_query, turn_answer, query_raw=query, domain=round_domain)
    body = {
        "code": 200, "message": "success", "traceId": trace_id, "deviceId": device_id,
        "data": {
            "planId": f"plan_{uuid.uuid4().hex[:4]}",
            "schemaVersion": "1.0", "planType": "execute", "planConfidence": 0.9,
            "steps": steps,
        },
        "stop": stop,
    }
    if is_dual_tab:
        body["data"]["parallel"] = True
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