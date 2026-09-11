"""Mock 编排逻辑：把 query 确定性拆步为 steps，无 LLM / 无域工具。

覆盖 server.py 的对外行为：
  · 空 query / 缺 data → code 400 错误响应
  · 卡通/动漫 双域并行（children + vod），parallel=True
  · 普通整句 → 按连接词拆步；依赖串行(dependsOn)，无依赖并行
  · 无步骤且结束 → data.final=True
"""
from __future__ import annotations

import re
import uuid
from typing import Any

from .models import SlowRequest

# 连接词：用于把整句拆成多个子意图（mock 版拆步器）。
_SPLIT_RE = re.compile(r"然后|接着|再|之后|，然后|,然后|;|；")
# 依赖判据：后续意图带指代（第N部/那部/该片/它）→ 依赖前一步结果。
_DEPEND_RE = re.compile(r"第[一二三四五六七八九十\d]+部|那部|那个|该片|它|上面|刚才")
# 卡通/动漫意图 → children + vod 双域并行短路。
_CARTOON_RE = re.compile(r"动漫|动画片|卡通")


def build_slow_response(req: SlowRequest) -> dict[str, Any]:
    query = (req.data.query or "").strip()
    if not query:
        return _error("empty query")

    trace_id = req.traceId or ""
    device_id = req.deviceId or ""

    history = req.data.toolHistory or []
    # 卡通双域：仅首轮无历史时短路。
    if _CARTOON_RE.search(query) and not history:
        return _dual_domain(query, trace_id, device_id)

    subqueries = _split(query)
    steps: list[dict[str, Any]] = []
    for n, sub in enumerate(subqueries, start=1):
        depends = n > 1 and bool(_DEPEND_RE.search(sub))
        steps.append(_step(n, sub, depends))

    body = _response(steps, stop=True, trace_id=trace_id, device_id=device_id)
    if not steps:
        body["data"]["final"] = True
    if req.data.debug is True:
        body["data"]["debug"] = {
            "planId": body["data"]["planId"],
            "intents": [
                {"index": i, "query": s, "depends_on_previous": bool(st["dependsOn"])}
                for i, (s, st) in enumerate(zip(subqueries, steps), start=1)
            ],
        }
    return body


def _split(query: str) -> list[str]:
    parts = [p.strip() for p in _SPLIT_RE.split(query) if p.strip()]
    return parts or [query]


def _step(n: int, sub: str, depends: bool) -> dict[str, Any]:
    return {
        "id": f"s{n}",
        "toolName": _mock_tool(sub),
        "parameters": {"query": sub},
        "plan": None,
        "dependsOn": [f"s{n - 1}"] if depends else [],
        "retext": sub,
    }


def _mock_tool(sub: str) -> str:
    """按关键词映射到一个 mock 的域工具名（仅示意）。"""
    if re.search(r"歌|音乐|听|专辑|歌曲", sub):
        return "music_search"
    if re.search(r"动画|少儿|儿歌|宝宝|动漫|卡通", sub):
        return "children_search"
    if re.search(r"电影|电视剧|影视|看|播放|片", sub):
        return "vod_search"
    if re.search(r"故事|有声|广播|评书", sub):
        return "audio_search"
    if re.search(r"设备|音量|亮度|开机|关机|投屏", sub):
        return "device_control"
    return "execute"


def _dual_domain(query: str, trace_id: str, device_id: str) -> dict[str, Any]:
    steps = [
        _step(1, query, depends=False),
        _step(2, query, depends=False),
    ]
    steps[0]["toolName"] = "children_search"
    steps[1]["toolName"] = "vod_search"
    body = _response(steps, stop=True, trace_id=trace_id, device_id=device_id)
    body["data"]["parallel"] = True
    return body


def _response(
    steps: list[dict[str, Any]],
    stop: bool,
    *,
    trace_id: str = "",
    device_id: str = "",
) -> dict[str, Any]:
    return {
        "code": 200,
        "message": "success",
        "traceId": trace_id,
        "deviceId": device_id,
        "data": {
            "planId": f"plan_{uuid.uuid4().hex[:4]}",
            "schemaVersion": "1.0",
            "planType": "execute",
            "planConfidence": 0.9,
            "steps": steps,
        },
        "stop": stop,
    }


def _error(message: str) -> dict[str, Any]:
    return {"code": 400, "message": message, "data": {"steps": []}, "stop": True}
