"""多轮改写 badcase：上轮 query + 本轮 query → 固定改写结果。

参照 hcTools/domains/audio/badcase.py 的 L2 精确止血模型，把多轮改写里
已知会让 LLM(MT_REWRITE_PROMPT) 改坏的『(上轮, 本轮)』句对，硬编码成权威改写结果。

为什么需要这一层：
  - 多轮改写结果直接进 _SM.tick → hcTools，选工具/参数以改写后的自包含 query 为准。
  - round2..N 的瓶颈是 LLM 多轮改写保真（丢指代/压扁多意图/改错属性），而非 hcTools。
  - badcase 用『(上轮真实末轮 query, 本轮原请求)』双键精确归一匹配，命中即直出固定改写串，
    优先级最高（在 _mt_rewrite 里先于 LLM），从源头消除该轮改写噪声。

设计：
  - 零泛化精确覆盖：只认数据里写死的 (prev, cur) 对，不做任何模式推理；
    泛化/兜底仍交给 LLM 与下游规则。
  - prev 取该 device 对话历史『真实执行的末轮 query』(_MT_CONTEXT 的 q，
    它是已改写/执行的 query 链)。评测里上一轮 query == 上一轮 gold query。
  - 归一化只做空白折叠，不做近义规约（避免误命中）。
  - 加载失败不可拖垮引擎：try/except → 空库 + 告警（与 hcTools.badcase 同构）。
  - 冲突双键（两条数据归一后同形）必须告警 = 数据问题。

命中返回 fixed 改写串；未命中返回 None（引擎继续走 LLM 改写）。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("hcAgent.mt_rewrite_badcase")

_BADCASES_PATH = Path(__file__).resolve().parent / "mt_rewrite_badcases.json"

_WS = re.compile(r"\s+")


def _normalize(s: str) -> str:
    """归一：去首尾空白 + 折叠内部空白。不改字符/近义，零泛化。"""
    return _WS.sub(" ", (s or "").strip())


class MtRewriteBadcaseStore:
    def __init__(self, entries: list[dict]):
        # key = (prev_norm, cur_norm) -> {target, why, id}
        self._map: dict[tuple[str, str], dict] = {}
        for e in entries or []:
            prev = _normalize(e.get("prev", ""))
            cur = _normalize(e.get("cur", ""))
            target = e.get("target", "")
            if not cur or not target:
                continue  # 本轮或目标改写缺一即无效，跳过
            key = (prev, cur)
            if key in self._map:
                logger.warning(
                    "mt改写badcase 冲突 key=(%r, %r)，后者覆盖（%r）",
                    prev, cur, self._map[key],
                )
            self._map[key] = {
                "target": target,
                "id": e.get("id", ""),
                "explain": e.get("explain", ""),
            }

    @classmethod
    def load(cls, path: Path = _BADCASES_PATH) -> "MtRewriteBadcaseStore":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cls(payload.get("entries", []))
        except Exception as exc:  # noqa: BLE001 加载失败不可拖垮引擎
            logger.error("mt改写 badcases 加载失败（%s），使用空库：%s", path, exc)
            return cls([])

    def lookup(self, prev_query: str, cur_query: str) -> dict | None:
        """(上轮query, 本轮query) 归一后精确命中返回 {target, id, explain}，否则 None。"""
        if not prev_query or not cur_query:
            return None
        return self._map.get((_normalize(prev_query), _normalize(cur_query)))

    def __len__(self) -> int:
        return len(self._map)


# 进程级单例（load once，与 badcases_domain 缓存同风格）
_MT_BAD = None


def mt_rewrite_badcase() -> "MtRewriteBadcaseStore":
    global _MT_BAD
    if _MT_BAD is None:
        _MT_BAD = MtRewriteBadcaseStore.load()
    return _MT_BAD