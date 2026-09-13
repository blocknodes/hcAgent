"""hcAgent 判域规则库内核：把 detect.py 里散在 if/正则堆栈的顺序判定提成显式规则条目。

与 hcTools/app/rulebase.py 同一组诉求，但语义不同：这里是**判域**（返回域名），
hcTools 那边是**选工具+参数**。故单独轻量实现，不耦合工具语义。

设计：
- `DecideRule(q, llm_domain, tv_mode)` -> str|None（返回命中的域名；None=不命中，继续下一条）。
- 每条规则带 id / priority / title / explain / enabled —— 可审计、可拔插。
- `DetectRuleSet.select()` 按 priority 升序跑 decide，首条命中即返回 (domain, rule)。
- default 是兜底（无 decide 恒跑，返回 None 表示整链放弃交 LLM）。

语义保证与改造前 detect.py 完全一致：
- decide 返回域名=命中；返回 None=未命中继续。
- 命中边界/闰注只由 decide 内部正则集决定，改造不改变任何判定结果（差异测试兜底）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("hcAgent.detect_rulebase")

# (query, llm_domain, tv_mode) -> str | None ；返回命中的域名
Hist = tuple[str, str, str | int]
Decider = Callable[[str, str, str | int], str | None]


@dataclass
class Rule:
    id: str
    decide: Decider = field(default=None)   # 完整判定；None=仅 metadata（如 default）
    priority: int = 100
    title: str = ""
    explain: str = ""
    scope: str = "both"      # bright | off | both — 亮屏/息屏适用性
    enabled: bool = True
    note: str = ""

    def run(self, q: str, llm_domain: str, tv_mode: str | int):
        if not self.enabled:
            return None
        if self.decide is None:
            return None
        try:
            return self.decide(q, llm_domain, tv_mode)
        except Exception as exc:  # noqa: BLE001 单规则失败不拖垮整链
            logger.error("判域规则 %s 异常：%s", self.id, exc)
            return None


@dataclass
class DetectRuleSet:
    rules: list[Rule] = field(default_factory=list)
    default: Rule | None = None

    def __post_init__(self) -> None:
        self.rules.sort(key=lambda r: r.priority)

    def select(self, q: str, llm_domain: str, tv_mode: str | int = "0"):
        """按 priority 升序找首条命中。返回 (domain, rule) 或 (None, None)。"""
        for r in self.rules:
            if r.scope != "all":
                is_off = str(tv_mode) == "6"
                if r.scope == "bright" and is_off:
                    continue
                if r.scope == "off" and not is_off:
                    continue
            d = r.run(q, llm_domain, tv_mode)
            if d is not None:
                return d, r
        if self.default is not None and self.default.enabled:
            d = self.default.run(q, llm_domain, tv_mode)
            if d is not None:
                return d, self.default
        return None, None


__all__ = ["Rule", "DetectRuleSet"]