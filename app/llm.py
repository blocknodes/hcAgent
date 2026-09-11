"""LLM 客户端：OpenAI-format /v1/chat/completions + 重试 + JSON 提取。

对齐 juagent/orchestrator/llm.py 契约，够编排侧拆解/改写用即可。
会以 info 级打印每次调用的输入与输出（便于观测 LLM 链路）。
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from . import config

logger = logging.getLogger("hcAgent.llm")


def chat(
    messages: list[dict[str, Any]],
    *,
    model: str = config.MODEL,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """返回 choices[0].message；失败返回 {"__error__": ...}。

    带小型重试：429/反序列化失败重试，4xx 非 429 直接放弃。
    """
    payload = {"model": model, "messages": messages, "temperature": temperature}
    body = json.dumps(payload, ensure_ascii=False).encode()
    last = ""
    started = time.perf_counter()
    for attempt in range(max(1, config.MAX_RETRY)):
        req = urllib.request.Request(
            f"{config.API_BASE}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {config.API_KEY}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=config.TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'ignore')[:200]}"
            if exc.code != 429 and 400 <= exc.code < 500:
                break
            continue
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            continue
        message = data["choices"][0]["message"]
        logger.info(
            "LLM req %s | %s\n"
            "  >> IN  %s\n"
            "  << OUT %s  (%.0fms)",
            model,
            " ".join((m.get("role", "?") for m in messages)),
            json.dumps(messages, ensure_ascii=False),
            json.dumps(message, ensure_ascii=False),
            (time.perf_counter() - started) * 1000,
        )
        return message
    return {"__error__": last}


def text_of(message: dict[str, Any]) -> str:
    content = message.get("content")
    return content.strip() if isinstance(content, str) else ""


def parse_json(text: str) -> Any:
    """取文本里第一个可解析的 JSON 顶层值（容忍 thinking 前缀/代码块）。"""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1].lstrip("json").lstrip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(stripped):
        if ch not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[i:])
            return value
        except json.JSONDecodeError:
            continue
    return None