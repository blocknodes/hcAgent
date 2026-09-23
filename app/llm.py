"""LLM 客户端（异步）：OpenAI-format /v1/chat/completions + 重试 + JSON 提取。

对齐 hcTools 的做法：用 httpx.AsyncClient 实现真并发（单连接池复用）。
会以 info 级打印每次调用的输入与输出（便于观测 LLM 链路）。

metadata：可选的业务上下文，透传到 OpenAI-format 请求体的 `metadata` 字段
（网关可采集/审计；LLM 侧多数实现会忽略，不改写 messages/prompt）。
调用方用 prompts.build_llm_metadata() 构造统一 schema。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from . import config

logger = logging.getLogger("hcAgent.llm")

# 复用连接池：并发请求共享一个客户端，避免每次新建连接。
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=config.TIMEOUT)
    return _client


async def chat(
    messages: list[dict[str, Any]],
    *,
    model: str = config.MODEL,
    temperature: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """返回 choices[0].message；失败返回 {"__error__": ...}。

    带小型重试：429/反序列化失败重试，4xx 非 429 直接放弃。
    metadata 透传到 OpenAI-format 请求体的 metadata 字段，用于链路审计/计费标签。
    """
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if not model.startswith("gpt-5"):
        payload["temperature"] = temperature
        # 参考 hcTools/compare：禁用 thinking，baseline 不开只会回 " thinking"
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if metadata:
        payload["metadata"] = metadata

    headers = {"Content-Type": "application/json"}
    if config.API_KEY:
        headers["Authorization"] = f"Bearer {config.API_KEY}"

    last = ""
    started = time.perf_counter()
    client = _get_client()
    for attempt in range(max(1, config.MAX_RETRY)):
        try:
            resp = await client.post(
                f"{config.API_BASE}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            last = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            if exc.response.status_code != 429 and 400 <= exc.response.status_code < 500:
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


async def close() -> None:
    """关闭复用的连接池（应用退出时调用）。"""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


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