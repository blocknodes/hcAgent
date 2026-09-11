"""hcTools 调解析客户端（异步）：把 (query, domain) 交给 hcTools 拿最终工具与参数。

hcAgent 编排出工具后不再 mock 参数，而是把 (query, domain) 交给 hcTools，
由 hcTools 的 LLM 两段式解析出最终 tool 与 params，作为 step 的真实参数。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from . import config

logger = logging.getLogger("hcAgent.hctools")

TIMEOUT = 90.0
MAX_RETRY = 2

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=TIMEOUT)
    return _client


async def predict(query: str, domain: str, metadata: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """调用 hcTools，返回 (tool, params)。失败返回 ("", {})，由调用方决定降级。"""
    payload = {"query": query, "domain": domain}
    if metadata:
        payload["metadata"] = metadata
    url = f"{config.HCTOOLS_BASE}/api/predict"
    last = ""
    started = time.perf_counter()

    client = _get_client()
    for attempt in range(MAX_RETRY):
        try:
            resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            data = resp.json()
            tool = data.get("tool", "")
            params = data.get("params") if isinstance(data.get("params"), dict) else {}
            logger.info(
                "HCTOOLS %s | %s\n  >> %s\n  << tool=%s params=%s  (%.0fms)",
                domain, query, payload, tool, json.dumps(params, ensure_ascii=False),
                (time.perf_counter() - started) * 1000,
            )
            return tool, params
        except httpx.HTTPStatusError as exc:
            last = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            if exc.response.status_code != 429 and 400 <= exc.response.status_code < 500:
                break
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
    logger.warning("hcTools predict 失败 (%s)：%s", url, last)
    return "", {}


async def close() -> None:
    """关闭复用的连接池（应用退出时调用）。"""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None