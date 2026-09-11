"""hcAgent 配置：只保留 LLM 网关的两个环境变量（api base / key），其余均为常量默认值。

模型与其他阈值改动直接在此改默认值即可，不再暴露环境影响。
"""
from __future__ import annotations

import os

# ---- 唯二环境变量 ----
# LLM 网关（OpenAI-format /v1/chat/completions），默认 juagent 真实网关。
API_BASE = (os.environ.get("HC_LLM_API_BASE", "http://10.19.96.219:4003/v1")).rstrip("/")
API_KEY = os.environ.get("HC_LLM_API_KEY", "")

# hcTools 意图解析服务地址（POST /api/predict，拿最终工具与参数）。
HCTOOLS_BASE = (os.environ.get("HC_HCTOOLS_BASE", "http://127.0.0.1:8084")).rstrip("/")

# ---- 常量默认值（不读环境）----
MODEL = "baseline"
TIMEOUT = 120.0
MAX_RETRY = 3
MAX_PLAN_STEPS = 6
PLAN_TRACE_TTL = 300.0