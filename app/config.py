"""hcAgent 配置：LLM 网关 / hcTools 地址 / 模型名走环境变量，其余为常量默认值。

编排层已无确定性规则，故没有规则开关。
"""
from __future__ import annotations

import os

# ---- LLM 网关环境变量 ----
# LLM 网关（OpenAI-format /v1/chat/completions），默认 juagent 真实网关。
API_BASE = (os.environ.get("HC_LLM_API_BASE", "http://127.0.0.1:7000/v1")).rstrip("/")
API_KEY = os.environ.get("HC_LLM_API_KEY", "")

# hcTools 意图解析服务地址（POST /api/predict，拿最终工具与参数）。
HCTOOLS_BASE = (os.environ.get("HC_HCTOOLS_BASE", "http://127.0.0.1:8084")).rstrip("/")

# LLM 请求体里的 model 名（对应 vllm --served-model-name 或 --lora-modules 名）。
# 默认 baseline；切 LoRA 时设 HC_LLM_MODEL=lora1 等。plan/多轮改写/兜底qa 全部共用此值。
MODEL = os.environ.get("HC_LLM_MODEL", "baseline")

# ---- 常量默认值（不读环境）----
TIMEOUT = 120.0
MAX_RETRY = 3
MAX_PLAN_STEPS = 6
PLAN_TRACE_TTL = 300.0
# T1 指代改写时携带的上一步候选媒资上限(避免超长)
CANDIDATE_CONTEXT_LIMIT = 30
