"""hcAgent —— juagent poc 慢任务接口的 FastAPI mock 实现。

对齐 juagent/orchestrator/server.py 的对外契约：

  POST /slowAgent/poc_<id>
    {"traceId":"...","deviceId":"...","deviceType":"...",
     "data":{"query":"...","tvMode":"...","segment":{},
             "memory":{"shortMemory":[...],"longMemory":{}},
             "toolHistory":[...],"toolList":[...],"timestamp":"..."}}
  → {"code":200,"message":"success","traceId":"...","deviceId":"...",
     "data":{"planId","schemaVersion","planType","planConfidence","steps":[...],
             "final"?:true,"parallel"?:true},"stop":bool}

只做 mock：不接 LLM / 域工具，用确定性规则生成 steps。
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .models import SlowRequest
from .mock import build_slow_response

app = FastAPI(
    title="hcAgent",
    description="juagent poc 慢任务接口的 FastAPI mock",
    version="0.1.0",
)


@app.get("/")
def index() -> dict:
    return {
        "ok": True,
        "service": "hcAgent",
        "endpoints": {"poc": "POST /slowAgent/poc_<id>"},
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True, "status": "healthy"}


@app.post("/slowAgent/{poc_id}")
def slow_agent(poc_id: str, req: SlowRequest) -> JSONResponse:
    """唯一对外接口：接收 poc 慢任务请求，返回编排 steps（mock）。"""
    body = build_slow_response(req)
    return JSONResponse(content=body, status_code=body.get("code", 200))
