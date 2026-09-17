"""hcAgent —— juagent poc 慢任务接口的 FastAPI 实现。

对齐 juagent/orchestrator/server.py 的对外契约：

  POST /slowAgent/poc_<id>
    {"traceId":"...","deviceId":"...","deviceType":"...",
     "data":{"query":"...","tvMode":"...","segment":{},
             "memory":{"shortMemory":[...],"longMemory":{}},
             "toolHistory":[...],"toolList":[...],"timestamp":"..."}}
  → {"code":200,"message":"success","traceId":"...","deviceId":"...",
     "data":{"planId","schemaVersion","planType","planConfidence","steps":[...],
             "final"?:true,"parallel"?:true},"stop":bool}

恒定走目标架构 **纯 LLM 主干**（app.engine，T0/T1 全 LLM，一个 tick 恰一次调用，
编排层零语言规则）。
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .engine import build_response as build_response_llm
from .models import SlowRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("hcAgent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # 关闭复用的异步连接池（LLM 网关 / hcTools）
    from . import hctools, llm
    await llm.close()
    await hctools.close()


app = FastAPI(
    title="hcAgent",
    description="juagent poc 慢任务接口（纯 LLM 编排入口）",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """把 422 打成可读日志：原始 body + 具体哪个字段、什么类型没过校验。

    runtime 回调本服务时若某字段传了 null / 类型不符，FastAPI 默认只回 422 无细节，
    这里把 body 和 errors 打出来，便于定位是哪个字段。
    """
    raw = (await request.body()).decode("utf-8", "replace")
    logger.warning("422 on %s body=%s errors=%s", request.url.path, raw, exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "code": 422,
            "message": "validation error",
            "data": {"steps": []},
            "stop": True,
            "errors": [
                {"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")}
                for e in exc.errors()
            ],
        },
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
async def slow_agent(poc_id: str, req: SlowRequest, request: Request) -> JSONResponse:
    """唯一对外接口：接收 poc 慢任务请求，返回编排 steps（纯 LLM 主干）。"""
    # 原始请求体（runtime 实际发的字节，含所有字段，未经模型裁剪）
    raw = (await request.body()).decode("utf-8", "replace")
    logger.info("REQ %s raw_body=%s", poc_id, raw)
    from .reqlog_tmp import tlog
    tlog(req)
    # 解析后的参数（经 Pydantic 归一化，含默认值填充）
    logger.info(
        "REQ %s params=%s",
        poc_id,
        json.dumps(req.model_dump(), ensure_ascii=False),
    )

    # 目标架构：恒定走 LLM 主干（T0/T1 全 LLM，编排层零规则）。异步执行以支持并发。
    body = await build_response_llm(req)
    logger.info("RESP %s body=%s", poc_id, json.dumps(body, ensure_ascii=False))
    return JSONResponse(content=body, status_code=body.get("code", 200))
