"""请求/响应的 Pydantic 模型（对齐 poc 慢任务接口契约）。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Memory(BaseModel):
    shortMemory: list[dict[str, Any]] = Field(default_factory=list)
    longMemory: dict[str, Any] = Field(default_factory=dict)


class SlowData(BaseModel):
    query: str = ""
    tvMode: str | int | None = None
    segment: dict[str, Any] = Field(default_factory=dict)
    memory: Memory = Field(default_factory=Memory)
    toolHistory: list[dict[str, Any]] = Field(default_factory=list)
    toolList: list[Any] = Field(default_factory=list)
    timestamp: str | None = None
    debug: bool | None = None

    class Config:
        extra = "allow"


class SlowRequest(BaseModel):
    traceId: str = ""
    deviceId: str = ""
    deviceType: str | None = None
    data: SlowData = Field(default_factory=SlowData)

    class Config:
        extra = "allow"


class Step(BaseModel):
    id: str
    toolName: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    plan: Any | None = None
    dependsOn: list[str] = Field(default_factory=list)
    retext: str = ""
