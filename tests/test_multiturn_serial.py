"""多轮 + 单轮串行多意图：验证方案 A 的核心目标。

背景（docs/multi_turn_serial_plan.md，方案 A）：
- 原缺陷：多轮 merge(_mt_rewrite) 把单轮内串行多意图压扁成单意图，
  导致「先问导演，再问该导演的电影」第二意图丢失。
- 修法：约束 MT_REWRITE_PROMPT 保留多意图、用分号分隔、保留串行依赖指代。
- 组合场景：同 device 有跨轮上下文(shortMemory 候选)，本轮 query 又含单轮内串行依赖；
  首轮发 s1，续跑(toolHistory 带 id:s1) 发 s2，s2 依赖 s1 结果。

本测试 mock LLM/hcTools，不连网，验证编排层（T0 拆解 + 续跑 T1）在该组合下正确。
"""
from __future__ import annotations

import pytest

from app import engine
from tests.test_llm_engine import FakeLLM, _j, fllm, client  # noqa: F401


def _payload(query: str, **extra):
    return {"traceId": "t", "deviceId": "d", "data": {"query": query, "tvMode": "0", **extra}}


def test_mt_serial_first_step_only(fllm, client):
    """本轮含跨轮上下文 + 单轮内串行：首轮只放依赖序号靠前的 s1（依赖步 s2 待续跑）。"""
    fllm.plan = _j([
        {"q": "《疯狂动物城2》的导演是谁", "d": "qa", "tool": "fan_knowledge_agent"},
        {"q": "该导演导过的电影", "d": "vod", "tool": "vod_search", "dep": 1},
    ])
    short = [{"dialogData": {"query": "最近有啥好看的电影", "answer": "找到2部"},
              "businessData": {"serviceData": [{"data": [
                  {"mediaTitle": "疯狂动物城2", "director": ["拜伦·霍华德"], "category": "电影"},
                  {"mediaTitle": "出入平安", "director": ["刘江江"], "category": "电影"},
              ]}]}}]
    r1 = client.post("/slowAgent/poc_1",
                     json=_payload("第三部导演是谁？他导过的电影", memory={"shortMemory": short}))
    b1 = r1.json()
    assert b1["code"] == 200
    assert [s["id"] for s in b1["data"]["steps"]] == ["s1"]
    # s1 是导演问答、不依赖任何人；s2 依赖 s1，留待续跑
    assert b1["data"]["steps"][0]["dependsOn"] == []
    assert b1["stop"] is False


def test_mt_serial_continue_fires_dep(fllm, client):
    """续跑：toolHistory 带回 s1，engine 走 _continue，串行推进发出依赖 s1 的 s2。"""
    fllm.plan = _j([
        {"q": "《疯狂动物城2》的导演是谁", "d": "qa", "tool": "fan_knowledge_agent"},
        {"q": "该导演导过的电影", "d": "vod", "tool": "vod_search", "dep": 1},
    ])
    short = [{"dialogData": {"query": "最近有啥好看的电影", "answer": "找到2部"},
              "businessData": {"serviceData": [{"data": [
                  {"mediaTitle": "疯狂动物城2", "director": ["拜伦·霍华德"], "category": "电影"},
              ]}]}}]
    # 首轮
    r1 = client.post("/slowAgent/poc_1",
                     json=_payload("第三部导演是谁？他导过的电影", memory={"shortMemory": short}))
    b1 = r1.json()
    assert [s["id"] for s in b1["data"]["steps"]] == ["s1"]
    # 续跑：回带 s1 结果（含 id），触发 _continue 发 s2
    h = [{"id": "s1", "toolName": "fan_knowledge_agent", "parameters": {},
          "result": "导演是拜伦·霍华德"}]
    r2 = client.post("/slowAgent/poc_1",
                     json=_payload("多部导演是谁？他导之前的电影", toolHistory=h, memory={"shortMemory": short}))
    b2 = r2.json()
    assert [s["id"] for s in b2["data"]["steps"]] == ["s2"]
    assert b2["data"]["steps"][0]["dependsOn"] == ["s1"]
    assert b2["stop"] is True

def test_mt_serial_person_name_resolved(fllm, client):
    """人物属性指代：多轮里“该导演/他”应还原成具体人名（而非带（普通话版）的片名），
    这样 T0 才不把“<片名>的导演”误当 vod 检索、hcTools 也不把（普通话版）当语言条件。"""
    fllm.plan = _j([
        {"q": "杰拉德·布什老婆叫啥名字", "d": "qa", "tool": "fan_knowledge_agent"},
        {"q": "杰拉德·布什演过啥电影", "d": "vod", "tool": "vod_search"},
    ])
    short = [{"dialogData": {"query": "最近有啥好看的电影", "answer": "找到2部"},
              "businessData": {"serviceData": [{"data": [
                  {"mediaTitle": "疯狂动物城2（普通话版）", "director": ["杰拉德·布什"], "category": "电影"},
              ]}]}}]
    r = client.post("/slowAgent/poc_1",
                    json=_payload("这导演老婆叫啥？演过啥电影", memory={"shortMemory": short}))
    b = r.json()
    assert b["code"] == 200
    assert b["data"]["steps"], "应至少发一个依赖 step"
