"""纯 LLM 主干测试：注入假 LLM，离线验证 T0/T1 与「一个 tick 恰一次 LLM」。

编排层零规则（无连接词拆、无词法落域、无关键词换工具、无规则指代替换）。
用 monkeypatch 覆盖 app.engine.llm.chat，按 system 提示区分：
  - T0（含 T0_PLAN_PROMPT）→ 返回顶层数组计划，每条含 q/d/tool/dep
  - T1（含 T1_STEP_PROMPT）→ 返回改写后的自包含 query（纯 LLM）
覆盖：单意图、并行、依赖串行（首轮只放 s1，续跑 T1 改写后放 s2）、LLM 失败兜底。
"""
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.prompts import T0_PLAN_PROMPT, T1_STEP_PROMPT  # noqa: E402


@pytest.fixture(autouse=True)
def _reset():
    engine.reset()
    yield
    engine.reset()


def _j(intents: list[dict]) -> str:
    import json
    return json.dumps(intents, ensure_ascii=False)


class FakeLLM:
    def __init__(self):
        # 默认单意图：给 tool，domain
        self.plan = _j([{"q": "播放流浪地球", "d": "vod", "tool": "vod_search"}])
        self.t1 = "查询《无间道》的导演是谁"
        self.t0_times = 0
        self.t1_times = 0

    async def chat(self, messages, model="", temperature=0.0):
        sys_prompt = messages[0]["content"]
        if T0_PLAN_PROMPT in sys_prompt:
            self.t0_times += 1
            return {"content": self.plan}
        if T1_STEP_PROMPT in sys_prompt:
            self.t1_times += 1
            return {"content": self.t1}
        return {"content": ""}


@pytest.fixture
def fllm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(engine.llm, "chat", fake.chat)
    # 测试不连网：mock hcTools，域 vod 直接返回占位 (tool, params)
    async def fake_hctools(query, domain, metadata=None):
        return "vod_search", {"query": query}
    monkeypatch.setattr(engine.hctools, "predict", fake_hctools)
    return fake


@pytest.fixture
def client():
    return TestClient(app)


def _payload(query: str, **extra):
    return {"traceId": "t", "deviceId": "d", "data": {"query": query, "tvMode": "0", **extra}}


# ---- M1 单意图 ----
def test_single_intent(fllm, client):
    r = client.post("/slowAgent/poc_1", json=_payload("我要看流浪地球"))
    b = r.json()
    assert b["code"] == 200
    assert len(b["data"]["steps"]) == 1
    assert b["data"]["steps"][0]["toolName"] == "vod_search"
    assert b["data"]["steps"][0]["dependsOn"] == []
    assert b["stop"] is True
    assert fllm.t0_times == 1 and fllm.t1_times == 0


# ---- 独立并行：T0 一次出两条，无 dep ----
def test_parallel(fllm, client):
    fllm.plan = _j([{"q": "点播战狼2", "d": "vod", "tool": "vod_search"},
                    {"q": "同时调大音量", "d": "device", "tool": "device_control"}])
    r = client.post("/slowAgent/poc_1", json=_payload("点播战狼2同时调大音量"))
    b = r.json()
    assert b["code"] == 200
    assert len(b["data"]["steps"]) == 2
    for st in b["data"]["steps"]:
        assert st["dependsOn"] == []
    assert fllm.t0_times == 1  # 并行 → 1 tick = 1 T0


# ---- 依赖链：T0 出 dep；首轮只放 s1；续跑放 s2 ----
def test_dependency_serial(fllm, client):
    fllm.plan = _j([
        {"q": "搜索科幻电影", "d": "vod", "tool": "vod_search"},
        {"q": "找它的评分最高的一部", "d": "qa", "tool": "execute", "dep": 1},
    ])
    r1 = client.post("/slowAgent/poc_1", json=_payload("搜索科幻电影然后看评分最高的一部"))
    b1 = r1.json()
    # 首轮：s1 可发，s2 依赖 s1 → 首批只放 s1
    assert [s["id"] for s in b1["data"]["steps"]] == ["s1"]
    assert b1["stop"] is False

    # 续跑：toolHistory 带回 s1 结果，T1 改写 s2 自包含
    h = [{"id": "s1", "result": {"mediaTitle": "沙丘"}}]
    r2 = client.post("/slowAgent/poc_1",
                     json=_payload("搜索科幻电影然后查询评分最高的一部", toolHistory=h))
    b2 = r2.json()
    assert [s["id"] for s in b2["data"]["steps"]] == ["s2"]
    s2 = b2["data"]["steps"][0]
    assert s2["dependsOn"] == ["s1"]
    assert b2["stop"] is True
    assert fllm.t0_times == 1 and fllm.t1_times == 1  # 依赖链深度 2 = 2 tick，各一次 LLM


# ---- LLM 失败：返回空 → 兜底为单条意图（不 inject 规则）----
def test_llm_failure_falls_back(fllm, client):
    fllm.plan = ""  # 空 → parse 失败
    r = client.post("/slowAgent/poc_1", json=_payload("我要看流浪地球"))
    b = r.json()
    assert b["code"] == 200
    assert len(b["data"]["steps"]) == 1
    assert b["data"]["steps"][0]["toolName"] == "execute"  # 中性，不启发
    assert fllm.t0_times == 1


# ---- 空 query 契约 ----
def test_empty_query(fllm, client):
    r = client.post("/slowAgent/poc_1", json=_payload(""))
    assert r.json()["code"] == 400
    assert r.json()["data"]["steps"] == []