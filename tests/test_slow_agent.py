"""hcAgent poc 接口的 HTTP 契约测试（纯 LLM 路径，注入假 LLM，离线）。

只验契约层：/ 与 /health、空 query、缺 data、单自包含意图的结构（id/toolName/
parameters/dependsOn/retext）、debug 字段。意图拆分/依赖等行为测试见 test_llm_engine.py。
"""
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import engine  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    engine.reset()
    # 假 LLM：T0 返回单个自包含意图（编层不连网）。
    async def fake_chat(messages, **kw):
        return {"content": '{"q":"播放流浪地球","d":"vod","tool":"vod_search"}'}
    async def fake_hctools(query, domain, metadata=None):
        return "vod_search", {"query": query}
    monkeypatch.setattr(engine.llm, "chat", fake_chat)
    monkeypatch.setattr(engine.hctools, "predict", fake_hctools)
    yield
    engine.reset()


def test_index():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "hcAgent"


def test_health():
    assert client.get("/health").json()["ok"] is True


def test_empty_query_returns_400():
    r = client.post("/slowAgent/poc_1", json={
        "traceId": "t", "deviceId": "d", "data": {"query": ""}})
    body = r.json()
    assert body["code"] == 400
    assert body["data"]["steps"] == []
    assert body["stop"] is True


def test_missing_data_returns_400():
    r = client.post("/slowAgent/poc_1", json={"traceId": "t"})
    assert r.json()["code"] == 400


def test_single_query_structure():
    r = client.post("/slowAgent/poc_1", json={
        "traceId": "t1", "deviceId": "d1",
        "data": {"query": "我要看流浪地球", "tvMode": "0"}})
    body = r.json()
    assert body["code"] == 200
    assert body["traceId"] == "t1"
    assert body["deviceId"] == "d1"
    assert body["data"]["schemaVersion"] == "1.0"
    step = body["data"]["steps"][0]
    for key in ("id", "toolName", "parameters", "dependsOn", "retext"):
        assert key in step


def test_debug_flag():
    r = client.post("/slowAgent/poc_1", json={
        "traceId": "t", "deviceId": "d", "data": {"query": "看电影", "debug": True}})
    body = r.json()
    assert "debug" in body["data"]
    assert "intents" in body["data"]["debug"]