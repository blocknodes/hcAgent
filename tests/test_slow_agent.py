"""hcAgent poc 接口的端到端测试（FastAPI TestClient）。"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _payload(query: str, **data_extra):
    return {
        "traceId": "t-1",
        "deviceId": "d-1",
        "deviceType": "tv",
        "data": {"query": query, "tvMode": "0", **data_extra},
    }


def test_index():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "hcAgent"


def test_health():
    assert client.get("/health").json()["ok"] is True


def test_empty_query_returns_400():
    r = client.post("/slowAgent/poc_1", json=_payload(""))
    assert r.status_code == 400
    body = r.json()
    assert body["code"] == 400
    assert body["data"]["steps"] == []
    assert body["stop"] is True


def test_single_query():
    r = client.post("/slowAgent/poc_1", json=_payload("我要看流浪地球"))
    assert r.status_code == 200
    body = r.json()
    assert body["code"] == 200
    assert body["traceId"] == "t-1"
    steps = body["data"]["steps"]
    assert len(steps) == 1
    assert steps[0]["id"] == "s1"
    assert steps[0]["toolName"] == "vod_search"
    assert steps[0]["dependsOn"] == []
    assert body["stop"] is True


def test_multi_step_split():
    r = client.post("/slowAgent/poc_1", json=_payload("我要听周杰伦的歌然后看复仇者联盟"))
    body = r.json()
    steps = body["data"]["steps"]
    assert len(steps) == 2
    assert steps[0]["toolName"] == "music_search"
    assert steps[1]["toolName"] == "vod_search"


def test_dependency_serial():
    r = client.post("/slowAgent/poc_1", json=_payload("搜索科幻电影然后播放第二部"))
    body = r.json()
    steps = body["data"]["steps"]
    assert len(steps) == 2
    assert steps[1]["dependsOn"] == ["s1"]


def test_cartoon_dual_domain():
    r = client.post("/slowAgent/poc_1", json=_payload("推荐动漫"))
    body = r.json()
    assert body["data"]["parallel"] is True
    tools = {s["toolName"] for s in body["data"]["steps"]}
    assert tools == {"children_search", "vod_search"}


def test_cartoon_with_history_not_dual():
    r = client.post(
        "/slowAgent/poc_1",
        json=_payload("推荐动漫", toolHistory=[{"toolName": "vod_search"}]),
    )
    body = r.json()
    assert "parallel" not in body["data"]


def test_debug_flag():
    r = client.post("/slowAgent/poc_1", json=_payload("看电影", debug=True))
    body = r.json()
    assert "debug" in body["data"]
    assert body["data"]["debug"]["intents"][0]["index"] == 1
