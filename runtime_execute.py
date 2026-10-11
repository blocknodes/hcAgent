#!/usr/bin/env python3
"""调用远端 runtime，自动跑完整 ReAct 多步，打印每步的工具、参数与媒资结果。

直连远端 runtime(10.18.210.7:31392/api/runtime/execute, SSE)，真实执行工具。
从 TOOL 帧提取:
  llmSemantic.intent   -> 工具名(vod_keyword_search / fan_knowledge_agent ...)
  memoryContent.memoryQuery -> 工具参数
  ttscontent + memoryData   -> 结果自然语言 + 候选媒资(完整字段)
开 debug:true 时从 DEBUG_MAP 帧解析 rpc.SlowAgentApi 计划，得到数据集命名的真工具序列。

用法:
  python3 tools/runtime_execute.py "刘德华的电影"
  python3 tools/runtime_execute.py --device-id 1234 "帮我查下无间道的导演是谁,然后再搜下他的片子"
  python3 runtime_execute.py --device-id 6 "最近有啥好看的电影" "第三部导演是谁？他导过哪些电影" ...
  python3 tools/runtime_execute.py --json "query"
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import secrets
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

RUNTIME_URL = "http://10.18.210.7:31392/api/runtime/execute"
DEFAULT_FEATURE_CODE = "861003009000056000000505"
DEFAULT_DEVICE_ID = 9900111205682291335156314657157449362655530101519471739595839462
SSE_TIMEOUT = 90  # 单次 SSE 连接超时(秒)

# 真实电视终端请求上下文。retext / device / feature / tvMode / debug 等运行参数
# 在 _run_runtime 中随后覆盖，确保 CLI 显式参数始终优先。
TERMINAL_CONTEXT: dict[str, Any] = {
    "clientsid": "74D082860B85,1789608633788",
    "coreversion": "217890050",
    "tvdemover": "482000111",
    "vidaaidever": "2026020012",
    "calendarver": "618000061",
    "deviceid": 9900111205682291335156314657157449362655530101519471739595839462,
    "tvtype": "UXS",
    "productname": "116UXS",
    "tvfeaturecode": "861003009000056000000505",
    "tvsystemversion": "07.02.70.202604240920;6.6",
    "rekey": "528cb8b668b5f7005a19cfbd2a664cb9",
    "terminaltype": "vidaa4tv",
    "excard": "1",
    "requestversion": "7.8.9",
    "crequestversion": "7.8.9",
    "vrequestversion": "7.8.8",
    "viewversion": "lui.7.8.8.35",
    "storeversioncode": "-1",
    "tutorialver": "2129030080",
    "moonmodever": "21.29.03.008.0",
    "jubaover": "-1",
    "snnumber": "00000000000000000000000",
    "audioversion": "3rd.audio.690",
    "micarrayfac": "aispeecharray",
    "micarrayver": "96550385",
    "token": "19ovMOAF4JKtNyDzPpcp-CJRE_CM7Qb37VOAizDpbCFxxpIjAQZDhut_CP2BSQQubbslFNd_b75F_vTEMgal4URBtc_trcw71VMKsogU2LIvIyf0G6NurBE0vqzsK8GfdHJ6Q1GSFNhCLhkvDJYfrppl7kWN5Hcg6cH2n4ctK63BLCz1ZAA4BKzsVEwrTboqJ1SedbscH4",
    "tokencreatetime": "1789608627",
    "tokenexpiretime": "259200",
    "tokensubscribeid": "1550256658",
    "customerid": "1517197036",
    "vodversion": "2026.8.4.09.10380000.0",
    "hisappversion": "@01.102.805",
    "license": "1015",
    "frontscene": "com.kurt.demo",
    "frontactivity": "activity.MainActivity",
    "switchstatus": "0",
    "tvarea": "北京 北京",
    "serialnumber": "1",
    "tvversion": "MT9655_Fri_Apr_24_09:21:13_CST_2026",
    "ismenuon": "0",
    "vodability": "{\"juuiAbility\":{\"aihome\":\"1\",\"edu\":\"1\",\"edubox\":\"0\",\"hichangSupportType\":\"1\",\"hising\":\"1\",\"kids\":\"1\",\"kidsSubMode\":\"0\",\"ksongbasequanmin\":\"1\",\"musicSupportPlayHistoryCollect\":\"1\",\"supportSearchCollection\":\"1\",\"terse\":\"1\",\"widgetpro\":\"0\"},\"juuiDevice\":{\"deviceLevel\":\"Social\"},\"juuiStatus\":{\"areaCode\":\"000000000000\",\"hiSingPlayState\":\"0\",\"ispCode\":\"0\",\"kugoupid\":\"200081\",\"regionCode\":\"000000000000\",\"runDesk\":\"standard\",\"runMode\":\"standard\",\"runSubMode\":\"undefined\",\"sceneCode\":\"\",\"sceneCodeValue\":\"\",\"speechControl\":\"0\",\"vipRight\":\"[{\\\"rightProduct\\\":\\\"vod\\\",\\\"rightValue\\\":\\\"2\\\"},{\\\"rightProduct\\\":\\\"edu\\\",\\\"rightValue\\\":\\\"2\\\"},{\\\"rightProduct\\\":\\\"educ\\\",\\\"rightValue\\\":\\\"2\\\"}]\",\"audiobookPlayState\":\"\"}}",
    "wpability": "{\"aiviewVersion\":\"7.8.8\",\"isJUAi\":\"0\",\"luiMode\":\"0\",\"screenOffMode\":\"1.1.1\"}",
    "mcability": "{\"connected\":false}",
    "isextconn": "0",
    "tvmode": "0",
    "speakermodever": "2.1",
    "tvpicver": "702010030",
    "mediaver": "703040320",
    "wakeupword": "0",
    "prodability": "{\"productType\":\"0\",\"nguser\":\"1\",\"aidebug\":\"1.0\",\"aiotapp\":\"304032000\",\"tvbrand\":\"Hisense\",\"social\":\"1.1\",\"uses2s\":\"0\",\"continuedspeech\":\"1.0\",\"iflytektts\":\"1.0\",\"tvcalendar\":\"1.0\",\"aodsupport\":\"true\",\"vpAudio\":\"1\",\"voicestartup\":\"1\",\"tvcontrolvers\":\"1,1,1,1,1,1\",\"aispeechasr\":\"1.0\",\"aispeechtts\":\"1.0\",\"iflytekvprint\":\"1.0\",\"commandwakeup\":\"1.0\",\"customwakeup\":\"1.0\",\"hisensetts\":\"2.0\",\"updateAppPage\":\"1\",\"farfieldswitch\":\"1\",\"voicestartupswitch\":\"0\"}",
    "speechchild": "0",
    "realbdswitch": "0",
    "electronictutorialability": "1",
    "mac": "74D082860B85",
    "avatartheme": "common",
    "systemstate": "{\"devpicmode\":\"23\",\"livetvstrategy\":\"1\",\"devsource\":\"2562\"}",
    "paireddevices": "[{\"name\":\"X8 Pro\",\"type\":7}]",
    "marketmode": "0",
    "mictype": "wechat",
    "position": "36.196773,120.517897",
    "posinfo": "山东省,青岛市,崂山区",
    "juappversionswitch": "0",
    "fdswitch": "0",
    "fdstatus": "0",
    "voiceid": "mandarin,null,,null",
    "vocalprint": "-1,-1",
    "speechviewscene": "-1",
    "mediainfo": "",
    "personinfo": "",
    "clientinfo": "",
    "servicecode": "",
    "modelType": "v3",
    "enableSearch": "0"
}


def device_id_with_suffix(suffix: str) -> int:
    """用指定数字替换默认 deviceId 的末尾数字，并保持总长度不变。"""
    base = str(DEFAULT_DEVICE_ID)
    if not suffix or not suffix.isdigit():
        raise ValueError("device id 尾号必须是纯数字")
    if len(suffix) > len(base):
        raise ValueError(f"device id 尾号最多 {len(base)} 位")
    return int(base[:-len(suffix)] + suffix)


def random_device_id() -> int:
    """随机替换默认 deviceId 的最后 4 位，并保持总长度不变。"""
    suffix = f"{secrets.randbelow(10_000):04d}"
    return device_id_with_suffix(suffix)


def _sse(url: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """发送请求并逐行消费 SSE，在 runtime 终止帧到达后立即返回。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
    )
    frames: list[dict[str, Any]] = []
    with urllib.request.urlopen(request, timeout=SSE_TIMEOUT) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            text = line[len("data:"):].strip()
            if not text or text == "[DONE]":
                continue
            try:
                frame = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(frame, dict):
                continue
            frames.append(frame)
            if (frame.get("header") or {}).get("status") == 2:
                break
    return frames


def _summarize_debug_trace(trace: dict[str, Any]) -> dict[str, Any] | None:
    """从 DEBUG 帧的 rpc.SlowAgentApi trace 抽出计划步骤(真工具名, 数据集命名)。

    trace.response 是双重转义 JSON 字符串 -> 解开后 data.steps=[{id,toolName,retext,parameters}]
    """
    if (trace.get("phase") or "") != "rpc.SlowAgentApi":
        return None
    resp_s = trace.get("response")
    if not isinstance(resp_s, str):
        return None
    try:
        obj = json.loads(json.loads(resp_s))
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    return obj.get("data") or {}


def _plan_steps_dedup(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """拍平所有计划帧, 去掉重复(同一查询在 multi-intent 里可能被 rpc 调多次)。"""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for d in plans:
        for s in d.get("steps") or []:
            if not isinstance(s, dict) or not s.get("toolName"):
                continue
            key = f"{s.get('toolName')}|{s.get('retext','')}|{json.dumps(s.get('parameters') or {}, ensure_ascii=False)}"
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "tool": s.get("toolName"),
                "retext": s.get("retext", ""),
                "params": s.get("parameters") or {},
            })
    return out


def _align_tool_names(tools: list[dict[str, Any]], plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """以计划序列为准(慢端规划, 真工具名 vod_search 数据集命名)。

    每个 plan step 生成一条展示记录, 再从 TOOL 帧里按顺序就近补 tts与候选结果。
    """
    plan = _plan_steps_dedup(plans)
    if not plan:
        return tools
    used: set[int] = set()
    out: list[dict[str, Any]] = []
    for idx, s in enumerate(plan, 1):
        rec: dict[str, Any] = {
            "tool": s["tool"],
            "retext": s.get("retext", ""),
            "params": s.get("params") or {},
            "tts": "",
            "candidates": [],
        }
        i = idx - 1
        if i < len(tools):
            t = tools[i]
            used.add(i)
            if t.get("tts"):
                rec["tts"] = t.get("tts")
            if t.get("candidates"):
                rec["candidates"] = t.get("candidates")
        out.append(rec)
    return out


def _summarize_tool_frame(data: dict[str, Any]) -> dict[str, Any]:
    """从 runtime TOOL 帧里提取"工具名/意图 + 参数 + 结果摘要"。"""
    app = data.get("application_data") or {}
    td = app.get("data") if isinstance(app, dict) else {}
    if not isinstance(td, dict):
        td = {}
    semantic = td.get("llmSemantic") or {}
    intent = semantic.get("intent") or semantic.get("domain") or "unknown-tool"
    memory_content = td.get("memoryContent")
    params: dict[str, Any] = {}
    if isinstance(memory_content, str):
        try:
            mc = json.loads(memory_content)
            for key in ("memoryQuery", "query"):
                if key in mc:
                    params[key] = mc[key]
        except (ValueError, TypeError):
            params["memoryContent"] = memory_content
    candidates = _memory_data_of({"application_data": app})
    if not candidates:
        candidates = _search_result_of({"application_data": app})
    candidates = candidates[:20]
    tts = (td.get("ttscontent") or "")[:120]
    if not tts:
        vague = td.get("vagueTtsContent") or ""
        show = td.get("data").get("showText") if isinstance(td.get("data"), dict) else ""
        tts = (vague or show or "")[:120]
    return {
        "tool": intent,
        "params": params,
        "tts": tts,
        "candidates": candidates,
    }


def _memory_data_of(tool_frame: dict[str, Any]) -> list[dict[str, Any]]:
    app = (tool_frame.get("application_data") or {})
    d = app.get("data") or {} if isinstance(app, dict) else {}
    if isinstance(d, dict):
        md = d.get("memoryData")
        rows = []
        if isinstance(md, list):
            for item in md:
                if isinstance(item, dict):
                    inner = item.get("data")
                    if isinstance(inner, list):
                        rows.extend(x for x in inner if isinstance(x, dict))
                    else:
                        rows.append(item)
        return rows
    return []


def _search_result_of(tool_frame: dict[str, Any]) -> list[dict[str, Any]]:
    """从模糊搜索业务结果提取候选媒资(基础字段)。

    真机返回结构(TOOL 帧, application.data):
      data.content.searchResultList[].data[]      # 内容桶(通常嵌套在含 total/ranking 的桶结构里)
    少数帧没有外层桶, 只有:
      data.content.searchResultList[] = 直接媒资对象 或 {data: 单对象}
    两者都吃, data 可能为单 dict(非 list)。
    """
    app = (tool_frame.get("application_data") or {})
    d = app.get("data") or {} if isinstance(app, dict) else {}
    if not isinstance(d, dict):
        return []
    # data 下可能再嵌一层 data(真机 content/loginfo/total 都在这层)
    for level in (d, d.get("data") if isinstance(d.get("data"), dict) else None):
        if not isinstance(level, dict):
            continue
        content = level.get("content")
        if not isinstance(content, dict):
            continue
        rows: list[dict[str, Any]] = []
        for bucket in content.get("searchResultList") or []:
            if not isinstance(bucket, dict):
                continue
            bd = bucket.get("data")
            if isinstance(bd, dict):
                rows.append(bd)
            elif isinstance(bd, list):
                rows.extend(x for x in bd if isinstance(x, dict))
            elif any(k in bucket for k in ("mediaTitle", "mediaId", "title")):
                rows.append(bucket)
        if rows:
            return rows
    return []


def _run_runtime(retext: str, *, feature_code: str, device_id: int,
                 client_sid: str | None, tv_mode: str = "0",
                 debug: bool = True,
                 slow_agent_url: str | None = None,
                 print_sse_params: bool = False) -> dict[str, Any]:
    """直连远端 runtime，聚合 TOOL 帧(含 llmSemantic 工具名/意图 + memoryContent 参数)。"""
    # 从真实电视终端上下文起步，仅覆盖终端协议中的原始字段。
    # 保持默认 payload 与真机请求一致，避免同时发送 snake_case 别名导致歧义。
    payload: dict[str, Any] = dict(TERMINAL_CONTEXT)
    payload.update({
        "tvfeaturecode": feature_code,
        "deviceid": device_id,
        "retext": retext,
        "tvmode": tv_mode,
        "enable_slow": "true",
        "debug": bool(debug),
    })
    if client_sid:
        payload["clientsid"] = client_sid
    if slow_agent_url:
        payload["slow_agent_url"] = slow_agent_url

    if print_sse_params:
        print(f"[debug] SSE URL: {RUNTIME_URL}", file=sys.stderr)
        print("[debug] SSE payload:", file=sys.stderr)
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)

    frames = _sse(RUNTIME_URL, payload)
    tools: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    texts: list[str] = []
    stop = False
    for frame in frames:
        msg_type = (frame.get("header") or {}).get("messageType", "TEXT")
        data = frame.get("data") or {}
        if msg_type == "FIRST_PACKAGE":
            continue
        # 计划帧可能落在 DEBUG 或结束型 TEXT 帧的 application_data.debug.traces 里，
        # 故不按 messageType 过滤，凡带 traces 的帧都解析。
        app = data.get("application_data") or {}
        dbg = (app.get("debug") or {}) if isinstance(app, dict) else {}
        if dbg and dbg.get("traces"):
            for trace in dbg.get("traces") or []:
                d = _summarize_debug_trace(trace)
                if d:
                    plans.append(d)
        if msg_type == "TOOL":
            summary = _summarize_tool_frame(data)
            tools.append(summary)
        elif msg_type in ("DEBUG_MAP", "TEXT") and not dbg:
            content = data.get("content") or ""
            if content:
                texts.append(content)
        header = frame.get("header") or {}
        if header.get("status") == 2:
            stop = True
    tools = _align_tool_names(tools, plans)
    return {"tools": tools, "plans": plans, "texts": texts, "stop": stop}


_QUEST_STOPWORDS = ("叫什么名字", "是谁", "是哪个", "是啥", "叫什么", "谁",
                    "老婆", "妻子", "老公", "丈夫", "导演")


def _answer_key(retext: str, assistant: str) -> str:
    """从 retext 里提取一个会在回答正文中出现的片段, 用于把流式回答切到对应步骤。"""
    s = retext.replace("？", "").replace("?", "")
    s = s.strip("《》 「」“”‘’（）() 　")
    for _ in range(3):
        for w in _QUEST_STOPWORDS:
            if s.endswith(w) and len(s) > len(w):
                s = s[: -len(w)]
    if len(s) <= 1:
        return ""
    for length in range(min(len(s), 10), 1, -1):
        for i in range(0, len(s) - length + 1):
            frag = s[i:i + length]
            if frag in assistant:
                return frag
    return ""


def _split_plan_stream(plans: list[dict[str, Any]], assistant: str) -> dict[int, str]:
    """把拼接的流式回答, 按 plan 里 fan_knowledge_agent 步骤拆分。"""
    if not assistant:
        return {}
    dedup = _plan_steps_dedup(plans)
    qa_idx = [i for i, t in enumerate(dedup) if t["tool"] == "fan_knowledge_agent"]
    if not qa_idx:
        return {}
    if len(qa_idx) == 1:
        return {qa_idx[0]: assistant}
    anchors: list[tuple[int, int, str]] = []
    for qi in qa_idx:
        s = dedup[qi]
        key = _answer_key(s.get("retext", ""), assistant)
        pos = assistant.find(key) if key else -1
        if pos >= 0:
            anchors.append((qi, pos, key))
    if not anchors:
        return {qa_idx[-1]: assistant}
    anchors.sort(key=lambda x: x[1])
    out: dict[int, str] = {}
    for n, (qi, pos, _key) in enumerate(anchors):
        end = anchors[n + 1][1] if n + 1 < len(anchors) else len(assistant)
        out[qi] = assistant[pos:end].strip()
    return out


def _show_result(result: dict[str, Any]) -> None:
    """终端展示一轮结果（tools 为准，带 tts/candidates/流式回答切片）。"""
    tools = result["tools"]
    plans = result["plans"]
    assistant = "".join(result["texts"]).replace("[DONE]", "").strip()
    step_answers = _split_plan_stream(plans, assistant)
    print(f"\ntotal tool frames: {len(tools)}  plans: {len(plans)}  stop={result['stop']}")

    ti = [0]
    def dump_step(label: str, t: dict[str, Any]) -> None:
        print(f"  {label}: {t.get('tool') or 'execute'}")
        retext = t.get("retext") or ""
        params = t.get("params") or {}
        if retext:
            print(f"           retext: {retext}")
        if params:
            print(f"           params: {json.dumps(params, ensure_ascii=False)[:300]}")
        if t.get("tts"):
            print(f"           tts: {t['tts']!r}")
        for j, item in enumerate(t.get("candidates") or [], 1):
            if isinstance(item, dict):
                brief = {k: item[k] for k in
                         ("mediaTitle", "director", "category", "childCategory",
                          "doubanRate", "pubdate", "mediaId", "episodeTitle", "summary")
                         if item.get(k) not in (None, "", [])}
                print(f"           [{j}] {json.dumps(brief, ensure_ascii=False)[:300]}")
            else:
                print(f"           [{j}] {item}")
        if not t.get("candidates") and step_answers.get(ti[0] - 1):
            print(f"           answer: {step_answers[ti[0] - 1]}")

    plan_steps = _plan_steps_dedup(plans)
    if not plan_steps:
        for i, t in enumerate(tools, 1):
            ti[0] = i
            dump_step(f"tool {i}", t)
    else:
        print(f"  plan: {len(plan_steps)} steps")
        for s in plan_steps:
            ti[0] += 1
            t = tools[ti[0] - 1] if ti[0] - 1 < len(tools) else s
            dump_step(f"step {ti[0]}", t)


# ---------------------------------------------------------------------------
# CSV 批处理：读入含 query 列的 csv，逐条走 runtime，输出首工具域 + 工具链到 csv。
# ---------------------------------------------------------------------------
_BATCH_TOOL_DOMAIN = [
    ("vod", r"^vod_|^tvchannel|^live_"),
    ("music", r"^music_"),
    ("audio", r"^audio_"),
    ("education", r"^edu"),
    ("sports", r"^sports_"),
    ("children", r"^child|^kid"),
    ("device", r"^(numeric_adjust|playback_control|mode_control|power_control|"
               r"source_switch|timer_control|screen_|open$)"),
    ("qa", r"^knowledge_agent|^fan_knowledge|^qa"),
]


def _batch_tool_to_domain(tool: str) -> str:
    if not tool:
        return "(空execute)"
    for dom, pat in _BATCH_TOOL_DOMAIN:
        if re.search(pat, tool):
            return dom
    return f"(其他:{tool})"


def _batch_load_queries(path: str) -> list[tuple[str, str]]:
    """读 csv，返回 [(query, reason)]；去重、保留首列 query / 可选 reason 列。"""
    rows, seen = [], set()
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        qcol = None
        for cand in ("query", "Query", "retext", "问题", "用户query"):
            if reader.fieldnames and cand in reader.fieldnames:
                qcol = cand
                break
        if qcol is None:
            qcol = (reader.fieldnames or ["query"])[0]
        for r in reader:
            q = re.sub(r"\s+", "", (r.get(qcol, "") or "").strip())
            if not q or q in seen:
                continue
            seen.add(q)
            rows.append((q, (r.get("reason", "") or "").strip()))
    return rows


def _batch_run_one(i: int, q: str, reason: str, *, feature_code: str,
                   random_each: bool, device_id: int, client_sid: str | None,
                   tv_mode: str, debug: bool) -> dict[str, Any]:
    """批处理单条：跑一次 runtime 取首工具。

    random_each=True(--device-random-all)：本条用独立随机 device，隔离短记忆/多轮上下文；
    否则统一用传入的 device_id（--device-random 时为全体共用的同一随机 id，默认为固定 id）。
    """
    did = random_device_id() if random_each else device_id
    try:
        res = _run_runtime(q, feature_code=feature_code, device_id=did,
                           client_sid=client_sid, tv_mode=tv_mode, debug=debug)
        tools = [t.get("tool") for t in (res.get("tools") or [])]
        params = [t.get("params") for t in (res.get("tools") or [])]
        dom = _batch_tool_to_domain(tools[0] if tools else "")
        return {"i": i, "q": q, "tools": tools, "domain0": dom,
                "params": params, "reason": reason, "err": ""}
    except Exception as exc:  # noqa: BLE001
        return {"i": i, "q": q, "tools": [], "domain0": "(ERR)",
                "params": [], "reason": reason, "err": f"{type(exc).__name__}: {exc}"}


def run_batch(csv_in: str, csv_out: str, *, feature_code: str, random_each: bool,
              device_id: int, client_sid: str | None, tv_mode: str, debug: bool,
              workers: int, limit: int = 0) -> int:
    """批处理入口：csv_in → 逐条 runtime → csv_out + 首工具域分布。"""
    rows = _batch_load_queries(csv_in)
    if limit:
        rows = rows[:limit]
    print(f"批处理 {len(rows)} 条 → 远端 runtime (workers={workers})  输入={csv_in}\n")

    out: list[dict[str, Any]] = []
    dist: Counter[str] = Counter()
    errs = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            pool.submit(_batch_run_one, i, q, rs, feature_code=feature_code,
                        random_each=random_each, device_id=device_id,
                        client_sid=client_sid, tv_mode=tv_mode, debug=debug): i
            for i, (q, rs) in enumerate(rows, 1)
        }
        done = 0
        for fut in as_completed(futs):
            o = fut.result()
            out.append(o)
            done += 1
            if o["err"]:
                errs += 1
                print(f"[{o['i']:>3}] !ERR {o['q']}  ({o['err']})")
            else:
                dist[o["domain0"]] += 1
                t0n = o["tools"][0] if o["tools"] else "(空)"
                print(f"[{o['i']:>3}] {o['domain0']:<12s} {t0n or '(空)':<24s} {o['q']}")
            if done % 30 == 0:
                print(f"    ... {done}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)

    out.sort(key=lambda x: x["i"])
    with open(csv_out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["序号", "query", "首工具域", "工具链", "参数", "err", "reason"])
        for o in out:
            w.writerow([
                o["i"], o["q"], o["domain0"],
                " | ".join(str(t) for t in (o.get("tools") or [])),
                json.dumps(o.get("params"), ensure_ascii=False) if o.get("params") else "",
                o.get("err", ""), o.get("reason", ""),
            ])
    print(f"\n完成 {len(out)} 条, 用时 {time.time()-t0:.0f}s, 错误 {errs} 条")
    print(f"结果 → {csv_out}\n")
    print("=== 首工具域分布 ===")
    for d, n in dist.most_common():
        print(f"  {d:<14s} {n:>3d}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("retext", nargs="*", default=None,
                        help="用户请求(可多个；同一 device_id 依序发送，多轮/串行会话延续)。"
                             "批处理模式(--csv-in)下忽略")
    parser.add_argument("--csv-in", default=None,
                        help="批处理：输入 csv(含 query 列)，逐条跑 runtime")
    parser.add_argument("--csv-out", default=None,
                        help="批处理：输出 csv(默认 <csv-in>_runtime_out.csv)")
    parser.add_argument("--workers", type=int, default=5,
                        help="批处理并发数(默认 5)")
    parser.add_argument("--limit", type=int, default=0,
                        help="批处理只跑前 N 条(0=全部)")
    parser.add_argument("--device-random-all", action="store_true",
                        help="批处理：每条用不同的随机 device id(隔离短记忆/多轮上下文)")
    parser.add_argument("--feature-code", default=DEFAULT_FEATURE_CODE)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--device-id", default=None, metavar="SUFFIX",
                              help="用指定数字替换默认 deviceId 的末尾几位")
    device_group.add_argument("--device-random", action="store_true",
                              help="随机替换默认 deviceId 的最后 4 位")
    parser.add_argument("--client-sid", default=None)
    parser.add_argument("--tv-mode", default="0")
    parser.add_argument("--expect-steps", dest="expect_steps", type=int, default=None,
                        help="期望工具步数(最后一轮), 不满足则退出码非0(回归断言)")
    parser.add_argument("--no-debug", dest="debug", action="store_false",
                        help="关闭 debug:true(默认开, 用于从 DEBUG 帧拿到数据集命名的真工具名)")
    parser.add_argument("--debug", dest="print_sse_params", action="store_true",
                        help="调用 SSE 前将最终请求 URL 和 payload 打印到标准错误")
    parser.add_argument("--slow-agent-url", default=None,
                        help="透传给远端 runtime 的 slow_agent 回调地址(如 http://localhost:6006)；"
                             "默认不传其在 payload 中缺失")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="以 JSON 输出(多轮为数组)")
    args = parser.parse_args()

    if args.device_id is not None:
        try:
            device_id = device_id_with_suffix(args.device_id)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.device_random:
        device_id = random_device_id()
    else:
        device_id = DEFAULT_DEVICE_ID

    # 批处理模式：--csv-in 存在时逐条跑 runtime，输出 csv，忽略位置参数 retext。
    if args.csv_in:
        csv_out = args.csv_out or re.sub(r"\.csv$", "", args.csv_in) + "_runtime_out.csv"
        # device 语义：
        #   --device-random-all → 每条用不同随机 id(random_each=True)
        #   --device-random     → 全体共用同一随机 id(device_id 已在上方算好)
        #   --device-id / 默认   → 全体共用固定 id
        return run_batch(
            args.csv_in, csv_out,
            feature_code=args.feature_code,
            random_each=args.device_random_all,
            device_id=device_id,
            client_sid=args.client_sid,
            tv_mode=args.tv_mode,
            debug=args.debug,
            workers=args.workers,
            limit=args.limit,
        )

    retexts = args.retext
    if not retexts:
        print("错误: 缺少 retext(用户请求) 或 --csv-in(批处理)", file=sys.stderr)
        parser.print_usage(file=sys.stderr)
        return 2

    results: list[dict[str, Any]] = []
    for i, retext in enumerate(retexts, 1):
        if len(retexts) > 1 and not args.as_json:
            sep = "=" * 24
            print(f"\n{sep} 第 {i}/{len(retexts)} 轮: {retext} {sep}")
        r = _run_runtime(
            retext,
            feature_code=args.feature_code,
            device_id=device_id,
            client_sid=args.client_sid,
            tv_mode=args.tv_mode,
            debug=args.debug,
            slow_agent_url=args.slow_agent_url,
            print_sse_params=args.print_sse_params,
        )
        results.append(r)
        if not args.as_json:
            _show_result(r)

    if args.as_json:
        output: dict[str, Any] | list[dict[str, Any]]
        output = results[0] if len(results) == 1 else results
        print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.expect_steps is not None:
        actual = len(results[-1]["tools"])
        if actual != args.expect_steps:
            print(f"\nFAIL: expected {args.expect_steps} tool frames (最后一轮), got {actual}",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())