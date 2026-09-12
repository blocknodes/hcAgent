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
  python3 tools/runtime_execute.py --device-id 您的device "帮我查下无间道的导演是谁，然后再搜下他的片子"
  python3 tools/runtime_execute.py --device-random "俄罗斯队，顺便声音调节"
  python3 tools/runtime_execute.py --json "query"
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import urllib.request
from typing import Any

RUNTIME_URL = "http://10.18.210.7:31392/api/runtime/execute"
DEFAULT_FEATURE_CODE = "861003009000014000000712"
DEFAULT_DEVICE_ID = "86100300900001400000071212345678"
SSE_TIMEOUT = 90  # 单次 SSE 连接超时(秒)


def random_device_id() -> str:
    """生成随机的 deviceId(每条请求独立, 避免 sessions/多样性复用)。"""
    return secrets.token_hex(8)  # 16 位十六进制随机串


def _sse(url: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=SSE_TIMEOUT) as response:
        raw = response.read().decode("utf-8")
    frames: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        text = line[len("data:"):].strip()
        if not text:
            continue
        try:
            frames.append(json.loads(text))
        except json.JSONDecodeError:
            continue
    return frames


def _summarize_debug_trace(trace: dict[str, Any]) -> dict[str, Any] | None:
    """从 DEBUG_MAP 帧的 rpc.SlowAgentApi trace 抽出计划步骤(真工具名, 数据集命名)。

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
    """拍平所有 DEBUG_MAP 计划帧, 去掉重复(同一查询在 multi-intent 里可能被 rpc 调多次)。"""
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
    """以 DEBUG_MAP 计划序列为准(慢端规划, 真工具名 vod_search 数据集命名)。

    每个 plan step 生成一条展示记录, 再从 TOOL 帧里按顺序就近补 tts与候选结果;
    plan 之外的 TOOL 帧(runtime 已消费执行流)不再展示, 避免 llmSemantic 帧重复。
    """
    plan = _plan_steps_dedup(plans)
    if not plan:
        return tools
    used: set[int] = set()
    out: list[dict[str, Any]] = []
    # 1) plan = 权威序列
    for idx, s in enumerate(plan, 1):
        rec: dict[str, Any] = {
            "tool": s["tool"],
            "retext": s.get("retext", ""),
            "params": s.get("params") or {},
            "tts": "",
            "candidates": [],
        }
        # 该步骤对应的 TOOL 帧: 同一执行流的两种描述, 数量/顺序一致,
        # 直接按顺序一一对应(zip), 参数形态不同(plan=结构化query vs 帧=memoryQuery串),
        # 字符串亲和判据会把它们判成"无关"导致媒资丢失, 不采用。
        i = idx - 1
        if i < len(tools):
            t = tools[i]
            used.add(i)
            if t.get("tts"):
                rec["tts"] = t.get("tts")
            if t.get("candidates"):
                rec["candidates"] = t.get("candidates")
            rec["frame_tool"] = t.get("tool")
        out.append(rec)
    return out


_QUEST_STOPWORDS = ("叫什么名字", "是谁", "是哪个", "是啥", "叫什么", "谁", "老婆", "妻子", "老公", "丈夫", "导演")

def _answer_key(retext: str, assistant: str) -> str:
    """从 retext 里提取一个会在回答正文中出现的片段, 用于把流式回答切到对应步骤。

    流式 TEXT 帧把每步问答的回答连在一起、无分隔; 但每步回答通常
    会带上该步问题里的实体(如"吴宇森的妻子叫牛春龙"), 用 retext 去掉
    疑问词后, 从长到短找一段在 assistant 中真实出现的子串作为锚点。
    """
    s = retext.replace("？", "").replace("?", "")
    s = s.strip("《》 「」“”‘’（）() 　")
    for _ in range(3):  # 剥掉常见疑问/称谓尾词, 留下实体
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
    """把拼接的流式回答, 按 plan 里 fan_knowledge_agent 步骤拆分。

    返回 {plan_step_index: 该步的回答}; 找不到锚点的多余步骤不返回。
    只有一条 fan 回答时直接整段给它。
    """
    if not assistant:
        return {}
    qa_idx = [i for i, t in enumerate(_plan_steps_dedup(plans)) if t["tool"] == "fan_knowledge_agent"]
    if not qa_idx:
        return {}
    if len(qa_idx) == 1:
        return {qa_idx[0]: assistant}
    # 多步问答: 用关键词锚点定位每段
    anchors: list[tuple[int, int, str]] = []
    for qi in qa_idx:
        s = _plan_steps_dedup(plans)[qi]
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


def _summarize_tool_frame(data: dict[str, Any]) -> dict[str, Any]:
    """从 runtime TOOL 帧里提取"工具名/意图 + 参数 + 结果摘要"。

    runtime 每帧 TOOL(即一次工具执行) 的 data.application_data.data 里:
      llmSemantic        -> {domain, intent} 工具域/意图
      memoryContent      -> 含记忆查询参数(memoryQuery)
      ttscontent         -> 给用户的自然语言结果
      memoryData         -> 候选媒体列表(如 vod_search 的 [{"subsort","data":[媒资]}])
      data.content.searchResultList -> 模糊搜索的候选媒资([{"total","data":[媒资]}])
      vagueTtsContent    -> 模糊搜索的定位文案
      data.showText      -> 控制类工具的确认语
    """
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
    # 阈值: 候选媒资优先记忆搜索结果, 缺(如模糊搜索 vod_vague_search 不带 memoryData)则取 data.content.searchResultList
    candidates = _memory_data_of({"application_data": app})
    if not candidates:
        candidates = _search_result_of({"application_data": app})
    candidates = candidates[:20]
    tts = (td.get("ttscontent") or "")[:120]
    if not tts:
        vague = (td.get("vagueTtsContent") or "")
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
    """从模糊搜索帧的 data.content.searchResultList 提取候选媒资(基础字段)。"""
    app = (tool_frame.get("application_data") or {})
    d = app.get("data") or {} if isinstance(app, dict) else {}
    if not isinstance(d, dict):
        return []
    content = d.get("content")
    if not isinstance(content, dict):
        return []
    rows: list[dict[str, Any]] = []
    for bucket in content.get("searchResultList") or []:
        if not isinstance(bucket, dict):
            continue
        for item in bucket.get("data") or []:
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _run_runtime(retext: str, *, feature_code: str, device_id: str,
                 client_sid: str | None, tv_mode: str = "0",
                 debug: bool = True) -> dict[str, Any]:
    """直连远端 runtime，聚合 TOOL 帧(含 llmSemantic 工具名/意图 + memoryContent 参数)。"""
    payload: dict[str, Any] = {
        "feature_code": feature_code,
        "device_id": device_id,
        "retext": retext,
        "tv_mode": tv_mode,
        "enable_slow": True,
    }
    if client_sid:
        payload["client_sid"] = client_sid
    if debug:
        payload["debug"] = True

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
        # 计划帧可能落在 DEBUG_MAP 或结束型 TEXT 帧(status=2)的 application_data.debug.traces 里
        # (播放控制类短命令常只回一条 TEXT 结束帧)，故不按 messageType 过滤，凡带 traces 的帧都解析。
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("retext", nargs="?", default=None, help="用户原始请求")
    parser.add_argument("--feature-code", default=DEFAULT_FEATURE_CODE)
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID,
                        help="固定 deviceId(多轮会话复用)")
    parser.add_argument("--device-random", action="store_true",
                        help="每次调用生成随机 device id, 等价于 --device-id random")
    parser.add_argument("--client-sid", default=None)
    parser.add_argument("--tv-mode", default="0")
    parser.add_argument("--expect-steps", dest="expect_steps", type=int, default=None,
                        help="期望工具步数, 不满足则退出码非0(回归断言)")
    parser.add_argument("--no-debug", dest="debug", action="store_false",
                        help="关闭 debug:true(默认开, 用于从 DEBUG_MAP 拿到数据集命名的真工具名)")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="以 JSON 完整输出")
    args = parser.parse_args()

    retext = args.retext
    if not retext:
        print("错误: 缺少 retext(用户请求)", file=sys.stderr)
        parser.print_usage(file=sys.stderr)
        return 2
    if args.device_random:
        device_id = random_device_id()
    else:
        device_id = args.device_id

    result = _run_runtime(
        retext,
        feature_code=args.feature_code,
        device_id=device_id,
        client_sid=args.client_sid,
        tv_mode=args.tv_mode,
        debug=args.debug,
    )
    tools = result["tools"]
    plans = result["plans"]
    # 流式 TEXT 帧拼接成完整回答(泛问答工具 fan_knowledge 的结果, 不是最终总结语);
    # 多步问答时按 plan 步骤 retext 锚点切分到各自 step 下(流式无边界, 只能近似归属)。
    assistant = "".join(result["texts"])
    assistant = assistant.replace("[DONE]", "").strip()
    step_answers = _split_plan_stream(plans, assistant)
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"\ntotal tool frames: {len(tools)}  plans: {len(plans)}  stop={result['stop']}")
        # 展示二元结构: plans 有则按 plan 顺序展示每步(已带 tts/candidates),
        # plans 为空(runtime 有时不发 DEBUG_MAP 帧)则只按 tools 原始执行帧展示,
        # 避免"啥都没有"。
        def dump_step(label: str, t: dict[str, Any], s: dict[str, Any]) -> None:
            print(f"  {label}: {t.get('tool') or s.get('toolName')}")
            retext = t.get("retext") or s.get("retext") or ""
            params = t.get("params") or s.get("parameters") or {}
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
            # 纯问答工具(无媒资候选)的结果 = 流式 TEXT 帧拼接的回答
            if not t.get("candidates") and step_answers.get(ti - 1):
                print(f"           answer: {step_answers[ti - 1]}")

        ti = 0
        if not plans:
            # 没 plan: 直接展示每条 TOOL 执行帧(流式回答整段给最后一个无候选的)
            for i, t in enumerate(tools, 1):
                dump_step(f"tool {i}", t, t)
        else:
            # 有 plan: 以 plan step 为权威序列展示
            for pi, d in enumerate(plans, 1):
                steps = d.get("steps") or []
                print(f"  plan {pi}: {len(steps)} steps")
                for s in steps:
                    if not isinstance(s, dict) or not s.get("toolName"):
                        continue
                    t = tools[ti] if ti < len(tools) else {}
                    ti += 1
                    dump_step(f"step {ti}", t, s)
    if args.expect_steps is not None and len(tools) != args.expect_steps:
        print(f"\nFAIL: expected {args.expect_steps} tool frames, got {len(tools)}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())