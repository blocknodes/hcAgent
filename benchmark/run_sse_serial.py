"""多意图串行 SSE 评测：喂【原始 query】一次，链路串行拆出 多个 tool（步间依赖，串行串），
评测对齐到 首步(tool1) 与 尾步(tool2) 两个 gold 槽。

口径(工具为主, 参数为辅)：
  step1_ok  = 某 step.tool == tool1 且(其参数 == param1)
  step2_ok  = 某 step.tool == tool2 且(其参数 == param2)
  both_ok   = step1_ok AND step2_ok
参数对齐用 order-insensitive canonical(params_equal)；gold 无参数时仅比工具名。
与 multiintent(并行)不同，serial 两槽工具名都精确比对(首尾是不同类工具：
vod_* 检索 vs fan_knowledge_agent)，不做同域工具族放宽。

表: 飞书 0821「多意图串行」sheet=pQYZry 100 例；
case 由 build_sheet0821_serial.py 生成 (query, tool1, param1, tool2, param2)。
用法:
  python benchmark/run_sse_serial.py            # 100 全量
  python benchmark/run_sse_serial.py -n 20 -w 6
输出:
  benchmark/output/serial_detail.csv   (row,query,step1_ok,step2_ok,both_ok,step1_pred,step2_pred)
  benchmark/output/serial_summary.csv / .json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.sse_7domain_eval import run_one, params_equal  # noqa: E402

BENCH = Path(__file__).resolve().parent
CASEFILE = BENCH / "cases" / "sheet0821_serial.csv"
OUTDIR = BENCH / "output"


def _load(s):
    if not s or not str(s).strip():
        return None
    try:
        return json.loads(s)
    except Exception:
        return {"_raw": str(s)}


def _T(b):
    return "TRUE" if b else "FALSE"


# gold 里派生的冗余字段（与 multiintent 同口径），链路未必给 → 判对时忽略。
_REDUNDANT_FIELDS = {"figures"}


def after_trim(s):
    return s.strip() if isinstance(s, str) else ""


def canonical_copy(x):
    """去掉 None / 空串值, 排序列表, 用于默认比较。"""
    if isinstance(x, dict):
        return {k: canonical_copy(v) for k, v in x.items()
                if not (v is None or (isinstance(v, str) and not v.strip()))}
    if isinstance(x, list):
        return sorted((canonical_copy(i) for i in x),
                      key=lambda e: json.dumps(e, ensure_ascii=False, sort_keys=True))
    return x


def _drop_redundant_list(items):
    """从 query.and 这类 dict 列表里剥掉承载冗余字段(figures/retext)的项。递归处理。"""
    out = []
    for it in items:
        if isinstance(it, dict):
            f = it.get("field") or it.get("k") or it.get("field_name")
            if f in _REDUNDANT_FIELDS or f == "retext":
                continue
            if any(k in _REDUNDANT_FIELDS or k == "retext" for k in it.keys()):
                if (set(it.keys()) & (_REDUNDANT_FIELDS | {"retext"})) == set(it.keys()) \
                   or it.get("field") in _REDUNDANT_FIELDS:
                    continue
        if isinstance(it, list):
            it = _drop_redundant_list(it)
        out.append(it)
    return out


_QUOTE_BLOCK = re.compile(r"[“\"]([^“”\"]{2,})[”\"]")


def _same_quote(a: str, b: str) -> bool:
    """两支 fuzzy query 是否指向同一句被引用台词/引文。

    用户口语（“X是哪部电影里的”）与检索改写（搜索包含台词X的电影）常带不同 wrapper，
    但它们检索的是同一条引语。若能各自抽出引号内的引文块且有一致（互为子串），判为等价。
    """
    def blocks(s):
        return [m.group(1) for m in _QUOTE_BLOCK.finditer(s)]
    ga = [x for x in blocks(a) if len(x) >= 2]
    gb = [x for x in blocks(b) if len(x) >= 2]
    for qa in ga:
        for qb in gb:
            if qa in qb or qb in qa:
                return True
    return False


def _params_ok(gold, pred, depth=0):
    """参数比对（与 multiintent/multiturn 同口径的宽容对齐）：
    - retext 字段完全不比（串行第一步 retext 常为整句，gold 是裁剪干净子句）；
    - query 只做子串宽容（pred 的 query 含 gold 子串即过，整句含子句）；
    - figures 等派生冗余字段忽略；一方缺字段视为可选不判负；
    - 其余结构化字段严格对齐。
    gold 为 None 时视为无参数，只看工具名。"""
    if gold is None:
        return True
    if isinstance(gold, dict) and isinstance(pred, dict):
        g = {k: v for k, v in canonical_copy(gold).items()
             if k not in _REDUNDANT_FIELDS and k != "retext"}
        p = {k: v for k, v in canonical_copy(pred).items()
             if k not in _REDUNDANT_FIELDS and k != "retext"}
        keys = set(g) | set(p)
        for k in keys:
            gv = g.get(k)
            pv = p.get(k)
            if gv is None or pv is None:
                continue  # 一侧有值另一侧无 → 可选字段不判负
            if k == "query" and isinstance(gv, str) and isinstance(pv, str):
                if after_trim(gv) and after_trim(pv):
                    if gv in pv or pv in gv:
                        continue
                    # fuzzy 引语等价：gold 与 pred 引用同一段台词/引文（wrapper 不同，
                    # 如“X是哪部电影里的” vs “搜索包含台词X的电影”）→ 视为同一次检索。
                    if _same_quote(gv, pv):
                        continue
                    return False
                continue
            if isinstance(gv, list) and isinstance(pv, list):
                gv = _drop_redundant_list(gv)
                pv = _drop_redundant_list(pv)
                # gold 每一项都须在 pred 侧匹配；允许 pred 有额外约束项
                # (链路检索常附加 category/kind 等黄金里没有的字段，不影响判定语义)。
                for a in gv:
                    if not any(_params_ok(a, b, depth + 1) for b in pv):
                        return False
                continue
            if isinstance(gv, dict) and isinstance(pv, dict):
                if not _params_ok(gv, pv, depth + 1):
                    return False
                continue
            if params_equal(gv, pv) is False:
                return False
        return True
    return params_equal(gold, pred)


def read_cases():
    cases = []
    with open(CASEFILE, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    hdr = [h.strip() for h in rows[0]]
    gi = {n: i for i, n in enumerate(hdr)}
    for r in rows[1:]:
        q = (r[gi["query"]] if "query" in gi else "").strip()
        if not q:
            continue

        def gv(n):
            i = gi.get(n)
            return (r[i] if i is not None and len(r) > i else "").strip()

        cases.append({
            "row": gv("源行号") or (r[0].strip() if r else ""),
            "query": q,
            "reason": gv("是否合理"),
            "t1": gv("tool1"),
            "p1": _load(gv("param1")),
            "t2": gv("tool2"),
            "p2": _load(gv("param2")),
        })
    return cases


def _find(steps, tool, gold_param):
    """steps 里是否有命中指定工具且参数对齐的 step；返回首个命中的 params。"""
    for s in steps:
        if (s["tool"] or "") != tool:
            continue
        if _params_ok(gold_param, s["params"]):
            return s["params"]
    return None


import urllib.request as _ur  # noqa: E402

LOCAL_URL = "http://127.0.0.1:8082/slowAgent/poc_serial"


def _post(payload):
    req = _ur.Request(LOCAL_URL, data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with _ur.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def _to_steps(body) -> list[dict]:
    out = []
    for s in (body.get("data", {}) or {}).get("steps") or []:
        out.append({
            "tool": s.get("toolName") or "",
            "retext": s.get("retext") or "",
            "params": s.get("parameters") or {},
            "hit_source": "",
        })
    return out


def run_local(case) -> dict:
    """打本地 8082 engine(慢链路)，映射成 run_one 同构的 {steps:[{tool,retext,params}]}。

    串行意图依赖 step1 执行结果，step2 需 toolHistory 续跑轮才下发（engine 的
    TraceStateMachine 按 device_id 存计划，单 POST 只发首批依赖步）。故首轮取回
    step1 后用其返回构造候选媒资(供 T1 指代改写)，回带 toolHistory 触发 step2。
    """
    dev = f"serial_dev_{case['row']}"
    body = _post({
        "traceId": f"serial_{case['row']}", "deviceId": dev,
        "data": {"query": case["query"], "tvMode": "0", "debug": True,
                 "memory": {"shortMemory": []}, "toolHistory": []},
    })
    steps = _to_steps(body)
    first = next((s for s in steps if s["tool"] in (
        "vod_search", "vod_search_all", "vod_fuzzy_search", "vod_relate_search")), None)
    if first is not None:
        # 提供 step1 的候选媒资：T1 改写用 toolHistory.result 里的候选做指代解析
        # ("评分最高的一部"→具体片名)。候选内容对评测判对无关键影响(step2 只看
        # tool==fan_knowledge_agent)，仅让指代改写可落定。
        cand = {"memoryData": [{"data": [{
            "mediaTitle": "候选影片", "director": ["导演"], "actor": ["主演"],
            "rate": 9.5, "pubdate": "2020-01-01",
        }]}]}
        history = [{"id": "s1", "toolName": first["tool"], "parameters": first["params"],
                    "result": json.dumps(cand, ensure_ascii=False)}]
        body2 = _post({
            "traceId": f"serial_{case['row']}b", "deviceId": dev,
            "data": {"query": case["query"], "tvMode": "0", "debug": True,
                     "memory": {"shortMemory": []}, "toolHistory": history},
        })
        steps += _to_steps(body2)
    return {"steps": steps}


DETAIL_HEADER = ["row", "query",
                 "gold_tool1", "gold_param1", "gold_tool2", "gold_param2",
                 "pred_tool1", "pred_param1", "pred_tool2", "pred_param2",
                 "step1_ok", "step2_ok", "both_ok"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=0)
    ap.add_argument("-w", "--workers", type=int, default=10)
    ap.add_argument("--cases", default=str(CASEFILE))
    ap.add_argument("--local", action="store_true",
                    help="打本地 8082 engine(慢链路) 而非远端 runtime")
    args = ap.parse_args()

    cases = read_cases()
    if args.n:
        cases = cases[: args.n]
    total = len(cases)
    print(f"多意图串行 SSE 评测 {total} 条 (并发 {args.workers}) ...", flush=True)

    def _one(c):
        try:
            if args.local:
                r = run_local(c)
            else:
                r = run_one({"row": c["row"], "domain": "serial", "query": c["query"]})
            return {"ok": True, "c": c, "steps": r["steps"]}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "c": c, "error": f"{type(exc).__name__}: {exc}"}

    per = []
    err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_one, c) for c in cases]
        for fut in as_completed(futs):
            res = fut.result()
            c = res["c"]
            if not res["ok"]:
                err += 1
                per.append([c["row"], c["query"],
                            c["t1"], json.dumps(c["p1"], ensure_ascii=False) if c["p1"] else "",
                            c["t2"], json.dumps(c["p2"], ensure_ascii=False) if c["p2"] else "",
                            "ERR", "", "ERR", res["error"],
                            "ERR", "ERR", "ERR"])
                continue
            steps = res["steps"]
            # predict 裸槽：取链路下发的前两个 step，不管对没对上 gold。
            pt1 = steps[0]["tool"] if steps else ""
            pp1 = steps[0]["params"] if steps else {}
            pt2 = steps[1]["tool"] if len(steps) > 1 else ""
            pp2 = steps[1]["params"] if len(steps) > 1 else {}
            # ok 判定：gold 工具是否出现且参数对齐。
            s1 = _find(steps, c["t1"], c["p1"])
            s2 = _find(steps, c["t2"], c["p2"])
            step1_ok = s1 is not None
            step2_ok = s2 is not None
            both = step1_ok and step2_ok
            per.append([c["row"], c["query"],
                        c["t1"], json.dumps(c["p1"], ensure_ascii=False) if c["p1"] else "",
                        c["t2"], json.dumps(c["p2"], ensure_ascii=False) if c["p2"] else "",
                        pt1, json.dumps(pp1, ensure_ascii=False),
                        pt2, json.dumps(pp2, ensure_ascii=False),
                        _T(step1_ok), _T(step2_ok), _T(both)])

    dn = total - err
    s1_ok = sum(1 for r in per if r[10] == "TRUE")
    s2_ok = sum(1 for r in per if r[11] == "TRUE")
    b_ok = sum(1 for r in per if r[12] == "TRUE")
    print(f"\n总={total}  错误={err}")
    print(f"step1_ok  = {s1_ok}/{dn} = {s1_ok/dn*100:.1f}%")
    print(f"step2_ok  = {s2_ok}/{dn} = {s2_ok/dn*100:.1f}%")
    print(f"both_ok   = {b_ok}/{dn} = {b_ok/dn*100:.1f}%")

    OUTDIR.mkdir(exist_ok=True)
    with open(OUTDIR / "serial_detail.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(DETAIL_HEADER)
        w.writerows(per)
    with open(OUTDIR / "serial_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "ok", "total", "pct"])
        w.writerow(["step1_ok", s1_ok, dn, f"{s1_ok/dn*100:.2f}"])
        w.writerow(["step2_ok", s2_ok, dn, f"{s2_ok/dn*100:.2f}"])
        w.writerow(["both_ok", b_ok, dn, f"{b_ok/dn*100:.2f}"])
    json.dump({"total": total, "err": err, "step1_ok": f"{s1_ok}/{dn}",
               "step2_ok": f"{s2_ok}/{dn}", "both_ok": f"{b_ok}/{dn}"},
              open(OUTDIR / "serial_summary.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()