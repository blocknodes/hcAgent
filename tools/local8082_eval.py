"""用本地 8082 编排接口(/slowAgent)重跑失败用例, 输出 CSV+汇总。

   python tools/local8082_eval.py                          # 跑 data/sse0827_diffs.json 全部失败用例
   python tools/local8082_eval.py -o retry8082.csv
   python tools/local8082_eval.py -w 8
"""
from __future__ import annotations
import argparse, csv, json, sys, time, uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import urllib.request

TOOLS = Path(__file__).resolve().parent
AGENT = TOOLS.parent
DIFFS = AGENT / "data" / "sse0827_diffs.json"
CASESET = AGENT / "data" / "sheet0827_caseset.json"
URL = "http://127.0.0.1:8082/slowAgent/poc_local8082"

sys.path.insert(0, str(AGENT))
from tools.sse_7domain_eval import (   # noqa: E402 复用比对函数
    canonical, params_equal, normalize_golden_params,
)


def call_local(query: str, trace_id: str | None = None) -> dict:
    # /slowAgent 是 traceId 维度的高状态编排机：同一个 traceId 视为续跑。
    # 评测里每条独立 query 必须给唯一 traceId，否则会复用上一条 trace 的 plan→参数串号。
    if trace_id is None:
        trace_id = f"ev_{uuid.uuid4().hex[:12]}"
    body = {
        "traceId": trace_id, "deviceId": "d1",
        "data": {"query": query, "tvMode": "0", "debug": True},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        r = json.loads(resp.read().decode("utf-8"))
    steps = []
    for it in (r.get("data") or {}).get("steps") or []:
        steps.append({
            "tool": it.get("toolName") or "",
            "retext": it.get("retext", ""),
            "params": it.get("parameters") or {},
        })
    return {"ok": True, "steps": steps, "raw_code": r.get("code")}


def run_one(row):
    q = row["query"]
    try:
        out = call_local(q)
    except Exception as exc:
        return {"row": row["row"], "query": q, "gold": row["gold"], "pred": "ERR",
                "tool_ok": False, "param_ok": False, "joint_ok": False,
                "error": f"{type(exc).__name__}: {exc}"}
    pred_tool = out["steps"][0]["tool"] if out["steps"] else ""
    pred_params = out["steps"][0]["params"] if out["steps"] else {}
    # golden params 从 caseset 依 row 取
    et = row.get("gold") or ""
    gparams = row.get("gold_params")
    normalize = normalize_golden_params(gparams)
    ok_t = pred_tool == et
    ok_p = (not normalize) or params_equal(pred_params, normalize)
    return {
        "row": row["row"], "domain": row.get("domain", ""), "query": q,
        "gold": et, "pred": pred_tool,
        "gold_params": json.dumps(normalize, ensure_ascii=False) if normalize else "",
        "pred_params": json.dumps(pred_params, ensure_ascii=False),
        "tool_ok": ok_t, "param_ok": ok_p, "joint_ok": ok_t and ok_p,
        "error": "",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="local8082_retry.csv",
                    help="输出文件名(data/[name])")
    ap.add_argument("-w", "--workers", type=int, default=8)
    ap.add_argument("--src-json", default=str(DIFFS), help="失败用例来源 json")
    args = ap.parse_args()

    src = json.loads(Path(args.src_json).read_text("utf-8"))
    diffs = src["diffs"] if isinstance(src, dict) else src
    print(f"失败用例 {len(diffs)} 条, 调本地 8082 重跑...")

    # caseset 依 row 取 golden params
    cs = json.loads(CASESET.read_text("utf-8"))
    records = cs["records"]
    by_row = {r["row"]: r for r in records}
    for row in diffs:
        gi = by_row.get(row["row"], {})
        row["expected_tool"] = gi.get("expected_tool", row.get("gold"))
        row["expected_params"] = gi.get("expected_params")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = [r for r in pool.map(lambda x: run_one(x), diffs) if r]

    # 汇总
    tot_t = sum(1 for r in results if r["tool_ok"])
    tot_j = sum(1 for r in results if r["joint_ok"])
    errs = [r for r in results if r.get("error") or r["pred"] == "ERR"]
    print(f"\n调 8082 重跑 {len(diffs)} 条失败用例:")
    print(f"  tool 对 {tot_t}/{len(diffs)}  joint 对 {tot_j}/{len(diffs)}  ERR {len(errs)}")

    wby = Counter((r["gold"], r["pred"]) for r in results if not r["tool_ok"])
    if wby:
        print("  仍错分布(gold->pred):")
        for (g, p), n in wby.most_common():
            print(f"    {g:28}->{p:26} n={n}")

    out_path = AGENT / "data" / args.out
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=[
            "row", "domain", "query", "gold", "pred",
            "gold_params", "pred_params", "tool_ok", "param_ok", "joint_ok", "error"])
        wr.writeheader()
        for r in results:
            wr.writerow(r)
    print(f"\nCSV → {out_path}")


if __name__ == "__main__":
    main()