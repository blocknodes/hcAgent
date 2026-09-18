"""设备域 sheet_rules 本地 8082 评测（快迭代用）。

对 sheet_rules_device.csv 每条 query 直连本地 hcAgent(8082) —— 与 run_one 相同：
每条 query 用随机 deviceId + 独立 traceId（无跨轮上下文），复刻 SSE 链路里
runtime → hcAgent 的编排：detect 判域 + hcTools 解析 tool/params。
retext 键不参与比对（与 run_sse_bench 口径一致）。
用法: python tools/local_device_eval.py [--ok-only] [--workers 16]
"""
from __future__ import annotations
import argparse, csv, json, sys, time, uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sse_7domain_eval import canonical, params_equal

BENCH = Path(__file__).resolve().parent.parent / "benchmark"
CASE = BENCH / "cases" / "sheet_rules_device.csv"
URL = "http://127.0.0.1:8082/slowAgent/poc_local_device"


def call_local_fresh(query: str) -> dict:
    """全新会话调用：随机 traceId + 随机 deviceId（与 run_one 一致）。"""
    tid = f"ld_{uuid.uuid4().hex[:12]}"
    did = uuid.uuid4().hex[:12]
    body = {
        "traceId": tid, "deviceId": did, "deviceType": None,
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
    return {"ok": True, "code": r.get("code"), "steps": steps}


def _eq(a, b):
    if isinstance(a, dict) and "retext" in a:
        a = {k: v for k, v in a.items() if k != "retext"}
    if isinstance(b, dict) and "retext" in b:
        b = {k: v for k, v in b.items() if k != "retext"}
    return canonical(a) == canonical(b)


def _gp(row):
    raw = (row.get("期望参数") or "").strip()
    if not raw:
        return None
    try:
        s = raw
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1].replace('""', '"')
        return json.loads(s)
    except Exception:
        try:
            return json.loads(raw)
        except Exception:
            return {"_raw": raw}


def _run(row):
    q = row["query"]
    try:
        out = call_local_fresh(q)
        pt = out["steps"][0]["tool"] if out["steps"] else ""
        pp = out["steps"][0]["params"] if out["steps"] else {}
        err = ""
    except Exception as exc:  # noqa: BLE001
        return {"query": q, "gold": row["期望工具"], "pred": "ERR",
                "tool_ok": False, "param_ok": False, "both_ok": False, "err": repr(exc)}
    gp = _gp(row)
    ot = pt == row["期望工具"]
    op = (not gp) or _eq(gp, pp)
    return {"query": q, "gold": row["期望工具"], "pred": pt,
            "tool_ok": ot, "param_ok": op, "both_ok": ot and op,
            "gold_params": gp, "pred_params": pp, "err": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ok-only", action="store_true")
    ap.add_argument("-w", "--workers", type=int, default=16)
    args = ap.parse_args()
    rows = []
    with open(CASE, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if args.ok_only and r.get("是否合理", "").strip() not in ("", "合理"):
                continue
            rows.append(r)
    n = len(rows)
    print(f"device 本地8082(fresh) {n} 条 ...", flush=True)
    res = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_run, r) for r in rows]
        for fut in as_completed(futs):
            res.append(fut.result())
    dt = time.perf_counter() - t0
    ok_t = sum(1 for r in res if r["tool_ok"])
    ok_p = sum(1 for r in res if r["param_ok"])
    ok_b = sum(1 for r in res if r["both_ok"])
    errs = sum(1 for r in res if r["err"])
    print(f"local8082f device  {n} 条, err={errs}, {dt:.0f}s")
    print(f"tool   {ok_t}/{n} = {ok_t/n*100:.2f}%")
    print(f"param  {ok_p}/{n} = {ok_p/n*100:.2f}%")
    print(f"joint  {ok_b}/{n} = {ok_b/n*100:.2f}%")
    mc = Counter((r["gold"], r["pred"]) for r in res if not r["tool_ok"])
    print("tool 失配:")
    for (g, p), m in mc.most_common():
        print(f"   {g:28}->{p:26} x{m}")
    # 保存失败用例供后续对比
    out = BENCH / "output" / "local_device_fails.json"
    fails = [r for r in res if not r["both_ok"]]
    out.write_text(json.dumps(fails, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"fails -> {out}  ({len(fails)} 条)")


if __name__ == "__main__":
    main()