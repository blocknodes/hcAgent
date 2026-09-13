"""多轮(会话)SSE 真实链路评测：飞书「0821多意图&多业务」多轮子表。

每张 query 处于一个**会话语境**(多轮依赖：第 N 轮的意图要承接前 N-1 轮累积的约束)。
评测必须按会话组织，把同会话的多轮 query **顺序**喂给远端 runtime，让链路在同一
设备会话内累积上下文，再逐轮比对工具 + 参数。

device id 规则(用户要求)：
  - **同一会话 id** 内所有轮次用**同一个** device_id
  - **不同会话 id** 用**不同的** device_id
  为可复现，device_id 由会话 id 确定性派生，不依赖随机。

入口:
  python benchmark/run_sse_multiturn.py               # 全量 30 会话 × 5 轮 = 150 条
  python benchmark/run_sse_multiturn.py -n 2          # 只跑前 2 个会话(冒烟)
  python benchmark/run_sse_multiturn.py -b 影视,少儿   # 只跑指定业务
  python benchmark/run_sse_multiturn.py -w 4          # 并发会话数(默认 4)
输出:
  benchmark/output/detail_multiturn.csv  # 逐轮明细(会话id/轮次/gold/pred/tool_ok/param_ok/both_ok)
  benchmark/output/summary_multiturn.csv # 按会话 + 按业务 + 全局汇总
  benchmark/output/summary_multiturn.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BENCH = Path(__file__).resolve().parent
AGENT = BENCH.parent
OUTDIR = BENCH / "output"
_CSV = BENCH / "cases" / "sheet0821_multiturn.csv"

sys.path.insert(0, str(AGENT))
from runtime_execute import _run_runtime, DEFAULT_FEATURE_CODE  # noqa: E402
from tools.sse_7domain_eval import canonical, params_equal       # noqa: E402


def _params(v):
    """解析 CSV 里的参数串(可能空 / 多行 JSON) -> dict 或 {_raw:..}。"""
    if not v or not v.strip():
        return {}
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return {"_raw": v}   # 不可比 -> 只比 tool


def device_id_for(session_id: str) -> str:
    """由会话 id 派生稳定 device_id：同会话 id 恒同、异会话 id 恒异、可复现。"""
    h = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    # 沿用 runtime 示例的 32 位数字签名 (861003009000014000000712 + 16 hex)
    return "861003009000014000000712" + h[:16]


def load_sessions():
    """读 CSV -> [{session_id,biz,grp,device_id,turns:[...]}], 每会话按轮次排好。"""
    with open(_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    sessions: dict[str, dict] = {}
    for r in rows[1:]:
        if len(r) < 7:
            continue
        sess_id, biz, grp, rnd, q, et, ep = (x.strip() for x in r[:7])
        if not q:
            continue
        # 会话分组与 device id 一律以“会话id”为准：同会话id复用同一个 device id，
        # 不同会话 id 使用不同的 device id。
        s = sessions.setdefault(sess_id, {
            "session_id": sess_id, "biz": biz, "grp": grp,
            "device_id": device_id_for(sess_id),
            # 多轮会话标记(client_sid)：同会话各轮共享，让远端 runtime 维护跨轮上下文。
            "client_sid": f"mt_{sess_id}", "turns": [],
        })
        s["turns"].append({
            "round": rnd, "query": q, "row": r[7] if len(r) > 7 else "",
            "expected_tool": et, "expected_params": _params(ep),
        })
    for s in sessions.values():
        s["turns"].sort(key=lambda t: t["round"])
    return list(sessions.values())


def run_session(sess) -> dict:
    """顺序轮完一个会话的全部轮次，共享同一 device_id + client_sid(多轮上下文)。"""
    return {
        "session": sess,
        "turns": [_run_turn(t, sess["device_id"], sess.get("client_sid")) for t in sess["turns"]],
    }


def _run_turn(turn, device_id, client_sid=None):
    started = time.perf_counter()
    try:
        r = _run_runtime(turn["query"], feature_code=DEFAULT_FEATURE_CODE,
                         client_sid=client_sid, device_id=device_id, tv_mode="0", debug=True)
        steps = [
            {"tool": s.get("toolName") or "", "retext": s.get("retext", ""),
             "params": s.get("parameters") or {},
             "hit_source": s.get("hitSource") or s.get("hit_source") or ""}
            for p in (r.get("plans") or []) for s in (p.get("steps") or [])
            if isinstance(s, dict) and s.get("toolName")
        ]
        if not steps:
            steps = [{"tool": t.get("tool") or "", "retext": t.get("retext", ""),
                      "params": t.get("params") or {},
                      "hit_source": t.get("hitSource") or t.get("hit_source") or ""}
                     for t in (r.get("tools") or [])]
        return {"ok": True, "turn": turn,
                "pred_tool": (steps[0]["tool"] if steps else ""),
                "pred_params": (steps[0]["params"] or {} if steps else {}),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "turn": turn, "error": f"{type(exc).__name__}: {exc}"}


def _param_repr(p):
    return json.dumps(p, ensure_ascii=False, sort_keys=True) if p else ""


# ---- 参数比对口径(与 run_sse_multiintent 一致) ----
# retext 字段完全不看(用户已定：retext 不作为评判标准，不影响准确率)；
# figures 等派生冗余字段也忽略；query 检索词子串宽容(整句含子句)。其余结构化字段严格对齐。
_REDUNDANT_FIELDS = {"figures"}


def after_trim(s):
    return s.strip() if isinstance(s, str) else ""


def canonical_copy(x):
    """去掉 None / 空串值, 排序列表。"""
    if isinstance(x, dict):
        return {k: canonical_copy(v) for k, v in x.items()
                if not (v is None or (isinstance(v, str) and not v.strip()))}
    if isinstance(x, list):
        return sorted((canonical_copy(i) for i in x), key=lambda e: json.dumps(e, ensure_ascii=False, sort_keys=True))
    return x


def _drop_redundant_list(items):
    """从 query.and 这类 dict 列表里剥掉承载冗余字段(figures/retext)的项。递归。"""
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


def _params_ok(gold, pred, depth=0):
    """参数对齐：retext 完全不看；figures 冗余字段忽略；query 字符串子串宽容；其余严格。"""
    if isinstance(gold, dict) and isinstance(pred, dict):
        g = {k: v for k, v in canonical_copy(gold).items()
             if k not in _REDUNDANT_FIELDS and k != "retext"}
        p = {k: v for k, v in canonical_copy(pred).items()
             if k not in _REDUNDANT_FIELDS and k != "retext"}
        for k in set(g) | set(p):
            gv = g.get(k)
            pv = p.get(k)
            if gv is None or pv is None:
                continue  # 一方有值另一方无 → 视为可选字段不判负
            if k == "query" and isinstance(gv, str) and isinstance(pv, str):
                gs, ps = after_trim(gv), after_trim(pv)
                if gs and ps and (gs in ps or ps in gs):
                    continue
                return False
            if isinstance(gv, list) and isinstance(pv, list):
                gv2 = _drop_redundant_list(gv)
                pv2 = _drop_redundant_list(pv)
                if len(gv2) != len(pv2):
                    return False
                for a, b in zip(gv2, pv2):
                    if not _params_ok(a, b, depth + 1):
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


def score(turn, pred_tool, pred_params):
    """-> (tool_ok, param_ok, both_ok, diff)。gold 无参数则 param 由 tool 决定。"""
    et = (turn.get("expected_tool") or "").strip()
    gp = turn.get("expected_params") or {}
    if isinstance(gp, dict) and set(gp.keys()) == {"_raw"}:
        gp = None
    ot = bool(et) and bool(pred_tool) and pred_tool == et
    op = (not gp) or _params_ok(pred_params, gp)
    ob = ot and op
    diff = ""
    if not ot:
        diff = f"tool: gold={et} pred={pred_tool}"
    elif op:
        diff = ""
    else:
        gpc, ppc = canonical(gp), canonical(pred_params)
        if isinstance(gpc, dict) and isinstance(ppc, dict):
            keys = sorted(set(gpc) | set(ppc))
            parts = [f"{k}: gold={_param_repr(gpc.get(k))} pred={_param_repr(ppc.get(k))}"
                     for k in keys if gpc.get(k) != ppc.get(k)]
            diff = "; ".join(parts)
        else:
            diff = f"gold={_param_repr(gp)} pred={_param_repr(pred_params)}"
    return ot, op, ob, diff


# 逐轮明细列(prefix: 会话 列)
DETAIL_HEADER = ["会话id", "业务", "组号", "轮次", "query", "源行",
                 "gold_tool", "gold_params", "pred_tool", "pred_params",
                 "tool_ok", "param_ok", "both_ok", "latency_ms", "param_diff"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-b", "--biz", default=None, help="逗号分隔业务: 影视,少儿,设备控制,教育,音乐,体育")
    ap.add_argument("-w", "--workers", type=int, default=4, help="并发会话数")
    ap.add_argument("-n", type=int, default=0, help="只跑前 N 个会话(按会话id升序, 冒烟)")
    ap.add_argument("-c", "--config", default=None, help="覆盖 cases 数据集路径")
    ap.add_argument("--json", action="store_true", help="机器可读汇总输出")
    args = ap.parse_args()

    global _CSV
    if args.config:
        _CSV = Path(args.config)

    sessions = load_sessions()
    if args.biz:
        want = {x.strip() for x in args.biz.split(",") if x.strip()}
        sessions = [s for s in sessions if s["biz"] in want]
    sessions.sort(key=lambda s: (s["biz"], int(s["grp"]) if s["grp"].isdigit() else s["grp"]))
    if args.n:
        sessions = sessions[: args.n]

    print(f"SSE 多轮评测: {len(sessions)} 会话, {sum(len(s['turns']) for s in sessions)} 轮 "
          f"(并发 {args.workers}) ...", flush=True)

    detail: list[list] = []
    total_ok_t = total_ok_b = err = 0
    per_biz_t: Counter = Counter()
    per_biz_b: Counter = Counter()
    per_sess_t: dict = {}
    per_sess_b: dict = {}
    done = 0

    def fold(res):
        nonlocal total_ok_t, total_ok_b, err, done
        s = res["session"]
        sid = s["session_id"]
        biz = s["biz"]
        ok_t = ok_b = 0
        for t in res["turns"]:
            if not t["ok"]:
                err += 1
                detail.append([sid, s["device_id"], s["grp"], t["turn"]["round"],
                               t["turn"]["query"], t["turn"]["row"],
                               "", "", "", "", "ERR", "ERR", "ERR", "", t.get("error", "")])
                continue
            tt = t["turn"]
            et = (tt.get("expected_tool") or "").strip()
            gp = tt.get("expected_params") or {}
            if isinstance(gp, dict) and set(gp.keys()) == {"_raw"}:
                gp = None
            ot, op, ob, diff = score(tt, t["pred_tool"], t["pred_params"])
            total_ok_t += ot
            total_ok_b += ob
            ok_t += ot
            ok_b += ob
            per_biz_t[biz] += ot
            per_biz_b[biz] += ob
            detail.append([sid, s["device_id"], s["grp"], tt["round"], tt["query"], tt["row"],
                           et, _param_repr(gp), t["pred_tool"], _param_repr(t["pred_params"]),
                           "Y" if ot else "N", "Y" if op else "N", "Y" if ob else "N",
                           t.get("latency_ms", ""), diff])
        done += 1
        per_sess_t[sid] = ok_t
        per_sess_b[sid] = ok_b
        n_done = done * len(res["turns"])  # 已完成轮数(近似, 会话内全跑完才计)
        print(f"\r[{done}/{len(sessions)} 会话] tool_acc={total_ok_t/n_done*100:.1f}%  "
              f"both_acc={total_ok_b/n_done*100:.1f}%  err={err}",
              end="", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for fut in as_completed([pool.submit(run_session, s) for s in sessions]):
            try:
                fold(fut.result())
            except Exception as exc:  # noqa: BLE001
                err += 1
                print(f"\n[会话异常] {exc}")
    print()

    n_turns = sum(len(s["turns"]) for s in sessions)
    n_ok = n_turns - err

    # 业务级轮数
    biz_total = Counter()
    for s in sessions:
        biz_total[s["biz"]] += len(s["turns"])

    # ---- 汇总 ----
    if args.json:
        print(json.dumps({
            "sessions": len(sessions), "turns": n_turns, "errors": err,
            "by_biz": {b: {"turns": biz_total[b], "tool_ok": per_biz_t[b], "both_ok": per_biz_b[b]}
                       for b in biz_total},
            "summary": {
                "tool_ok": total_ok_t, "tool_acc": round(total_ok_t / n_ok * 100, 1) if n_ok else 0,
                "both_ok": total_ok_b, "both_acc": round(total_ok_b / n_ok * 100, 1) if n_ok else 0,
            },
        }, ensure_ascii=False, indent=2))
        return

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / "detail_multiturn.csv"
    with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(DETAIL_HEADER)
        w.writerows(detail)

    # 会话级汇总
    rows = [["会话id", "业务", "组号", "轮数", "tool_ok", "both_ok", "会话tool%", "会话both%"]]
    for s in sessions:
        sid = s["session_id"]
        n = len(s["turns"])
        rows.append([sid, s["biz"], s["grp"], n,
                     f"{per_sess_t.get(sid, 0)}/{n}", f"{per_sess_b.get(sid, 0)}/{n}",
                     f"{per_sess_t.get(sid, 0)/n*100:.1f}" if n else "0",
                     f"{per_sess_b.get(sid, 0)/n*100:.1f}" if n else "0"])
    # 业务级汇总
    rows.append([])
    rows.append(["业务", "轮数", "tool_ok", "both_ok", "tool%", "both%"])
    for b, n in biz_total.items():
        rows.append([b, n, per_biz_t[b], per_biz_b[b],
                     f"{per_biz_t[b]/n*100:.1f}", f"{per_biz_b[b]/n*100:.1f}"])
    # 全局
    rows.append([])
    rows.append(["全局", n_turns, total_ok_t, total_ok_b,
                 f"{total_ok_t/n_ok*100:.1f}%", f"{total_ok_b/n_ok*100:.1f}%"])
    summary_path = OUTDIR / "summary_multiturn.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)

    with open(OUTDIR / "summary_multiturn.json", "w", encoding="utf-8") as f:
        json.dump({
            "sessions": len(sessions), "turns": n_turns, "errors": err,
            "by_biz": {b: {"turns": biz_total.get(b, 0), "tool_ok": per_biz_t[b],
                           "both_ok": per_biz_b[b]} for b in biz_total},
            "summary": {"tool_ok": total_ok_t, "tool_acc": round(total_ok_t / n_ok * 100, 1) if n_ok else 0,
                        "both_ok": total_ok_b, "both_acc": round(total_ok_b / n_ok * 100, 1) if n_ok else 0},
        }, ensure_ascii=False, indent=2, fp=f)

    print(f"\n逐轮明细 -> {detail_path}")
    print(f"会话/业务/全局汇总 -> {summary_path}")
    print(f"全局: {n_turns} 轮, 错误 {err}, tool_acc={total_ok_t/n_ok*100:.1f}%, "
          f"tool+param_acc={total_ok_b/n_ok*100:.1f}%")


if __name__ == "__main__":
    main()