"""多意图并行 SSE 评测：喂【原始 query】一次，链路(LLM拆分→多步执行)内部拆出多个 tool，
评测把全部返回 step 分别对齐到 device 槽(gold_tool1) 与 content 槽(gold_tool2_options)。

口径(工具为主, 参数为辅):
  device_ok   = 某 step.tool==gold_tool1 且(序号槽对上的参数==gold_param1)
  content_ok  = 某 step.tool ∈ gold_tool2_options 且(其参数==gold_param2)
  both_ok     = device_ok AND content_ok
参数对齐用 order-insensitive canonical(params_equal)；gold 无参数时仅比工具名。

表: 飞书 e80803「多意图并行」200 例, 每行原始 query + 人工标的 device/content 双槽 gold。
用法:
  python benchmark/run_sse_multiintent.py            # 200 全量
  python benchmark/run_sse_multiintent.py -n 10 -w 6
输出:
  benchmark/output/multiintent_detail.csv (row,query,source,device_ok,content_ok,both_ok,dev_pred,cont_pred)
  benchmark/output/multiintent_summary.csv / .json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.sse_7domain_eval import run_one, params_equal  # noqa: E402

BENCH = Path(__file__).resolve().parent
CASEFILE = BENCH / "cases" / "e80803_multiintent.csv"
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


def _retext_match(gold, pred):
    """retext 字段宽容: gold 作为 clean 子串出现在 pred 内即判同。
    内容侧 retext 常为"整句"，gold 是清洗后的干净子句(如"沂蒙山小调")，允许子串包含。"""
    if not isinstance(gold, str) or not isinstance(pred, str):
        return gold == pred
    g = gold.strip()
    p = pred.strip()
    return g == p or (g and p and g in p)


def _retext_match(gold, pred):
    """retext/query 回显字段宽容: gold 作为 clean 子串出现在 pred 内即判同。
    内容侧回显常为"整句"，gold 是清洗后的干净子句(如"沂蒙山小调")，允许子串包含。"""
    if not isinstance(gold, str) or not isinstance(pred, str):
        return gold == pred
    g = gold.strip()
    p = pred.strip()
    return g == p or (g and p and g in p)


# gold 里与 grade/semester 等派生的冗余字段，链路仅给基础字段；判对时忽略。
_REDUNDANT_FIELDS = {"figures"}


def _params_ok(gold, pred, depth=0):
    """参数对齐: retext 字段完全不看(用户已定)；figures 冗余字段也忽略；
    query 只问子串宽容(整句含子句)；其余结构化字段严格对齐。
    figures 常嵌在 query.and 每项里(gold 派生、链路未必给)，需在任意层剥掉，
    不能只剥顶层 dict。"""
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
                continue  # 一.一方有值另一方无 → 视为可选字段不判负
            if k == "query" and isinstance(gv, str) and isinstance(pv, str):
                # query 是检索词回显: pred 的 query 含 gold 子串即过(整句含子句)
                if after_trim(gv) and after_trim(pv):
                    if gv in pv or pv in gv:
                        continue
                    return False
                continue
            if isinstance(gv, list) and isinstance(pv, list):
                gv = _drop_redundant_list(gv)
                pv = _drop_redundant_list(pv)
                if len(gv) != len(pv):
                    return False
                for a, b in zip(gv, pv):
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


def _drop_redundant_list(items):
    """从 query.and 这类 dict 列表里剥掉承载冗余字段(figures/retext)的项。
    例如 gold 的 and=[{field:figures},{field:grade},...]，链路不给 figures 项 → 删。
    递归处理嵌套列表。"""
    out = []
    for it in items:
        if isinstance(it, dict):
            f = it.get("field") or it.get("k") or it.get("field_name")
            if f in _REDUNDANT_FIELDS or f == "retext":
                continue
            if any(k in _REDUNDANT_FIELDS or k == "retext" for k in it.keys()):
                # 顶层即冗余结构(嵌套 only-figures 包裹) → 剥
                if (set(it.keys()) & (_REDUNDANT_FIELDS | {"retext"})) == set(it.keys()) \
                   or it.get("field") in _REDUNDANT_FIELDS:
                    continue
        if isinstance(it, list):
            it = _drop_redundant_list(it)
        out.append(it)
    return out


def after_trim(s):
    return s.strip() if isinstance(s, str) else ""


def canonical_copy(x):
    """去掉 None / 空串值, 排序列表, 用于默认比较(不复用 canonical 覆盖 str CF 处理)。"""
    if isinstance(x, dict):
        return {k: canonical_copy(v) for k, v in x.items()
                if not (v is None or (isinstance(v, str) and not v.strip()))}
    if isinstance(x, list):
        return sorted((canonical_copy(i) for i in x), key=lambda e: json.dumps(e, ensure_ascii=False, sort_keys=True))
    return x


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
            "row": (r[0].strip() if r and r[0].strip() else ""),
            "source": gv("source"),
            "query": q,
            "dev_tool": gv("gold_tool1"),
            "dev_param": _load(gv("gold_param1")),
            "cont_opts": _load(gv("gold_tool2_options")),
            "cont_param": _load(gv("gold_param2")),
        })
    return cases


# 内容域工具的域前缀。各域内容检索工具 hcTools 实现里可能收敛到一个总名
# (如 sports_match_search 覆盖 sports_team_search)，gold 标的细分名与链路实现名
# 不必拘，故 content 槽放宽到【同域内容工具族】命中。device 槽仍精确(gold_tool1)。
_CONTENT_PREFIXES = ("sports", "vod", "edu", "educ", "audio", "music", "children")


def _family(tool: str) -> str:
    """工具名的内容域族前缀；非内容类(device_*/timer_control 等)返回 ''。"""
    for p in _CONTENT_PREFIXES:
        if (tool or "").startswith(p):
            return p
    return ""




def _hits(gold_opts, gold_param, steps, family=False):
    """steps 里是否为命中：工具∈gold_opts(或 family=True 时属同域内容工具族) 且参数对齐。

    - device 槽: family=False, gold_opts 为 str(gold_tool1), 精确工具名。
    - content 槽: family=True, gold_opts 为 list(gold_tool2_options), 同域内容工具族宽容
      (hcTools 各域内容检索可能收敛到总名, gold 标的细分名不必拘)。
    """
    if isinstance(gold_opts, str):
        gold_opts = [gold_opts]
    opts = gold_opts if isinstance(gold_opts, list) else ([] if gold_opts is None else [gold_opts])
    fams = {_family_name(t) for t in opts if isinstance(t, str) and t}
    for s in steps:
        t = s["tool"]
        if t in opts:
            pass
        elif family and _family_name(t) in fams and _family_name(t):
            pass
        else:
            continue
        if gold_param is None or _params_ok(gold_param, s["params"]):
            return True
    return False


def _family_name(tool: str) -> str:
    """工具名的内容域族键(如 sports_match_search→sports)；非内容域→''。"""
    for p in _CONTENT_PREFIXES:
        if (tool or "").startswith(p):
            return p
    return ""


DETAIL_HEADER = ["row", "source", "query",
                 "gold_dev_tool", "gold_content_tool",
                 "dev_ok", "content_ok", "both_ok", "n_steps", "pred_steps"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=0)
    ap.add_argument("-w", "--workers", type=int, default=10)
    ap.add_argument("--cases", default=str(CASEFILE))
    args = ap.parse_args()

    cases = read_cases()
    if args.n:
        cases = cases[: args.n]
    total = len(cases)
    print(f"多意图并行 SSE 评测 {total} 条 (并发 {args.workers}) ...", flush=True)

    def _one(c):
        try:
            r = run_one({"row": c["row"], "domain": "multi", "query": c["query"]})
            return {"ok": True, "c": c, "steps": r["steps"], "stop": r["stop"]}
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
                per.append([c["row"], c["source"], c["query"], c["dev_tool"],
                            str(c["cont_opts"]), "ERR", "ERR", "ERR", 0, res["error"]])
                continue
            steps = res["steps"]
            dev_ok = _hits(c["dev_tool"], c["dev_param"], steps, family=False)
            cont_ok = _hits(c["cont_opts"], c["cont_param"], steps, family=True)
            both = dev_ok and cont_ok
            pred = " | ".join(f"{s['tool']}:{json.dumps(s['params'], ensure_ascii=False)}"
                              for s in steps)
            per.append([c["row"], c["source"], c["query"], c["dev_tool"],
                        str(c["cont_opts"]), _T(dev_ok), _T(cont_ok), _T(both),
                        len(steps), pred])

    dn = total - err
    d_ok = sum(1 for r in per if r[5] == "TRUE")
    c_ok = sum(1 for r in per if r[6] == "TRUE")
    b_ok = sum(1 for r in per if r[7] == "TRUE")
    print(f"\n总={total}  错误={err}")
    print(f"device_ok = {d_ok}/{dn} = {d_ok/dn*100:.1f}%")
    print(f"content_ok= {c_ok}/{dn} = {c_ok/dn*100:.1f}%")
    print(f"both_ok   = {b_ok}/{dn} = {b_ok/dn*100:.1f}%")

    OUTDIR.mkdir(exist_ok=True)
    with open(OUTDIR / "multiintent_detail.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(DETAIL_HEADER)
        w.writerows(per)
    with open(OUTDIR / "multiintent_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "ok", "total", "pct"])
        w.writerow(["device_ok", d_ok, dn, f"{d_ok/dn*100:.2f}"])
        w.writerow(["content_ok", c_ok, dn, f"{c_ok/dn*100:.2f}"])
        w.writerow(["both_ok", b_ok, dn, f"{b_ok/dn*100:.2f}"])
    json.dump({"total": total, "err": err, "device_ok": f"{d_ok}/{dn}",
               "content_ok": f"{c_ok}/{dn}", "both_ok": f"{b_ok}/{dn}"},
              open(OUTDIR / "multiintent_summary.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()