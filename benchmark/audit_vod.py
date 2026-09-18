"""离线分析：对 sheet_rules_vod 266 条，模拟 (llm_domain=vod) 的 detect + vod rules 管道。

报告：
 1) detect(vod) 把 vod 误判成其它域的 query —— 本地 detect 可修
 2) detect=vod 但 vod rules 给的 tool != golden —— rules.py 可修
 3) detect=vod 且 tool OK，但 params 与 golden 不符 —— rules.py/dsl.py 可修
评测口径：retext 不算；canonical 去空。
"""
from __future__ import annotations
import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT = ROOT / "hcAgent"
HCT = ROOT / "hcTools"
BM = AGENT / "benchmark"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HCT))
sys.path.insert(0, str(AGENT / "app"))

from hcTools.domains import vod as _vod          # noqa: E402


def load_detect():
    spec = importlib.util.spec_from_file_location("detv", str(AGENT / "app" / "detect.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_pipe(q):
    bc = _vod.badcase.BadcaseStore.load(HCT / "domains" / "vod" / "badcases.json")
    hit = bc.lookup(q)
    if hit:
        t, p = hit
        return t, _vod.postproc.normalize(t, p or {})
    r = _vod.rules.apply(q)
    if r is not None:
        t, p, rid = r
        return t, _vod.postproc.normalize(t, p)
    t, p = _vod.fallback.fallback(q)
    return t, _vod.postproc.normalize(t, p)


def _can(x):
    if isinstance(x, dict):
        return {k: _can(v) for k, v in x.items() if k != "retext" and not _empty(v)}
    if isinstance(x, list):
        return sorted((_can(i) for i in x if not _empty(i)),
                      key=lambda e: json.dumps(e, ensure_ascii=False, sort_keys=True))
    if isinstance(x, str):
        return x.strip()
    return x


def _empty(v):
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, dict)):
        return len(v) == 0
    return False


def peq(a, b):
    return _can(a) == _can(b)


def parse_gold(raw):
    import sys as _s
    sys.path.insert(0, str(AGENT / "tools"))
    from build_sheet0901_cases import parse_params
    from sse_7domain_eval import normalize_golden_params
    return normalize_golden_params(parse_params(raw))


def main():
    det = load_detect()
    rows = list(csv.DictReader(open(BM / "cases" / "sheet_rules_vod.csv", encoding="utf-8-sig")))
    print(f"total {len(rows)}")

    # 1) detect(w=vod) 落域
    over = []
    vodrows = []
    for r in rows:
        q = r["query"]
        dom = det.detect_domain(q, "vod", tv_mode="0")
        if dom != "vod":
            over.append((q, dom, r["意图"], r["期望工具"]))
        else:
            vodrows.append(r)
    print("\n=== 1) detect(llm=vod) 把 vod 误判成其它域 ===")
    for q, dom, intent, et in over:
        print(f"  [{dom:>6}] 期望工具={et:<24} q={q!r}")
    print(f"  count={len(over)}")

    # 2/3) 对 detect=vod 的 query 求 rules 输出
    print("\n=== 2) detect=vod 但 rules tool != golden ===")
    tmiss = 0
    for r in vodrows:
        q = r["query"]
        t, p = run_pipe(q)
        if t != r["期望工具"]:
            tmiss += 1
            print(f"  exp={r['期望工具']:<22} act={t:<22} q={q!r}")
    print(f"  count={tmiss}")

    print("\n=== 3) detect=vod, tool OK, params mismatch ===")
    pmiss = 0
    for r in vodrows:
        q = r["query"]
        t, p = run_pipe(q)
        if t != r["期望工具"]:
            continue
        gp = parse_gold(r["期望参数"])
        if gp is None:
            continue
        if not peq(p or {}, gp):
            pmiss += 1
            print(f"  q={q!r}")
            print(f"    predict={json.dumps(p, ensure_ascii=False)}")
            print(f"    gold   ={json.dumps(gp, ensure_ascii=False)}")
    print(f"  count={pmiss}")


if __name__ == "__main__":
    main()