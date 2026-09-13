"""0901 vod 测试集 · 规则层回归 + 每条 hitSource 审计。

用 hcTools 确定性管线(badcase → rules.apply → fallback)对飞书 0901测试集/影视
(sheet0901_vod_caseset.json, 445 条)打分，验证 RuleSet 迁移零回归，并打印每条 query
命中的判定路径：
    badcase:<id> / general_rule:<rule_id> / fallback

用法:
  python tools/vod0901_rule_regress.py                # 汇总 + 前 10 条 miss
  python tools/vod0901_rule_regress.py --all          # 全部 miss 明细
  python tools/vod0901_rule_regress.py --rules        # 打印每条 query 命中规则
  python tools/vod0901_rule_regress.py -n 30          # miss 明细条数
  python tools/vod0901_rule_regress.py --json         # JSON 汇总
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "hcTools"   # hc/hcTools/
sys.path.insert(0, str(ROOT.parent))  # hc/ → from hcTools… 以包名导入
sys.path.insert(0, str(ROOT))         # hcTools/ → from hcTools… / app… 都能解析

from hcTools.domains.vod import badcase, fallback  # noqa: E402
from hcTools.domains.vod.rules import _RULE_SET    # noqa: E402
from hcTools.domains.vod import textkey            # noqa: E402

AGENT = ROOT.parent / "hcAgent"
CASESET = AGENT / "data" / "sheet0901_vod_caseset.json"

_BAD = badcase.BadcaseStore([])


def _load_badcases():
    global _BAD
    p = ROOT / "domains" / "vod" / "badcases.json"
    _BAD = badcase.BadcaseStore.load(p) if p.exists() else badcase.BadcaseStore([])


def _j_key(c):
    return json.dumps(c, ensure_ascii=False, sort_keys=True)


def canonical(x):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k == "and" and isinstance(v, list):
                out[k] = sorted((canonical(i) for i in v), key=_j_key)
            elif k == "values" and isinstance(v, list):
                out[k] = sorted(v)
            else:
                out[k] = canonical(v)
        return out
    if isinstance(x, list):
        return sorted((canonical(i) for i in x), key=_j_key)
    return x


def params_equal(a, b):
    return canonical(a) == canonical(b)


def _norm(q):
    return textkey.normalize(q)


def _badcase_rule_id(query):
    p = ROOT / "domains" / "vod" / "badcases.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return "badcase"
    entries = data.get("entries", data) if isinstance(data, dict) else data
    if isinstance(entries, dict):
        for k, v in entries.items():
            if _norm(query) == k:
                return (v.get("id") if isinstance(v, dict) and v.get("id") else "badcase")
    for e in (entries if isinstance(entries, list) else []):
        if isinstance(e, dict) and (e.get("query") == query or
                                    _norm(e.get("query", "")) == _norm(query)):
            return e.get("id", "badcase")
    return "badcase"


def _det_hit(q):
    """badcase → rules(带 rule id) → fallback。返回 (tool, params, source_label)。"""
    hit = _BAD.lookup(q)
    if hit is not None:
        return hit[0], hit[1] or {}, f"badcase:{_badcase_rule_id(q)}"
    sel = _RULE_SET.select_with_rule(q)
    if sel is not None:
        tool, params, rule = sel
        return tool, params or {}, f"general_rule:{rule.id}"
    tool, params = fallback.fallback(q)
    return tool, params or {}, "fallback"


def compute(records):
    stats = {"tool": 0, "param": 0, "both": 0, "diff": 0}
    buckets = Counter()
    hit_src = Counter()
    rows = []
    for rec in records:
        q = rec["query"]
        et, ep = rec["expected_tool"], rec.get("expected_params") or {}
        gt, gp, src = _det_hit(q)
        hit_src[src] += 1
        ok_t, ok_p = gt == et, params_equal(gp, ep)
        ok_b = ok_t and ok_p
        stats["tool"] += ok_t
        stats["param"] += ok_p
        stats["both"] += ok_b
        if ok_b:
            buckets["pass"] += 1
            continue
        stats["diff"] += 1
        if ok_t and not ok_p:
            kind = "only-param-mismatch"
        elif et == "vod_fuzzy_search" or gt == "vod_fuzzy_search":
            kind = "fuzzy-involved"
        else:
            kind = "tool-divergence"
        buckets[kind] += 1
        rows.append({"query": q, "expected_tool": et, "expected_params": ep,
                     "got_tool": gt, "got_params": gp, "kind": kind,
                     "hit": src})
    return stats, buckets, rows, hit_src


def _fmt(num, den):
    return f"{num}/{den} {num / den:.1%}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-n", type=int, default=10, help="miss 明细条数")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rules", action="store_true", help="打印每条命中规则")
    args = ap.parse_args()

    if not CASESET.exists():
        sys.exit(f"评测集不存在: {CASESET}")
    records = json.loads(CASESET.read_text(encoding="utf-8"))["records"]
    N = len(records)
    _load_badcases()

    stats, buckets, rows, hit_src = compute(records)

    if args.json:
        print(json.dumps({"n": N, "stats": stats, "buckets": dict(buckets),
                          "hit_source": dict(hit_src)}, ensure_ascii=False))
        return

    print(f"== vod 0901 规则层回归 ({N} 条, sheet0901_vod_caseset.json) ==")
    print(f"  tool      : {_fmt(stats['tool'], N)}")
    print(f"  param     : {_fmt(stats['param'], N)}")
    print(f"  tool+param: {_fmt(stats['both'], N)}   miss(bucket): {dict(buckets)}")
    print(f"  命中来源分布 hit_source:")
    for k, v in hit_src.most_common():
        print(f"    {k:<40} {v}")

    if args.rules:
        print("\n=== 每条 query 命中规则 (hitSource) ===")
        for rec in records:
            q = rec["query"]
            gt, gp, src = _det_hit(q)
            print(f"  [{rec['row']}] tool={gt:<22} hitSource={src:<34} {q!r}")

    n_show = len(rows) if (args.all or not rows) else min(args.n, len(rows))
    if n_show:
        print(f"\n=== miss 明细 (前 {n_show}) ===")
        for r in rows[:n_show]:
            print(f"  [{r['kind']}] {r['query']!r}")
            print(f"      expected={r['expected_tool']} got={r['got_tool']}")
            print(f"      expected_params={json.dumps(r['expected_params'], ensure_ascii=False)}")
            print(f"      got_params={json.dumps(r['got_params'], ensure_ascii=False)}  hit={r['hit']}")


if __name__ == "__main__":
    main()