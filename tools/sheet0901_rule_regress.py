"""0901 多域测试集 · 规则层确定性回归 + 每条 hitSource 审计。

对 hcTools 各域确定性管线(badcase → rules.apply → fallback)打分，验证 RuleSet 迁移
零回归，并打印每条 query 命中的判定路径(badcase / general_rule:<id> / fallback)。

用法:
  python tools/sheet0901_rule_regress.py                  # 7 域汇总 + 每域前5条miss
  python tools/sheet0901_rule_regress.py -d vod,music    # 只跑指定域
  python tools/sheet0901_rule_regress.py --rules         # 打印每条 query 命中规则
  python tools/sheet0901_rule_regress.py --all           # 全部 miss 明细
  python tools/sheet0901_rule_regress.py -n 30           # miss 明细条数
  python tools/sheet0901_rule_regress.py --json          # JSON 汇总
"""
from __future__ import annotations
import argparse
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

HCTOOLS = Path(__file__).resolve().parents[2] / "hcTools"   # hc/hcTools/
sys.path.insert(0, str(HCTOOLS.parent))
sys.path.insert(0, str(HCTOOLS))
AGENT = Path(__file__).resolve().parent.parent

DOMAINS = ["vod", "device", "audio", "music", "sports", "children", "education"]
CN = {"vod": "影视", "device": "设备", "audio": "有声", "music": "音乐",
      "sports": "体育", "children": "少儿", "education": "教育"}


def _jkey(c):
    return json.dumps(c, ensure_ascii=False, sort_keys=True)


def canonical(x):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k == "and" and isinstance(v, list):
                out[k] = sorted((canonical(i) for i in v), key=_jkey)
            elif k == "values" and isinstance(v, list):
                out[k] = sorted(v)
            else:
                out[k] = canonical(v)
        return out
    if isinstance(x, list):
        return sorted((canonical(i) for i in x), key=_jkey)
    return x


def params_equal(a, b):
    return canonical(a) == canonical(b)


_pipe_cache = {}


def _load_pipeline(domain):
    if domain in _pipe_cache:
        return _pipe_cache[domain]
    mod = importlib.import_module(f"hcTools.domains.{domain}")
    badcase_mod = importlib.import_module(f"hcTools.domains.{domain}.badcase")
    bad = badcase_mod.BadcaseStore([])
    p = HCTOOLS / f"domains/{domain}/badcases.json"
    if p.exists():
        bad = badcase_mod.BadcaseStore.load(p)
    rules_mod = importlib.import_module(f"hcTools.domains.{domain}.rules")
    # 各域规则表命名不一：vod/device 用 _RULE_SET，其余用 RULE_SET
    rule_set = getattr(rules_mod, "_RULE_SET", None) or getattr(rules_mod, "RULE_SET", None)
    fallback_mod = importlib.import_module(f"hcTools.domains.{domain}.fallback")
    _pipe_cache[domain] = (bad, rule_set, fallback_mod)
    return _pipe_cache[domain]


def detect_hit(domain, q):
    bad, rule_set, fallback_mod = _load_pipeline(domain)
    hit = bad.lookup(q)
    if hit is not None:
        return hit[0], hit[1] or {}, "badcase"
    if rule_set is not None:
        sel = rule_set.select_with_rule(q)
        if sel is not None:
            tool, params, rule = sel
            return tool, params or {}, f"general_rule:{rule.id}"
    tool, params = fallback_mod.fallback(q)
    return tool, params or {}, "fallback"


def evaluate(records, domain):
    stats = {"tool": 0, "param": 0, "both": 0, "diff": 0}
    buckets = Counter()
    hit_src = Counter()
    rows = []
    for rec in records:
        q = rec["query"]
        et, ep = rec["expected_tool"], rec.get("expected_params") or {}
        gt, gp, src = detect_hit(domain, q)
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
        elif "fuzzy" in et or "fuzzy" in gt:
            kind = "fuzzy-involved"
        elif ok_t:
            kind = "only-param-mismatch"
        else:
            kind = "tool-divergence"
        buckets[kind] += 1
        rows.append({"query": q, "expected_tool": et, "expected_params": ep,
                     "got_tool": gt, "got_params": gp, "kind": kind, "hit": src})
    return stats, buckets, rows, hit_src


def _fmt(num, den):
    return f"{num}/{den} {num / den:.1%}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-d", "--domains", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-n", type=int, default=5, help="每个域 miss 明细条数")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rules", action="store_true")
    args = ap.parse_args()

    want = {x.strip() for x in args.domains.split(",") if x.strip()} if args.domains else set(DOMAINS)
    want = [d for d in DOMAINS if d in want]

    summary = {}
    for domain in want:
        caseset = AGENT / "data" / f"sheet0901_{domain}_caseset.json"
        if not caseset.exists():
            print(f"!! {domain}: caseset 缺失, 跳过")
            continue
        records = json.loads(caseset.read_text(encoding="utf-8"))["records"]
        stats, buckets, rows, hit_src = evaluate(records, domain)
        N = len(records)
        summary[domain] = {
            "tool": f"{stats['tool']}/{N} {stats['tool']/N:.1%}",
            "param": f"{stats['param']}/{N} {stats['param']/N:.1%}",
            "both": f"{stats['both']}/{N} {stats['both']/N:.1%}",
            "buckets": dict(buckets),
        }
        if not args.json:
            print(f"== {CN[domain]}({domain}) {N} 条 ==")
            print(f"  tool      : {summary[domain]['tool']}")
            print(f"  param     : {summary[domain]['param']}")
            print(f"  tool+param: {summary[domain]['both']}   bucket={dict(buckets)}")
            print(f"  命中来源: {dict(hit_src.most_common())}")
        if args.rules:
            print(f"\n=== {domain} 每条 query 命中规则 (hitSource) ===")
            for rec in records:
                q = rec["query"]
                gt, gp, src = detect_hit(domain, q)
                print(f"  [{rec['row']}] ({CN[domain]}) tool={gt:<24} "
                      f"hitSource={src:<32} {q!r}")
        if args.all or rows:
            n_show = len(rows) if args.all else min(args.n, len(rows))
            if n_show and not args.json:
                print(f"\n--- {domain} miss 明细 (前 {n_show}) ---")
                for r in rows[:n_show]:
                    print(f"  [{r['kind']}] {r['query']!r}")
                    print(f"      expected={r['expected_tool']} got={r['got_tool']}")
                    print(f"      hit={r['hit']}")

    if args.json:
        print(json.dumps({"domains": summary}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()