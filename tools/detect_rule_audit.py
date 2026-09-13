"""hcAgent 判域可解释性审计。

对给定 query（或 0901 域 caseset）逐条打印：
  detect_domain 命中的路径（_TRACE step 分支）、走信号层时命中的具体 Rule id，
  以及最终 domain 与期望的比对。

用法:
  python tools/detect_rule_audit.py --query "播放庆余年 第三集" --tv 0
  python tools/detect_rule_audit.py --domain vod          # 跑 sheet0901 该域并打印命中规则
  python tools/detect_rule_audit.py -d vod,music -n 20
  python tools/detect_rule_audit.py --ruleset             # 打印全部信号规则清单
  python tools/detect_rule_audit.py --json                 # 结构化
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

AGENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT))

DOMAINS = ["vod", "device", "audio", "music", "sports", "children", "education"]
CN = {"vod": "影视", "device": "设备", "audio": "有声", "music": "音乐",
      "sports": "体育", "children": "少儿", "education": "教育"}


def load_detect():
    import app.detect as d
    return d


def trace_domain(d, q, llm_domain, tv_mode="0"):
    """跑 detect_domain + 记录命中路径 + 信号规则命中明细。"""
    d._TRACE.clear()
    domain = d.detect_domain(q, llm_domain, tv_mode)
    trace = dict(d._TRACE)
    trace["final_domain"] = domain
    # 主链命中 signal_match 时，进一步解析命中的具体高置信信号 Rule id
    if trace.get("step") == "signal_match":
        sig = d._match_rule(q) if hasattr(d, "_match_rule") else None
        trace["rule"] = sig or trace.get("rule")
    return trace


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", default=None)
    ap.add_argument("--llm", default="qa", help="LLM 判的域（默认 qa 看能否救回）")
    ap.add_argument("--screen", default="0", choices=["0", "6"], help="0亮屏 6息屏")
    ap.add_argument("-d", "--domain", default=None, help="跑指定 sheet 0901 域 caseset")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--rules", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    d = load_detect()

    if args.rules:
        print(f"DetectSignalSet 规则数：{len(d._SIGNALS.rules)}")
        for r in d._SIGNALS.rules:
            print(f"  id={r.id:<26} prio={r.priority:<4} scope={r.scope:<6} {r.title}")
        return

    if args.query:
        tr = trace_domain(d, args.query, args.llm, args.screen)
        print(f"query     : {args.query!r}")
        print(f"llm_domain: {args.llm!r}  tv_mode={args.screen}")
        print(f"最终域    : {tr['final_domain']!r}")
        print(f"命中分支  : {tr.get('step')!r}")
        if tr.get("rule"):
            print(f"信号规则  : {tr['rule']}")
        return

    want = {x.strip() for x in args.domain.split(",") if x.strip()} if args.domain else set(DOMAINS)
    want = [x for x in DOMAINS if x in want]
    for dom in want:
        caseset = AGENT / "data" / f"sheet0901_{dom}_caseset.json"
        if not caseset.exists():
            print(f"!! {dom}: caseset 缺失")
            continue
        records = json.loads(caseset.read_text(encoding="utf-8"))["records"]
        print(f"\n== {CN[dom]}({dom}) {len(records)} 条 ==")
        for rec in records[: args.n]:
            q = rec["query"]
            tr = trace_domain(d, q, dom, args.screen)
            et = rec.get("expected_tool")
            print(f"  [{rec.get('row')}] 域={tr['final_domain']!r:<9} 分支={tr.get('step','')!r:<26} 规则={tr.get('rule')!r:<26} {q}")
        if len(records) > args.n:
            print(f"  ... 余 {len(records)-args.n} 条略")


if __name__ == "__main__":
    main()