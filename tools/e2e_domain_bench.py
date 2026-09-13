"""hcAgent 跨域端到端判域 Bench。

度量「detect 判域 → hcTools 各域三层流水线 → tool+param」的真实端到端准确率，
并量化 detect 相对「直接用 golden 真值域」的下滑。

判定流水线（与各域 bench 同契约）：
    badcase(L2) → rules.apply(L1) → fallback(L3)

指标：tool / tool+param（joint，最严，order-insensitive canonical）。
对比两路：
  native  = 直接用 query 所属 golden 域跑流水线（理想判域，天花板）
  detect  = detect.detect_domain(query, llm_domain) 判域后路由（真实链路）
DIFF = detect - native：>0 说明 detect 修正了流水线误判；<0 说明 detect 引入了下滑。

用法：
  python tools/e2e_domain_bench.py
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

HCROOT = Path(__file__).resolve().parents[2] / "hcTools"
AGENT = Path(__file__).resolve().parents[1]  # hcAgent
DOMS = ["vod", "audio", "music", "children", "education", "sports", "device"]
DOMAINS_DIR = HCROOT / "domains"

# 让 hcTools 及其 app.* / domains.* 可 import
sys.path.insert(0, str(HCROOT))
sys.path.insert(0, str(HCROOT / "app"))


def _load_agent_detect():
    """加载 hcAgent/app/detect.py（避免与 hcTools/app 的 'app' 包名冲突）。"""
    # detect.py 依赖同级 detect_rulebase.py，需临时加入 sys.path 供裸 import 解析
    _deps = str(AGENT / "app")
    if _deps not in sys.path:
        sys.path.insert(0, _deps)
    spec = importlib.util.spec_from_file_location("agentdetect", str(AGENT / "app" / "detect.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def canonical(x):
    if isinstance(x, dict):
        return {k: canonical(v) for k, v in x.items()}
    if isinstance(x, list):
        return sorted((canonical(i) for i in x), key=lambda e: json.dumps(e, ensure_ascii=False, sort_keys=True))
    return x


def params_equal(a, b):
    return canonical(a) == canonical(b)


def load_pipeline(dom):
    """加载单域 (badcase, rules.apply, fallback)，契约与各域 bench 一致。"""
    b = importlib.import_module(f"domains.{dom}.badcase")
    r = importlib.import_module(f"domains.{dom}.rules")
    f = importlib.import_module(f"domains.{dom}.fallback")
    bc_path = DOMAINS_DIR / dom / "badcases.json"
    store = b.BadcaseStore.load(bc_path) if bc_path.exists() else b.BadcaseStore([])

    def run(q):
        hit = store.lookup(q)
        if hit is not None:
            return hit[0], hit[1] or {}
        rr = r.apply(q)
        if rr is not None:
            return rr[0], rr[1] or {}
        return f.fallback(q)

    return run


def main():
    detect = _load_agent_detect()
    pipes = {d: load_pipeline(d) for d in DOMS}

    native_t = native_b = 0
    detect_t = detect_b = 0
    tot = 0
    det_mis: Counter[tuple[str, str, str]] = Counter()  # (golden_domain, pred_domain, golden_tool)

    for dom in DOMS:
        ts = json.load(open(DOMAINS_DIR / dom / "testset.json"))
        for rec in ts["records"]:
            q = rec["query"]
            et, ep = rec["expected_tool"], rec.get("expected_params") or {}
            tot += 1

            # native：直接用 golden 真值域
            gt, gp = pipes[dom](q)
            native_t += gt == et
            native_b += (gt == et) and params_equal(gp, ep)

            # detect：判域后路由（llm_domain 用 golden 作为理想 LLM 上界）
            pred = detect.detect_domain(q, dom)
            pt, pp = pipes[pred](q) if pred in pipes else pipes[dom](q)
            detect_t += pt == et
            ok = (pt == et) and params_equal(pp, ep)
            detect_b += ok
            if not ok:
                det_mis[(dom, pred, et)] += 1

    print(f"=== 端到端 单域 tool accuracy  ({tot} 条) ===")
    print(f"  native (golden 域):   {native_t:5d}/{tot} = {native_t/tot*100:.2f}%")
    print(f"  detect 判域路由:      {detect_t:5d}/{tot} = {detect_t/tot*100:.2f}%")
    print(f"=== 端到端 单域 tool+param joint ===")
    print(f"  native (golden 域):   {native_b:5d}/{tot} = {native_b/tot*100:.2f}%")
    print(f"  detect 判域路由:      {detect_b:5d}/{tot} = {detect_b/tot*100:.2f}%")
    print(f"  DIFF (detect - native): tool {detect_t-native_t:+d}, tool+param {detect_b-native_b:+d}")
    print(f"\n=== detect 引入的 tool+param 失败，按 (golden域 -> 判域, golden tool) ===")
    for (d, pred, et), n in det_mis.most_common(30):
        marker = "" if d == pred else "  <-- 判错域"
        print(f"  {d:10s}->{pred:10s} tool={et:22s} x{n}{marker}")


if __name__ == "__main__":
    main()