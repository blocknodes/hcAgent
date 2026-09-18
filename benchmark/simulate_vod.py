"""本地模拟 vod 域 SSE 规则：给 query + llm_domain → detect 判域 + hcTools vod 流水线取 (tool, params)。

与真实 SSE 对齐：
- detect_domain(query, llm_domain) 判域（detect.py 热加载）
- domain=vod -> hcTools vod 域 badcase→rules→fallback→postproc，得到 tool+params
- engine._build_steps 把 params.retext 覆写为用户原句（评测对 golden 时会剥 retext）

用法: python3 benchmark/simulate_vod.py -q "火的电影" -q "..." [-llm qa music ...]
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]           # hc
AGENT = ROOT / "hcAgent"
HCT = ROOT / "hcTools"

sys.path.insert(0, str(ROOT))                        # hc  → from hcTools.… 可解析
sys.path.insert(0, str(HCT))                         # hcTools → from app.… 可解析（域 __init__ 用）
sys.path.insert(0, str(AGENT / "app"))               # detect.py 内 fallback `from detect_rulebase import …`

from hcTools.domains import vod as _vod

_badcase_mod = _vod.badcase
_postproc_mod = _vod.postproc
_fallback_mod = _vod.fallback
rules_mod = _vod.rules
postproc_mod = _vod.postproc

_BAD = None

def _load_bad():
    global _BAD
    if _BAD is None:
        _BAD = _vod.badcase.BadcaseStore.load(HCT / "domains" / "vod" / "badcases.json")
    return _BAD

def run_pipe(q):
    """badcase → rules.apply → fallback，过 postproc，与 bench 同源。"""
    hit = _load_bad().lookup(q)
    if hit is not None:
        t, p = hit
        return t, _postproc_mod.normalize(t, p or {})
    r = rules_mod.apply(q)
    if r is not None:
        t, p, rid = r
        return t, postproc_mod.normalize(t, p)
    t, p = _fallback_mod.fallback(q)
    return t, postproc_mod.normalize(t, p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", nargs="*", default=[])
    ap.add_argument("-llm", default="vod")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("advdet", str(AGENT / "app" / "detect.py"))
    det = importlib.util.module_from_spec(spec); spec.loader.exec_module(det)

    for q in args.q:
        dom = det.detect_domain(q, args.llm)
        if dom != "vod":
            print(f"q={q!r} llm={args.llm} detect={dom}  (NOT vod)")
            continue
        tool, params = run_pipe(q)
        print(f"q={q!r} llm={args.llm} detect=vod")
        print(f"    tool={tool}")
        print(f"    params={json.dumps(params, ensure_ascii=False)}")

if __name__ == "__main__":
    main()