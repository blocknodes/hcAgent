"""detect.py 规则化重构的差异回归：旧实现 vs 新实现必须逐 (query, llm_domain, tv_mode) 完全一致。

覆盖语料：
  - hcTools 各域 testset.json 的 query（e2e 域判的入口）
  - sheet0821 caseset（含亮/息）
  - 0901 各域 caseset query
遍历每个 (query, llm_domain ∈ 全域 + "", tv_mode ∈ {0,6})，比对老/新 detect 输出。

用法：
  python tools/detect_diff_test.py            # 跑差异
  python tools/detect_diff_test.py --json     # 结构化输出
目录必须放 hcTools 与 hcAgent 同级。
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from collections import Counter
from pathlib import Path

HC = Path(__file__).resolve().parents[2]      # workspace/hc
hcTools = HC / "hcTools"
AGENT = HC / "hcAgent"

DOMAINS = ["vod", "device", "audio", "music", "sports", "children", "education", "qa"]
TV_MODES = ["0", "6"]

sys.path.insert(0, str(hcTools))


def load_detect_via_path(p):
    return load_detect_path(str(p))


def collect_queries() -> list[str]:
    qs: set[str] = set()
    for dom in ["vod", "device", "audio", "music", "sports", "children", "education"]:
        ts = hcTools / "domains" / dom / "testset.json"
        if ts.exists():
            for rec in json.loads(ts.read_text(encoding="utf-8")).get("records", []):
                if isinstance(rec, dict) and rec.get("query"):
                    qs.add(rec["query"])
    # sheet0821 + 0901
    for f in (AGENT / "data").glob("sheet*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            recs = d.get("records") if isinstance(d, dict) else d
            for r in recs or []:
                if isinstance(r, dict) and r.get("query"):
                    qs.add(r["query"])
        except Exception:
            continue
    # 坏 case 语料
    for f in (AGENT / "app").glob("badcases*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            for b in d.get("badcases", []):
                if isinstance(b, dict) and b.get("query"):
                    qs.add(b["query"])
        except Exception:
            continue
    return sorted(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=str(AGENT / "app" / "detect.py.bak"))
    ap.add_argument("--new", default=str(AGENT / "app" / "detect.py"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    old = load_detect_path(args.old)
    new = load_detect_path(args.new)

    queries = collect_queries()
    diffs: list[tuple] = []
    n_cases = 0
    llm_pool = DOMAINS + [""]
    for q in queries:
        for dom in llm_pool:
            for tv in TV_MODES:
                n_cases += 1
                try:
                    a = old.detect_domain(q, dom, tv_mode=tv)
                except Exception as e:
                    a = f"OLD_ERR:{e}"
                try:
                    b = new.detect_domain(q, dom, tv_mode=tv)
                except Exception as e:
                    b = f"NEW_ERR:{e}"
                if a != b:
                    diffs.append((q, dom, tv, a, b))

    if args.json:
        print(json.dumps({"total_cases": n_cases, "total_queries": len(queries),
                          "diff_count": len(diffs),
                          "diffs": [{"query": d, "llm_domain": dm, "tv_mode": t,
                                     "old": o, "new": n}
                                    for d, dm, t, o, n in diffs[:200]]},
                         ensure_ascii=False, indent=1))
    else:
        print(f"语料 {len(queries)} query; 遍历 (q, llm_domain, tv_mode) 组合 {n_cases} 次判定")
        print(f"差异数: {len(diffs)}")
        by_reason = Counter((o, n) for _, _, _, o, n in diffs)
        for (o, n), c in by_reason.most_common(20):
            print(f"  {o!r} -> {n!r}  x{c}")
        print("\n=== 差异明细 (前 40) ===")
        for d in diffs[:40]:
            print(f"  [{tv_mode_img(d[2])}] llm={d[1]!r} q={d[0]!r}\n      old={d[3]!r}\n      new={d[4]!r}")


_TMP = Path(__file__).resolve().parent / "_detect_tmp"
_TMP.mkdir(exist_ok=True)


def load_detect_path(p):
    import hashlib, shutil
    name = "detect_mod_" + hashlib.md5(str(p).encode()).hexdigest()[:8]
    tmp = _TMP / f"{name}.py"
    shutil.copyfile(p, tmp)
    # 把 detect 的同级依赖（detect_rulebase.py）一并拷入临时目录，
    # 并把临时目录加入 sys.path，使新 detect.py 的裸 import 可解析到。
    src_dir = Path(p).resolve().parent
    for dep in ("detect_rulebase.py",):
        dep_src = src_dir / dep
        if dep_src.exists() and not (_TMP / dep).exists():
            shutil.copyfile(dep_src, _TMP / dep)
    _TMP_str = str(_TMP)
    if _TMP_str not in sys.path:
        sys.path.insert(0, _TMP_str)
    spec = importlib.util.spec_from_file_location(name, str(tmp))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def tv_mode_img(t):
    return f"tv_mode={t}" if t else "tv_mode=''(默认0)"


if __name__ == "__main__":
    main()