"""sheet0821 用例走本地确定性流水线（可改进）。

detect.detect_domain(query, golden业务域) 判域 → 各域 badcase→rules→fallback
→ tool/tool+param，与 golden bright 列比对。焦点 tool 准确率。
用法：python tools/sheet_pipe_bench.py [--json]
"""
from __future__ import annotations
import importlib, importlib.util, json, sys, re
from collections import Counter
from pathlib import Path

HCROOT = Path(__file__).resolve().parents[2] / "hcTools"
AGENT = Path(__file__).resolve().parents[1]
CASESET = AGENT / "data" / "sheet0821_caseset.json"
DOMS = ["vod", "audio", "music", "children", "education", "sports", "device", "qa"]
DMAP = {"影视": "vod", "有声": "audio", "音乐": "music",
        "少儿": "children", "教育": "education", "设备控制": "device"}
sys.path.insert(0, str(HCROOT)); sys.path.insert(0, str(HCROOT / "app"))


def _canon(x):
    if isinstance(x, dict):
        # retext 是展示层回显（= 用户原句），非可执行槽位，各域 pipe 与 golden 时有/时无，不计入参数一致
        return {k: _canon(v) for k, v in x.items() if k != "retext" and not _empty(v)}
    if isinstance(x, list):
        return sorted((_canon(i) for i in x if not _empty(i)),
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


def params_equal(a, b):
    return _canon(a) == _canon(b)


def load_detect():
    spec = importlib.util.spec_from_file_location("agentdetect", str(AGENT / "app" / "detect.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_pipe(dom):
    b = importlib.import_module(f"domains.{dom}.badcase")
    r = importlib.import_module(f"domains.{dom}.rules")
    f = importlib.import_module(f"domains.{dom}.fallback")
    bc = HCROOT / "domains" / dom / "badcases.json"
    store = b.BadcaseStore.load(bc) if bc.exists() else b.BadcaseStore([])

    def run(q):
        h = store.lookup(q)
        if h is not None:
            return h[0], h[1] or {}
        rr = r.apply(q)
        if rr is not None:
            return rr[0], rr[1] or {}
        return f.fallback(q)
    return run


def _json_fix(txt):
    """尽力修复 golden 常见 JSON 语法瑕疵：键间缺逗号、值串残留引号、多余逗号。
    返回解析后的 dict/None。"""
    candidates = [txt]
    # 1) 修复 `"retext": "x"\n  "query"` 这类键间缺逗号（值以引号结尾，紧跟行开始的新字符串键）
    candidates.append(re.sub(r'(["])\s*\n\s*(?="[a-zA-Z_]+"\s*:)', r'\1,\n', txt))
    # 2) 数字/花括号值结尾后紧跟新字符串键缺逗号
    candidates.append(re.sub(r'(\d|})\s*\n\s*(?="[a-zA-Z_]+"\s*:)', r'\1,\n', txt))
    # 3) 修复值串尾部多引号 `"value": "x""` → `"value": "x"`
    candidates.append(re.sub(r'":\s*"([^"]*)"+', r'": "\1"', txt))
    # 4) 去掉对象/数组中多余的尾部逗号
    candidates.append(re.sub(r',\s*}', '}', txt))
    candidates.append(re.sub(r',\s*\]', ']', txt))
    # 5) 链式：先修值串多引号，再补键间缺逗号（row36 双重瑕疵）
    _t = re.sub(r'":\s*"([^"]*)"+', r'": "\1"', txt)
    candidates.append(re.sub(r'(["])\s*\n\s*(?="[a-zA-Z_]+"\s*:)', r'\1,\n', _t))
    candidates.append(re.sub(r'(\d|})\s*\n\s*(?="[a-zA-Z_]+"\s*:)', r'\1,\n', _t))
    seen = set()
    for t2 in candidates:
        if t2 in seen:
            continue
        seen.add(t2)
        try:
            v = json.loads(t2)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            continue
    return None


def gold_params(raw):
    """从 messy golden 参数文本里提取目标 JSON block；失败返回 None(golden 不可比)。

    多分支 golden（息屏多 tab：影视/少儿各有 param）优先取含 action=play 的 影视 分支，
    与 bright=vod_search 的评测口径一致；否则退而取整个文本里首个可解析 JSON。
    """
    raw = (raw or "").strip()
    if not raw:
        return {}  # golden 预期空参数
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    txt = m.group(0)
    # 多分支：按 {} 配平切出顶层完整 JSON 对象
    blocks = _split_top_objs(txt) or [txt]
    # 优先 action=play（影视播放，与 bright=vod_search 口径一致）
    for block in blocks:
        v = _json_fix(block)
        if v and v.get("action") == "play":
            return v
    for block in blocks:
        v = _json_fix(block)
        if v is not None:
            return v
    return None


def _split_top_objs(txt):
    """按大括号配平切出顶层若干个完整 JSON 对象文本（用于息屏多分支 golden）。"""
    out, depth, start = [], 0, None
    in_str, esc = False, False
    for i, ch in enumerate(txt):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                out.append(txt[start:i + 1])
                start = None
    return out


def main():
    detect = load_detect()
    pipes = {d: load_pipe(d) for d in DOMS}
    cases = json.loads(CASESET.read_text(encoding="utf-8"))
    tot = dt = p_hit = param_best = 0
    mis = Counter()
    tool_mis = Counter()
    for c in cases:
        q = c["query"]
        gd = DMAP.get(c["domain"], "")
        et = c["bright_tool"] or ""
        if not gd or gd not in pipes or not et:
            continue
        tot += 1
        pred = detect.detect_domain(q, gd)
        pt, pp = pipes[pred](q) if pred in pipes else pipes[gd](q)
        dt_ok = (pt == et)
        dt += dt_ok
        # param
        ep = gold_params(c.get("bright_params"))
        if ep is None:
            pass  # golden 不可比，不参与 param
        elif not ep:
            p_ok = dt_ok
            param_best += 1
            p_hit += dt_ok
        else:
            p_ok = dt_ok and params_equal(pp, ep)
            param_best += 1
            p_hit += p_ok
        if not dt_ok:
            tool_mis[(c["domain"], et, pt)] += 1
        if not (dt_ok and p_ok):
            mis[(c["domain"], et, pt)] += 1

    print(f"=== sheet0821 亮屏 tool 基准 {tot} 条 (detect判域路由) ===")
    print(f"  tool      {dt}/{tot} = {dt/tot*100:.2f}%")
    if param_best:
        print(f"  tool+param {p_hit}/{param_best} = {p_hit/param_best*100:.2f}%  (在 {param_best} 可打分 golden 上)")
    print("\n=== 工具未命中 (gold域 -> 目标工具 <-> 实际工具) ===")
    for (g, et, pt), n in tool_mis.most_common(60):
        print(f"  {g:6s} {et:24s} -> {pt:24s} x{n}")
    print("\n=== 全未命中(工具或参数) ===")
    for (g, et, pt), n in mis.most_common(60):
        print(f"  {g:6s} {et:24s} -> {pt:24s} x{n}")


if __name__ == "__main__":
    main()
