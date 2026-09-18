#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gold 金标准 vs schema 对照审计 —— 找出 sheet_rules_<domain>.csv 中「gold 本身不合理/标注错误」的行。

将每个问题追加到 CSV 新列「问题（gold审计）」；并输出汇总 output/gold_audit_summary.csv。
判定类别：
  G1 工具名不在该域 schema 工具列表
  G2 参数的键不在该工具 schema 属性中（未知参数 / 未知 field / 未知操作对象）
  G3 枚举非法：action / operation / sort.order / 状态值(is_fee/is_over) / 集合缺 operator
  G4 query 叶子 field 不在该域合法字段集；value/values 并存
  G5 gold 参数为空但该工具必须显式提参（判为 gold 标注不完整，非模型问题）
"""
import csv, json, os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DOMAIN_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..', 'hcTools', 'domains'))
DOMS = ['vod', 'education', 'music', 'sports', 'children', 'audio', 'device']

# ---------------------------------------------------------------- schema tools ---
def _collect(node, defs, valid, _seen=None):
    """递归展开 $ref / oneOf / 收集 field const + 枚举值。带环保护。"""
    if _seen is None:
        _seen = set()
    if isinstance(node, dict):
        rid = id(node)
        if rid in _seen:
            return
        _seen.add(rid)
        if node.get('$ref'):
            t = node['$ref'].split('/')[-1]
            sub = defs.get(t)
            if sub:
                _collect(sub, defs, valid, _seen)
            return
        props = node.get('properties')
        if isinstance(props, dict):
            for pn, pv in props.items():
                if isinstance(pv, dict):
                    if isinstance(pv.get('enum'), list) and pn in ('value', 'operation', 'operator'):
                        valid['enums'].update(str(x) for x in pv['enum'])
                    if pn == 'field':
                        f = pv
                        if isinstance(f, dict):
                            if f.get('oneOf'):
                                for o in f['oneOf']:
                                    if 'const' in o:
                                        valid['fields'].add(o['const'])
                            if f.get('enum'):
                                valid['fields'].update(f['enum'])
                            if 'const' in f:
                                valid['fields'].add(f['const'])
                        elif isinstance(f, str):
                            valid['fields'].add(f)
        for v in node.values():
            if isinstance(v, dict):
                _collect(v, defs, valid, _seen)
            elif isinstance(v, list):
                for i in v:
                    _collect(i, defs, valid, _seen)
    elif isinstance(node, list):
        for i in node:
            _collect(i, defs, valid, _seen)


def load_schema(dom):
    """返回 {tool_name: {'props':set,'required':set,'fields':set,'enums':set}}"""
    with open(os.path.join(DOMAIN_ROOT, dom, 'schema.json'), encoding='utf-8') as f:
        raw = json.load(f)
    tools = {}
    for t in raw.get('tools', []):
        name = t['tool_name']
        params = t.get('parameters') or {}
        defs = params.get('definitions') or {}
        valid = {'fields': set(), 'enums': set()}
        # 展开整个 parameters（含 definitions），收集 field const 与值枚举
        for dn in list(defs.keys()) + ['']:
            node = defs.get(dn)
            if dn == '':
                node = {'properties': params.get('properties') or {}}
            _collect(node, defs, valid)
        tools[name] = {
            'props': set((params.get('properties') or {}).keys()),
            'required': set(params.get('required') or []),
            'fields': valid['fields'],
            'enums': valid['enums'],
        }
    return tools


def _leaf_query(ts, q, pfx='', depth=0):
    """返回 query 树问题列表"""
    if depth > 8:
        return [f'{pfx}query嵌套>8']
    if q is None:
        return []
    out = []
    if isinstance(q, dict):
        if 'field' in q:
            f = q['field']
            if ts['fields'] and f not in ts['fields']:
                out.append(f'{pfx}field[{f}]非法')
            if 'value' in q and 'values' in q:
                out.append(f'{pfx}value与values并存')
            if q.get('values') is not None and not isinstance(q['values'], list):
                out.append(f'{pfx}values应数组')
            if q.get('operator') is not None and 'values' not in q:
                out.append(f'{pfx}operator需配values')
            if isinstance(q.get('values'), list) and len(q['values']) > 1 and 'operator' not in q:
                out.append(f'{pfx}多值缺operator')
            if f in ('is_fee', 'is_over') and ts['enums']:
                v = q.get('value')
                if v is not None and str(v) not in ts['enums']:
                    out.append(f'{pfx}状态值[{v}]非法')
            return out
        for k in ('and', 'or', 'not'):
            ch = q.get(k)
            if k == 'not':
                out += _leaf_query(ts, ch, f'{pfx}.not', depth + 1)
            elif isinstance(ch, list):
                for i, c in enumerate(ch):
                    out += _leaf_query(ts, c, f'{pfx}.{k}[{i}]', depth + 1)
            elif isinstance(ch, dict):
                out += _leaf_query(ts, ch, f'{pfx}.{k}', depth + 1)
        if not any(kk in q for kk in ('and', 'or', 'not', 'field')):
            out.append(f'{pfx}query节点为空')
        return out
    if isinstance(q, list):
        for i, c in enumerate(q):
            out += _leaf_query(ts, c, f'{pfx}[{i}]', depth + 1)
        return out
    return [f'{pfx}query非对象[{q!r}]']


def audit_row(dom, tool, params, ts):
    """返回问题列表（不在则空）"""
    issues = []
    if ts is None:
        return [f'G1:工具[{tool}]不在schema']
    if params is None:
        # gold 空参数
        if dom in ('vod', 'education', 'children', 'audio', 'sports') and tool in (
                'vod_search_all', 'vod_search', 'vod_relate_search', 'vod_fuzzy_search',
                'edu_search', 'edu_fuzzy_search', 'educ_search', 'educ_search_all',
                'audio_search', 'sports_match_search', 'sports_vod_search',
                'music_song_search', 'music_song_recommend'):
            issues.append('G5:gold参数为空(应显式)')
        return issues
    # 参数键
    for k in params:
        if k not in ts['props'] and k not in ('action', 'retext'):
            issues.append(f'G2:未知参数[{k}]')
    for req in ts['required']:
        if req not in params or (params.get(req) in (None, '', [], {})):
            issues.append(f'G2:缺必填[{req}]')
    act = params.get('action')
    if act is not None and act not in ('search', 'play', 'query', 'view', 'check'):
        issues.append(f"G3:action[{act}]非法")
    if isinstance(params.get('sort'), dict):
        srt = params['sort']
        for kk, vv in srt.items():
            if kk not in ('rate', 'hot', 'new', 'play'):
                issues.append(f"G3:sort非法键[{kk}]")
            elif not isinstance(vv, dict) or vv.get('order') not in ('asc', 'desc'):
                issues.append(f"G3:sort.{kk}缺order")
    if 'query' in params:
        for x in _leaf_query(ts, params['query']):
            issues.append(f'G4:{x}')
    return issues


def main():
    summary = []
    for dom in DOMS:
        case_p = os.path.join(ROOT, f'cases/sheet_rules_{dom}.csv')
        if not os.path.exists(case_p):
            continue
        tools = load_schema(dom)
        with open(case_p, encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        flag = '问题（gold审计）'
        flagged = 0
        for r in rows:
            tool = (r.get('期望工具') or '').strip()
            g = (r.get('期望参数') or '').strip()
            params = None
            if g:
                try:
                    v = json.loads(g)
                    params = v if isinstance(v, dict) and set(v.keys()) != {'_raw'} else None
                except Exception:
                    params = None
            issues = audit_row(dom, tool, params, tools.get(tool))
            r[flag] = ';'.join(issues)
            if issues:
                flagged += 1
        # 写回（保持原列顺序，新增列放最后）
        fnames = list(rows[0].keys())
        with open(case_p, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in fnames})
        print(f'=== {dom}: total={len(rows)} flagged={flagged}')
        # 打印部分带问题的行，便于核对
        for r in rows:
            if r.get(flag):
                print('    ', r['query'][:34], ' | ', r[flag][:160])
        summary.append((dom, len(rows), flagged))
    with open(os.path.join(ROOT, 'output', 'gold_audit_summary.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['domain', 'total', 'gold_issue_rows'])
        for d, t, n in summary:
            w.writerow([d, t, n])
    print('\n汇总 → output/gold_audit_summary.csv\n')


if __name__ == '__main__':
    DOMAINS = sys.argv[1:] or DOMS
    total = 0
    main()