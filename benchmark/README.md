# benchmark —— hcAgent SSE 真实链路评测集

本目录存放 hcAgent 的 **SSE 真链路评测脚本 + 用例集**。评测不是单元测试：每条 query 直连
真实链路（远端 runtime → 本地 hcAgent(8082) → 本地 hcTools(8084)），从 DEBUG_MAP 计划帧取
真实工具名 + 参数，与人工标的 golden 比对。

## 入口脚本

| 脚本 | 覆盖范围 | 输入 | 说明 |
|---|---|---|---|
| `run_sse_bench.py` | 7 个单意图域 | `cases/sheet0901_<domain>.csv` | 每 query 一步 SSE，判 tool / param / joint。类似 `tools/sse_domain_output.py` 但保留“是否合理”列。 |
| `run_sse_multiintent.py` | 多意图并行 | `cases/e80803_multiintent.csv` | 一条 query 含 device + content 两个独立意图，一次 SSE 拆多步，双槽分别判。 |

两者复用 `tools/sse_7domain_eval.run_one`（喂原始 query → 返回所有 plan steps）。

---

## 1. run_sse_bench.py — 单域/单意图 7 域评测

每个 query 拆成一个意图，链路返回单一步骤。分别判：
- **tool_ok**：pred 工具名 == golden 工具名
- **param_ok**：pred 参数与 golden 参数 order-insensitive 对齐（`params_equal`）
- **joint(both_ok)**：tool_ok and param_ok

输入 `benchmark/cases/sheet0901_<domain>.csv`，前 6 列：
`业务域, query, 意图(线上意图), 期望工具, 期望参数, 是否合理`

```
python benchmark/run_sse_bench.py                  # 7 域全量
python benchmark/run_sse_bench.py --ok-only        # 只跑 标注为空/合理 的用例
python benchmark/run_sse_bench.py -d vod,music     # 只跑指定域
python benchmark/run_sse_bench.py -n 10            # 每域前 10 条(冒烟)
python benchmark/run_sse_bench.py -w 16            # 并发(默认 8)
```

输出：
- `output/detail_<domain>.csv` 逐条明细（含 是否合理/tool_ok/param_ok/both_ok/param_diff）
- `output/summary.csv` + `output/summary.json` 按域 + 总体汇总

## 2. run_sse_multiintent.py — 多意图并行评测

考察一张 query 同时含 **设备操作意图** 与 **内容检索意图**（如“打开多声道音响，同时小聚小聚，
请播放王菲免费专辑”）。链路内部由 LLM(T0) 拆 query、定 dep、落域，拆出**多个 step**。

评测对 200 条案例判两个独立槽：
- **device 槽**：`gold_tool1` 精确命中某 step 的 tool，且参数对齐
- **content 槽**：某 step 的 tool 属 `gold_tool2_options` 的**同域内容工具族**（hcTools 各内容域
  可能收敛到一个总名，如 `sports_match_search` 覆盖 `sports_team_search`），且参数对齐
- **both_ok**：两槽同时命中

参数对齐规则：
- **`retext` 字段不算考核**（链路回显常为整句，gold 是干净子句，判对时整体忽略）
- 字符串 `query` / 结构化 `query.and`：pred 的 query 含 gold 子串即可（整句含子句）
- 其余结构化字段（grade/song/action/...）严格对齐
- `figures` 等派生冗余字段忽略

输入 200 条来自飞书 sheet `e80803`「多意图并行」，每条含人工标的：
`query, gold_tool1(+gold_param1, device), gold_tool2_options(+gold_param2, content)`

```
python benchmark/run_sse_multiintent.py            # 200 全量
python benchmark/run_sse_multiintent.py -n 10 -w 6 # 冒烟
```

输出：
- `output/multiintent_detail.csv`：每设备/内容/both 命中 + 实际 steps
- `output/multiintent_summary.{csv,json}`：device_ok / content_ok / both_ok

## 评测口径关键约定

- 全程喂**原始 query**（不做外部分词）；拆分/依赖/落域由链路内 LLM 决策。
- 单步（`run_sse_bench`）取 steps[0]；多步（`run_sse_multiintent`）取全部 steps 分槽对齐。
- 参数对齐用 `tools/sse_7domain_eval.canonical` / `params_equal`（列表排序、剥零宽字符 U+200C）。
- 多意图判定放于 content 槽按**同域工具族**（tool 名前缀）放宽，device 槽精确匹配。

## 历史 / 提示

- 多意图 device 判定原用**整句**喂 `detect` 判域，会把内容意图带偏成 device
  （如“八年级的物理，然后切鲜艳模式”→整句被判 device）。已修为用**意图子句**判域
  （`app/engine.py::_with_source`），多意图 both_ok 由个位提升到 ~67%。