# 规则层开关

hcAgent 的确定性规则全部由**层/域级总闸**控制，集中登记在
`app/config.py::_RULE_DEFAULTS`，运行时可用环境变量覆盖。

- **默认全开**：不设任何环境变量时，行为与开关引入前**逐字节一致**
  （已用 `tools/detect_diff_test.py` 对 4272 query × 76896 次判定验证，差异 0）。
- **关闭语义**：整组短路，行为**回退到下一层**（通常是 LLM 判定 / 原始 query），
  不是"报错"也不是"返回空"。
- **粒度**：只到层/域级，**不提供单条规则开关**。规则之间有 priority 依赖，
  单条关掉容易产生「本该由 A 抢答、A 关了落到 B」的隐性漂移；层/域级语义明确、
  便于 A/B 对拍。要看单条命中情况请用 `tools/detect_rule_audit.py`。

## 覆盖方式

环境变量名 = `HC_RULE_` + 组名大写（`.` 换成 `_`）。值为 `0`/`false`/`off`/`no`
表示关闭，其余非空值表示开启。

```bash
HC_RULE_DETECT_MUSIC=0        # 关掉判域链里全部 →music 的规则
HC_RULE_MULTIINTENT_SERIAL_QA=0   # 关掉「检索→提问」确定性拆分
HC_RULE_PLAN_SORT_MERGE=0     # 关掉排序词回填
```

启动时会打印被关闭的组：

```
WARNING hcAgent.config 规则层开关已关闭 2 组：detect.music, plan.sort_merge
```

## 开关清单

### 判域链（`app/detect.py::_DETECT_RULES`，20 条规则）

| 开关 | 覆盖的规则 | 关闭后 |
|---|---|---|
| `detect.off` | `off_routing`（息屏 tvMode=6 分诊链） | 息屏不做分诊 |
| `detect.badcase` | `badcase`（L2 精确句判域） | 精确句 badcase 失效 |
| `detect.music` | `music_override`、`music_discovery` | 不强拉 music，保 LLM 判定 |
| `detect.audio` | `_audio_listen_carry`、`audio_discovery` | 不强拉 audio |
| `detect.vod` | `media_query`、`media_locator`、`vod_auteur_anim`、`vod_recommend_genre` | 不做 vod 媒资保护/定位/推荐 |
| `detect.children` | `children_locator`、`children_ergou_bright` | 不强拉 children |
| `detect.education` | `edu_bright_strong`、`edu_no_anchor_qa` | 不强拉 education |
| `detect.qa` | `media_knowledge_qa`、`plot_qa`、`qa_open_knowledge`、`qa_greeting` | 不强拉 qa |
| `detect.sports` | `sports_prediction` | 不做赛事预测 |
| `detect.signal` | `signal_match`（高置信信号表） | 信号表失效 |
| `detect.keep` | `llm_domain_keep`（保 LLM 兜底） | **整链不兜底，`detect_domain` 返回 None** |

> 关闭某一组后，最终 domain 与开启时**可能仍然相同** —— 因为其它规则或 LLM 判定
> 可能落到同一域。这是预期行为：开关控制的是"这组规则是否参与判定"，
> 而不是"最终结果必须不同"。

### 确定性拆分（`app/multiintent.py`）

| 开关 | 覆盖 | 关闭后 |
|---|---|---|
| `multiintent.content_device` | `split_multiintent`，「内容+设备」双目标 | 不拆，整句交给 LLM（T0） |
| `multiintent.serial_qa` | `split_serial_qa`，串行「检索→提问」 | 不拆，整句交给 LLM |
| `multiintent.multi_tab` | 卡通/动漫双域并行（children+vod 双 tab） | 不生成双 tab |

### 多轮（`app/engine.py`）

| 开关 | 覆盖 | 关闭后 |
|---|---|---|
| `mt.rewrite_badcase` | `_mt_rewrite` 里的改写 badcase | 改写全交 LLM |
| `mt.inherit_domain` | `_inherit_mt_domain` 多轮域继承 | 弱句不再继承上轮域 |
| `mt.bare_song_reseed` | `_reseed_bare_song` 裸书名号歌曲兜底 | 裸《X》不救回 music |

### 计划后处理（`app/engine.py`）

| 开关 | 覆盖 | 关闭后 |
|---|---|---|
| `plan.sort_merge` | `_merge_sort_word_to_retrieval` 排序词回填 | 排序词不回填到检索句 |

## 旧开关兼容

`HC_DISABLE_SORT_MERGE=1` 仍然有效，语义等价于 `HC_RULE_PLAN_SORT_MERGE=0`
（保留是为了不让已写好的 `run.sh` / 评测脚本失效）。

## 验证

```bash
python tools/detect_diff_test.py                     # 判域链默认行为零漂移
python tools/detect_rule_audit.py                    # 单条规则命中审计
```
