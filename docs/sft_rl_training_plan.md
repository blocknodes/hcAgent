# hcAgent SFT 与强化学习规划（V0.1）

> 本文将“强化学校”按“强化学习（RL）”理解。当前先做规划，不修改在线推理链路。

## 1. 结论先行

当前不建议直接做端到端 Agent 强化学习，而应采用以下路线：

1. **先冻结 `hcTools`，只训练 `hcAgent` 编排模型**，避免工具选择/参数抽取的变化污染奖励。
2. 将现有模型任务拆成三类：
   - **T0 Plan**：拆意图、改写子请求、落域、判断依赖，输出 `[{"q","d","dep?"}]`；
   - **T1 Rewrite**：结合上一步工具结果，把依赖意图改写成自包含请求；
   - **MT Rewrite**：结合多轮历史，补全当前请求并保留多意图结构。
3. **先 SFT，后偏好优化/强化学习**：
   - SFT 建立稳定格式、任务边界和基本能力；
   - T1/MT 优先做 DPO，纠正“丢意图、错继承、压缩问句”等错误；
   - T0 优先做 GRPO，利用结构和端到端结果的可验证奖励；
   - PPO 暂不作为首选，只有后期做长程整会话优化时再评估。
4. 当前 benchmark 分数较高，很大程度上来自规则补丁。因此训练验收不能只看“规则开启”的总分，必须同时比较**规则开启、规则关闭/消融**两种模式。

## 2. 当前系统与训练边界

### 2.1 当前 Agent 链路

核心代码位置：

- `app/engine.py`
  - `TraceStateMachine._plan()`：T0 计划；
  - `TraceStateMachine._rewrite_step()`：T1 依赖改写；
  - `_mt_rewrite()`：多轮改写；
  - `_with_source()`、`multiintent`、`detect.py` 等：规则修正和兜底；
  - `_hctools_params()`：调用 `hcTools` 获取最终工具和参数。
- `app/prompts.py`
  - `T0_PLAN_PROMPT`、`T1_STEP_PROMPT`、`MT_REWRITE_PROMPT`。
- `app/llm.py`
  - OpenAI-compatible 模型网关；当前 `MODEL="baseline"`，关闭 thinking，温度为 0。
- `hcTools`
  - 负责域内工具选择和参数抽取，现阶段应视作固定环境。

### 2.2 第一阶段训练范围

训练范围只覆盖 `hcAgent` 的语言决策：

| 任务 | 输入 | 输出 | 第一阶段方法 |
|---|---|---|---|
| T0 Plan | 当前 query | JSON plan：`q/d/dep` | SFT → GRPO |
| T1 Rewrite | 待执行意图 + 工具结果 | 自包含 query | SFT → DPO |
| MT Rewrite | 对话历史 + 当前 query | 一条或分号分隔的意图序列 | SFT → DPO |

第一阶段不训练：

- `hcTools` 的工具选择和参数抽取；
- runtime 的执行协议与状态机；
- 规则引擎本身。

第二阶段只有在编排模型稳定后，才考虑 `hcAgent + hcTools` 联合优化或端到端轨迹训练。

## 3. 已有数据与基线

### 3.1 可用数据资产

现有 case 共约 3461 行原始记录，尚未经过清洗、去重和训练/测试隔离：

| 数据集 | 规模 | 可用于 |
|---|---:|---|
| `sheet0901_<domain>.csv`（7 域） | 2912 | 单意图落域、原句保持、端到端工具参数奖励 |
| `e80803_multiintent.csv` | 200 | T0 多意图拆分、并行关系、子句保持 |
| `sheet0821_serial.csv` | 100 | T0 依赖拓扑、T1 结果回填 |
| `sheet0821_multiturn.csv` | 150 轮/30 会话 | MT 上下文继承、跨轮工具参数验证 |
| `sheet0821_brightoff.csv` | 99 | `tvMode` 条件下的路由回归 |

注意事项：

- 单意图表可以较直接构造 `[{"q": 原query, "d": 业务域}]`。
- 多意图表已有 `device_query`、`content_query`，是质量较高的 T0 拆分监督。
- 串行表的 `param2` 已包含工具执行后解析出的具体实体，**不能直接作为 T0 目标**，否则发生未来信息泄漏；它适合做 T1 target。T0 的第二步应保留“这一部/第三部/该导演”等待解析指代，并标注 `dep: 1`。
- 多轮表可从“修正后参数”的 `retext/query` 构造 MT target，但需抽样人工复核，避免把工具参数噪声当成语言目标。

### 3.2 当前可见基线

仓库现有最新结果：

- 单意图最新 `summary.json` 覆盖音乐/体育/教育 631 条：joint **95.40%**；
- 多意图：both **183/200 = 91.5%**；
- 串行：both **91/100 = 91.0%**；
- 多轮：both **146/150 = 97.3%**；
- 亮/息屏 99 条：亮屏 both **94.9%**，息屏 both **97.0%**。

这些是“当前模型 + 当前规则 + 当前 hcTools”的系统分，不能直接视作基础模型能力。M0 必须补跑规则消融基线。

## 4. 数据建设方案

### 4.1 统一训练格式

建议使用 Chat JSONL，每条样本结构如下：

```json
{
  "id": "serial_0001_t1",
  "task": "t0_plan",
  "messages": [
    {"role": "system", "content": "<对应线上 prompt>"},
    {"role": "user", "content": "<线上实际输入格式>"}
  ],
  "target": "[{\"q\":\"...\",\"d\":\"vod\"}]",
  "meta": {
    "source": "sheet0821_serial.csv",
    "group_id": "conversation-or-case-id",
    "domain": "vod",
    "tv_mode": "0",
    "label_origin": "golden",
    "label_version": "v1"
  }
}
```

约束：

- T0 target 必须能被 `llm.parse_json()` 和 `engine.parse_plan()` 正确解析；
- T0 只包含 `q/d/dep`，不输出工具和参数；
- T1、MT target 是纯文本，不包含 markdown、thinking 或 JSON 外壳；
- 用户说“听”不能被改写为“看/播放”，原始动作词要保留；
- 并列属性不能拆成多个动作；
- 多轮多意图必须用分号保留完整意图链，不能压成单一请求。

### 4.2 数据来源分层

按可信度分四层：

1. **人工 Golden**：现有 sheet 中人工修正的 query/tool/params，权重最高。
2. **规则弱监督**：利用 `multiintent`、`detect.py` 和现有规则生成标签；与 Golden 冲突时以 Golden 为准，冲突样本进入人工复核池。
3. **拒绝采样**：模型对同一输入采样多个候选，通过结构校验和真链路评分选正负例。
4. **线上日志回流**：从 `app/llm.py` 的输入输出日志构造候选，经脱敏、去重、质量过滤后进入训练池。

线上数据必须先处理：设备标识脱敏、可能的个人信息清理、重复 query 聚类、低置信标签隔离、训练数据版本留痕。

### 4.3 数据切分

不能把全部 benchmark 同时用于训练和验收。建议：

- 按归一化 query 去重后切分，防止近重复泄漏；
- 多轮数据严格按 `会话id` 整组切分，同一会话不得跨 train/dev/test；
- 串行、多意图按语义模板聚类后切分，避免只记住固定连接词模板；
- 现有 benchmark 若进入训练，必须另建 **hidden test v2**；否则保留现有 benchmark 作为纯回归集；
- 建议比例为 80/10/10，但小类以保证每个域和错误类型均有覆盖为先。

### 4.4 首批数据目标

首轮不追求大规模，先追求可审计：

- Golden/高置信 SFT：3k～5k；
- 规则和模板增强：10k～30k；
- DPO 偏好对：每个任务 1k～3k，重点覆盖 badcase；
- GRPO prompts：先 2k～5k 去重输入，在线生成候选。

数据量是启动建议，不是硬指标；若可训练底模较大，优先扩展高质量 hard cases，而不是机械复制样本。

## 5. SFT 方案

### 5.1 前置条件

当前代码只暴露名为 `baseline` 的推理 API。在开始 SFT 前必须确认：

- baseline 对应的真实基础模型、权重和 tokenizer；
- 权重是否可训练、许可证是否允许；
- 上下文长度与线上 chat template；
- GPU 类型、数量和可用时长；
- 训练后模型如何注册到现有网关，并通过 `MODEL` 切换。

如果只有 API、拿不到权重，则不能直接 SFT，只能先做教师蒸馏到一个可训练的开源底模。

### 5.2 训练方式

建议第一版使用 LoRA/QLoRA 多任务 SFT：

- 三类任务共享底模，通过不同 system prompt 区分；
- 对 T0、多意图、串行和 MT 样本做分桶采样，避免 2912 条单意图淹没少量串行/多轮样本；
- target-only loss，prompt 部分不计 loss；
- 加入一定比例的空计划、非法输入和无需执行样本，防止所有输入都强行出计划；
- 使用严格格式验证，未通过解析器的样本不得入库。

建议采样权重从以下范围起步，再按 dev 集调整：

- T0 单意图：35%；
- T0 多意图/串行：35%；
- T1：15%；
- MT：15%。

### 5.3 SFT 验收

除 loss 外必须看任务指标：

- T0：parse rate、意图数准确率、domain accuracy、dependency edge F1、核心动作词保持率；
- T1：实体替换准确率、问句主体保持率、孤立实体输出率；
- MT：上下文继承准确率、多意图保留率、错误继承率；
- 端到端：tool、param、joint，以及 p50/p95 延迟和输出 token 数。

SFT 模型首先要做到“规则开启不回退”，然后再进入规则消融和 RL。

## 6. 偏好优化与强化学习方案

### 6.1 算法选择

#### DPO：用于 T1 和 MT

适合原因：

- 已有明确的 Golden 改写和大量 baseline badcase；
- 错误主要是相对偏好问题，例如 chosen 保留所有意图，rejected 丢失第二意图；
- 不需要在线调用 `hcTools`，训练稳定、成本较低。

偏好对示例：

- chosen：保留“导演是谁；该导演还导过哪些电影”；
- rejected：只保留“导演是谁”；
- chosen：输出完整问句；
- rejected：只输出一个片名；
- chosen：只继承相关历史条件；
- rejected：把无关上一轮条件拼进来。

#### GRPO：用于 T0

适合原因：

- T0 输出可做程序化解析；
- 意图数量、域、依赖拓扑、下游执行结果都可打分；
- 无需单独训练 value model，适合对同一 prompt 生成多候选并做组内相对优化。

建议分两步：

1. **离线/轻量 GRPO**：只使用格式、结构、domain、dependency 奖励；
2. **真链路 GRPO**：固定版本 `hcTools`，加入最终 tool/param/joint 奖励。

#### PPO：暂缓

PPO 需要 value model、轨迹服务和更复杂的稳定性调参。当前单 tick 任务已有可验证奖励，GRPO 和 DPO 的投入产出比更高。只有后期需要优化跨多轮、跨工具的长程收益时再考虑 PPO。

### 6.2 奖励设计

建议采用“硬门禁 + 分层奖励”，而不是只看最终 joint：

```text
若 T0 无法解析：reward = -1
否则：
reward = 0.15 * format
       + 0.25 * structure
       + 0.15 * semantic_preservation
       + 0.45 * e2e_joint
       - penalties
```

奖励项：

- `format`：无 thinking、无 markdown、JSON 可解析、字段合法；
- `structure`：意图数、顺序、domain、dep 边是否匹配；
- `semantic_preservation`：听/看/搜索/推荐等关键动作词和实体是否保留；
- `e2e_joint`：使用现有 `canonical/params_equal` 口径评估最终工具和参数；
- `penalties`：丢意图、增加无关意图、依赖环、未来信息泄漏、输出超长、把完整问句压成孤立实体。

真链路训练期间必须：

- 固定 `hcTools` 代码、模型和配置版本；
- 对相同候选缓存执行结果，减少成本和服务抖动；
- 设置超时和错误奖励，区分环境失败与模型错误；
- 定期人工抽查高奖励样本，防止 reward hacking。

## 7. 评测与实验矩阵

每个模型至少跑四组：

| 组别 | 模型 | 规则 | 目的 |
|---|---|---|---|
| A | baseline | 全开 | 当前线上系统基线 |
| B | SFT/RL | 全开 | 确认替换模型不回退 |
| C | baseline | 关闭目标规则 | 测基础模型真实能力 |
| D | SFT/RL | 关闭目标规则 | 测训练带来的净增益 |

需要补齐的规则开关包括：多意图规则、串行拆分、domain 覆盖、排序词回填、多轮 badcase 映射；按模块逐个消融，不能一次全关后无法归因。

核心验收指标：

1. **格式稳定性**：T0 parse rate ≥ 99.9%，非法 domain/dep 为 0；
2. **回归护栏**：全规则模式下，现有高分集不显著下降；
3. **重点提升**：在独立 hidden test 上，多意图和串行 both 应优先提升；
4. **规则消融收益**：关闭对应规则后，训练模型相对 baseline 的错误数至少降低 50%；
5. **效率护栏**：p95 延迟、平均输出 token、调用次数不超过现网预算；
6. **稳健性**：同义改写、口语、错别字、连接词变化、跨域弱信号均需专项测试。

建议首版目标（最终以 M0 重跑结果校准）：

- 规则全开：单意图/多轮不低于当前结果；
- 多意图 both：91.5% → ≥94%；
- 串行 both：91.0% → ≥95%；
- 规则消融集：错误数相对 baseline 至少下降 50%；
- 任何指标需同时报告样本数和置信区间，避免小样本 1～2 条波动被误判为收益。

## 8. 工程目录建议

```text
hcAgent/training/
├── README.md
├── configs/
│   ├── sft.yaml
│   ├── dpo.yaml
│   └── grpo.yaml
├── data/
│   ├── build_t0.py
│   ├── build_t1.py
│   ├── build_mt.py
│   ├── build_preferences.py
│   ├── split_dataset.py
│   └── schemas.py
├── rewards/
│   ├── format_reward.py
│   ├── plan_reward.py
│   ├── semantic_reward.py
│   └── hctools_reward.py
├── train/
│   ├── run_sft.py
│   ├── run_dpo.py
│   └── run_grpo.py
├── eval/
│   ├── eval_offline.py
│   ├── eval_e2e.py
│   └── ablation.py
└── artifacts/
    └── .gitkeep
```

原则：训练奖励和离线校验应尽量复用 `engine.parse_plan`、`benchmark` 中的 `canonical/params_equal`，避免训练和线上评测使用不同口径。

## 9. 分阶段里程碑

### M0：模型与基线冻结（约 3～5 天）

- 确认 baseline 权重/tokenizer/chat template 和训练资源；
- 固定 hcTools 版本；
- 重跑全部 benchmark；
- 建立规则开关与消融基线；
- 保存模型、代码、数据、评测结果的版本号。

**交付物**：baseline report、规则贡献表、可训练性结论。

### M1：数据管线（约 1～2 周）

- 构建 T0/T1/MT JSONL；
- 完成解析校验、去重、分组切分和数据审计；
- 新建 hidden test v2；
- 对串行和多轮高风险标签做人工抽检。

**交付物**：dataset v1、data card、质量报告。

### M2：多任务 SFT（约 1 周）

- LoRA/QLoRA 训练与超参小规模搜索；
- 跑离线指标和全链路回归；
- 分析按域、任务、错误类型的收益。

**准入下一阶段**：规则全开模式无明显回退，格式指标达标。

### M3：T1/MT DPO（约 1 周）

- 从 badcase 和拒绝采样构造偏好对；
- 优化问句保持、多意图保持、历史条件继承；
- 重点验证多轮和串行 step2。

### M4：T0 GRPO（约 1～2 周）

- 先结构奖励，再接固定 hcTools 的真链路奖励；
- 控制奖励投机、格式退化和过度拆分；
- 验证多意图、串行和跨域 hard cases。

### M5：规则消融与灰度（约 1 周）

- 按模块逐步下线或降级规则；
- 离线双跑 baseline 与训练模型；
- 影子流量 → 小比例灰度 → 扩大流量；
- 预留模型和规则的一键回滚。

整体约 6～8 周，前提是可训练权重、GPU 和标注复核资源已就绪；资源未确认前不承诺具体日期。

## 10. 主要风险

1. **拿不到 baseline 权重**：改走“教师 API 蒸馏到可训练底模”。
2. **评测集泄漏**：现有 benchmark 一旦用于训练，必须另建 hidden test。
3. **高分由规则贡献**：必须做规则消融，否则无法证明模型真的学会。
4. **奖励受 hcTools 波动影响**：冻结服务版本并缓存结果。
5. **串行标签未来信息泄漏**：T0 与 T1 标签必须分开构造。
6. **弱监督复制规则错误**：规则标签与 Golden 冲突时剔除或人工复核。
7. **领域样本极不均衡**：设备域远多于其他域，训练时分桶采样。
8. **格式奖励压过语义**：采用硬门禁和分层奖励，不能只奖励 JSON 合法。
9. **线上日志数据风险**：训练前脱敏、授权确认、留存周期和访问审计。

## 11. 启动前需要确认的决策

1. `baseline` 实际是哪一个模型，是否能拿到权重和 tokenizer？
2. 可用 GPU 资源及期望训练周期是多少？
3. 当前第一目标是提高总分，还是逐步减少 `detect/multiintent/badcase` 规则？
4. 是否允许使用现有 benchmark 训练？若允许，谁负责补充 hidden test v2？
5. 第一阶段是否同意冻结 `hcTools`，只训练 `hcAgent`？
6. 线上日志是否允许用于训练，脱敏和审批流程是什么？

## 12. 推荐的第一步

先完成 M0，不急于开训：确认底模与资源，固定 hcTools，补齐全部 benchmark 的“规则全开/逐项消融”基线。只有得到“模型错误、规则贡献、hcTools 错误”的责任分层后，SFT 和 RL 的数据、奖励、验收才不会互相污染。
