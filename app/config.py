"""hcAgent 配置：LLM 网关走环境变量，其余为常量默认值；另含**规则层开关**。

模型与其他阈值改动直接在此改默认值即可，不再暴露环境影响。
规则开关是唯一例外：它们是运维/回归用的总闸，按「层/域」粒度暴露环境变量。
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("hcAgent.config")

# ---- LLM 网关环境变量 ----
# LLM 网关（OpenAI-format /v1/chat/completions），默认 juagent 真实网关。
API_BASE = (os.environ.get("HC_LLM_API_BASE", "http://127.0.0.1:7000/v1")).rstrip("/")
API_KEY = os.environ.get("HC_LLM_API_KEY", "")

# hcTools 意图解析服务地址（POST /api/predict，拿最终工具与参数）。
HCTOOLS_BASE = (os.environ.get("HC_HCTOOLS_BASE", "http://127.0.0.1:8084")).rstrip("/")

# ---- 常量默认值（不读环境）----
MODEL = "baseline"
TIMEOUT = 120.0
MAX_RETRY = 3
MAX_PLAN_STEPS = 6
PLAN_TRACE_TTL = 300.0
# T1 指代改写时携带的上一步候选媒资上限(避免超长)
CANDIDATE_CONTEXT_LIMIT = 30


# ===========================================================================
# 规则层开关
# ===========================================================================
# 粒度：**层/域**级，不是单条规则级。每个 key 是一组同源规则的总闸。
# 关闭语义：整组短路，行为**回退到下一层**（通常是 LLM 判定 / 原始 query）。
#
#   detect.*       判域链（app/detect.py 的 20 条规则，按目标域分组）
#   multiintent.*  确定性拆分层（app/multiintent.py）
#   mt.*           多轮改写层（app/engine.py 的多轮相关规则）
#   plan.*         计划后处理层（app/engine.py 的 plan 改写）
#
# 覆盖方式：环境变量 HC_RULE_<组名大写、点转下划线>=0 关闭。
#   例：HC_RULE_DETECT_MUSIC=0      关掉判域链里全部 →music 的规则
#       HC_RULE_MULTIINTENT_SERIAL_QA=0   关掉「检索→提问」确定性拆分
# 默认**全开**（= 与开关引入前逐字节一致）；关闭只用于 A/B 回归与线上止血。
#
# 为什么不做单条规则级：规则之间有 priority 依赖，单条关掉极易产生
# 「本该由 A 抢答、A 关了落到 B」的隐性行为漂移；层/域级总闸的语义明确、
# 便于对拍。需要单条控制时走 app/rule_audit 的审计而非改开关。
_RULE_DEFAULTS: dict[str, bool] = {
    # ---- 判域链（app/detect.py::_DETECT_RULES）----
    "detect.off": True,        # 息屏（tvMode=6）分诊链
    "detect.badcase": True,    # L2 精确句判域 badcase（最高优先）
    "detect.music": True,      # →music 的高置信歌曲/音乐频道强信号、music Discovery
    "detect.audio": True,      # →audio 的「听/播放+有声载体」、audio Discovery
    "detect.vod": True,        # →vod 的媒资载体保护、台词/集数定位、署名动画作者、推荐类型片
    "detect.children": True,   # →children 的内容定位、亮屏具名/播放儿歌
    "detect.education": True,  # →education 的亮屏教育强词、无锚问答
    "detect.qa": True,         # →qa 的媒体/明星信息咨询、剧情问答、泛知识、问候
    "detect.sports": True,     # →sports 的赛事预测
    "detect.signal": True,     # 高置信信号表命中
    "detect.keep": True,       # 保 LLM 兜底（关掉=整链不兜底，返回 None）
    # ---- 确定性拆分（app/multiintent.py）----
    "multiintent.content_device": True,  # 「内容+设备」双目标拆成两条并行意图
    "multiintent.serial_qa": True,       # 串行「检索 → 提问」确定性拆分
    "multiintent.multi_tab": True,       # 卡通/动漫双域并行（children+vod 双 tab）
    # ---- 多轮改写（app/engine.py）----
    "mt.rewrite_badcase": True,   # 多轮改写 badcase（(上轮,本轮) 命中即固定改写，绕过 LLM）
    "mt.inherit_domain": True,    # 多轮域继承（本轮判空时沿用上轮稳定域）
    "mt.bare_song_reseed": True,  # 裸书名号歌曲兜底（上轮应答里的歌名 → 强置 music）
    # ---- 计划后处理（app/engine.py）----
    "plan.sort_merge": True,      # 串行「检索+排序提问」排序词回填
    "plan.single_obj_qa": True,   # 单目标属性/知识直问 → 直接一条 qa（不拆检索+提问）
}


def _env_name(group: str) -> str:
    return "HC_RULE_" + group.upper().replace(".", "_")


def _resolve(group: str, default: bool) -> bool:
    """环境变量覆盖：仅认 "0"/"false"/"off"/"no" 为关，其余非空值视为开。"""
    raw = os.environ.get(_env_name(group))
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


#: 生效的规则层开关（进程启动时定格一次）
RULE_SWITCHES: dict[str, bool] = {
    g: _resolve(g, d) for g, d in _RULE_DEFAULTS.items()
}

# 兼容旧开关：历史上只有这一个规则开关，语义等价于 plan.sort_merge。
# 保留它是为了避免已写好的 run.sh / 评测脚本失效。
if os.environ.get("HC_DISABLE_SORT_MERGE"):
    RULE_SWITCHES["plan.sort_merge"] = False


def rule_on(group: str) -> bool:
    """该规则组是否启用。未登记的组按「开」处理并告警（防拼写错误静默失效）。"""
    val = RULE_SWITCHES.get(group)
    if val is None:
        logger.warning("未登记的规则组 %r（按启用处理）；已登记：%s",
                       group, ", ".join(sorted(RULE_SWITCHES)))
        return True
    return val


def disabled_rule_groups() -> list[str]:
    """当前被关闭的规则组（供启动日志/审计打印）。"""
    return sorted(g for g, on in RULE_SWITCHES.items() if not on)


_off = disabled_rule_groups()
if _off:
    logger.warning("规则层开关已关闭 %d 组：%s", len(_off), ", ".join(_off))