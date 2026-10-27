"""弱设备词提示：供 T0 计划 LLM 判断是否需要拆出设备意图。

弱设备词（色彩/画质/图像…）语义两可：既可能是设备调参，也可能是剧情/美学描述。
这里只生成一条提示串注入 _plan 的 user 消息，不做任何确定性拆分。
"""
from __future__ import annotations

_DEV_WEAK = [
    "色彩", "画质", "图像", "色温", "原画", "原音", "音画", "氛围", "场景细节",
    "细节增强", "动态范围", "数字艺术",
]


def weak_device_hint(query: str) -> str:
    """检测 query 是否含【弱设备词】。命中返回提示串（供 _plan 提示 LLM），未命中返回空串。"""
    q = (query or "").strip()
    if not q:
        return ""
    hits = [w for w in _DEV_WEAK if w in q]
    if not hits:
        return ""
    return f"含弱设备词({'/'.join(hits)})：可能是设备调节，也可能是剧情/美学描述，请结合上下文判定是否拆成设备意图。"
