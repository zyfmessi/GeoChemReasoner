"""Task-type classification: map natural-language task text onto the agent's two capability directions.

The deposit prediction and analysis agent answers two kinds of scientific
questions (+ one comprehensive):
- prospectivity   prospectivity targeting (mineral prediction: where ore might be)
- genesis         metallogenic mechanism analysis (why mineralization happened, what the process was)
- comprehensive   combined analysis when both kinds are hit or neither can be decided

Classification is pure keyword scoring (deterministic, auditable, works offline):
hitting both kinds means the task itself spans prediction and analysis, so it goes
to comprehensive; hitting neither also falls back to comprehensive -- prefer
executing more rather than silently dropping any capability direction.
"""

from __future__ import annotations  # Enable deferred type-annotation evaluation (PEP 563)

# Capability direction → keyword table (substring matching, same style as planner._relevance)
_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {  # Keyword table (dict spans multiple lines)
    "prospectivity": (  # Task words for the prospectivity-targeting direction (spans lines)
        "成矿预测", "靶区", "远景", "找矿", "成图", "异常图",  # Prediction/targeting words
        "圈定", "潜力", "勘查", "勘探", "普查", "空间分布", "矿化强度"),  # Delineation/potential/exploration words
    "genesis": (  # Task words for the metallogenic-mechanism direction (spans lines)
        "成因", "机制", "过程", "源区", "物质来源", "构造环境",  # Genesis/mechanism words
        "判别", "分异", "稀土", "同位素", "比值", "蚀变", "热液", "交代", "岩浆", "变质"),  # Process-discrimination words
}


def classify_task(question: str) -> str:  # Define the task-type classifier: question text → type string
    """Classify the task text by keyword scoring; both kinds hit or neither hit → comprehensive."""
    text = str(question or "").lower()  # Lowercase the task text (English keywords match case-insensitively)
    scores = {t: sum(1 for kw in kws if kw in text)  # Accumulate keyword hits per direction (comprehension spans lines)
              for t, kws in _TYPE_KEYWORDS.items()}  # Iterate directions and their keyword tables
    hits = [t for t, n in scores.items() if n > 0]  # List of hit directions
    if len(hits) == 1:  # Exactly one direction hit: classify by that hit
        return hits[0]  # Return the single hit direction
    return "comprehensive"  # Double hit (spanning both) or zero hit → comprehensive fallback


def capability_label(task_type: str) -> str:  # Define the Chinese capability-label function: type → human-readable description
    """Task type → human-readable capability description (shared by the report "task understanding" section and the planner prompt)."""
    return {  # Type → description mapping (literals span multiple lines)
        "prospectivity": "远景靶区圈定（成矿预测：综合地学数据圈定找矿有利区）",  # Prospectivity direction description
        "genesis": "成矿机制分析（成因过程：物质来源-运移-沉淀与判别证据）",  # Genesis direction description
        "comprehensive": "综合分析（靶区圈定 + 成矿机制分析）",  # Comprehensive direction description
    }.get(task_type, "综合分析（靶区圈定 + 成矿机制分析）")  # Unknown types fall back to comprehensive; close the mapping
