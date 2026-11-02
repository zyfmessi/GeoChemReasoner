"""LLM-augmented tool family: visual description, text semantic enrichment, and analytical narrative synthesis.

All three tools treat the large model as a "registry-constrained perceiver/synthesizer"; outputs always carry limitation statements:
- vlm_describe_image: image DataUnit -> objective description by a multimodal model (the description participates in reasoning as evidence);
- analyze_geological_text: text DataUnit -> geological entity/observation/process extraction (the tool form of blueprint (6)
  Semantic Enrichment; degrades to a deterministic keyword profile without a model, so pure-text
  input is no longer a dead end - text data thereby gains a tool path equal to that of numeric data);
- llm_genesis_synthesis: evidence ledger digest -> analytical interpretation narrative (written into the report's analysis section,
  marked as LLM-generated and requiring expert review, never posing as quantitative observation).
"""

from __future__ import annotations  # enable deferred evaluation of type annotations (PEP 563)

import base64  # encode image bytes as base64
import json  # parse the JSON extraction result returned by the LLM
import re  # extract the JSON fragment from the reply with a regex
from pathlib import Path  # read image files by reference path, take corpus file names
from typing import Any  # any-type annotation

from oregenesis.tools.registry import ToolCall, ToolSpec  # import the ToolCall and ToolSpec dataclasses

_DEFAULT_PROMPT = ("客观描述这张地质图像：可见矿物、结构、构造、颜色与"
                   "可能的照片类型（单偏光 PPL / 正偏光 XPL / 手标本）。"
                   "只描述可见事实，不做成因推断。")

# Small lexicon for rule-mode text profiling (deterministic, auditable; coverage != LLM extraction, fallback only)
_LEXICON: dict[str, tuple[str, ...]] = {  # lexicon grouped by entity/process (dict spans multiple lines)
    "rock_type": ("花岗岩", "伟晶岩", "闪长岩", "辉长岩", "玄武岩",  # rock-type keywords (spans lines)
                  "流纹岩", "大理岩", "灰岩", "片麻岩", "砂岩"),  # rock-type keywords cont.
    "mineral": ("石英", "斜长石", "钾长石", "白云母", "黑云母",  # mineral keywords (spans lines)
                "黄铁矿", "黄铜矿", "磁铁矿", "锆石", "磷灰石", "钠长石", "电气石"),  # mineral keywords cont.
    "process": ("蚀变", "矿化", "分离结晶", "交代", "热液",  # geological process keywords (spans lines)
                "变质", "风化", "构造叠加", "气成"),  # geological process keywords cont.
}


def _pick_image(state: Any) -> Any:
    """Return the first successfully parsed image DataUnit."""
    for u in state.get("data_units", []):
        if u.data_modality == "image" and u.data_status == "success":
            return u
    return None


def _pick_text_units(state: Any) -> list[Any]:
    """Return all text DataUnits that parsed successfully and are non-empty (the tool entry for pure-text input)."""
    return [u for u in state.get("data_units", [])  # iterate over data units (comprehension spans lines)
            if u.data_modality == "text" and u.data_status == "success"  # condition 1: text modality and successfully parsed
            and str(u.raw_data.get("text", "")).strip()]  # condition 2: non-blank body text


def run_vlm_describe_image(state: Any, call: ToolCall) -> dict[str, Any]:
    """Call the VLM to describe an image; the prompt can be overridden via parameters.prompt."""
    backend = state.get("backend")
    if backend is None:
        raise ValueError("VLM 工具需要语言模型在线（state 未注入 backend）")
    unit = _pick_image(state)
    if unit is None:
        raise ValueError("没有可用的图像 DataUnit")
    image_bytes = Path(unit.raw_data_reference).read_bytes()
    prompt = str((call.parameters or {}).get("prompt") or _DEFAULT_PROMPT)
    resp = backend.chat(
        [{"role": "user", "content": prompt}],
        system_prompt="你是岩相学助手，只描述图像中可见的客观事实。",
        images=[base64.b64encode(image_bytes).decode("ascii")],
        temperature=0.1, max_tokens=512)
    description = (resp.content or "").strip()
    if not description:
        raise ValueError("VLM 返回空描述")
    return {"observation": f"图像描述（{unit.raw_data.get('format', '')}）：{description}",
            "description": description,
            "supports": ["vlm_image_described"],
            "limitations": ["VLM 描述是模型感知，非定量观测，可靠度按中等处理"]}


def _text_blob(units: list[Any], cap: int = 6000) -> str:  # private function: text chunks -> provenance-traceable corpus
    """Join text chunks into a corpus with line-number provenance (capped on both chunk count and total characters, to control tokens)."""
    chunks: list[str] = []  # initialize the corpus chunk list
    for u in units[:12]:  # take at most 12 text chunks
        text = str(u.raw_data.get("text", "")).strip()  # take the chunk body and strip surrounding whitespace
        origin = u.data_origin or {}  # take the provenance dict (empty by default)
        name = Path(str(origin.get("source_path") or u.raw_data_reference or "文本")).name  # provenance file name (short display)
        chunks.append(f"[{name} L{origin.get('start_line', '?')}-{origin.get('end_line', '?')}]\n{text}")  # prefixed chunk with the line-number range
    return "\n\n".join(chunks)[:cap]  # join with blank lines and truncate to the total character cap


def run_analyze_geological_text(state: Any, call: ToolCall) -> dict[str, Any]:  # text semantic enrichment run function
    """Text -> geological entity/observation/process extraction: LLM extraction when online, lexicon profiling when offline."""
    units = _pick_text_units(state)  # get usable text chunks
    if not units:  # no text chunks: raise directly (the applicability check already gates this; fallback here)
        raise ValueError("没有可用的文本 DataUnit")
    blob = _text_blob(units)  # join into a corpus with line-number provenance
    backend = state.get("backend")  # get the LLM backend (may be None)
    if backend is None:  # ── rule mode: deterministic lexicon profiling, pure-text runs are never a dead end ──
        hits: list[str] = []  # initialize the hit keyword list
        for words in _LEXICON.values():  # iterate over the lexicon groups
            hits += [f"{w}×{blob.count(w)}" for w in words if blob.count(w)]  # record keywords with non-zero counts (comprehension spans lines)
        obs = (f"文本地质画像（规则模式）：{len(units)} 个文本块、共 "  # assemble the first half of the profile observation (string spans lines)
               f"{sum(int(u.data_quality.get('character_count', 0)) for u in units)} 字；"  # profile observation: chunk count and total characters
               f"检出关键词：{'、'.join(hits[:12]) or '无（词表未覆盖，建议在线模式深析）'}")  # profile observation: first 12 hit keywords
        return {"observation": obs, "supports": [],  # return the profile result: no process inference (avoids over-reading the lexicon)
                "limitations": ["规则模式仅做词表计数画像；接入在线模型可获得实体/过程抽取"]}  # state limitations honestly
    # ── LLM mode: structured extraction of entities/observations/processes ──
    prompt = ("你是矿床地质文本分析助手。从下面的野外记录/文献文本中抽取地质信息，"  # first half of the extraction prompt: task and output format (string spans lines)
              "严格输出 JSON：{\"observations\":[\"客观观测\"],\"entities\":[\"岩石/矿物/构造\"],"  # first half of the JSON template
              "\"processes\":[\"地质过程\"]}。只抽取文本明确提到的事实，不推断。\n\n文本（含行号溯源）：\n" + blob)  # second half of the JSON template, with the corpus appended
    resp = backend.chat([{"role": "user", "content": prompt}],  # call the backend for extraction (parentheses span lines)
                        system_prompt="你是严格的地质信息抽取器，只输出一个 JSON 对象。",  # system constraint: JSON output only
                        temperature=0.1, max_tokens=768,  # low temperature for stable extraction, 768 tokens to fit structured output
                        response_format={"type": "json_object"})  # request server-side JSON mode
    m = re.search(r"\{[\s\S]*\}", resp.content or "")  # greedily extract the JSON fragment from the reply
    try:  # wrap JSON parsing in try; bad JSON degrades
        data = json.loads(m.group(0)) if m else {}  # take the dict on success, empty dict when no match
    except json.JSONDecodeError:  # catch bad JSON
        data = {}  # degrade to an empty dict
    observations = [str(x) for x in (data.get("observations") or []) if str(x).strip()]  # observation list (empties removed)
    entities = [str(x) for x in (data.get("entities") or []) if str(x).strip()]  # entity list (empties removed)
    processes = [str(x) for x in (data.get("processes") or []) if str(x).strip()]  # process list (empties removed)
    if not observations and not entities and not processes:  # total JSON failure: degrade the whole reply into a single observation
        fallback = (resp.content or "").strip()  # take the raw reply text
        if not fallback:  # empty reply too: treat as tool failure
            raise ValueError("LLM 文本分析返回空结果")  # raise -> ToolResult error Observation
        observations = [fallback[:300]]  # truncated reply as the observation fallback
    parts: list[str] = []  # initialize the observation assembly parts list
    if observations:  # assemble the observations part only if present
        parts.append("观测——" + "；".join(observations[:6]))  # observations part (at most 6)
    if entities:  # assemble the entities part only if present
        parts.append("实体——" + "；".join(entities[:8]))  # entities part (at most 8)
    if processes:  # assemble the processes part only if present
        parts.append("过程——" + "；".join(processes[:5]))  # processes part (at most 5)
    supports = [f"text_process::{p}" for p in processes[:3]]  # processes -> support tags (flow back into the evidence layer with the tool observation)
    return {"observation": f"文本地质分析（LLM，{len(units)} 块）：" + "。".join(parts),  # return the extraction result: joined observations (dict spans lines)
            "supports": supports,  # support tags (for the report's "supported processes" summary)
            "limitations": ["实体/过程由 LLM 从文本抽取，属模型感知而非定量观测",  # limitation 1: model perception
                            "引用需对照原文行号复核"]}  # limitation 2: verify against source line numbers


def run_llm_genesis_synthesis(state: Any, call: ToolCall) -> dict[str, Any]:  # narrative synthesis run function
    """Evidence ledger digest -> analytical interpretation narrative (report "analysis process" section, marked as requiring expert review)."""
    backend = state.get("backend")  # get the LLM backend
    if backend is None:  # no model: this tool does not apply (the applicability check gates this; fallback here)
        raise ValueError("叙述综合工具需要语言模型在线")
    evidence = [e for e in state.get("evidence", []) if e.get("observation")]  # keep only evidence with observation content
    if not evidence:  # no evidence to synthesize
        raise ValueError("没有可综合的证据（证据账本为空）")
    digest = "\n".join(f"- [{e.get('evidence_id', '')}] {str(e['observation'])[:300]}"  # per-entry digest: id + observation (truncated to 300 chars) (comprehension spans lines)
                       for e in evidence[:40])[:12000]  # at most 40 entries, 12000 chars total (deep task 2026-10-13: the stage-grouping table needs longer entries)
    prompt = (f"基于以下证据摘要，写一段矿床预测与分析解释叙述（250 字以内）。\n"  # synthesis prompt: task and length constraints
              f"要求：只引用已列出的证据编号（如 [E-001]）；如有反证或未决之处必须提及；"  # synthesis requirements: cite only listed evidence, mention counter-evidence
              f"结尾一句指出最有价值的下一步检验。\n\n证据摘要：\n{digest}")  # synthesis requirements: give the next step and attach the material
    resp = backend.chat([{"role": "user", "content": prompt}],  # call the backend for synthesis (parentheses span lines)
                        system_prompt="你是严谨的矿床学家，只依据给定证据写作，不编造数据；"
                                      "必须给出带置信度的最佳判断——置信度与证据量匹配（小样本给区间不给单一值），"
                                      "不确定通过置信度与反证表达，不得以\"证据不足\"拒答。",  # system constraint: anti-fabrication kept, refusing to answer made a violation, calibration aligned (same standard as the synthesis template)
                        temperature=0.3, max_tokens=512)  # moderate temperature for fluent prose, 512 tokens of length
    narrative = (resp.content or "").strip()  # take the narrative body
    if not narrative:  # an empty reply counts as failure
        raise ValueError("LLM 叙述综合返回空结果")
    return {"observation": narrative, "narrative": narrative,  # return the narrative: the observation is the body (dict spans lines)
            "supports": ["llm_genesis_narrative"],  # marker tag: the report layer uses it to place the narrative in section 1
            "limitations": ["本段为 LLM 生成的叙述综合，仅基于已列证据，需专家审查"]}  # limitation: honestly mark how it was generated


def synthesize_analysis(state: Any, backend: Any) -> dict[str, str]:  # deterministic narrative and conclusion generation function
    """In-graph deterministic "conclusion-first" analysis generation (called directly by the synthesize node, without tool selection).

    One template per task type: the prospectivity-task conclusion = preferred target zone + confidence + basis; the genesis-task
    conclusion = genetic type + family + confidence + basis; comprehensive tasks cover both. It assembles the task text, input
    text dossier, and evidence digest, and asks the LLM to output the two sections 【最终结论】+【分析过程】 in Chinese.
    Writing relies only on the given material; failures are handled by the caller (blanked and logged) and never block the report.
    """
    from oregenesis.reasoning.task_types import classify_task  # local import of task-type classification (avoids a package-level circular import)

    task = str((state.get("task") or {}).get("question") or "（未提供任务）")  # raw task text
    task_type = str((state.get("task") or {}).get("task_type")  # task type: the state value takes precedence (ternary spans lines)
                    or classify_task(task))  # fallback: classify from the task text
    evidence = [e for e in state.get("evidence", []) if e.get("observation")]  # evidence with observations
    digest = "\n".join(f"- [{e.get('evidence_id', '')}] {str(e['observation'])[:300]}"  # evidence digest lines (comprehension spans lines)
                       for e in evidence[:40])[:12000]  # at most 40 entries, 12000 chars (deep task 2026-10-13: relaxed to match narrative synthesis)
    # Tool-call track (2026-10-15 report restructuring): the factual substrate of the step-by-step narrative - the LLM's "parsed... called... got..."
    # The step-by-step section must match this track, to prevent fabricated steps and tool names (only tool-result evidence is taken, the narrative channel excluded)
    track_lines: list[str] = []  # track line buffer
    for e in evidence:  # iterate in ledger order (i.e. execution order)
        if e.get("kind") != "tool_result":  # keep tool results only
            continue  # narrative/text evidence stays out of the track
        track_lines.append(f"第{len(track_lines) + 1}步 调用 {e.get('tool_name', '?')}："
                           f"{str(e['observation'])[:160]}")  # step line: tool name + observation summary
        if len(track_lines) >= 8:  # at most 8 steps (aligned with the max_rounds cap)
            break  # stop once enough
    tool_track = "\n".join(track_lines) if track_lines else "（本轮无工具调用）"  # track text
    blob = _text_blob(_pick_text_units(state), cap=4000)  # input text dossier (the semantic main channel of the key-value case file)
    sample_lines: list[str] = []  # data-volume hint lines (the factual basis of the calibration rule: confidence must answer to these n)
    from pathlib import Path as _P  # local import of the path type (to take source file names)
    for u in state.get("data_units", []):  # iterate over numeric tables for sample sizes
        if u.data_modality == "numeric" and u.data_status != "error":  # list only usable numeric tables
            n = len(u.raw_data.get("rows", []))  # the row count is the sample size
            if n:  # skip empty tables
                sample_lines.append(f"- {_P(u.data_origin.get('source_path', '')).name}[{u.data_unit_id}] n={n}")  # one line per table
    sample_tip = ("\n【数据量提示（置信度必须与这些样本量匹配，n<30 的统计给区间不给单一值）】\n"  # hint section title
                  + "\n".join(sample_lines[:25])) if sample_lines else ""  # at most 25 lines to prevent blow-up
    material = (f"任务：{task}\n\n【输入档案/文本材料】\n{blob or '（无文本材料）'}\n\n"  # material part 1: task and text
                f"【工具调用轨迹（逐步叙事必须与该轨迹一致，不得虚构未发生的步骤或工具）】\n{tool_track}\n\n"  # material part 2: execution track
                f"【证据摘要（工具/画像产出）】\n{digest or '（无）'}{sample_tip}")  # material part 3: evidence digest + data-volume hint
    if task_type == "prospectivity":  # ── prospectivity task template ──
        role, body = ("严谨的矿产勘查学家",  # role setting
                      "【最终结论】1-2 句：首选找矿靶区区带（A/B/C 类）+ 置信度 0-1 + 一句话核心依据；"
                      "证据不足以支撑精细分级时，给\"数据支撑的下限结论\"（例：\"B 类远景区，置信度低\"），"
                      "不得以\"证据不足\"整体拒答。\n"
                      "【分析过程】≤450 字：依次讨论数据覆盖→成矿相关元素组合→异常空间特征→"
                      "融合分级结果→工程验证建议，最后一句写不确定性。")  # two-section format requirements
    elif task_type == "genesis":  # ── genesis task template ──
        # Note (2026-10-13 rollback): this template once appended three items - "ore-carrier mineral rules / interval-narrowing rules / 700 characters";
        # all three validations regressed (average 38.7->28.3, task6 51->15.3, furnace-trap 3->10) (instruction overload collapsed enumeration +
        # Goodhart-style interval grabbing), so it was rolled back entirely. Lesson kept: change only one structural variable at a time.
        # 2026-10-15 report restructuring (user-defined "evidence-chain narrative"): step-by-step = data -> tool -> conclusion,
        # synthesis step = combine input dossier/literature -> overall judgment, then list supporting evidence; the step-by-step section is
        # constrained by the 【工具调用轨迹】 factual substrate (see material), replacing the former "commodity assemblage -> wall rock -> ..." free-discussion outline.
        role, body = ("严谨的矿床学家",  # role setting
                      "【分析过程】按下方【工具调用轨迹】逐步叙事（步骤数与轨迹一致，只写真实发生的步骤），"
                      "每步以序号开头（1. 2. 3. …），一句话格式：\"解析出某文件某表某字段数据，调用某工具"
                      "进行分析，得到结论：一句话地质解读。\"（直接写自然句，不要输出尖括号<>或占位符）；"
                      "逐步之后写一段综合：综合以上各步结论，结合输入档案中的区域地质背景或文献观点"
                      "（引用须注明\"文献观点\"），得到总体判断。\n"
                      "【最终结论】4-6 句：首选成因模型与置信度 0-1；用 2-3 条关键证据线支撑"
                      "（引用步骤号或证据编号）；给出成因链（物质来源-运移-沉淀）；"
                      "指出未排除的替代模型与数据缺口。证据不充分的部分给\"数据支撑的下限结论\"，不得拒答。\n"
                      "类比规则：类比外推不是编造，但必须写明类比依据（与哪个矿床哪个阶段类比、为何可比）。\n"
                      "【支持证据】3-5 条，每条一行，格式：[E-证据编号] 一句话说明该证据支持结论的哪一点。")  # three-section format requirements
    else:  # ── comprehensive task template: covers both targeting and mechanism ──
        role, body = ("严谨的矿床预测与分析专家",  # role setting
                      "【分析过程】按下方【工具调用轨迹】逐步叙事（步骤数与轨迹一致，只写真实发生的步骤），"
                      "每步以序号开头（1. 2. 3. …），一句话格式：\"解析出某文件某表某字段数据，调用某工具"
                      "进行分析，得到结论：一句话地质解读。\"（直接写自然句，不要输出尖括号<>或占位符）；"
                      "逐步之后写一段综合：综合以上各步结论，结合输入档案中的区域地质背景或文献观点"
                      "（引用须注明\"文献观点\"），分别给出靶区判断与成因判断的总体结论。\n"
                      "【最终结论】4-6 句：先给靶区判断（区带+置信度），再给成因判断（模型+置信度）；"
                      "逐条列关键证据线（引用步骤号或证据编号）；指出未排除的替代解释与数据缺口。"
                      "证据不充分的部分给\"数据支撑的下限结论\"，不得整段拒答。\n"
                      "【支持证据】3-5 条，每条一行，格式：[E-证据编号] 一句话说明该证据支持结论的哪一点。")  # three-section format requirements
    prompt = (f"你是{role}。只依据下面给出的材料，用中文写一份\"结论先行\"的分析，"
              "严格按以下两段格式输出：\n" + body + "\n"
              "硬性要求：只引用材料中出现的字段/数值/证据编号，不得编造；与材料冲突时如实指出；"
              "基于已给材料的类比外推必须显式标注类比依据，不算编造。\n"
              "校准规则：置信度必须与证据量匹配——统计基于小样本（n<30）时给区间不给单一值、"
              "置信度 ≤0.5 并在括号注明 n；仅由类比外推支持的数值标注\"类比\"；"
              "给出精确置信度数值时，用一句话说明它由什么证据量决定。\n"
              "多单元综合规则：任务涉及多个矿床/单元时，分析过程必须先给出逐单元对比表"
              "（每行一个单元：金属组合|关键同位素|温度证据|年代），统计推断必须附统计独立性声明"
              "（共享区域背景的单元不是独立样本，汇总需说明限制）。\n\n" + material)
    resp = backend.chat([{"role": "user", "content": prompt}],  # call the backend to generate (parentheses span lines)
                        system_prompt=f"你是{role}，只依据给定材料用中文写作，不编造数据；"
                                      "必须给出带置信度的最佳判断——不确定通过置信度与不确定性说明表达，"
                                      "不得以\"证据不足\"拒答。",  # system constraint: the anti-fabrication red line kept, refusing to answer made a violation (2026-10 baseline diagnosis)
                        temperature=0.2, max_tokens=1600)  # moderate temperature for fluent prose; 1600 tokens fit the three sections "step-by-step + synthesis + conclusion + supporting evidence" (2026-10-15 revision; the former 900 only fit two)
    content = (resp.content or "").strip()  # take the body and strip surrounding whitespace
    if not content:  # an empty reply counts as failure (the caller handles the fallback)
        raise ValueError("分析生成返回空结果")  # raise and let the caller degrade
    i_proc, i_conc, i_evid = (content.find("【分析过程】"), content.find("【最终结论】"),  # locate the three section markers (spans lines)
                              content.find("【支持证据】"))  # new structure order = analysis process -> final conclusion -> supporting evidence
    if i_proc >= 0 and i_conc >= 0:  # split by position when both markers exist (compatible with both old and new orders)
        if i_proc < i_conc:  # new structure (genesis/comprehensive 2026-10-15 revision): analysis process first
            narrative = content[i_proc + len("【分析过程】"):i_conc].strip()  # analysis process section (step-by-step narrative + synthesis)
            conc = content[i_conc + len("【最终结论】"):(i_evid if i_evid > i_conc else len(content))].strip()  # conclusion section
            if i_evid > i_conc:  # append the supporting-evidence section to the end of the conclusion when present (section 1 shows the full "conclusion + evidence")
                evid = content[i_evid + len("【支持证据】"):].strip()  # supporting evidence list
                if evid:  # append only when non-empty
                    conc = f"{conc}\n\n支持的证据：\n{evid}"  # user-defined structure: supporting evidence right after the conclusion
            conclusion = conc or content[:150]  # fallback to the first 150 chars when the conclusion section is empty
        else:  # old structure (prospectivity: conclusion-first): conclusion = between the two markers, analysis process = after 【分析过程】
            conclusion = content[i_conc + len("【最终结论】"):i_proc].strip() or content[:150]  # conclusion section
            narrative = content[i_proc + len("【分析过程】"):].strip()  # full analysis process text
    else:  # model broke the format: whole text as the analysis process, first 150 chars as the conclusion
        conclusion, narrative = content[:150], content  # fallback split
    return {"conclusion": conclusion, "narrative": narrative}  # return the conclusion and analysis process sections


def build_vlm_tools() -> list[ToolSpec]:
    """Build the VLM visual tool family (objective image description; the description participates in reasoning as evidence)."""
    return [ToolSpec(
        name="vlm_describe_image", tool_family="vlm_petrography",
        description="用视觉大模型（Ollama qwen3-vl 等）客观描述图像数据中的可见事实",
        required_data=("image_data",),
        capabilities=("petrography", "image_description"),
        applicable=lambda s: (s.get("backend") is not None and _pick_image(s) is not None,
                              "需要在线视觉模型与已解析的图像 DataUnit"),
        run=run_vlm_describe_image)]


def build_text_analysis_tools() -> list[ToolSpec]:  # factory function that builds the text semantic enrichment tool family
    """Build the text analysis tool family: the Semantic Enrichment tool path for pure-text input."""
    return [ToolSpec(  # return a single-tool list (parentheses span lines)
        name="analyze_geological_text", tool_family="llm_text_analysis",  # tool name and family
        description="从文本数据（野外记录/文献）抽取地质实体、观测与过程（无模型时退化为关键词画像）",  # one-line description (goes into the planner menu)
        required_data=("text_data",),  # requires text-modality data
        capabilities=("text_mining", "semantic_enrichment"),  # capability tags
        applicable=lambda s: (bool(_pick_text_units(s)),  # applicability: applicable whenever a non-empty text chunk exists (online not required) (ternary spans lines)
                              "需要已解析成功的非空文本 DataUnit"),  # reason when not applicable
        run=run_analyze_geological_text)]  # bind the run function and close the list


def build_llm_synthesis_tools() -> list[ToolSpec]:  # factory function that builds the narrative synthesis tool family
    """Build the narrative synthesis tool family: evidence -> analytical interpretation narrative (the report's narrative channel)."""
    return [ToolSpec(  # return a single-tool list (parentheses span lines)
        name="llm_genesis_synthesis", tool_family="llm_synthesis",  # tool name and family
        description="把证据账本摘要综合为分析解释叙述段（写入报告分析过程节，需专家审查）",  # one-line description
        required_data=(),  # consumes the evidence ledger, not data units
        capabilities=("genesis_interpretation", "llm_synthesis"),  # capability tags
        applicable=lambda s: (s.get("backend") is not None  # applicability condition 1: model online (comprehension spans lines)
                              and bool([e for e in s.get("evidence", []) if e.get("observation")]),  # condition 2: synthesizable evidence exists
                              "需要在线模型与至少一条有观测内容的证据"),  # reason when not applicable
        run=run_llm_genesis_synthesis)]  # bind the run function and close the list
