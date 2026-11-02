"""Scientific Output: aggregate the final state into an explainable scientific output (deposit prediction and analysis edition).

The agent output is not a single deposit type but a task-adaptive, explainable result: task
understanding, final conclusion, analysis process, prospectivity targets (prediction tasks),
mechanism discovery (analysis tasks), key observations, supporting/conflicting evidence,
uncertainties, recommended tests, and full provenance. The section skeleton is fixed and
capability blocks are filled per task (blocks not involved keep empty values with an
explanatory note, keeping the machine-readable key structure stable). This module provides
both Markdown rendering (human-readable) and a dict (machine-readable).
"""

from __future__ import annotations  # enable PEP 563 deferred type-annotation evaluation
import os  # figure relative paths (for figures embedded in the report)

import re  # import re: parse confidence statements and sample sizes (for the programmatic calibration gate)

from typing import Any  # import the Any type annotation

from oregenesis.reasoning.task_types import capability_label, classify_task  # import task-type classification and the Chinese capability label

# Mechanism-tool exclusion set: their results stay out of the "成因机制分析" block (each has its own dedicated section/channel)
_NON_MECHANISM_TOOLS = {  # exclusion set (literal spans multiple lines)
    "geochemical_prospectivity_mapping",  # mapping tool -> prospectivity targets section
    "element_inventory",  # data inventory -> data inventory section
    "table_column_stats",  # targeted column stats (data description, not mechanism evidence) -> data inventory section
    "llm_genesis_synthesis",  # narrative synthesis -> analysis process section
    "plot_data_overview",  # generic overview plot -> figures section
    "plot_discrimination_diagram",  # discrimination diagram plotting -> figures section
    "plot_xy_scatter", "plot_ternary",  # generic plotting family -> figures section
    "plot_ree_spider", "plot_category_stats",  # continued: generic plotting family
}


def build_final_result(state: dict[str, Any]) -> dict[str, Any]:  # define the final-state to scientific-output aggregation function
    """Final state -> scientific output (dict form; called by the synthesize node, which fills conclusion/narrative back in)."""
    evidence = state.get("evidence", [])  # evidence list from the final state, empty by default
    tool_results = state.get("tool_results", [])  # tool run result list from the final state
    bundles = state.get("bundles", [])  # sample bundle list from the final state
    task = state.get("task") or {}  # task fields (question text and task type live here)
    task_type = task.get("task_type") or classify_task(str(task.get("question") or ""))  # task type: state value wins; re-classify from text when missing
    key_obs = [e["observation"] for e in evidence  # collect non-empty observation facts as key observations (comprehension spans lines)
               if e.get("observation") and "llm_genesis_narrative" not in (e.get("supports") or [])]  # condition: the narrative goes to Section 2, so do not double count
    uncertainties = sorted({lim for e in evidence for lim in e.get("limitations", [])})  # collect limitation notes from all evidence, deduplicated and sorted
    recommended = sorted({t for e in evidence for t in e.get("missing_tests", [])})  # collect missing tests from all evidence, deduplicated and sorted
    return {  # return the scientific output dict (literal spans lines)
        "task_understanding": {"task_type": task_type,  # task understanding block: classified task type
                               "capability": capability_label(task_type),  # task understanding block: Chinese capability description
                               "question": str(task.get("question") or "")},  # task understanding block: original task text
        "conclusion": "",                                           # final conclusion (filled back in when the synthesize node runs)
        "narrative": "",                                            # analysis process narrative (filled back in when the synthesize node runs)
        "key_observations": key_obs,                                # key observations (all traceable)
        "supported_processes": sorted({s for e in evidence for s in e.get("supports", [])  # collect process labels supported by the evidence, deduplicated and sorted (comprehension spans lines)
                                      if s != "llm_genesis_narrative"}),  # condition: exclude the narrative channel marker (not a geological process)
        "target_zones": _extract_zones(tool_results),               # prospectivity targets block (empty dict when the mapping tool never ran)
        "mechanism_analysis": _extract_mechanism(tool_results),     # mechanism discovery block (empty dict when no mechanism tool ran)
        "supporting_evidence": [e for e in evidence  # filter tool evidence carrying support labels (comprehension spans lines)
                                if e.get("supports") and e.get("kind") == "tool_result"  # condition: has support labels and originates from a tool result
                                and "llm_genesis_narrative" not in e["supports"]],  # condition: exclude narrative synthesis (its text goes to Section 2)
        "conflicting_evidence": [e for e in evidence if e.get("contradicts")],  # conflicting evidence carrying contradict labels
        "remaining_uncertainties": uncertainties,                   # remaining uncertainties (not the same as contradictions)
        "recommended_tests": recommended,                           # recommended further tests
        "figures": _collect_figures(tool_results),  # figures: collect image paths from the artifacts of each tool run
        "provenance": [],                                           # filled back in by the audit node
        "data_inventory": {"n_samples_datasets": len(bundles),  # data inventory: number of sample datasets (dict spans lines)
                           "tool_runs": [r["tool_name"] for r in tool_results]},  # list of tool names, one per tool run
    }


def _extract_zones(tool_results: list[dict[str, Any]]) -> dict[str, Any]:  # private helper: tool results -> prospectivity targets block
    """Take the most recent successful mineralization prediction result and shape it into the target zone deliverable structure (empty dict if none)."""
    for r in reversed(tool_results):  # search backwards from the newest (last run wins across rounds)
        if r.get("tool_name") != "geochemical_prospectivity_mapping" or r.get("status") != "success":  # only accept successful mapping tool runs
            continue  # skip other tools
        payload = r.get("result") or {}  # tool result payload
        return {"zones": list(payload.get("zones") or []),  # ranked target zone list (name/area/ratio/capture rate)
                "fusion_method": str(payload.get("fusion_method") or ""),  # fusion model name
                "model_metrics": dict(payload.get("model_metrics") or {}),  # model metrics (AUC / known deposit count)
                "elements": [str(x) for x in (payload.get("elements") or [])],  # element set used in the fusion
                "maps": [str(x) for x in ((payload.get("artifacts") or {}).get("maps") or [])],  # map artifact path list
                "cell_scores": str((payload.get("artifacts") or {}).get("cell_scores") or ""),  # per-cell score CSV path
                "summary": str(payload.get("observation") or "")}  # one-sentence tool result summary
    return {}  # mapping tool never succeeded: empty targets block (rendered as "本任务未涉及" in the report)


def _extract_mechanism(tool_results: list[dict[str, Any]]) -> dict[str, Any]:  # private helper: tool results -> mechanism discovery block
    """Aggregate successful findings from mechanism tools (mapping/inventory/narrative tools excluded; empty dict if none)."""
    findings = [{"tool": r.get("tool_name", ""),  # one finding per entry: tool name (comprehension spans lines)
                 "observation": str((r.get("result") or {}).get("observation", ""))}  # finding body is the tool observation text
                for r in tool_results  # iterate all tool run results
                if r.get("status") == "success"  # condition 1: executed successfully
                and r.get("tool_name") not in _NON_MECHANISM_TOOLS  # condition 2: not a mapping/inventory/narrative tool
                and (r.get("result") or {}).get("observation")]  # condition 3: has observation text
    return {"findings": findings}  # return the mechanism block (rendered as "本任务未涉及" when findings is empty)


def _collect_figures(tool_results: list[dict[str, Any]]) -> list[dict[str, str]]:  # private helper: collect figure artifacts
    """Walk successful tool results and collect figure/table artifact paths from artifacts (deduplicated, order preserved)."""
    seen: set[str] = set()  # dedup set
    out: list[dict[str, str]] = []  # result list
    for r in tool_results:  # iterate tool run results
        if r.get("status") != "success":  # collect successful runs only
            continue  # skip failures
        arts = (r.get("result") or {}).get("artifacts") or {}  # artifacts dict
        for key, val in arts.items():  # per artifact key
            paths = val if isinstance(val, list) else [val]  # normalize list or single value
            for fp in paths:  # per path
                fp = str(fp)  # cast to string
                if fp and fp not in seen and fp.lower().endswith((".png", ".csv", ".md")):  # collect figure/table artifacts only
                    seen.add(fp)  # mark as seen
                    out.append({"tool": r.get("tool_name", ""), "kind": key, "path": fp})  # include
    return out  # return the artifact list


def _narrative(evidence: list[dict[str, Any]]) -> str:  # private helper: fetch the latest LLM narrative from the evidence ledger
    """Take the body of the latest narrative-synthesis evidence (supports contains llm_genesis_narrative).

    The normal path has the synthesize node generate the narrative deterministically; this
    function is its fallback channel — when the ReAct planner happens to pick the
    llm_genesis_synthesis tool mid-loop, its product is reused here.
    """
    for e in reversed(evidence):  # search backwards from the newest: narrative synthesis is usually the closing move
        if "llm_genesis_narrative" in (e.get("supports") or []) and e.get("observation"):  # narrative marker hit and body non-empty
            return str(e["observation"])  # return the narrative body
    return ""  # return an empty string when no narrative synthesis exists


# ── Programmatic confidence calibration gate (deep-work phase 2, 2026-10-13) ──
# Diagnosis: template-level calibration rules proved ineffective (overconfidence hits 10->9) —
# matching confidence to evidence quantity is a deterministic computation (evidence observations already carry n=...) and
# must not rely on model self-discipline; this gate caps the conclusion after synthesize and before review, every action deterministic and auditable (the calibration record is written back into final_result).
_CONF_RE = re.compile(r"置信度\s*(?:为|约|估计)?\s*(0\.\d+|1\.0|0\.0)")  # confidence statement pattern (numeric form)
_CITED_RE = re.compile(r"\[([EM]-\d{3})\]")  # evidence citation ids inside the conclusion
_ANALOGY_HINT_RE = re.compile(r"类比|外推")  # analogy/extrapolation wording
_CONF_CAP_SMALL_N = 0.5  # confidence cap for small samples
_SMALL_N_THRESHOLD = 30  # small-sample threshold (n<30 counts as insufficient)


def calibrate_confidence(final_result: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:  # define the programmatic calibration gate
    """Cap the confidence stated in the conclusion by the sample sizes of the evidence it cites (deterministic rules, no reliance on model self-discipline).

    Rules: (1) the smallest n extractable from the evidence cited by the conclusion is < 30
    -> cap confidence at 0.5; (2) the conclusion uses analogy/extrapolation wording and the
    cited evidence carries no n information at all -> cap at 0.5 as well; (3) confidence
    values above the cap are rewritten in place and logged in the return value, which the
    caller merges into remaining_uncertainties. No action when there is no confidence
    statement or nothing exceeds the cap.
    """  # function docstring: rules and boundaries
    concl = str(final_result.get("conclusion") or "")  # original conclusion text
    if not concl or "置信度" not in concl:  # no conclusion or no confidence statement
        return {"adjusted": False}  # no action
    n_by_id: dict[str, int] = {}  # evidence index: id -> max sample size
    for e in evidence:  # walk the evidence ledger
        eid = str(e.get("evidence_id") or "")  # evidence id
        ns = [int(x) for x in re.findall(r"n=(\d+)", str(e.get("observation") or ""))]  # sample sizes declared in the observation
        if ns and (eid not in n_by_id or max(ns) > n_by_id[eid]):  # multiple entries with one id: take the max
            n_by_id[eid] = max(ns)  # record
    cited = set(_CITED_RE.findall(concl))  # set of evidence ids cited by the conclusion
    ns_cited = [n_by_id[c] for c in cited if c in n_by_id]  # sample sizes of the cited evidence
    min_n = min(ns_cited) if ns_cited else None  # minimum sample size (None when no n information)
    analogy = bool(_ANALOGY_HINT_RE.search(concl))  # whether analogy/extrapolation wording is present
    cap: float | None = None  # cap value (None = no cap)
    reasons: list[str] = []  # capping justifications (feed the annotation and the audit)
    if min_n is not None and min_n < _SMALL_N_THRESHOLD:  # rule 1: small sample
        cap = _CONF_CAP_SMALL_N  # cap at 0.5
        reasons.append(f"统计样本 n={min_n}<30")  # log the reason
    if analogy and not ns_cited:  # rule 2: pure analogy/extrapolation with no statistical sample at all
        cap = _CONF_CAP_SMALL_N  # cap at 0.5 as well
        reasons.append("结论仅由类比外推支持")  # log the reason
    if cap is None:  # no capping rule triggered
        return {"adjusted": False, "min_n": min_n}  # return unchanged
    changed: list[float] = []  # rewrite log (list of original values)

    def _sub(m: "re.Match[str]") -> str:  # replacement callback: rewrite above-cap values in place
        val = float(m.group(1))  # original confidence
        if val > cap:  # rewrite only above the cap
            changed.append(val)  # record the original value
            return f"{cap}（已按证据量校准：{'；'.join(reasons)}）"  # rewrite with an inline annotation
        return m.group(0)  # keep the original text when within the cap

    new_concl = _CONF_RE.sub(_sub, concl)  # perform the rewrite
    if new_concl != concl:  # write back only when the conclusion changed
        final_result["conclusion"] = new_concl  # write back the calibrated conclusion
    return {"adjusted": bool(changed), "cap": cap, "changed": changed,  # return the calibration record (for audit)
            "reasons": reasons, "min_n": min_n}  # close the return literal


def _fig_rel(path: str, base_dir: str) -> str:  # helper to relativize figure paths
    """Relativize artifact paths against the report directory (so embedded markdown figures display directly); fall back to the original path on failure."""
    try:  # relpath raises when crossing drive letters
        rel = os.path.relpath(path, base_dir) if base_dir else path  # relativize (fall back on failure)
    except Exception:  # noqa: BLE001 -- fallback for cross-drive/empty base
        rel = path  # fall back to the absolute path
    return rel.replace("\\", "/")  # normalize backslashes to forward slashes (markdown friendly)


def render_markdown(final_result: dict[str, Any], task: str = "",  # define the Markdown report renderer (signature spans lines)
                    base_dir: str = "") -> str:  # param cont: report output directory (base for relative figure paths; empty = absolute paths)
    """Final result -> human-readable Markdown report (deposit prediction and analysis edition; empty capability sections state the reason instead of staying blank).

    When base_dir is the report output directory, figure artifacts are embedded with relative
    paths (markdown viewers display the images directly).
    """
    r = final_result  # short alias to simplify field access below
    inv = r.get("data_inventory", {})  # data inventory (for the input overview)
    audit = r.get("audit", {})  # audit info (the decision trail lives here)
    decisions = audit.get("decisions", [])  # per-round ReAct decision trail
    tool_calls = r.get("tool_calls") or audit.get("tool_calls", [])  # tool call records
    warnings = r.get("warnings", [])  # report-level warnings (e.g. narrative failure)
    understand = r.get("task_understanding", {})  # task understanding block
    zones = r.get("target_zones") or {}  # prospectivity targets block (None coerced to empty dict)
    mechanism = r.get("mechanism_analysis") or {}  # mechanism discovery block (None coerced to empty dict)

    def bullets(items: list[str]) -> list[str]:  # render subsection bullet items (placeholder for empty sections)
        return [f"- {i}" for i in items] or ["- （无）"]  # prefix each item with a dash; placeholder entry for an empty list

    # ── Section 3: reasoning decision trail table (merge adjacent same-goal rows — when the LLM repeats one sub-goal round after round, a human only needs one row) ──
    decision_rows: list[str] = []  # decision row buffer
    i = 0  # merge cursor
    while i < len(decisions):  # walk the decision sequence
        j = i  # end of the same-goal run
        while j + 1 < len(decisions) and decisions[j + 1].get("goal") == decisions[i].get("goal"):  # adjacent goals identical verbatim
            j += 1  # extend the same-goal run
        d0 = decisions[i]  # first decision of the run
        if j > i:  # same goal across multiple rounds: merge into one row
            acts = list(dict.fromkeys(str(x.get("next_action", "")) for x in decisions[i:j + 1]))  # dedupe actions within the run, order preserved
            act_txt = f"{acts[0]} ×{j - i + 1} 次" if len(acts) == 1 else "；".join(acts)  # same action: record the count; different actions: list them all
            decision_rows.append(f"| {i + 1}–{j + 1} | 同一目标：{str(d0.get('goal', ''))[:60]} | {act_txt} | "  # merged row (spans lines)
                                 f"{str(d0.get('reason', ''))[:60]} |")  # reason from the first round of the run only (truncated to 60 chars)
        else:  # single-round goal: one row as-is
            decision_rows.append(f"| {i + 1} | {d0.get('goal', '')} | {d0.get('next_action', '')} | "  # decision row: round/goal/action (comprehension spans lines)
                                 f"{str(d0.get('reason', ''))[:80]} |")  # decision row cont: reason truncated to 80 chars
        i = j + 1  # jump to the next run
    # ── Section 3: tool call record table ──
    tool_rows = [f"| {i} | {t.get('tool', '')} | {t.get('status', '')} | "  # tool row: index/tool/status (comprehension spans lines)
                 f"{str(t.get('observation', ''))[:120]} |"  # tool row cont: output summary truncated to 120 chars
                 for i, t in enumerate(tool_calls, 1)]  # number each call
    # ── Section 4 content: prospectivity targets block (one explanatory line when not involved/not run) ──
    zone_rows = [f"| {z.get('name', '')} | {z.get('area_km2', '?')} | "  # zone row: zone name/area (comprehension spans lines)
                 f"{z.get('ratio', '?')} | {z.get('capture_ratio', '—')} |"  # zone row cont: area ratio/deposit capture rate
                 for z in zones.get("zones", [])]  # walk the ranked zone list
    if zones:  # expand the deliverable structure only when target results exist
        deliverables = [*(f"- 图件: {m}" for m in zones.get("maps", []))]  # map artifact entries (comprehension spans lines)
        if zones.get("cell_scores"):  # list the entry only when a per-cell score table exists
            deliverables.append(f"- 逐格得分: {zones.get('cell_scores')}")  # score table entry
        section4 = [f"- 融合模型: {zones.get('fusion_method') or '—'} · "  # targets summary line: model name (string spans lines)
                    f"参与元素: {'、'.join(zones.get('elements', [])[:10]) or '—'}",  # summary line cont: first 10 elements involved
                    f"- 模型指标: {zones.get('model_metrics') or '—'}",  # summary line: model metrics (AUC/deposit count)
                    f"- 结果摘要: {zones.get('summary', '')}", "",  # summary line: one-sentence result plus a blank line
                    "### 靶区分级",  # subsection title (section numbers are generated dynamically; subsections carry no fixed number)
                    "| 区带 | 面积(km²) | 面积占比 | 矿点捕获率 |", "|---|---|---|---|",  # the two header rows of the ranking table
                    *zone_rows, "",  # ranking data rows plus a blank line
                    "### 交付产物",  # deliverables subsection title
                    *(deliverables or ["- （无产物文件）"])]  # deliverable entries (placeholder when empty)
    else:  # no target results: state why this task does not involve them
        section4 = ["（本任务未涉及靶区圈定，或成矿预测工具未执行/未成功）"]  # empty-block explanation line
    # ── Section 5 content: mechanism discovery block (one explanatory line when not involved/not run) ──
    if mechanism.get("findings"):  # expand only when mechanism findings exist
        section5 = [f"- **{f.get('tool', '')}**：{f.get('observation', '')}"  # mechanism finding line: bold tool name + observation (comprehension spans lines)
                    for f in mechanism.get("findings", [])]  # walk the findings list
        section5 += ["", "支持的过程标签:",  # append a blank line and the process label subheading
                     *bullets(r.get("supported_processes", []))]  # process label entry list
    else:  # no mechanism findings: state why this task does not involve them
        section5 = ["（本任务未涉及机制分析，或机制类工具未执行/未成功）"]  # empty-block explanation line

    # ── Section 6 content: data and execution overview (sort and dedupe — tool-style observations belong to the supporting evidence section, not repeated here) ──
    covered = {str(e.get("observation", "")) for e in r.get("supporting_evidence", [])} | \
              {str(e.get("observation", "")) for e in r.get("conflicting_evidence", [])}  # set of observation texts already covered by Section 7
    seen_obs: set[str] = set()  # normalized keys of seen observations (first 120 chars; the per-unit template once repeated the same statistics sentence 18 times)
    key_obs_lines: list[str] = []  # rendered line buffer
    dup_obs = 0  # count of duplicate-content observations folded away
    for o in r.get("key_observations", []):  # walk key observations
        o = str(o)  # normalize to string
        if o in covered:  # this observation is already listed as supporting/conflicting evidence in Section 7: do not repeat
            continue  # skip
        if not (o.startswith(("数据表 ", "文本档案 ")) or o.startswith(("防重闸", "工具 ", "ValueError"))):  # keep only profile-style ("数据表 "/"文本档案 ") and alert-style ("防重闸"/"工具 "/"ValueError") observations; the full text of other tool observations lives in Section 7
            continue  # observations that are neither profile nor alert stay out of the overview section
        norm = o[:120]  # normalization key: first 120 chars (evidence id differences do not affect same-content detection)
        if norm in seen_obs:  # fold duplicate-content observations
            dup_obs += 1  # count it
            continue  # skip
        seen_obs.add(norm)  # mark as seen
        key_obs_lines.append(f"- {o[:300]}" + ("…（全文见 final_state.json）" if len(o) > 300 else ""))  # render truncated with a pointer to the full text
    if dup_obs:  # state the folded count explicitly when non-zero
        key_obs_lines.append(f"- （另有 {dup_obs} 条同内容观测已折叠，全文见 final_state.json）")  # note line

    # ── Report header (slimmed: keep only the first sentence of the task; the review verdict stays readable in place) ──
    task_head = (task or "").split("\n", 1)[0].strip()  # first line of the task (the rest of the task statement stays out of the header)
    if len(task_head) > 80:  # truncate further when the first line is too long
        task_head = task_head[:80] + "…"  # truncate to 80 chars
    task_note = "（任务书全文见 final_state.json）" if "\n" in (task or "") or len(task or "") > 80 else ""  # pointer when more content exists
    _review = r.get("review", {})  # review block
    if _review.get("passed"):  # passed
        review_line = "通过"  # one-word verdict
    else:  # failed: list the first two problems in place (readers need not dig through sections for problems)
        review_line = "未通过：" + "；".join(str(p) for p in (_review.get("problems") or ["详见 final_state.json"])[:2])  # display in place

    sections: list[tuple[str, list[str]]] = []  # dynamic section registry: (title, body lines); empty sections are not registered, numbers are assigned consecutively at render time
    sections.append(("任务理解与最终结论",  # Section 1: conclusion first (spans lines)
                     [(r.get("conclusion") or "（未能形成结论——模型未在线，或证据不足以支撑判断）")]))  # explain why when the conclusion is empty
    sections.append(("分析过程",  # Section 2: full step-by-step narrative
                     [(r.get("narrative") or "（无叙述——模型未在线，或证据不足以生成分析过程）")]))  # explain why when the narrative is empty
    sec3: list[str] = []  # Section 3 body (decision table + tool table)
    if decision_rows:  # decision subsection only when decision records exist
        sec3 += ["### 决策轨迹", "| 轮次 | 目标 | 动作 | 理由 |", "|---|---|---|---|",  # subsection title carries no number (section numbers are dynamic)
                 *decision_rows, ""]  # decision rows
    if tool_rows:  # tool subsection only when tool calls exist
        sec3 += ["### 工具调用记录", "| # | 工具 | 状态 | 关键输出（摘要） |", "|---|---|---|---|",  # subsection title
                 *tool_rows]  # tool rows
    if sec3:  # emit the trail section only when at least one kind of record exists
        sections.append(("推理与工具调用轨迹", sec3))  # register the trail section
    if zones:  # register the targets section only when target results exist (eliminates "本任务未涉及" placeholder sections)
        sections.append(("远景靶区圈定", section4))  # register the targets section
    if mechanism.get("findings"):  # register the mechanism section only when mechanism findings exist
        sections.append(("成因机制分析", section5))  # register the mechanism section
    if key_obs_lines:  # register the overview section only when profile/alert entries exist
        sections.append(("数据与执行概览", key_obs_lines))  # register the overview section
    sec7 = [f"支持 [{e['evidence_id']}] {str(e['observation'])[:120]}"  # supporting evidence entry: id + observation (truncated to 120 chars; full text in final_state.json)
            + ("…（全文见 final_state.json）" if len(str(e.get("observation", ""))) > 120 else "")  # truncation note
            for e in r.get("supporting_evidence", [])]  # walk supporting evidence
    sec7 += [f"冲突 [{e['evidence_id']}] {str(e['observation'])[:120]}"  # conflicting evidence entry: same truncation rule
             + ("…（全文见 final_state.json）" if len(str(e.get("observation", ""))) > 120 else "")  # truncation note
             for e in r.get("conflicting_evidence", [])]  # walk conflicting evidence
    if sec7:  # register the evidence section only when evidence exists
        sections.append(("支持/冲突证据", [f"- {s}" for s in sec7]))  # register the evidence section
    sec8 = [*(f"- {u}" for u in r.get("remaining_uncertainties", [])),  # uncertainty entries
            *(f"- 建议检验: {t}" for t in r.get("recommended_tests", []))]  # recommended test entries
    if sec8:  # skip the whole section when both kinds are empty (eliminates the orphan "（无）")
        sections.append(("不确定性与建议检验", sec8))  # register the uncertainty section
    sections.append(("溯源（结论 → 证据 → 文件）",  # provenance section is always emitted: one pointer line (spans lines)
                     [f"共 {len(r.get('provenance', []))} 条证据的完整血缘链（证据 → 数据单元 → 来源文件）"
                      "已完整存档于本目录 final_state.json 的 audit.evidence_chain 字段；"
                      "各证据的编号即上文章节中的 [E-xxx] / [M-xxx] 标记，可按编号在该文件中检索对应条目。"]))  # register the provenance section
    figs = r.get("figures", [])  # figure list
    if figs:  # register the figures section only when figures exist (the whole section disappears when empty)
        sections.append(("图件产物",  # figures section (spans lines)
                         [*(f"![{f.get('tool', '')}·{f.get('kind', '')}]({_fig_rel(f.get('path', ''), base_dir)})"  # embedded figure (markdown renders it directly)
                            for f in figs),  # walk figures
                          *(f"  - 来源 `{f.get('tool', '')}`：`{f.get('path', '')}`"  # absolute path text (for audit)
                            for f in figs)]))  # register the figures section

    lines = [  # collect report content line by line (list spans lines)
        "# OreGenesis-Agent 矿床预测与分析报告", "",  # report H1 title and a blank line
        f"- 任务: {task_head or '（未提供）'}{task_note}",  # metadata line: first task sentence + full-text pointer
        f"- 任务类型: {understand.get('task_type', '—')}（{understand.get('capability', '—')}）",  # metadata line: task type and capability description
        f"- 审查: {review_line}",  # metadata line: review verdict (problems listed in place when failed)
        f"- 输入概览: 样品数据集 {inv.get('n_samples_datasets', '?')} 组 · "  # metadata line: first half of the input overview
        f"工具运行 {len(tool_calls)} 次 · 证据 {len(r.get('key_observations', []))} 条",  # metadata line cont: tool and evidence counts
        *(f"- ⚠ {w}" for w in warnings), "",  # list report-level warnings one by one (empty when none) plus a blank line
    ]
    for no, (title, body) in enumerate(sections, 1):  # number dynamic sections consecutively (empty sections are never registered, so no placeholder lines)
        lines.append(f"## {no} {title}")  # section title (dynamic number)
        lines.extend(body)  # section body
        lines.append("")  # blank line between sections
    return "\n".join(lines) + "\n"  # join lines with newlines, append the trailing newline, and return
