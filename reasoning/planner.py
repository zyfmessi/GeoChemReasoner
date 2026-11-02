"""Reasoning Planner: structures "what is known / what is missing / next step" into a ReasoningDecision.

The ReAct Reason step does not ask the model to output its inner chain of thought,
only a structured decision (blueprint §18): goal / missing_information /
next_action / parameters / done / reason. When the LLM is unavailable or its
output is invalid, a deterministic rule-based fallback runs (Data-aware: chooses
by tool applicability and execution history), and the fallback reason is written
into the decision record.
"""

from __future__ import annotations  # Enable deferred annotation evaluation: type annotations have no runtime effect
import json  # Import json: parse the JSON decision text returned by the LLM
import re  # Import re: extract the JSON fragment from the reply with a regex
from dataclasses import dataclass, field  # Import the dataclass decorator and the field default-factory
from pathlib import Path  # Import Path: table-index lines take the source file name
from typing import Any  # Import Any for type annotations
from oregenesis.reasoning.task_types import classify_task  # Import the task-type classifier: the rule-based fallback picks chains by type
from oregenesis.tools import REGISTRY  # Import the tool registry REGISTRY: look up specs and names, execute tools
from oregenesis.understanding import summarize_table_columns  # Import the all-column summary: workbench index-line generation (reuses column statistics)


@dataclass  # dataclass decorator: auto-generates construction and comparison methods
class ReasoningDecision:  # Define one round of structured reasoning decision (the auditable product of the Reason step)
    """One round of structured reasoning decision (the minimal auditable decision basis; no chain of thought stored)."""  # Class docstring: stores only the decision basis, never the inner chain of thought

    goal: str = ""                                   # The scientific question this round should resolve
    missing_information: list[str] = field(default_factory=list)  # What is still missing
    next_action: str = ""                            # Next action (tool name or stop)
    parameters: dict[str, Any] = field(default_factory=dict)      # Tool parameters (empty = the tool chooses on its own)
    done: bool = False                               # Whether evidence gathering is finished
    reason: str = ""                                 # Decision reason (for audit)

    def to_dict(self) -> dict[str, Any]:  # Instance method: convert the decision object to a dict
        """Convert to dict (written into State.decisions as a record)."""  # Method docstring: the result is recorded in State.decisions
        return {"goal": self.goal, "missing_information": self.missing_information,  # Assemble the dict: goal and missing_information first (dict spans lines)
                "next_action": self.next_action, "parameters": self.parameters,  # Dict continued: next_action and parameters
                "done": self.done, "reason": self.reason}  # Dict continued: done and reason; close and return the dict


def _state_summary(state: dict[str, Any]) -> str:  # Private function: produce the state-summary text for the planner
    """State summary for the planner: data coverage + full table index + executed tools + evidence count + small-table content passthrough.
    Input fidelity (2026-09 benchmark diagnosis): key-value case files were complete at
    the parse layer, but the aggregate summary shrank to "n_samples≈1; variables≈none",
    leaving the planner fully blind to case-file content. Small tables (≤50 rows ×
    ≤12 columns) are now serialized in full and text blocks excerpted 300 chars each
    (total budget 1400 chars; large geochemical tables are unaffected).
    Workbench layer 1 (2026-10 benchmark diagnosis): every numeric table gets a
    one-line compact index (ID = filename + key column statistics) permanently in the
    prompt -- previously only the first 8 of 27 table profiles were visible, the
    planner did not know the Microtherm table's ID or column names, and targeted
    retrieval could never be initiated.
    """
    # Function docstring: explains the input-fidelity and full-table-index motivations
    executed = [r["tool_name"] for r in state.get("tool_results", [])]  # List comprehension: extract the tool names from all tool-result records
    variables: list[str] = []  # Initialize an empty list: collect variable names from each data unit
    n_samples = 0  # Initialize the sample-count counter to 0
    for u in state.get("data_units", []):  # Iterate over each data unit in state (default empty list)
        n_samples = max(n_samples, int(u.data_semantics.get("n_samples", 0)))  # Take the unit's declared sample count; keep the running maximum
        variables.extend(u.data_semantics.get("variables", []))  # Merge the unit's variable names into variables
    base = (f"样品数≈{n_samples}；变量≈{', '.join(sorted(set(variables))[:20]) or '无'}；"  # Summary first half: sample count and the first 20 deduplicated, sorted variables (displays "无" when empty)
            f"已执行工具={executed or '无'}；证据={len(state.get('evidence', []))} 条")  # Summary second half: executed-tool list and evidence count
    index_lines: list[str] = []  # Buffer for full-table index lines (workbench layer 1: a permanent catalog page)
    for u in state.get("data_units", []):  # Iterate all data units and build one index line per table
        if u.data_modality != "numeric" or u.data_status == "error":  # Index only usable numeric tables
            continue  # Text blocks / errored units do not enter the index
        header = [str(h) for h in u.raw_data.get("header", [])]  # Header
        rows = u.raw_data.get("rows", [])  # Data rows
        brief = summarize_table_columns(header, rows, max_cols=6, budget=320)  # Compact summary: first 6 informative columns (coordinate columns already demoted), 320-char budget
        index_lines.append(f"- {u.data_unit_id}={Path(u.data_origin.get('source_path', '')).name}"  # Index line first half: ID = filename (unit_id enables targeted table_column_stats queries)
                           f"（{len(rows)}行）: {brief or '（无可摘要列）'}")  # Index line second half: row count + key column statistics
    index = ("\n全表索引（unit_id=文件名，可用 table_column_stats 的 unit_id/columns/group_by 定向查询）:\n"  # Index section title: states the targeted-query usage
             + "\n".join(index_lines)) if index_lines else ""  # Concatenate the index section only when numeric tables exist
    content_parts: list[str] = []  # Content passthrough buffer
    budget = 1400  # Total character budget for passthrough (token control)
    for u in state.get("data_units", []):  # Iterate data units to collect passthrough content
        if budget <= 0:  # Stop once the budget is exhausted
            break  # Break out of the loop
        if u.data_modality == "numeric" and u.data_status != "error":  # Numeric tables: serialize small tables in full
            header = [str(h) for h in u.raw_data.get("header", [])]  # Header
            rows = u.raw_data.get("rows", [])  # Data rows
            if rows and len(header) <= 12 and len(rows) <= 50:  # Small-table criterion (key-value case files always match; large geochemical tables never do)
                lines = [", ".join(f"{h}={r[i]}" for i, h in enumerate(header)  # Serialize each row as "column=value" pairs
                                   if i < len(r) and str(r[i]).strip())  # Skip empty cells
                         for r in rows]  # Iterate over data rows
                blob = "；".join(lines)[:budget]  # Join and truncate to the remaining budget
                if blob:  # Consume budget only when non-empty
                    content_parts.append(f"[数据表 {u.data_unit_id}] {blob}")  # Append to the passthrough buffer
                    budget -= len(blob)  # Deduct from the budget
        elif u.data_modality == "text":  # Text blocks: excerpt the first 300 chars
            t = str(u.raw_data.get("text", "")).strip()  # Take the body text
            if t:  # Consume budget only when non-empty
                blob = t[:min(300, max(0, budget))]  # Excerpt and truncate to the remaining budget
                if blob:  # Record only when non-empty
                    content_parts.append(f"[文本 {u.data_unit_id}] {blob}")  # Append to the passthrough buffer
                    budget -= len(blob)  # Deduct from the budget
    content = ("\n输入内容（小表全量透传/文本摘录）：\n" + "\n".join(content_parts)) if content_parts else ""  # Concatenate only when there is content
    return base + index + content  # Return the summary joined with the full-table index (workbench layer 1) and the content passthrough


def _relevance(spec: Any, task_text: str) -> int:  # Private function: score the relevance of one tool against the task text
    """Task-tool relevance: hit count of keywords/name tokens in the task text (Task-aware Routing)."""  # Function docstring: keyword-hit scoring
    text = task_text.lower()  # Lowercase the task text for case-insensitive matching
    score = 0  # Reset the score counter to zero
    for kw in getattr(spec, "keywords", ()) or ():  # Iterate the tool's declared keywords (including name tokens)
        if kw.lower() and kw.lower() in text:  # Each keyword appearing in the task text scores 1
            score += 1  # Add 1 point per keyword hit
    return score  # Return the relevance score (0 = no keyword overlap with the task)


def _tool_menu(state: dict[str, Any], task_text: str, cap: int = 12) -> str:  # Private function: produce the applicable-tool menu text (sorted by task relevance, capped in count)
    """Planner menu: list only tools usable under the current data conditions, shown in descending task relevance (Task+Data-aware), capped to keep the prompt from exploding.
    Exception: text analysis and narrative synthesis are **forced into the menu**
    (never squeezed out by the cap) -- they are the only channels for text-input
    semantic enrichment and report narrative; historically a 12-entry cap pushed
    them out of the menu permanently, leaving pure-text tasks idling for six
    rounds (see the 2026-09 benchmark diagnosis).
    """  # Function docstring: task-aware sorting + capping + guaranteed key channels
    applicable = [s for s in REGISTRY.specs()  # List comprehension head: iterate all tool specs in the registry (brackets span lines)
                  if s.applicable is None or s.applicable(state)[0]]  # Filter: keep tools with no applicability check, otherwise keep those whose check tuple's first element is true
    applicable.sort(key=lambda s: -_relevance(s, task_text))  # Sort by descending task relevance: relevant tools always sit at the top of the menu
    chosen = applicable[:cap]  # Slice the first cap applicable tools for display
    chosen_names = {s.name for s in chosen}  # Set of chosen tool names (for dedup checks)
    for must in ("analyze_geological_text", "llm_genesis_synthesis", "table_column_stats"):  # Forced-guarantee channels: text analysis / narrative synthesis / targeted column retrieval (2026-10 benchmark diagnosis: the retrieval channel was once squeezed out by the cap)
        if must not in chosen_names:  # When not already chosen
            s = next((x for x in applicable if x.name == must), None)  # Find the tool in the applicable list
            if s is not None:  # Append only if applicable (inapplicable tools never enter the menu)
                chosen.append(s)  # Append to the menu tail so the channel stays visible
    menu = "\n".join(f"- {s.name} [{s.tool_family}]: {s.description}" for s in chosen)  # Join each chosen spec into a "- name [family]: description" line, newline-separated
    extra = f"\n（另有 {len(applicable) - len(chosen)} 个适用工具未列出，可按名称调用）" \
        if len(applicable) > len(chosen) else ""  # Conditional expression: emit an "N more not listed" notice when overflowing, else an empty string
    return menu + extra  # Return the menu text joined with the overflow notice


def _rule_decision(state: dict[str, Any]) -> ReasoningDecision:  # Private function: deterministic rule-based fallback decision when the LLM is unavailable
    """Deterministic rule-based fallback (Task+Data-aware): match tools by task relevance first, then follow fixed priority chains by task type when keywords run out.
    Two-level routing: (1) task relevance -- applicable tools whose keywords hit the
    task text come first (narrow tasks hit dedicated tools directly); (2) typed
    priority chains -- pick a fixed sequence by state.task.task_type
    (prospectivity=targeting chain / genesis=mechanism chain / comprehensive=mixed
    chain); all executed or nothing applicable → done.
    """
    executed = {r["tool_name"] for r in state.get("tool_results", [])}  # Set comprehension: collect executed tool names as a set for dedup checks
    task = state.get("task") or {}  # Task field (question text and task type live here)
    task_text = str(task.get("question") or "")  # Take the raw user task text as the basis for task-aware matching
    task_type = str(task.get("task_type") or classify_task(task_text))  # Task type: the state value wins; classify from text when missing
    # ── Level 1: task-relevance routing ──
    candidates = []  # Collect (relevance score, tool name, spec) triples
    for s in REGISTRY.specs():  # Iterate all tool specs in the registry
        if s.name in executed:  # Already-executed tools are not chosen again
            continue
        ok, reason = (s.applicable(state) if s.applicable else (True, ""))  # Data-condition check (Data-aware)
        score = _relevance(s, task_text)  # Task-relevance score (keyword hit count)
        if ok and score > 0:  # Enter the candidates only if data conditions hold and the tool is task-relevant
            candidates.append((score, s.name, s, reason))  # Record the candidate triple
    if candidates:  # Task-relevant applicable tools exist: take the top one by descending score
        candidates.sort(key=lambda t: -t[0])  # Sort by descending score (ties stay in registration order, stable)
        score, name, spec, why = candidates[0]  # Take the highest-scoring candidate
        _, reason = (spec.applicable(state) if spec.applicable else (True, ""))  # Re-fetch the applicability explanation
        return ReasoningDecision(goal=f"{name}（任务相关，得分 {score}）", next_action=name,  # Return the decision: the goal notes task relevance and score
                                 reason=f"任务路由：关键词命中且数据条件满足（{reason or '适用'}）")  # Write the task-routing reason and close the return
    # ── Level 2: fixed priority chains by task type (main capabilities first, then the common tail) ──
    common_tail = ["analyze_geological_text",  # Text semantic enrichment (applicable for text modality; the main path of text-only runs)
                   "table_column_stats",  # Targeted column retrieval (2026-10 benchmark diagnosis: the only read channel for non-element column values such as microthermometry/isotopes)
                   "element_inventory",  # Data inventory (fallback)
                   "llm_genesis_synthesis"]  # Narrative-synthesis finisher (applicable when the LLM is online and evidence exists)
    priority = {  # Task type → tool priority chains (capability heads differ, tail shared) (dict spans multiple lines)
        "prospectivity": ["geochemical_prospectivity_mapping", *common_tail],  # Targeting chain: mapping first
        "genesis": ["calculate_nb_ta_ratio", "calculate_zr_hf_ratio",  # Mechanism chain head: differentiation indices (spans lines)
                    "calculate_eu_anomaly", "calculate_la_yb_n", "tas_classification", *common_tail],  # Mechanism chain continued: REE indices + TAS
        "comprehensive": ["geochemical_prospectivity_mapping",  # Mixed chain: spatial first, then mechanism (spans lines)
                          "calculate_nb_ta_ratio", "calculate_zr_hf_ratio",  # Mixed chain continued: differentiation indices
                          "calculate_eu_anomaly", "calculate_la_yb_n", *common_tail],  # Mixed chain continued: REE indices + tail
    }.get(task_type, [*["geochemical_prospectivity_mapping"],  # Unknown types fall back to the mixed chain (spans lines)
                      *common_tail])  # Fallback chain: mapping + tail
    for name in priority:  # Examine candidate tools one by one in priority order
        spec = REGISTRY.get(name)  # Fetch the tool's spec object from the registry
        ok, reason = (spec.applicable(state) if spec.applicable else (True, ""))  # Ternary: run and unpack (ok, reason) when an applicability check exists, otherwise default to applicable
        if ok and name not in executed:  # Choose it when data conditions hold and the tool is unexecuted
            return ReasoningDecision(goal=f"执行 {name}", next_action=name,  # Return the decision: the goal is to execute the tool, the action is the tool name (parentheses span lines)
                                     reason=f"规则回退：{task_type} 链内数据条件满足且未执行（{reason or '适用'}）")  # Write the fallback reason (naming the task-type chain) and close the return
    return ReasoningDecision(goal="完成取证", next_action="stop", done=True,  # With no executable tool left, return a finishing decision: action stop, done set true (parentheses span lines)
                             reason="规则回退：无更多适用的未执行工具")  # Write the finishing reason and close the return


def plan(state: dict[str, Any], backend: Any = None) -> ReasoningDecision:  # ReAct Reason step entry: backend is an optional LLM backend
    """ReAct Reason step entry: LLM decision → parse failure/unavailable → rule-based fallback."""  # Function docstring: LLM decision first; rule-based fallback on parse failure or unavailability
    task_text = str((state.get("task") or {}).get("question") or "")  # Raw task text: feeds task-aware matching for menu sorting and the rule-based fallback
    task = state.get("task") or {}  # Task field (task type lives here)
    task_type = str(task.get("task_type") or classify_task(task_text))  # Task type: the state value wins; classify from text when missing
    tool_menu = _tool_menu(state, task_text)  # Sort by task relevance and cap the count, so 100+ tools do not blow up the prompt
    obs_evidence = [e for e in state.get("evidence", []) if e.get("observation")]  # Evidence with observation content (tool outputs are appended at the ledger tail)
    recent_evidence = obs_evidence[-10:]  # Last 10 entries (workbench layer 2): the planner must see its own recent tool outputs --
    # (2026-10 benchmark diagnosis: the old evidence[:8] was always the opening data profiles; from round 2 on
    #   the model was fully blind to its own tool results, so group_by hammered the wrong table repeatedly
    #   and 5 rounds were burned away)
    memory_lines = [str(x) for x in (state.get("memory_cards") or []) if str(x).strip()]  # Long-task memory lines (recalled and injected by the runner; empty when disabled by default)
    memory_sec = ("\n## 已有结论记忆（你自己在早先任务中的判断，可能需要复核——可引用，也可依据本题数据推翻）\n"  # Memory section title: makes explicit that cards may be cited or overturned
                  + "\n".join(memory_lines)) if memory_lines else ""  # Build the section only when memory exists (workbench layer 3; the tail window cannot reach the ledger head, so a dedicated standing section is used)
    prompt = (  # Start assembling the planner prompt (parentheses span lines)
        "你是矿床预测与分析智能体的规划器（能力：矿床成因分析）。"  # Prompt first segment: role and capability statement (implicit string concatenation)
        f"当前任务类型「{task_type}」，根据状态选择下一个动作。\n"  # Prompt continued: state the task type and the decision instruction
        f"## 用户任务（节选）\n{task_text[:500]}\n"  # Prompt inserts a task-text excerpt: the decision-maker was previously blind to task requirements (2026-10 benchmark diagnosis)
        f"## 当前状态\n{_state_summary(state)}\n"  # Prompt inserts the "current state" section: embeds the state summary and the full-table index
        f"## 最近证据（含你的工具产出）\n" +  # Prompt appends the "recent evidence" heading (tail window, see above) and prepares to join the evidence lines
        "\n".join(f"- {str(e['observation'])[:2000]}" for e in recent_evidence)  # Take the last 10 evidence entries with observation content, each truncated to 2000 chars
        # (2026-10-19 paper-H smoke diagnosis: 600 chars covered only group 1 of table_column_stats
        #  multi-group statistics; the decision-maker never saw inter-group numbers from group 2 on,
        #  so inter-group comparison tasks were planned blind -- aligned with the tool-side group budget)
        + memory_sec  # Append the "prior conclusions memory" section (long-task memory mode; empty string by default)
        + "\n\n## 可用工具\n" + tool_menu +  # Prompt appends the "available tools" heading and the tool menu
        "\n\n所选工具描述中声明了调用参数时，按任务要求把参数填进 parameters"  # Parameter guidance: the template previously anchored an empty dict, and the model never exercised its selection right (2026-10 benchmark diagnosis)
        "（如 fusion_method/interpolation），无参数可传才留空 {}。\n"  # Parameter guidance continued: state the default semantics
        "严格输出 JSON：\n"  # Prompt appends the output constraint: require strict JSON output
        '{"goal":"...","missing_information":["..."],"next_action":"工具名或stop",'  # Prompt gives the JSON template first half: goal/missing_information/next_action fields
        '"parameters":{"如fusion_method":"weights_of_evidence"},"done":false,"reason":"一句话理由"}')  # JSON template second half: parameters gets an example value to avoid empty-param anchoring; close the parentheses to finish the prompt
    if backend is not None:  # Attempt a model decision only when an LLM backend is provided
        try:  # The try wraps the call and parsing; exceptions fall through to the rule-based fallback
            resp = backend.chat([{"role": "user", "content": prompt}],  # Call backend chat: the user message is the planning prompt (parentheses span lines)
                                system_prompt="你是严格的 JSON 决策器，只输出一个 JSON 对象。",  # The system message constrains the model to output exactly one JSON object
                                temperature=0.1, max_tokens=512,  # Low temperature 0.1, 512 tokens: stable output and quota savings
                                response_format={"type": "json_object"})  # Request server-side JSON mode and close the call parentheses
            m = re.search(r"\{[\s\S]*\}", resp.content or "")  # Regex-greedily extract the first {...} JSON fragment from the reply (search an empty string when content is None)
            data = json.loads(m.group(0)) if m else {}  # Parse the extracted JSON; fall back to an empty dict when nothing matched
            action = str(data.get("next_action", "")).strip()  # Take the next_action field and strip surrounding whitespace
            if action != "stop" and action not in REGISTRY.names():  # Hallucination interception: when the action is not stop and is not in the registered tool names
                action, data["done"] = "stop", True  # Fabricated tool names are intercepted here (only registered tools may be chosen)
            decision = ReasoningDecision(  # Build the decision object from the parsed fields (parentheses span lines)
                goal=str(data.get("goal", "")),  # goal field: take the JSON goal as a string
                missing_information=[str(x) for x in (data.get("missing_information") or [])],  # missing_information: stringify each item (None falls back to an empty list)
                next_action=action, parameters=dict(data.get("parameters") or {}),  # next_action uses the intercepted action; parameters is dict-ized (None falls back to an empty dict)
                done=bool(data.get("done")) or action == "stop",  # done takes the JSON boolean, or is forced true when the action equals stop
                reason=str(data.get("reason", "")))  # reason field as a string; close the construction parentheses
            decision.reason = (decision.reason + "（LLM 决策）").strip("（ ")  # Append the "（LLM 决策）" marker to the reason, then strip the leading "（" and spaces
            return decision  # Return the LLM-produced decision
        except Exception as e:  # LLM failure does not abort reasoning: fall through to the rule-based fallback with the reason
            decision = _rule_decision(state)  # Exception branch: generate the fallback decision with deterministic rules instead
            decision.reason += f"（LLM 不可用回退: {type(e).__name__}）"  # Append the exception type name to the reason, leaving a fallback trace
            return decision  # Return the fallback decision
    decision = _rule_decision(state)  # No backend provided: generate the decision directly via the rule-based fallback
    decision.reason += "（无模型规则回退）"  # Append the "（无模型规则回退）" marker to the reason
    return decision  # Return the final decision
