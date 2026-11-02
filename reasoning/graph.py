"""LangGraph Reasoning & Control Layer: node definitions + ReAct main loop + routing.

Control flow:
    init → parse → enrich → seed → plan ─┬─(act → plan)… ReAct loop
                                         └─(done/limit exceeded)→ synthesize
        → review → audit → END

Golden rule of responsibility: the LLM only plans and explains (ReasoningDecision),
tools do the computation (ToolResult), and state is merged by LangGraph according
to field annotations (accumulating fields are defined in reasoning/state.py); tool
failures flow back as Observations for re-planning instead of aborting the run.
"""
from __future__ import annotations  # Enable deferred annotation evaluation: type annotations have no runtime effect
import json  # Import json: serializes call signatures for the duplicate-call gate
from typing import Any, Callable, Literal  # Import typing utilities: Any for arbitrary types, Callable for node functions, Literal for routing return literals
from langgraph.graph import END, StateGraph  # Import LangGraph's state graph builder and the END marker
from oregenesis.evidence import EvidenceLedger, EvidenceUnit, evidence_from_data_units  # Import evidence layer: evidence ledger, evidence unit type, and the data-profile evidence generator
from oregenesis.io import make_asset  # Import IO layer: register raw assets (parsing is done per asset by the parse node)
from oregenesis.reasoning.planner import ReasoningDecision, plan  # Import the planner: decision type and the Reason entry function
from oregenesis.reasoning.state import AgentState  # Import the global state schema
from oregenesis.tools import REGISTRY, ToolCall  # Import tool layer: the registry and the tool-call request type


def _stage(status: dict[str, str], name: str, value: str) -> dict[str, str]:  # Private helper: produce an updated node-status table
    """Node-status bookkeeping helper: return an updated copy of node_status (never mutate State in place)."""  # Function docstring: return a copy, never mutate State in place
    out = dict(status)  # Shallow-copy the status table to avoid in-place modification
    out[name] = value  # Write the node's latest status value
    return out  # Return the new status table


def make_nodes(ledger: EvidenceLedger, backend: Any = None) -> dict[str, Callable]:  # Node factory: closures share the evidence ledger and the LLM backend
    """Build all node functions (closures share the evidence ledger and LLM backend; no global mutable state)."""  # Function docstring: build all nodes with no global mutable state

    def node_init(state: dict) -> dict:  # Define the init node: task initialization
        """Task initialization: register task.files as InputAssets (identity cards of raw inputs)."""  # Node docstring: register task.files as InputAssets
        files = state["task"].get("files", [])  # Take the input file list from the task field (default to an empty list)
        return {"input_assets": [make_asset(f) for f in files],  # Register each file as a raw asset with make_asset (dict spans lines)
                "current_stage": "initialized",  # Mark the current stage as initialized
                "node_status": _stage(state.get("node_status", {}), "init", "completed")}  # Record init node completion and close the returned dict

    def node_parse(state: dict) -> dict:  # Define the parse node: data parsing
        """Data Parsing: InputAsset → DataUnit (read/structure/provenance only, no interpretation).
        Parse asset by asset with error isolation: a single file's parse failure is only
        logged into errors and does not sink the whole batch (previously one try wrapped
        the entire batch; one badly encoded file once left all 27 files parsed as empty).
        """  # Node docstring: documents the per-asset error-isolation robustness semantics
        units: list = []  # Successfully parsed data units
        errors: list[str] = []  # Error records for failed files
        from oregenesis.io.parser import parse_asset  # Local import of the single-asset parser
        from oregenesis.io.data_models import IdAllocator  # ID allocator needed by single-asset parsing
        alloc = IdAllocator()  # One allocator shared across the batch keeps IDs unique
        for a in state.get("input_assets", []):  # Parse asset by asset (error isolation)
            try:  # A failed file is only logged; it does not interrupt the other files
                units.extend(parse_asset(a, alloc))  # Parse and accumulate data units
            except Exception as e:  # noqa: BLE001 -- leave a trace for bad files (with filename and reason)
                errors.append(f"文件解析失败 {a.file_path}: {type(e).__name__}: {e}")  # Leave an error trace
        if not units and errors:  # Only treat parsing as wholly failed when everything failed (short circuit)
            return {"data_units": [], "current_stage": "parse_failed",  # With no data at all there is nothing to reason over
                    "errors": errors,  # Keep all error traces
                    "node_status": _stage(state.get("node_status", {}), "parse", "failed")}  # Mark as failed
        return {"data_units": units, "errors": errors, "current_stage": "data_parsed",  # Return successful units along with error traces
                "node_status": _stage(state.get("node_status", {}), "parse",  # Record the parse node status
                                      "completed" if not errors else "completed_with_errors")}  # Flag as completed-with-errors when bad files exist

    def node_enrich(state: dict) -> dict:  # Define the enrich node: enrichment and multi-source alignment
        """Entity + Semantic Enrichment + multi-source alignment (Data Understanding layer)."""  # Node docstring: entity + semantic enrichment + multi-source alignment
        from oregenesis.understanding import align_by_entity, enrich_all  # Local import of the understanding layer: entity alignment and batch enrichment functions

        units = enrich_all(state.get("data_units", []))  # Enrich all data units to get the new unit list
        bundles = [{**b.__dict__, "variables": b.variables(),  # After alignment, expand each bundle's attribute dict and append variable names (comprehension spans lines)
                    "unit_ids": [u.data_unit_id for u in b.units]}  # Append the bundle's member data-unit ID list and close the inner dict
                   for b in align_by_entity(units)]  # SampleBundle → serializable dict
        return {"data_units": units, "bundles": bundles, "current_stage": "understood",  # Return enriched units and aligned bundles; mark the stage as understood
                "node_status": _stage(state.get("node_status", {}), "enrich", "completed")}  # Record enrich node completion and close the dict

    def node_seed(state: dict) -> dict:  # Define the seed node: seed evidence generation
        """Seed evidence: data profiles (the data-before-reasoning starting point; later evidence comes from tool results)."""  # Node docstring: data profiles are the data-before-reasoning starting point
        seeds = [ledger.add(u) for u in evidence_from_data_units(state.get("data_units", []))]  # Add each data-profile evidence unit to the ledger and collect the return values
        return {"evidence": [u.to_dict() for u in seeds], "current_stage": "evidence_seeded",  # Serialize each seed evidence unit into the partial update (dict spans lines)
                "node_status": _stage(state.get("node_status", {}), "seed", "completed")}  # Record seed node completion and close the dict

    def node_plan(state: dict) -> dict:  # Define the plan node: the Reason step
        """Reason: produce a structured ReasoningDecision (LLM first, rule-based fallback)."""  # Node docstring: produce a structured decision; LLM first with rule-based fallback
        decision: ReasoningDecision = plan(state, backend=backend)  # Call the planner to get this round's structured decision (typed)
        return {"decisions": [decision.to_dict()], "rounds": state.get("rounds", 0) + 1,  # Serialize the decision for the record; increment the executed round count (dict spans lines)
                "current_stage": f"round_{state.get('rounds', 0) + 1}",  # Mark the current stage as round_N (N is the new round count)
                "node_status": _stage(state.get("node_status", {}), "plan", "completed")}  # Record plan node completion and close the dict

    def node_act(state: dict) -> dict:  # Define the act node: the Act + Observe step
        """Act + Observe: execute the registered tool chosen by the planner; the ToolResult is uniformly converted into evidence that flows back."""  # Node docstring: execute the registered tool and flow the result back as evidence
        decision = ReasoningDecision(**state["decisions"][-1])  # The most recent planning round
        # Duplicate-call gate (2026-10 benchmark diagnosis): reject a call whose tool +
        # parameters were already executed -- the LLM decision path originally had no
        # dedup (the rule-based fallback path had the executed set), so the same query
        # was fired verbatim 5 times, burning the scouting budget. The interception
        # record flows back as an Observation to force the planner to switch targets.
        _sig = json.dumps({"a": decision.next_action, "p": decision.parameters or {}},  # Signature of this call: action + parameters
                          sort_keys=True, ensure_ascii=False, default=str)  # Parameter structure is uncontrolled; default=str as a safety net
        _prev = {json.dumps({"a": str(d.get("next_action") or ""), "p": d.get("parameters") or {}},  # Set of already-executed signatures
                            sort_keys=True, ensure_ascii=False, default=str)  # Same fallback serialization as above
                 for d in state.get("decisions", [])[:-1]}  # The current decision is the last element of decisions; not counted as history
        if _sig in _prev:  # Duplicate hit: do not execute; return the interception observation
            dup = EvidenceUnit(  # Build the interception record into the evidence ledger (visible to the next planning round's tail window)
                observation=f"防重闸：{decision.next_action} 同参数已执行过，本次拒绝执行。"
                            "请更换工具、更换参数（如换列名/分组列/换表），或信息已足时推进综合。",
                source_tool_calls=[], reliability=0.3,
                limitations=["重复调用被防重闸拦截"], kind="tool_result")  # Close the construction
            entry = dup.to_dict() | {"status": "error", "tool_name": decision.next_action}  # Isomorphic to a normal tool observation
            return {"tool_results": [], "evidence": [entry],  # Produce no tool result; return only the interception observation (dict spans lines)
                    "current_stage": f"duplicate_blocked_round_{state.get('rounds', 1)}",  # Mark the stage as intercepted
                    "node_status": _stage(state.get("node_status", {}), "act", "duplicate_blocked")}  # Leave a trace
        call = ToolCall(  # Build the tool-call request (parentheses span lines)
            tool_call_id=f"toolcall_{state.get('rounds', 1):03d}",  # Call ID zero-padded to 3 digits by round, e.g. toolcall_003
            tool_name=decision.next_action,  # Tool name taken from the decision's next action
            input_data_unit_ids=[u.data_unit_id for u in state.get("data_units", [])],  # Input is the ID list of all current data units
            parameters=decision.parameters, purpose=decision.goal or decision.reason)  # Pass tool parameters and purpose (goal first, reason as fallback) and close the parentheses
        if (call.tool_name == "geochemical_prospectivity_mapping"  # Benchmark/application adaptation: the prospectivity mapping tool
                and "deposits" not in (call.parameters or {})  # and the LLM did not explicitly provide the deposits parameter
                and (state.get("task") or {}).get("deposits")):  # and the task layer carries a known-deposits table (injected by the runner via extra)
            call.parameters = {**(call.parameters or {}),  # Merge-in injection (does not override the LLM's explicit parameters)
                               "deposits": state["task"]["deposits"]}  # Known deposits enter fusion and evaluation as supervision data
        if (call.tool_name == "geochemical_prospectivity_mapping"  # Benchmark controlled experiment: the task layer may pin the fusion method
                and (state.get("task") or {}).get("fusion_method")  # and the task layer carries fusion_method (auto/random arms injected via extra)
                and "fusion_method" not in (call.parameters or {})):  # and the LLM did not explicitly choose a method (the LLM-free-choice arm is not overridden)
            call.parameters = {**(call.parameters or {}),  # Merge-in injection (same semantics as the deposits injection: fill only when absent)
                               "fusion_method": state["task"]["fusion_method"]}  # Force the auto hard rule or the randomly drawn method
        if (call.tool_name == "geochemical_prospectivity_mapping"  # Baseline-arm forcing: the experiment design requires ignoring model choice
                and (state.get("task") or {}).get("fusion_method_forced")):  # and the task layer carries a forced method (force arm injected via extra)
            call.parameters = {**(call.parameters or {}),  # Overriding injection: the baseline arm's method choice must be controlled by the experiment,
                               "fusion_method": state["task"]["fusion_method_forced"]}  # unaffected by whether the LLM happens to pass it (distinct from fusion_method's fill-only-when-absent)
        result = REGISTRY.execute(call, state)  # Execute the tool through the registry to get a uniform ToolResult
        ok = result.status == "success"  # Boolean flag: whether this tool execution succeeded
        unit = EvidenceUnit(  # Build an evidence unit wrapping this observation (parentheses span lines)
            observation=(str(result.result.get("observation", "")) if ok  # On success the observation comes from the tool result's observation field (ternary spans lines)
                         else f"工具 {result.tool_name} 执行失败: {result.error}"),  # On failure the observation records the failure instead
            source_data_units=result.source_data_unit_ids,  # Evidence provenance: record the source data-unit IDs
            source_tool_calls=[call.tool_call_id],  # Evidence provenance: record the source tool-call IDs
            supports=list(result.result.get("supports", [])) if ok else [],  # On success copy the tool-declared support tags; empty on failure
            contradicts=list(result.result.get("contradicts", [])) if ok else [],  # On success copy the contradict tags; empty on failure
            reliability=0.85 if ok else 0.3,  # Reliability assignment: 0.85 on success, 0.3 on failure
            limitations=(list(result.result.get("limitations", [])) if ok  # On success take the tool-declared limitations list (ternary spans lines)
                         else [result.error or "工具失败"]),  # On failure record the error message as the limitation (fallback wording) and close the parentheses
            missing_tests=list(result.result.get("missing_tests", [])) if ok else [],  # On success copy the missing-test suggestions; empty on failure
            kind="tool_result")  # Mark the evidence kind as tool_result and close the construction parentheses
        ledger.add(unit)  # Write the evidence unit into the evidence ledger
        entry = unit.to_dict() | {"status": result.status, "tool_name": result.tool_name,  # Merge execution status and tool name into the evidence dict (parentheses span lines)
                                  "artifacts": result.result.get("artifacts", {}),  # Attach the tool's produced artifacts (figures, etc.)
                                  "missing_tests": list(result.result.get("missing_tests", [])) if ok else []}  # Attach missing tests (empty on failure) and close the merged dict
        return {"tool_results": [result.to_dict()], "evidence": [entry],  # Return the tool result record and the evidence entry (dict spans lines)
                "current_stage": f"observed_round_{state.get('rounds', 1)}",  # Mark the current stage as observed_round_N
                "node_status": _stage(state.get("node_status", {}), "act", "completed")}  # Record act node completion and close the dict

    def node_synthesize(state: dict) -> dict:  # Define the synthesize node: result synthesis
        """Result Synthesis: assemble the scientific output + deterministically generate the Chinese narrative and conclusion.
        The narrative/conclusion no longer depends on the planner happening to pick the
        llm_genesis_synthesis tool -- when the model is online this node calls it directly
        (a deterministic in-graph step); when offline it falls back to the tool-generated
        narrative section (if one was produced during the ReAct loop), and failing that
        leaves it empty and states so in the report.
        """
        # Node docstring: explains the design motivation of the deterministic narrative channel and its fallback chain
        from oregenesis.report import _narrative, build_final_result  # Local import of the report layer: result assembly and narrative extraction
        # ── Auto-plot: for genesis/comprehensive tasks whose tables contain REE columns, deterministically add a normalized REE pattern plot ──
        # (No planning round consumed, no LLM involved -- guarantees the analysis report has a figure by default; failure only warns and does not block)
        extra_results: list = []  # Tool results produced by auto-plotting (appended back to state)
        extra_evidence: list = []  # Evidence produced by auto-plotting (appended back to state)
        auto_warn: list = []  # Auto-plot warnings (failure traces)
        _tt = str((state.get("task") or {}).get("task_type") or "")  # Task type
        _already = any(r.get("tool_name") == "plot_ree_spider"  # The ReAct loop already produced a pattern plot (comprehension spans lines)
                       for r in state.get("tool_results", []))  # Iterate over tool results
        _has_num = any(u.data_modality == "numeric" and u.data_status != "error"  # A usable numeric table exists (comprehension spans lines)
                       for u in state.get("data_units", []))  # Iterate over data units
        if _tt in ("genesis", "comprehensive") and _has_num and not _already:  # For genesis/comprehensive tasks without a plot yet
            try:  # Auto-plot failure only warns; it does not block
                from oregenesis.understanding import _norm_element  # Element-name recognition conventions
                from oregenesis.tools.ree import _CHONDRITE  # REE reference values table (used to count REE columns)
                _n_ree = sum(1 for u in state.get("data_units", [])  # Count REE columns (comprehension spans lines)
                             if u.data_modality == "numeric" and u.data_status != "error"  # Usable numeric tables
                             for h in u.raw_data.get("header", [])  # Iterate over the header
                             if _norm_element(str(h)) in _CHONDRITE)  # Count a header as REE when its normalized name hits the reference table
                if _n_ree >= 6:  # REE columns sufficient (a pattern needs at least 5 points to form a line)
                    acall = ToolCall(tool_call_id="toolcall_auto_ree",  # The auto-plot call (fixed ID)
                                     tool_name="plot_ree_spider", parameters={},  # No parameters (the tool auto-detects REE columns)
                                     purpose="自动配图：REE 球粒陨石标准化配分模式")  # The scientific purpose goes into the audit
                    aresult = REGISTRY.execute(acall, state)  # Execute through the registry (uniform contract)
                    aok = aresult.status == "success"  # Whether execution succeeded
                    aentry = EvidenceUnit(  # Build an evidence unit (same contract as node_act) (parentheses span lines)
                        observation=(str(aresult.result.get("observation", "")) if aok  # On success take the observation (ternary spans lines)
                                     else f"工具 plot_ree_spider 执行失败: {aresult.error}"),  # On failure record the reason
                        source_data_units=aresult.source_data_unit_ids,  # Lineage: source data units
                        source_tool_calls=[acall.tool_call_id],  # Lineage: tool-call ID
                        supports=list(aresult.result.get("supports", [])) if aok else [],  # Support tags
                        reliability=0.85 if aok else 0.3,  # Reliability tier
                        limitations=(list(aresult.result.get("limitations", [])) if aok  # Limitations (ternary spans lines)
                                     else [aresult.error or "自动配图失败"]),  # On failure record the reason as a limitation
                        kind="tool_result")  # Kind: tool result
                    ledger.add(aentry)  # Register into the evidence ledger
                    extra_results = [aresult.to_dict()]  # Append the tool result
                    extra_evidence = [aentry.to_dict() | {"status": aresult.status,  # Append the evidence entry (spans lines)
                                                          "tool_name": aresult.tool_name,  # Merge in status and tool name
                                                          "artifacts": aresult.result.get("artifacts", {}),  # Artifacts
                                                          "missing_tests": []}]  # No missing tests
            except Exception as exc:  # noqa: BLE001 -- the exception is only recorded
                auto_warn = [f"自动配图失败: {exc}"]  # Leave a warning trace
        _state_extra = {**state, "tool_results": list(state.get("tool_results", [])) + extra_results,  # State copy including auto-plot outputs (spans lines)
                        "evidence": list(state.get("evidence", [])) + extra_evidence}  # State copy continued: evidence

        final = build_final_result(_state_extra)  # Assemble the scientific output (including auto-plot evidence)
        final["narrative"] = ""  # Blank the narrative field first (filled by the deterministic step below when online)
        final["conclusion"] = ""  # Blank the conclusion field first (same as above)
        final["warnings"] = [] + auto_warn  # Report-level warnings (auto-plot warnings merged in)
        bk = state.get("backend") or backend  # LLM backend: prefer the one in State (injected by the runner); fall back to the graph-construction closure
        if bk is not None:  # The Chinese narrative and conclusion can only be generated when the model is online
            try:  # A single point of failure must not block the report (degrade to empty and leave a trace in warnings)
                from oregenesis.tools.vlm import synthesize_analysis  # Local import of the deterministic generator
                out = synthesize_analysis(_state_extra, bk)  # Generate the conclusion and analysis narrative by task type (the narrative may cite the pattern plot)
                final["narrative"] = out.get("narrative", "")  # Write the full analysis narrative
                final["conclusion"] = out.get("conclusion", "")  # Write the conclusion summary
            except Exception as exc:  # noqa: BLE001 -- record synthesis failure without raising
                final["warnings"].append(f"叙述/结论综合失败: {exc}")  # Record the failure reason
        else:  # Offline fallback: state honestly why there is no narrative
            final["warnings"].append("无在线模型：结论与分析过程叙述不可用（规则回退模式）")  # Record the reason
        if not final["narrative"]:  # When the deterministic narrative is empty (offline or failed)
            final["narrative"] = _narrative(state.get("evidence", []))  # Fall back to the tool-generated narrative section from the ReAct loop (if any)
        return {"final_result": final,  # Return the final result (including auto-plot evidence and warnings)
                "tool_results": extra_results, "evidence": extra_evidence,  # Auto-plot tool results and evidence (accumulating semantics)
                "current_stage": "synthesized",  # Return the stage marker
                "node_status": _stage(state.get("node_status", {}), "synthesize", "completed")}  # Record node completion and close the dict

    def node_review(state: dict) -> dict:  # Define the review node: scientific review
        """Scientific Review: check conclusion provenance / conflict retention / over-interpretation (problems are logged into errors)."""  # Node docstring: check provenance / conflict retention / over-interpretation
        from oregenesis.reasoning.review import scientific_review  # Local import of the review function

        return scientific_review(state)  # Forward the review node's partial update directly

    def node_audit(state: dict) -> dict:  # Define the audit node: audit trail
        """Audit: generate a back-traceable audit trail (conclusion → evidence → tool → data → raw file)."""  # Node docstring: build the back-tracing chain from conclusion to raw file
        from oregenesis.reasoning.review import audit_trail  # Local import of the audit function

        out = audit_trail(state)  # Produce the audit partial update
        return {**out, "current_stage": "audited",  # Merge the audit result and append the audited stage marker (dict spans lines)
                "node_status": _stage(state.get("node_status", {}), "audit", "completed")}  # Record audit node completion and close the dict

    return {"init": node_init, "parse": node_parse, "enrich": node_enrich,  # Return nodes as a name-to-function dict (init/parse/enrich, spans lines)
            "seed": node_seed, "plan": node_plan, "act": node_act,  # Continue registering the seed/plan/act nodes
            "synthesize": node_synthesize,  # Register the synthesize node
            "review": node_review, "audit": node_audit}  # Register the review and audit nodes and close the dict


def _route_after_plan(state: dict) -> Literal["act", "synthesize"]:  # Conditional routing function after plan: returns act or synthesize
    """Conditional routing after plan: decision stop/done or round limit exceeded → result synthesis; otherwise execute a tool."""  # Function docstring: finish/limit goes to synthesis, otherwise execute a tool
    last = (state.get("decisions") or [{}])[-1]  # Take the most recent decision (fall back to an empty dict when absent)
    if last.get("done") or last.get("next_action") in ("", "stop"):  # Enter the synthesis branch when the decision is done or the action is empty/stop
        return "synthesize"  # Route to the result-synthesis node
    if state.get("rounds", 0) >= state.get("max_rounds", 6):  # When the executed rounds reach the limit
        return "synthesize"  # Also route to synthesis to end the loop
    return "act"  # Otherwise route to the tool-execution node


def build_graph(backend: Any = None):  # Main graph construction entry: backend is an optional LLM backend
    """Build and compile the OreGenesis-Agent main graph (fresh per run; states never cross-contaminate)."""  # Function docstring: build a fresh graph per run so states never cross-contaminate
    nodes = make_nodes(EvidenceLedger(), backend=backend)  # Create a fresh evidence ledger and build all node closures
    g = StateGraph(AgentState)  # Schema is a TypedDict: Annotated fields merge by their annotations (accumulate)
    for name, fn in nodes.items():  # Iterate the node dict and register each into the graph
        g.add_node(name, fn)  # Register the function as the graph node of that name
    g.set_entry_point("init")  # Set the entry node to init
    g.add_edge("init", "parse")  # Fixed edge: init → parse
    g.add_edge("parse", "enrich")  # Fixed edge: parse → enrich
    g.add_edge("enrich", "seed")  # Fixed edge: enrich → seed
    g.add_edge("seed", "plan")  # Fixed edge: seed → plan, entering the ReAct loop
    g.add_conditional_edges("plan", _route_after_plan, {"act": "act", "synthesize": "synthesize"})  # Conditional edge: after plan the routing function jumps to act or synthesize
    g.add_edge("act", "plan")  # Observe → Reason Again (closing the ReAct loop)
    g.add_edge("synthesize", "review")  # Fixed edge: after synthesis enter scientific review
    g.add_edge("review", "audit")  # Fixed edge: after review enter audit
    g.add_edge("audit", END)  # After audit reach END and finish the run
    return g.compile()  # Compile the graph and return the executable
