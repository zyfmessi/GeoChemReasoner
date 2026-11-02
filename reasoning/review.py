"""Scientific Review + Audit: conclusion review and a back-traceable audit trail.

Review checks "whether conclusions have sources, whether counter-evidence is
retained, whether required sections are present"; Audit builds the complete
lineage chain (raw file → InputAsset → DataUnit → ToolCall → ToolResult →
EvidenceUnit → conclusion), so any conclusion can be traced back to where it
came from.
"""

from __future__ import annotations  # Enable deferred annotation evaluation: type annotations have no runtime effect

from typing import Any  # Import Any for type annotations

# Required sections of the scientific output (key presence suffices: empty targeting/mechanism capability blocks are legal states)
_REQUIRED_SECTIONS = ("task_understanding", "conclusion", "narrative",  # Required sections: task understanding / conclusion / narrative (tuple spans lines)
                      "key_observations", "supported_processes",  # Tuple continued: key observations and supported processes
                      #"target_zones", "mechanism_analysis",  # Tuple continued: target-zone and mechanism-finding blocks (empty values are legal)
                      "supporting_evidence", "conflicting_evidence",  # Tuple continued: supporting and conflicting evidence
                      "remaining_uncertainties", "recommended_tests",  # Tuple continued: uncertainties and recommended tests
                      "provenance", "data_inventory", "figures")  # Tuple continued: provenance / data inventory / figure outputs; close the tuple


def _stage(status: dict[str, str], name: str, value: str) -> dict[str, str]:  # Private helper: produce an updated node-status table
    """Node-status bookkeeping helper: return an updated copy of node_status (same semantics as graph.py)."""  # Function docstring: return a copy, never mutate State in place; same semantics as graph.py
    out = dict(status)  # Shallow-copy the status table to avoid in-place modification
    out[name] = value  # Write the node's latest status value
    return out  # Return the new status table


def scientific_review(state: dict) -> dict[str, Any]:  # Scientific-review node function: deterministic checks over the final result
    """Scientific Review Node: run one deterministic review over the synthesized result.

    Checks: (1) all required sections present (key presence suffices; empty capability
    blocks are legal); (2) every piece of evidence has a source (DataUnit or ToolCall);
    (3) when counter-evidence/limitation fields are non-empty, the conclusion must be
    flagged as conditional. Problems go into errors (never silently passed); the
    review verdict is written into final_result.review.
    """
    result = dict(state.get("final_result") or {})  # Copy the final-result dict (None falls back to an empty dict)
    problems: list[str] = []  # Initialize the problem list
    missing = [s for s in _REQUIRED_SECTIONS if s not in result]  # Key presence suffices (empty lists are legal:
    if missing:                                                   #  no counter-evidence / no alternatives is a normal state)
        problems.append(f"科学输出缺少章节: {', '.join(missing)}")  # When sections are missing, join the list into the problems
    for e in state.get("evidence", []):  # Evidence citation integrity: evidence without a source cannot be trusted
        if (not e.get("source_data_units") and not e.get("source_tool_calls")  # When there is neither a data-unit source nor a tool-call source
                and e.get("kind") != "memory"):  # Memory entries are exempt: their source is a prior task (cards carry their own provenance references and seen_dirs permission records)
            problems.append(f"证据 {e.get('evidence_id')} 缺少来源（既无 DataUnit 也无 ToolCall）")  # Record the problem that this evidence ID lacks a source
    if result.get("conflicting_evidence"):  # With counter-evidence present, the wording must be conditional
        result["conditional_note"] = "存在未消解反证，结论为条件性判断"  # Write the conditionality note field: unresolved counter-evidence exists
    review = {"passed": not problems, "problems": problems,  # Assemble the review verdict dict: passed when no problems (dict spans lines)
              "note": ("审查通过" if not problems else "审查发现问题，见 problems")}  # Attach the review note text and close the dict
    result["review"] = review  # Attach the review verdict to the final result's review field
    return {"final_result": result, "errors": problems,  # Return the updated result and problem list (errors leave a trace), dict spans lines
            "node_status": _stage(state.get("node_status", {}), "review",  # node_status continued: record the review node status
                                  "completed" if not problems else "completed_with_issues")}  # With problems the status is completed_with_issues; close the returned dict


def audit_trail(state: dict) -> dict[str, Any]:  # Audit node function: build the lineage chain from conclusions to raw files
    """Audit Node: full lineage back-traced from conclusions to raw files (blueprint §36, §44)."""  # Function docstring: satisfies the blueprint §36, §44 back-traceability requirement
    chain: list[dict[str, Any]] = []  # Initialize the evidence-chain list
    for e in state.get("evidence", []):  # Build one audit record per piece of evidence
        chain.append({  # Append one audit dict to the chain (spans lines)
            "evidence_id": e.get("evidence_id"),  # Record the evidence ID
            "observation": (e.get("observation") or "")[:80],  # Truncate the observation text to the first 80 chars (None falls back to an empty string)
            "data_units": e.get("source_data_units", []),  # Record the list of source data-unit IDs
            "tool_calls": e.get("source_tool_calls", []),  # Record the list of source tool-call IDs
            "files": sorted({u.data_origin.get("source_path", "")  # files field: set comprehension of source units' raw paths, sorted (spans lines)
                             for u in state.get("data_units", [])  # Iterate all data units in state
                             if u.data_unit_id in set(e.get("source_data_units", []))}),  # Keep only the units cited by this evidence; close the set/field/call
        })
    audit = {  # Assemble the audit summary dict (spans lines)
        "data_flow": [a.file_path for a in state.get("input_assets", [])],  # data_flow field: file-path list of all input assets
        "n_data_units": len(state.get("data_units", [])),  # Count total data units
        "n_evidence": len(state.get("evidence", [])),  # Count total evidence entries
        "n_tool_calls": len(state.get("tool_results", [])),  # Count total tool calls
        "decisions": state.get("decisions", []),   # Per-round planning rationale (auditable)
        "evidence_chain": chain,  # Attach the per-evidence lineage chain
    }
    tool_calls = [{  # Tool-call records: per-call name/status/output summary/artifact keys (rendered by the report "trail" section)
        "tool": r.get("tool_name", ""),  # Tool name
        "status": r.get("status", ""),  # Execution status (success/error)
        "observation": (str((r.get("result") or {}).get("observation", ""))  # On success take the payload's observation text
                        or r.get("error") or "")[:4000],  # On failure take the error message; uniformly truncate to 4000 chars (paper-H smoke 2026-10-19 diagnosis:
        # 200 chars cut table_column_stats group statistics from group 2 on; final_state/report/grading text
        # fields were collectively distorted, making inter-group comparison tasks (e.g. G03) structurally
        # unsolvable -- aligned with the tool-side group budget (8 groups × 900 chars ≈ 7.2k cap), rounded;
        # the report §7 display layer separately truncates to 120 chars, unaffected here)
        "artifacts": sorted(((r.get("result") or {}).get("artifacts") or {}).keys()),  # Artifact key list (figure/table names)
    } for r in state.get("tool_results", [])]  # Iterate all tool run results
    audit["tool_calls"] = tool_calls  # Keep a full call record in the audit
    result = dict(state.get("final_result") or {})  # Copy the final-result dict for write-back
    result["provenance"] = audit["evidence_chain"]  # Reverse index: conclusion → evidence → file
    result["tool_calls"] = tool_calls  # Tool-call records (rendered by the report "reasoning & tool-call trail" section)
    result["audit"] = audit  # Attach the full audit info to the result's audit field
    return {"final_result": result}  # Return a partial state updating only final_result
