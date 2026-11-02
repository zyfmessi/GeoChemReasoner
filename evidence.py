
from __future__ import annotations  # enable PEP 563 deferred type-annotation evaluation
from dataclasses import dataclass, field  # import the dataclass decorator and the field default factory
from pathlib import Path  # import the Path type for file names
from typing import Any  # import the Any type annotation
from oregenesis.io.data_models import DataUnit  # standard data plane unit (the io layer has no reverse dependency, no import cycle)
"""Evidence Layer: the unified scientific evidence unit EvidenceUnit and evidence generation.

From this layer on, the main object of agent reasoning is no longer the raw DataUnit but the
EvidenceUnit — each evidence record carries the observed fact, its source DataUnits, the
directional supports/contradicts labels, reliability, and limitations. Evidence can only be
generated from deterministic parse results and tool results; the LLM must never fabricate it.
"""

@dataclass  # declare as a dataclass
class EvidenceUnit:  # define the evidence unit class
    """One piece of scientific evidence: observed fact + lineage (which DataUnits/ToolCalls it came from) + directionality."""

    evidence_id: str = ""                            # evidence id (E-001 style, assigned by the ledger)
    observation: str = ""                            # observed fact (faithful to the source, no interpretation mixed in)
    source_data_units: list[str] = field(default_factory=list)   # source DataUnit ids
    source_tool_calls: list[str] = field(default_factory=list)   # source ToolCall ids
    supports: list[str] = field(default_factory=list)    # supported labels (processes/models)
    contradicts: list[str] = field(default_factory=list) # contradicted labels
    reliability: float = 0.5                         # reliability 0~1 (deterministic sources start at 0.9, downgraded when warnings exist)
    limitations: list[str] = field(default_factory=list)  # limitation/downweight notes
    missing_tests: list[str] = field(default_factory=list)  # missing/discriminating tests (surfaced by knowledge matching)
    kind: str = "observation"                        # observation / tool_result

    def to_dict(self) -> dict[str, Any]:  # define the to-dict method
        """Convert to dict for on-disk audit and reporting."""
        return {  # return the all-fields dict (literal spans lines)
            "evidence_id": self.evidence_id, "observation": self.observation,  # write the evidence id and observed fact
            "source_data_units": self.source_data_units,  # write the source data unit id list
            "source_tool_calls": self.source_tool_calls,  # write the source tool call id list
            "supports": self.supports, "contradicts": self.contradicts,  # write the supports and contradicts label lists
            "reliability": round(self.reliability, 2),  # reliability rounded to two decimals
            "limitations": self.limitations, "missing_tests": self.missing_tests,  # write the limitation and missing test lists
            "kind": self.kind,  # write the evidence kind marker
        }


class EvidenceLedger:  # define the evidence ledger class
    """Evidence ledger: the single registry of all evidence, assigning ids and providing referential integrity checks."""

    def __init__(self) -> None:  # constructor: initialize an empty ledger
        self.units: dict[str, EvidenceUnit] = {}  # id -> evidence unit registry
        self._counter = 0  # id counter reset to zero

    def add(self, unit: EvidenceUnit) -> EvidenceUnit:  # define the register method, returning the unit
        """Register evidence: assign an id (E-001 style auto-increment when absent); duplicate ids raise immediately."""
        if not unit.evidence_id:  # auto-assign branch when no id is present
            self._counter += 1  # increment the counter first
            unit.evidence_id = f"E-{self._counter:03d}"  # generate an E-001 style zero-padded three-digit id
        elif unit.evidence_id in self.units:  # when an id is present and conflicts with the ledger
            raise ValueError(f"证据编号重复: {unit.evidence_id}")  # raise the duplicate evidence id error
        self.units[unit.evidence_id] = unit  # store the unit in the registry by id
        return unit  # return the registered unit

    def require(self, evidence_ids: list[str]) -> list[EvidenceUnit]:  # define the batch fetch-by-id method
        """Fetch evidence by id; citing a non-existent id counts as fabricated evidence and raises immediately (referential integrity gate)."""
        missing = [e for e in evidence_ids if e not in self.units]  # find cited ids absent from the ledger
        if missing:  # any missing reference trips the integrity gate
            raise KeyError(f"引用了账本中不存在的证据编号: {missing}")  # raise the non-existent evidence reference error
        return [self.units[e] for e in evidence_ids]  # return the evidence units in the given order

    def all(self) -> list[EvidenceUnit]:               # noqa: A003 (semantically clear, name kept)
        """All evidence (registration order)."""
        return list(self.units.values())  # return the list of all units in the registry

    def to_dict(self) -> list[dict[str, Any]]:  # define the ledger-level to-dict method
        """Dict list of all evidence (for audit persistence)."""
        return [u.to_dict() for u in self.all()]  # call to_dict per unit and assemble the list


def evidence_from_data_units(units: list[DataUnit]) -> list[EvidenceUnit]:  # define the data-units-to-seed-evidence conversion
    """Seed evidence: turn the objective facts of numeric-table DataUnits into evidence (the data-before-reasoning starting point).
    Each numeric table yields one "data profile" evidence (sample count/variables/missing);
    text blocks produce no evidence (text is C4a material and knowledge clues — evidence must
    wait for tools or semantic enrichment to yield testable statements).
    """
    out: list[EvidenceUnit] = []  # initialize the output evidence list
    for u in units:  # walk each data unit
        if u.data_modality != "numeric" or u.data_status == "error":  # skip non-numeric tables and parse-error units
            continue  # move on to the next unit
        header = [str(h) for h in u.raw_data.get("header", [])]     # take the table facts directly
        rows = u.raw_data.get("rows", [])  # data row (sample) list
        from oregenesis.understanding import _norm_element, summarize_table_columns, stage_table_brief  # reuse the element column recognition rules and the all-column/stage summaries
        variables = [v for v in (_norm_element(h) for h in header) if v]  # normalize header columns into element names, dropping non-element columns
        summary = summarize_table_columns(header, rows)  # deterministic all-column summary (numeric ranges/categorical counts, visible up front)
        missing = int(u.data_semantics.get("n_missing_cells", 0))  # missing cell count (default 0)
        limitations = [f"表格含 {missing} 个缺失单元格"] if missing else []  # limitation note when missing cells exist, otherwise an empty list
        attributes = u.data_semantics.get("attributes") or {}  # attribute list of the key-value dossier (filled by the semantics layer; input fidelity)
        if attributes:  # key-value dossier: seed evidence faithfully carries every field-value pair —
            out.append(EvidenceUnit(  # geological content used to be dropped wholesale here as "N 样品×0 变量" (benchmark diagnosis)
                observation="案例属性清单（{} 项）：{}".format(  # observation text: attributes joined pair by pair (spans lines)
                    len(attributes),  # number of attributes
                    "；".join(f"{k}={v}" for k, v in attributes.items())[:800]),  # all fields as key=value (total truncated to 800 chars)
                source_data_units=[u.data_unit_id],  # lineage: record the source data unit id
                reliability=0.9,  # attributes are verbatim transcriptions of the dossier, high reliability
                limitations=["属性清单为案卷原文转写，未做数值核验"],  # limitation: declared faithfully
                kind="observation",  # kind marked as objective observation
            ))  # close the attribute-list evidence
        out.append(EvidenceUnit(  # construct and append the data-profile evidence (arguments span lines)
            observation=(f"数据表 {Path(u.data_origin.get('source_path', '')).name}"  # observation first segment: source file name (concatenated across lines)
                         f"[{u.data_unit_id}]"  # append the unit id (both the unit_id argument of table_column_stats and its column stats output refer to tables by id; the model needs this mapping)
                         f"（{u.data_origin.get('worksheet', 'CSV')}）包含 "  # continue with the worksheet name (CSV fallback)
                         f"{len(rows)} 个样品 × {len(variables)} 个变量"  # continue with sample count and variable count
                         f"（{', '.join(variables[:8])}）"  # continue with the first 8 variable names
                         + (f"｜列摘要: {summary}" if summary else "")),  # append the all-column summary (benchmark 2026-10 diagnosis: numeric columns were previously invisible)
            source_data_units=[u.data_unit_id],  # lineage: record the source data unit id
            reliability=0.9 if not limitations else 0.7,   # downgraded when missing cells exist (conservative aggregation)
            limitations=limitations,  # write the limitations field
            kind="observation",  # kind marked as objective observation
        ))
        stage_brief = stage_table_brief(header, rows)  # stage-grouped table summary (produced only for tables with a stage/阶段/期次 column)
        if stage_brief:  # deep work 2026-10-13: per-group fluid stage microthermometry becomes visible up front (the highest-scoring data for task1/2/6)
            out.append(EvidenceUnit(  # append as a separate entry (not squeezed into the profile entry, so it enters summaries in full)
                observation=f"阶段分组表 {Path(u.data_origin.get('source_path', '')).name}"  # observation first segment: source file
                            f"[{u.data_unit_id}]（按组大小排序）: {stage_brief}",  # with the unit id and per-group statistics
                source_data_units=[u.data_unit_id],  # lineage: source data unit
                reliability=0.9,  # deterministic statistics, high reliability
                limitations=["分组统计为确定性计算；阶段先后语义需结合穿切关系核验"],  # limitation stated faithfully
                kind="observation",  # objective observation
            ))
    return out  # return the generated seed evidence list
