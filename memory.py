"""Long-task memory pattern v1: one JSON conclusion card per task + permission-checked recall + off by default.
Scope (agreed with the user on 2026-10-11):
- This is **engineered context passing**, not the model having learned something — model
  weights are completely unchanged between two calls; a card stores the conclusions the
  model itself drew for earlier tasks, citable and overridable;
- Card writing = **deterministic verbatim copying** of existing final_result fields
  (conclusion / key_observations / remaining_uncertainties / provenance) with no AI
  re-processing — the numbers on a card match the answer sheet verbatim and stay traceable;
- Permission = each card records seen_dirs (the set of directories mounted for that task);
  recall enforces a **strict subset** filter: a card is injected only if seen_dirs is a
  subset of the directories mounted this time. Any unmounted directory mixed in blocks the
  whole card (if task6 blocks hangar_flat, its cards are simply never recalled, zero
  special-casing);
- Off by default: with memory_dir=None the entire chain does not exist.
"""
from __future__ import annotations  # enable PEP 563 deferred type-annotation evaluation
import json  # import json: card serialization and reading
from pathlib import Path  # import Path: workspace directory and card file paths
from typing import Any  # import the Any type annotation

MEMORY_EVIDENCE_PREFIX = "M"  # memory evidence id prefix (M-001; distinguishes the origin from ledger E-xxx)
MEMORY_RELIABILITY = 0.75  # memory entry reliability: prior judgment, citable and overridable (below direct observation at 0.9)


def card_from_result(task_label: str, final_result: dict[str, Any],  # define the conclusion card constructor (signature spans lines)
                     seen_dirs: set[str], mounted_dirs: set[str]) -> dict[str, Any]:  # param cont: directory permission sets
    """Deterministically extract a conclusion card from final_result: a verbatim copy of four existing fields, zero AI processing."""
    observations = [str(o) for o in (final_result.get("key_observations") or []) if str(o).strip()]  # key observation texts (including column summaries / per-stage microthermometry numbers produced by tools)
    provenance = final_result.get("provenance") or []  # evidence -> file lineage chain
    cited = sorted({str(e.get("evidence_id")) for e in provenance if e.get("evidence_id")})  # cited evidence ids, deduplicated
    return {  # conclusion card: six-key structure (everything written into taskN.json)
        "task": task_label,  # source task label (e.g. task1_yellow_pine)
        "conclusion": str(final_result.get("conclusion") or "")[:1200],  # judgment + confidence (conclusion text truncated to 1200 chars)
        "key_observations": observations,  # key number texts, verbatim (no distortion)
        "remaining_uncertainties": [str(x) for x in (final_result.get("remaining_uncertainties") or []) if str(x).strip()],  # unresolved questions
        "evidence_refs": cited,  # cited evidence ids
        "seen_dirs": sorted(seen_dirs),  # directories actually mounted for this task (permission basis, absolute paths)
    }


class MemoryStore:  # define the memory store class: one workspace = one folder
    """The workspace is a folder: <root>/card_<task>.json one card per task + an index.json index."""

    def __init__(self, root: str | Path) -> None:  # constructor: bind the workspace directory
        self.root = Path(root)  # workspace root directory (e.g. oregenesis/memory/Stibnite_chain)

    def save_card(self, card: dict[str, Any]) -> Path:  # define the save-card method
        """Write one conclusion card and update the workspace index; rerunning the same task overwrites the old card (last-write-wins)."""
        self.root.mkdir(parents=True, exist_ok=True)  # create the workspace directory recursively
        path = self.root / f"card_{card.get('task') or 'task'}.json"  # card file name: card_<task label>.json
        path.write_text(json.dumps(card, ensure_ascii=False, indent=2, default=str), encoding="utf-8")  # serialize to disk
        index_path = self.root / "index.json"  # workspace index file
        index: list[dict[str, Any]] = []  # index defaults to an empty list
        if index_path.exists():  # read the index when it exists
            try:  # a corrupt index is not fatal
                index = json.loads(index_path.read_text(encoding="utf-8"))  # read the old index
            except Exception:  # rebuild from an empty list when parsing fails
                index = []  # reset to empty
        index = [i for i in index if i.get("task") != card.get("task")]  # remove the old entry of the same task (overwrite semantics)
        index.append({"task": card.get("task"),  # index entry: task label
                      "conclusion_head": str(card.get("conclusion", ""))[:120],  # first 120 chars of the conclusion (for human directory review)
                      "n_observations": len(card.get("key_observations") or []),  # number of key-number entries
                      "file": path.name})  # card file name
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")  # write the index back
        return path  # return the card file path

    def recall_cards(self, mounted_dirs: set[str]) -> list[dict[str, Any]]:  # define the recall method
        """Recall conclusion cards under the strict-subset permission: a card passes only if its seen_dirs ⊆ the directories mounted this time."""
        if not self.root.exists():  # workspace absent (first round / memory disabled)
            return []  # nothing to recall
        cards: list[dict[str, Any]] = []  # recall result buffer
        for p in sorted(self.root.glob("card_*.json")):  # check card by card (stable file name order)
            try:  # skipping a corrupt card is not fatal
                card = json.loads(p.read_text(encoding="utf-8"))  # read the card
            except Exception:  # parse failure
                continue  # skip the card
            seen = set(card.get("seen_dirs") or [])  # set of directories the card originally saw
            if seen and seen.issubset(mounted_dirs):  # strict subset: not a single unmounted directory allowed
                cards.append(card)  # admit
        return cards  # return the admitted cards

    @staticmethod  # static method: rendering depends on no instance state
    def render_lines(cards: list[dict[str, Any]]) -> list[str]:  # define the prompt-line renderer
        """Render one prompt block per card (feeds the "已有结论记忆" section of the planner prompt and state injection)."""
        lines: list[str] = []  # render buffer (one block per card)
        for c in cards:  # render card by card
            part = f"[{c.get('task')}] {str(c.get('conclusion', ''))[:400]}"  # first line: task label + conclusion
            for o in (c.get("key_observations") or [])[:6]:  # at most 6 key numbers
                part += f"\n  · {str(o)[:220]}"  # number lines (verbatim, truncated to 220 chars)
            if c.get("remaining_uncertainties"):  # unresolved questions present
                part += "\n  · 未解决：" + "；".join(str(x)[:80] for x in c["remaining_uncertainties"][:3])  # at most 3 entries
            lines.append(part)  # include this card block
        return lines  # return the rendered lines

    @staticmethod  # static method
    def recall_evidence(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:  # define the evidence-entry converter
        """Convert recalled cards into evidence ledger entries (M-prefixed ids; the synthesis summary evidence[:40] naturally includes the header memories)."""
        out: list[dict[str, Any]] = []  # evidence entry buffer
        for i, c in enumerate(cards, 1):  # number cards one by one
            obs = f"[{c.get('task')}] {str(c.get('conclusion', ''))[:400]}"  # observation first segment: task + conclusion
            if c.get("key_observations"):  # append key numbers (at most 4, truncated to 160 chars each)
                obs += "｜关键数值：" + "；".join(str(o)[:160] for o in c["key_observations"][:4])  # join the numbers
            out.append({  # assemble the ledger entry (fields isomorphic to EvidenceUnit.to_dict)
                "evidence_id": f"{MEMORY_EVIDENCE_PREFIX}-{i:03d}",  # M-001 style id (no clash with E-xxx)
                "observation": obs,  # observation text
                "source_data_units": [],  # no data unit sources (memory originates from earlier tasks)
                "source_tool_calls": [],  # no tool call sources
                "reliability": MEMORY_RELIABILITY,  # reliability 0.75: prior judgment, below direct observation
                "limitations": ["来自早先任务的结论卡，可能需要复核"],  # limitation stated explicitly
                "kind": "memory",  # kind marker: the provenance check of review exempts this kind
            })  # close the entry
        return out  # return the entry list
