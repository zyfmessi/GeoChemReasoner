"""REE atomic tool family: Eu/Ce anomalies, chondrite-normalized ratios, ΣREE, and LREE/HREE fractionation.

Chondrite reference values from McDonough & Sun (1995); (La/Yb)N and (Gd/Yb)N are
post-normalization ratios. Interpretation thresholds (δEu<0.7 negative / >1.3 positive) are written into the tool descriptions for human reading; the tools themselves only
output deterministic index values and generate no candidate processes.
"""

from __future__ import annotations  # enable deferred evaluation of type annotations (PEP 563)

from typing import Any, Callable  # import Any and Callable type annotations

from oregenesis.tools.geochem import _load_table  # reuse the geochemistry family's table-loading function _load_table
from oregenesis.tools.registry import ToolCall, ToolSpec  # import the ToolCall and ToolSpec dataclasses
from oregenesis.understanding import _norm_element  # import the element-name normalization function

# Chondrite REE abundances (McDonough & Sun, 1995, ppm) -- normalization reference
_CHONDRITE = {  # define the chondrite REE abundance dict (normalization reference, ppm)
    "La": 0.487, "Ce": 1.246, "Pr": 0.192, "Nd": 0.954, "Sm": 0.693,  # reference abundances of the five LREE La/Ce/Pr/Nd/Sm
    "Eu": 0.260, "Gd": 0.947, "Tb": 0.179, "Dy": 1.171, "Ho": 0.265,  # reference abundances of the five Eu/Gd/Tb/Dy/Ho
    "Er": 0.793, "Tm": 0.125, "Yb": 0.436, "Lu": 0.070,  # reference abundances of the four Er/Tm/Yb/Lu (14 REE in total)
}
_LREE = ("La", "Ce", "Pr", "Nd", "Sm", "Eu")  # define the light REE tuple (6 elements)
_HREE = ("Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu")  # define the heavy REE tuple (8 elements)


def _ree_tool(name: str, description: str, required: tuple[str, ...],  # REE tool factory: name/description/required elements
              compute: Callable[[dict[str, float | None]], float | None]) -> ToolSpec:  # compute: single-sample element value dict -> index value or None
    """REE atomic tool factory: compute receives a single-sample element value dict and returns the index value or None."""

    def _applicable(state: Any) -> tuple[bool, str]:  # inner data-condition check function
        for u in state.get("data_units", []):  # iterate over all DataUnits in state
            if u.data_modality != "numeric":  # skip non-numeric tables
                continue  # continue to the next data unit
            syms = {_norm_element(str(h)) for h in u.raw_data.get("header", [])}  # normalize the header into a set of element symbols
            if all(s in syms for s in required):  # check whether all required element columns exist
                return True, ""  # all present: return usable
        return False, f"需要同时含 {'/'.join(required)} 列的数值表"  # not applicable, stating which columns are needed

    def _run(state: Any, call: ToolCall) -> dict[str, Any]:  # inner run function
        columns, meta = _load_table(state, call)  # load the numeric table: element-column dict and sample-id metadata
        per_sample: dict[str, float | None] = {}  # initialize the per-sample index value dict
        for i, sid in enumerate(meta["ids"]):  # compute the index per sample from column values
            vals = {s: columns[s][i] if s in columns else None for s in required}  # assemble this sample's required element value dict (missing columns padded with None)
            per_sample[sid] = (round(compute(vals), 3)  # call compute for the index, keeping 3 decimals
                               if compute(vals) is not None else None)  # store None when compute returns None
        valid = [v for v in per_sample.values() if v is not None]  # collect all non-missing index values
        obs = (f"{description}：均值 {statistics_mean(valid):.3f}，范围 "  # with valid values: assemble the first part of the observation, the mean
               f"{min(valid):.3f}~{max(valid):.3f}（n={len(valid)}）" if valid  # cont.: min~max range and valid sample count (ternary condition)
               else f"{description}：无可计算数据")  # fallback observation text when there are no valid values
        return {"observation": obs, "per_sample": per_sample,  # return the observation text and per-sample indices
                "parameters": {"normalization": "McDonough & Sun (1995)" if "_n" in name else "raw"},  # tool names containing _n are tagged with the chondrite normalization reference, otherwise raw
                "supports": [f"ree_{name}"]}  # attach the evidence tag ree_<tool name>

    return ToolSpec(  # assemble and return the REE tool spec
        name=f"calculate_{name}", tool_family="geochemistry",  # tool name = calculate_<name>, assigned to the geochemistry family
        description=description, required_data=required,  # pass in the description and the required elements tuple
        capabilities=("ree_pattern", "process_calculator"),  # capability tags: REE pattern + process calculator
        applicable=_applicable, run=_run,  # bind the applicability check and the run function
    )


def statistics_mean(values: list[float]) -> float:  # mean helper function
    """Small mean helper (avoids repeating the boilerplate of importing statistics everywhere)."""
    import statistics  # import the statistics module inside the function
    return statistics.mean(values)  # return the arithmetic mean of the list


def build_ree_tools() -> list[ToolSpec]:  # REE tool family constructor
    """Build the REE atomic tool family (6 tools)."""
    def eu(v: dict) -> float | None:  # δEu computation closure
        return v["Eu"] / (v["Sm"] * v["Gd"]) ** 0.5 if None not in (v["Eu"], v["Sm"], v["Gd"]) else None  # Eu divided by sqrt(Sm*Gd) gives the Eu anomaly; return None when any value is missing

    def ce(v: dict) -> float | None:  # δCe computation closure
        return v["Ce"] / (v["La"] * v["Pr"]) ** 0.5 if None not in (v["Ce"], v["La"], v["Pr"]) else None  # Ce divided by sqrt(La*Pr) gives the Ce anomaly; return None when any value is missing

    def norm_ratio(a: str, b: str) -> Callable:  # normalized-ratio closure factory (element a over b)
        f = _CHONDRITE[a] / _CHONDRITE[b]  # precompute the ratio of the chondrite reference values of a and b
        return lambda v: v[a] / v[b] / f if None not in (v[a], v[b]) else None  # return a lambda: measured a/b divided by the normalization factor; None for missing values

    def sum_ree(v: dict) -> float | None:  # ΣREE total computation closure
        vals = [v.get(s) for s in (*_LREE, *_HREE)]  # collect the 14 REE concentrations (missing entries are None)
        return sum(vals) if all(x is not None for x in vals) else None  # sum for the REE total only when all present, otherwise None

    def lree_hree(v: dict) -> float | None:  # LREE/HREE total-ratio closure
        l, h = sum(v.get(s) or 0 for s in _LREE), sum(v.get(s) or 0 for s in _HREE)  # LREE sum and HREE sum (None counted as 0)
        return l / h if l and h else None  # divide for the LREE/HREE ratio only when both are non-zero

    return [  # start the list of 6 REE tools
        _ree_tool(  # build tool 1: Eu anomaly
            "eu_anomaly", "Eu 异常 δEu=Eu/√(Sm×Gd)（<0.7 负=斜长石分离；>1.3 正=斜长石堆积）",  # tool name and description (with δEu interpretation thresholds)
            ("Eu", "Sm", "Gd"), eu),  # required elements Eu/Sm/Gd, compute function eu (interpretation thresholds go into the description for human reading only)
        _ree_tool(  # build tool 2: Ce anomaly
            "ce_anomaly", "Ce 异常 δCe=Ce/√(La×Pr)（氧化/还原环境指示）",  # tool name and description (δCe redox-environment indicator)
            ("Ce", "La", "Pr"), ce),  # required elements Ce/La/Pr, compute function ce
        _ree_tool(  # build tool 3: normalized La/Yb
            "la_yb_n", "(La/Yb)N 球粒陨石标准化轻重稀土分馏",  # tool name and description: overall LREE-HREE fractionation
            ("La", "Yb"), norm_ratio("La", "Yb")),  # required elements La/Yb, compute function is the normalized-ratio closure
        _ree_tool(  # build tool 4: normalized Gd/Yb
            "gd_yb_n", "(Gd/Yb)N 球粒陨石标准化 MREE-HREE 分馏",  # tool name and description: MREE-HREE fractionation
            ("Gd", "Yb"), norm_ratio("Gd", "Yb")),  # required elements Gd/Yb, compute function is the normalized-ratio closure
        _ree_tool(  # build tool 5: REE total
            "sum_ree", "ΣREE 稀土总量（14 项之和）",  # tool name and description: sum of 14 items
            ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"),  # required elements: all 14 REE
            sum_ree),  # compute function is the summation closure sum_ree
        _ree_tool(  # build tool 6: LREE/HREE ratio
            "lree_hree", "LREE/HREE 轻重稀土总量比",  # tool name and description: LREE/HREE total ratio
            ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"),  # required elements: all 14 REE
            lree_hree),  # compute function is the LREE/HREE-ratio closure lree_hree
    ]
