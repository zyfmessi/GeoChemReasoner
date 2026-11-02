"""Geochemistry Tool Family: atomic geochemistry tools (ratios / REE indicators / data inventory).

All tools follow the same contract — read columns from a DataUnit table, do deterministic
computation, and return a result payload with observation/summary/supports. Trend
readings state only deterministic facts (increasing/decreasing/stable), never genetic
attribution — interpretation is left to the report narration layer.
"""

from __future__ import annotations  # Enable deferred evaluation of type annotations (PEP 563)

import re  # Import re: column-name tokenization (for fuzzy rescue)
import statistics  # Import the statistics module (for ratio means)
from pathlib import Path  # Import Path: take file names from provenance paths (for column-stats observation text)
from typing import Any  # Import the Any type annotation

from oregenesis.tools.registry import ToolCall, ToolSpec  # Import the ToolCall and ToolSpec dataclasses
from oregenesis.understanding import _norm_element, summarize_table_columns  # Import the element-name normalization and all-column summary functions

# ── Element-pair ratio calculators (50 pairs of common geological-process indicators, each registered as a standalone tool) ──
# (key, numerator, denominator, geologic meaning) — the meaning goes into the tool description for Data-aware planner selection.
RATIO_PAIRS: list[tuple[str, str, str, str]] = [  # Define the element-pair ratio list: (key, numerator, denominator, meaning) four-tuples
    ("nb_ta", "Nb", "Ta", "高分异演化指标"),  # Nb/Ta ratio: high-differentiation evolution indicator
    ("zr_hf", "Zr", "Hf", "高分异演化指标"),  # Zr/Hf ratio: high-differentiation evolution indicator
    ("k_rb", "K", "Rb", "流体交代强度"),  # K/Rb ratio: fluid metasomatism intensity
    ("rb_sr", "Rb", "Sr", "演化/交代程度"),  # Rb/Sr ratio: degree of evolution/metasomatism
    ("sr_ba", "Sr", "Ba", "流体演化阶段"),  # Sr/Ba ratio: fluid evolution stage
    ("th_nb", "Th", "Nb", "俯冲组分加入"),  # Th/Nb ratio: subduction component input
    ("la_sm", "La", "Sm", "LREE 内部分馏"),  # La/Sm ratio: internal LREE fractionation
    ("gd_yb", "Gd", "Yb", "HREE 分馏（石榴子石）"),  # Gd/Yb ratio: HREE fractionation (garnet)
    ("la_yb", "La", "Yb", "LREE/HREE 总分馏"),  # La/Yb ratio: overall LREE/HREE fractionation
    ("sr_y", "Sr", "Y", "埃达克质/深部压力指标"),  # Sr/Y ratio: adakitic/deep-pressure indicator
    ("ce_pb", "Ce", "Pb", "壳源物质贡献"),  # Ce/Pb ratio: crustal material contribution
    ("th_u", "Th", "U", "风化/氧化再利用"),  # Th/U ratio: weathering/oxidation reworking
    ("sm_nd", "Sm", "Nd", "源区 REE 分异"),  # Sm/Nd ratio: source REE differentiation
    ("lu_hf", "Lu", "Hf", "Hf 同位素分异"),  # Lu/Hf ratio: Hf isotope differentiation
    ("ti_zr", "Ti", "Zr", "结晶分异程度"),  # Ti/Zr ratio: degree of fractional crystallization
    ("zr_y", "Zr", "Y", "构造环境判别坐标"),  # Zr/Y ratio: tectonic-setting discrimination coordinate
    ("nb_y", "Nb", "Y", "构造环境/碱度坐标"),  # Nb/Y ratio: tectonic-setting/alkalinity coordinate
    ("y_ho", "Y", "Ho", "流体活动异常"),  # Y/Ho ratio: fluid-activity anomaly
    ("ba_la", "Ba", "La", "俯冲流体交代"),  # Ba/La ratio: subduction-fluid metasomatism
    ("la_nb", "La", "Nb", "俯冲组分/源区"),  # La/Nb ratio: subduction component/source
    ("ta_yb", "Ta", "Yb", "构造环境坐标"),  # Ta/Yb ratio: tectonic-setting coordinate
    ("th_yb", "Th", "Yb", "构造环境坐标"),  # Th/Yb ratio: tectonic-setting coordinate
    ("ti_v", "Ti", "V", "MORB/弧火山岩判别"),  # Ti/V ratio: MORB/arc volcanic discrimination
    ("ba_sr", "Ba", "Sr", "蚀变程度"),  # Ba/Sr ratio: degree of alteration
    ("la_ce", "La", "Ce", "LREE 端元内部分馏"),  # La/Ce ratio: internal LREE end-member fractionation
    ("ce_y", "Ce", "Y", "稀土配分形态"),  # Ce/Y ratio: REE distribution pattern
    ("eu_sm", "Eu", "Sm", "中稀土演化"),  # Eu/Sm ratio: middle-REE evolution
    ("u_pb", "U", "Pb", "时代/源区"),  # U/Pb ratio: age/source
    ("th_pb", "Th", "Pb", "源区特征"),  # Th/Pb ratio: source characteristics
    ("rb_cs", "Rb", "Cs", "晚期流体演化"),  # Rb/Cs ratio: late-stage fluid evolution
    ("fe_mn", "Fe", "Mn", "热液/氧化条件（FeOt/MnO）"),  # Fe/Mn ratio: hydrothermal/oxidation conditions (FeOt/MnO)
    ("ni_co", "Ni", "Co", "幔源/岩浆硫化物"),  # Ni/Co ratio: mantle-derived/magmatic sulfides
    ("cr_ni", "Cr", "Ni", "超镁铁质亲和"),  # Cr/Ni ratio: ultramafic affinity
    ("v_cr", "V", "Cr", "氧化还原/构造"),  # V/Cr ratio: redox/tectonics
    ("al2o3_tio2", "Al2O3", "TiO2", "源区/分异程度"),  # Al2O3/TiO2 ratio: source/degree of differentiation
    ("cao_al2o3", "CaO", "Al2O3", "演化程度坐标"),  # CaO/Al2O3 ratio: degree-of-evolution coordinate
    ("k2o_mgo", "K2O", "MgO", "演化/钾交代"),  # K2O/MgO ratio: evolution/potassic metasomatism
    ("na2o_k2o", "Na2O", "K2O", "钠质/钾质碱度类型"),  # Na2O/K2O ratio: sodic/potassic alkalinity type
    ("feot_mgo", "FeO", "MgO", "分异指数（Miyashiro 轴）"),  # FeO/MgO ratio: differentiation index (Miyashiro axis)
    ("zr_th", "Zr", "Th", "源区/分异坐标"),  # Zr/Th ratio: source/differentiation coordinate
    ("th_la", "Th", "La", "俯冲/源区坐标"),  # Th/La ratio: subduction/source coordinate
    ("sr_zr", "Sr", "Zr", "流体/分异坐标"),  # Sr/Zr ratio: fluid/differentiation coordinate
    ("hf_ta", "Hf", "Ta", "矿化指标坐标"),  # Hf/Ta ratio: mineralization-indicator coordinate
    ("dy_yb", "Dy", "Yb", "HREE 分馏（石榴子石）"),  # Dy/Yb ratio: HREE fractionation (garnet)
    ("er_yb", "Er", "Yb", "HREE 端元分馏"),  # Er/Yb ratio: HREE end-member fractionation
    ("tb_yb", "Tb", "Yb", "MREE 分馏"),  # Tb/Yb ratio: MREE fractionation
    ("pr_nd", "Pr", "Nd", "LREE 内部分馏"),  # Pr/Nd ratio: internal LREE fractionation
    ("ba_th", "Ba", "Th", "俯冲流体通量"),  # Ba/Th ratio: subduction fluid flux
    ("w_sn", "W", "Sn", "钨锡矿化元素对"),  # W/Sn ratio: W-Sn mineralization element pair
    ("cu_mo", "Cu", "Mo", "斑岩 Cu-Mo 演化对"),  # Cu/Mo ratio: porphyry Cu-Mo evolution pair
    ("la_nd", "La", "Nd", "极轻稀土分馏"),  # La/Nd ratio: extremely light REE fractionation
    ("eu_yb", "Eu", "Yb", "Eu 异常与 HREE 组合"),  # Eu/Yb ratio: Eu anomaly combined with HREE
    ("ho_yb", "Ho", "Yb", "HREE 内部分馏"),  # Ho/Yb ratio: internal HREE fractionation
    ("ba_rb", "Ba", "Rb", "流体活动性对比"),  # Ba/Rb ratio: fluid mobility contrast
    ("ti_nb", "Ti", "Nb", "构造环境坐标"),  # Ti/Nb ratio: tectonic-setting coordinate
]


def _load_table(state: Any, call: ToolCall) -> tuple[dict[str, list[float | None]], dict[str, Any]]:  # Define the table loader: returns (element-column dict, metadata dict)
    """Load the numeric table specified by the call (or the first available one) from the State.
    Returns ({normalized element name: per-row value list}, {"unit": DataUnit, "ids": [sample ids or row numbers]}).
    """
    units = {u.data_unit_id: u for u in state.get("data_units", [])}  # Index DataUnits by id (all data units in the state)
    unit_ids = call.input_data_unit_ids or list(units)  # Input ids specified by the call; otherwise all DataUnit ids
    for uid in unit_ids:  # Take the first available numeric table in the planner-given input order
        u = units.get(uid)  # Get the DataUnit by id (None if not found)
        if u is None or u.data_modality != "numeric":  # Skip if the id is unknown or the modality is not numeric
            continue  # Try the next DataUnit
        header = [str(h) for h in u.raw_data.get("header", [])]  # Read the header and normalize it to a list of strings
        rows = u.raw_data.get("rows", [])  # Read the data rows (default empty list)
        columns: dict[str, list[float | None]] = {}  # Initialize the column dict: normalized element name -> per-row value list
        for i, h in enumerate(header):  # Iterate the header (i is the column index)
            sym = _norm_element(h)  # Normalize the column name to an element symbol (None for non-element columns)
            if sym is None:  # This column is not an element column
                continue  # Skip it
            vals: list[float | None] = []  # Initialize this column's value list (float or None per cell)
            for row in rows:  # Iterate each data row
                raw = row[i] if i < len(row) else ""  # Read cell i of this row; pad with an empty string if the row is short
                try:  # Try converting the cell value to float
                    v = float(raw)  # Convert each cell of the numeric column to float
                    vals.append(v if v > 0 else None)  # Physically non-positive contents (<=0, lab "below detection limit" negative placeholders) are treated as missing
                except (TypeError, ValueError):  # Catch conversion errors from None/non-numeric values
                    vals.append(None)  # On failure append None as a placeholder (missing value)
            if any(v is not None for v in vals):  # Keep the column only if it has at least one non-missing value
                columns[sym] = vals  # Store into the column dict keyed by normalized element name
        # Sample-id column (fall back to row_N by row order if absent) so results align with samples
        id_col = next((i for i, h in enumerate(header)  # Find the sample-id column index (generator matches common column names)
                       if h.strip().lower() in ("sample_id", "样品id", "样品编号", "编号")), None)  # Selected when the trimmed lowercase name hits any alias; None if not found
        ids = [str(row[id_col]).strip() if id_col is not None and id_col < len(row)  # Build a sample id per row: stringify the sample-id cell and trim
               and str(row[id_col]).strip() else f"row{n + 1}"  # Empty sample ids fall back to row_N by row order
               for n, row in enumerate(rows)]  # Iterate rows (n from 0) completing the list comprehension
        return columns, {"unit": u, "ids": ids}  # Return the column dict and metadata (DataUnit object + sample-id list)
    raise ValueError("未找到可用的数值型 DataUnit（表格）")  # Raise ValueError if no usable numeric table was found after the scan


def element_inventory(state: Any, call: ToolCall) -> dict[str, Any]:  # Define the data-inventory tool's run function
    """Data-inventory tool: summarize element/sample coverage across all numeric tables (the Router's data-condition ledger)."""
    inventory: dict[str, int] = {}  # Initialize the variable-coverage counter: variable name -> occurrence count
    n_samples = 0  # Initialize sample count to 0
    for u in state.get("data_units", []):  # Iterate over all DataUnits in the state
        if u.data_modality != "numeric":  # Skip non-numeric data units
            continue  # Continue to the next data unit
        n_samples = max(n_samples, int(u.data_semantics.get("n_samples", 0)))  # Sample count is the max of the tables' n_samples semantics
        for v in u.data_semantics.get("variables", []):  # Iterate this table's variable-name list
            inventory[v] = inventory.get(v, 0) + 1  # Increment the variable's count (get defaults to 0, then +1)
    return {"observation": f"数据覆盖：{n_samples} 个样品 × {len(inventory)} 个变量",  # Observation text: sample-count x variable-count overview
            "variables": inventory, "n_samples": n_samples}  # Attach the variable-coverage dict and total sample count


_COLUMN_UNIT_TOKENS = {"ppm", "ppb", "ppt", "wt", "pct", "wtpct", "wtper", "percent", "cps",  # Unit noise words stripped from column-name tokens (removed before matching)
                       "count", "counts", "icpoes", "icpms", "ms", "wds", "aq"}  # Continued: common instrument/method suffix words


def _col_tokens(name: str) -> set[str]:  # Define the column-name tokenizer: split alpha tokens, drop unit noise
    """Column name -> set of effective tokens: "As_ppm"->{as}; "As_WTpct"->{as}; "Th_tot"->{th, tot}.

    Purpose: **two-sided normalized matching** between requested column names and headers —
    the model often guesses column names by convention (As_ppm), while the actual header may
    be As_WTpct (the same data, different unit notation); neither contains the other literally,
    so both miss. Intersecting tokens are treated as two spellings of the same data (benchmark
    2026-10 diagnosis: task2 fired As_ppm nine rounds in a row, all misses; the As fingerprint
    never made the report).
    """  # Function docstring: motivation and examples
    raw = re.split(r"[^A-Za-z]+", str(name).lower())  # Split tokens on non-letters
    return {t for t in raw if t and t not in _COLUMN_UNIT_TOKENS}  # Drop empty and unit-noise tokens


def table_column_stats(state: Any, call: ToolCall) -> dict[str, Any]:  # Define the generic column-stats tool's run function
    """Generic column-stats tool: deterministic descriptive statistics on any column of a data table (not limited to element columns).

    Benchmark 2026-10 diagnosis: geothermometry/isotope/geochronology columns (Th, Tm_ice, δ34S…)
    bypass element-name normalization, and previously no tool could read their values. Parameters:
    columns=list of column names (when given, tables containing matching columns are located
    automatically across all tables, no unit ids needed from the model); unit_id=table id
    (restricts to one table); group_by=grouping column (e.g. Stage, statistics per group — staged
    temperature-salinity is the core test of TaskC tasks 1/2/6); all omitted=summary of all
    columns of all tables.
    """  # Function docstring: motivation and parameter semantics
    params = call.parameters or {}  # Read call parameters (planner-provided; may be empty)
    wanted_cols = [str(c).strip() for c in (params.get("columns") or []) if str(c).strip()]  # Requested column-name list (trimmed, blanks dropped)
    wanted_unit = str(params.get("unit_id") or "").strip()  # Requested table id (empty = all tables)
    group_by = str(params.get("group_by") or "").strip()  # Requested grouping column (empty = no grouping)
    units = {u.data_unit_id: u for u in state.get("data_units", [])}  # Index DataUnits by id
    order = (([wanted_unit] if wanted_unit in units else []) +  # The requested table goes first
             [uid for uid in (call.input_data_unit_ids or list(units)) if uid != wanted_unit])  # Remaining tables follow in call order

    def _fname(u: Any) -> str:  # Inner function: the table's source file name
        return Path(u.data_origin.get("source_path", "")).name  # Base file name of the provenance path

    hits: list[tuple[str, Any, list[int]]] = []  # Hit-table buffer: (id, DataUnit, hit column indices)
    found_cols: set[str] = set()  # Column names already hit in some table (the "not found anywhere" list is settled after the scan)
    for uid in order:  # Scan table by table
        u = units.get(uid)  # Get the DataUnit by id
        if u is None or u.data_modality != "numeric" or u.data_status == "error":  # Process only usable numeric tables
            continue  # Skip
        header = [str(h) for h in u.raw_data.get("header", [])]  # Header
        if wanted_cols:  # Columns requested -> find hits in this table
            idx: list[int] = []  # Hit column indices in this table
            for c in wanted_cols:  # Each requested column name
                if c in found_cols:  # Already hit in another table
                    continue  # Do not expand twice
                hit = next((i for i, h in enumerate(header) if h.strip() == c),  # Exact match first
                           next((i for i, h in enumerate(header) if c.lower() in h.lower()), None))  # Fall back to substring match
                if hit is not None:  # This table hits the column
                    idx.append(hit)  # Record the column index
                    found_cols.add(c)  # Mark as hit
            if not idx:  # No hits in this table
                continue  # Skip silently (only hit tables are expanded, avoiding junk lines)
            hits.append((uid, u, idx))  # Record the hit table
        else:  # No columns requested -> include the whole table
            hits.append((uid, u, list(range(len(header)))))  # All columns
        if wanted_unit and hits:  # When a single table is explicitly requested
            break  # Process only that table
    # ── Token fuzzy rescue (2026-10 benchmark diagnosis): two-sided normalized token matching between requested column names and headers —
    # the model guesses "As_ppm" by convention while the actual header is "As_WTpct"; neither contains the other literally, so both miss;
    # identical token sets such as {as} are treated as two spellings of the same data, returned as an approximate column so the model does not spin on retries.
    rescued: list[tuple[str, str, str]] = []  # Rescue records: (requested column name, actually matched header, file name)
    missed = [c for c in wanted_cols if c not in found_cols]  # Column names still unmatched after the exact/substring scan
    if missed:  # Run the rescue scan only when columns are still missing
        miss_tokens = {c: _col_tokens(c) for c in missed}  # Effective tokens of the requested column names
        for uid in order:  # Re-scan the whole library
            u = units.get(uid)  # Get the DataUnit by id
            if u is None or u.data_modality != "numeric" or u.data_status == "error":  # Look only at usable numeric tables
                continue  # Skip
            header = [str(h) for h in u.raw_data.get("header", [])]  # Header
            rows = u.raw_data.get("rows", [])  # Data rows (for validating the rescued column's effective values)
            fname = _fname(u)  # File name (for the rescue annotation)
            for c in list(missed):  # Find an approximate header for each unmatched column
                ct = miss_tokens.get(c) or set()  # This column's effective tokens
                if not ct:  # No effective tokens (purely numeric column names, etc.)
                    continue  # Next column
                best = None  # (intersection size, column index, header name)
                for i, h in enumerate(header):  # Compute the token intersection per header
                    inter = ct & _col_tokens(h)  # Token intersection
                    if inter and (best is None or len(inter) > best[0]):  # The larger intersection wins
                        best = (len(inter), i, h)  # Record the best
                if best:  # Approximate column found: validate effective values first; an all-sentinel column must not be returned as data
                    vals_c = [str(r[best[1]]).strip() for r in rows if best[1] < len(r) and str(r[best[1]]).strip()]  # Non-blank cells of this column
                    n_real = 0  # Real observation count
                    for v in vals_c:  # Per cell: parseable as float and >-1000 counts as real; non-numeric text also counts as real
                        try:  # try wraps the conversion
                            if float(v.replace(",", "")) > -1000:  # Large negative sentinels do not count
                                n_real += 1  # Real value +1
                        except ValueError:  # Text value (categorical column)
                            n_real += 1  # Also valid
                    if n_real == 0:  # Whole column sentinels/blanks -> record "exists but no data"; do not merge into hits
                        rescued.append((c, best[2], fname + "〔该列全缺失，无有效数据〕"))  # State the truth plainly to prevent model misreading
                        found_cols.add(c)  # Resolved (conclusion = no data)
                        missed = [x for x in missed if x != c]  # Remove from the unmatched list
                        continue  # Next column
                    already = next((x for x in hits if x[0] == uid), None)  # Whether this table is already in the hit list
                    if already is not None:  # Already there: append the column index (the list inside the tuple is mutable in place)
                        already[2].append(best[1])  # Append the column index
                    else:  # Not there: add a hit table
                        hits.append((uid, u, [best[1]]))  # New hit table
                    rescued.append((c, best[2], fname))  # Record the rescue
                    found_cols.add(c)  # Mark as hit
                    missed = [x for x in missed if x != c]  # Remove from the unmatched list
    lines: list[str] = []  # Observation line buffer (one line per table)
    stats_out: dict[str, Any] = {}  # Machine-readable statistics detail
    if rescued:  # Rescued hits exist: state first "the column you asked for was not found; this approximate column was used instead" (the model corrects its references accordingly)
        lines.append("近似列救援: " + "; ".join(f"你查询的 {c} 未找到，已按近似列 {h}（{fn}）返回"  # Per-item rescue note
                                              for c, h, fn in rescued))  # Join
    if missed:  # Columns still unmatched after rescue -> provide an "id=file name" index to aid retries
        index = "; ".join(f"{uid}={_fname(units[uid])}" for uid in order  # Index line: id=file name
                          if uid in units and units[uid].data_modality == "numeric")  # Numeric tables only
        lines.append(f"列 {missed} 未在任何表找到。可用数值表: {index}")  # Index line goes first
    for uid, u, idx in hits[:6]:  # Expand at most 6 hit tables (to keep the observation from ballooning)
        header = [str(h) for h in u.raw_data.get("header", [])]  # Header
        rows = u.raw_data.get("rows", [])  # Data rows
        sub_header = [header[i] for i in idx]  # Subset header
        sub_rows = [[r[i] if i < len(r) else "" for i in idx] for r in rows]  # Subset row matrix (short rows padded with blanks)
        if group_by:  # Grouping column given -> statistics per group (e.g. per-stage temperature-salinity by Stage)
            gi = next((i for i, h in enumerate(header) if h.strip() == group_by),  # Exact match on the grouping column
                      next((i for i, h in enumerate(header) if group_by.lower() in h.lower()), None))  # Substring fallback
            if gi is None:  # Grouping column absent
                lines.append(f"{uid}（{_fname(u)}）未找到分组列 {group_by}；表头: {', '.join(header[:15])}")  # Report the available header
                stats_out[uid] = {"error": f"group_by not found: {group_by}", "header": header}  # Machine-readable trace
                continue  # Next table
            groups: dict[str, list[list[str]]] = {}  # Grouped-row buffer: group key -> row matrix
            for r in rows:  # Assign each row to its group
                key = str(r[gi]).strip() if gi < len(r) else ""  # This row's group key
                groups.setdefault(key or "(空)", []).append(r)  # Blank keys go to the "(空)" group
            top = sorted(groups.items(), key=lambda t: -len(t[1]))  # Descending by group size
            parts = []  # Group-fragment buffer
            for g, grows in top[:8]:  # Expand at most 8 groups
                gsub = [[r[i] if i < len(r) else "" for i in idx] for r in grows]  # In-group column-subset matrix
                gsum = summarize_table_columns(sub_header, gsub, max_cols=20, budget=900)  # In-group summary (H-volume smoke 2026-10-19 diagnosis:
                # the old settings max_cols=6/budget=350 silently dropped the model's 7th requested column onward, so cross-group comparison saw only the first few columns —
                # aligned with the ungrouped summary (max_cols=20/budget=1400); the group count is still capped by top[:8])
                parts.append(f"[{g} n={len(grows)}] {gsum}")  # Group fragment: group name + row count + statistics
            tail = f"（另有 {len(top) - 8} 组未展开）" if len(top) > 8 else ""  # Note when groups exceed the cap
            lines.append(f"{uid}（{_fname(u)}，按 {header[gi]} 分组）: "  # Line head: table id/file name/grouping column
                         + " ‖ ".join(parts) + tail)  # Group fragments side by side
            stats_out[uid] = {"group_by": header[gi], "groups": {g: len(grows) for g, grows in top}}  # Machine-readable detail
        else:  # No grouping -> whole-table summary
            summary = summarize_table_columns(sub_header, sub_rows, max_cols=20, budget=1400)  # Column summary (enlarged tool-side budget)
            lines.append(f"{uid}（{_fname(u)}，{len(rows)}行）: {summary}")  # One line per table
            stats_out[uid] = {"summary": summary, "columns": sub_header}  # Machine-readable detail
    if len(hits) > 6:  # Hit tables exceed the expansion cap
        rest = "; ".join(f"{uid}={_fname(u)}" for uid, u, _ in hits[6:])  # id=file name of unexpanded tables
        lines.append(f"（另有 {len(hits) - 6} 张命中表未展开: {rest}，可用 unit_id 指定）")  # Hint at targeted querying
    if not lines:  # Not a single usable numeric table
        raise ValueError("未找到可用的数值型 DataUnit（表格）")  # Same error wording as the inventory tool
    return {"observation": "列统计｜" + " ｜ ".join(lines),  # Observation text: per-table statistics lines
            "stats": stats_out,  # Attach the machine-readable statistics detail
            "supports": ["column_stats_computed"]}  # Evidence tag: this table's column statistics computed


def _ratio_tool(pair_key: str, num: str, den: str, meaning: str) -> ToolSpec:  # Define the ratio tool factory: one standalone ToolSpec per element pair
    """Ratio tool factory: the 50 element pairs share one execution skeleton, differing only in pair and meaning."""

    def _applicable(state: Any) -> tuple[bool, str]:  # Define the inner applicability check function
        for u in state.get("data_units", []):  # Any numeric table containing both element columns suffices
            if u.data_modality != "numeric":  # Skip non-numeric tables
                continue  # Continue to the next data unit
            syms = {_norm_element(str(h)) for h in u.raw_data.get("header", [])}  # Normalize every header column into a set of element symbols
            if num in syms and den in syms:  # Check whether the numerator and denominator columns both exist
                return True, ""  # Both present: return applicable
        return False, f"需要同时含 {num} 与 {den} 列的数值表"  # Not applicable after the scan; the reason names the two missing columns

    def _run(state: Any, call: ToolCall) -> dict[str, Any]:  # Define the inner run function
        columns, meta = _load_table(state, call)  # Load the numeric table: element-column dict and sample-id metadata
        vals, ids = columns.get(num, []), columns.get(den, [])  # Read the numerator and denominator columns (missing columns give empty lists)
        ratios = {}  # Initialize the per-sample ratio dict
        for sid, n, d in zip(meta["ids"], vals, ids):  # Compute the ratio per sample (either missing -> None)
            ratios[sid] = round(n / d, 3) if (n is not None and d not in (None, 0)) else None  # Divide only when both values exist and the denominator is non-zero (3 decimals); otherwise None
        valid = [v for v in ratios.values() if v is not None]  # Collect all non-missing ratios for statistics
        trend = "stable"  # Trend defaults to stable
        if len(valid) >= 3:  # Judge the first-vs-last trend (deterministic factual statement, no genetic attribution)
            trend = "decreasing" if valid[-1] < valid[0] else \
                "increasing" if valid[-1] > valid[0] else "stable"  # Ternary chain continued: last greater than first is increasing, otherwise stable (joins the line above into the first-vs-last trend judgment)
        return {  # Begin building the result payload dict
            "observation": (f"{num}/{den} 比值（{meaning}）：均值 {statistics.mean(valid):.2f}，"  # First part of the observation text: ratio mean (via statistics.mean)
                            f"范围 {min(valid):.2f}~{max(valid):.2f}（n={len(valid)}），"  # Continued: min~max range and valid sample count
                            f"序列趋势 {trend}"),  # Continued: series trend conclusion
            "ratios": ratios, "trend": trend,  # Attach the per-sample ratio dict and trend string
            "parameters": {"numerator": num, "denominator": den},  # Attach algorithm parameters: numerator/denominator element names
            "supports": [f"ratio_{pair_key}_computed"],  # Attach the evidence tag: this ratio computed
        }

    return ToolSpec(  # Assemble and return the ToolSpec from factory parameters
        name=f"calculate_{pair_key}_ratio", tool_family="geochemistry",  # Tool name = calculate_<key>_ratio, in the geochemistry family
        description=f"计算 {num}/{den} 比值并给出序列趋势（{meaning}）",  # Description carries the element pair and meaning (goes into the planner prompt)
        required_data=(num, den),  # Declare required data: the numerator and denominator element columns
        capabilities=("element_ratio", "process_calculator"),  # Capability tags: element ratio + process calculator
        applicable=_applicable, run=_run,  # Bind the applicability check and run function
    )


def build_ratio_tools() -> list[ToolSpec]:  # Define the ratio tool family constructor
    """Build all element-pair ratio calculators."""
    return [_ratio_tool(key, num, den, meaning) for key, num, den, meaning in RATIO_PAIRS]  # Call the factory for every pair in RATIO_PAIRS to generate the tool list


def build_data_tools() -> list[ToolSpec]:  # Define the data tool family constructor
    """Build the data-inventory tool (the Router's data-condition ledger) and the generic column-stats tool."""
    return [ToolSpec(  # First item of the tool list: data inventory
        name="element_inventory", tool_family="data",  # Tool name element_inventory, in the data family
        description="汇总当前数据的样品数与元素/变量覆盖",  # Description: summarize sample count and element coverage
        capabilities=("data_inventory",),  # Capability tag: data inventory
        applicable=lambda s: (any(u.data_modality == "numeric"  # Applicability lambda: whether any numeric DataUnit exists
                                   for u in s.get("data_units", [])),  # Existence scan over all DataUnits
                              "需要数值型 DataUnit（表格）"),  # Reason text when unsatisfied
        run=element_inventory,  # Bind the run function element_inventory
    ), ToolSpec(  # Second item: generic column stats (added per the 2026-10 benchmark diagnosis, filling the non-element-column tool gap)
        name="table_column_stats", tool_family="data",  # Tool name table_column_stats, in the data family
        description=("对数据表的任意列做描述统计（数值列 n/范围/中位/四分位/均值；类别列取值计数）——"  # First half of the description: capability and semantics
                     "不限元素列，适用于测温/同位素/年代学等任何 CSV；"  # Middle of the description: applicability
                     "参数 columns=列名清单、unit_id=表编号、group_by=分组列（如 Stage，逐组统计各阶段）；"  # Parameter section of the description: three parameters
                     "可省略参数=全部表全列摘要"),  # Final part of the description: default semantics (goes into the planner prompt)
        required_data=(),  # No specific data requirement (any numeric table works)
        capabilities=("data_inventory", "descriptive_stats"),  # Capability tags: inventory + descriptive stats
        applicable=lambda s: (any(u.data_modality == "numeric" and u.data_status != "error"  # Applicability: an available numeric table exists
                                   for u in s.get("data_units", [])),  # Existence scan over DataUnits
                              "需要数值型 DataUnit（表格）"),  # Reason text when unsatisfied
        run=table_column_stats,  # Bind the run function table_column_stats
    )]
