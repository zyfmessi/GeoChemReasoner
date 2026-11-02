"""Rock classification and discrimination-diagram axis tool family.

Two kinds of tools: (1) rock type/index classifiers (four-level TAS, Shand alumina
saturation, Mg#/Fe#/SI/mafic index — thresholds follow standard geoscience textbook
conventions and are declared explicitly with the descriptions); (2) coordinate
calculators for common discrimination diagrams (they only compute axis values;
interpretation is left to the reasoning layer).
"""

from __future__ import annotations  # Enable deferred evaluation of type annotations (PEP 563)

from typing import Any, Callable  # Import the Any and Callable type annotations

from oregenesis.tools.geochem import _load_table  # Reuse the geochemistry family's table loader _load_table
from oregenesis.tools.registry import ToolCall, ToolSpec  # Import the ToolCall and ToolSpec dataclasses
from oregenesis.understanding import _norm_element  # Import the element-name normalization function

# Oxide molar masses (g/mol) — for molar-ratio indices such as CIA/Shand/Mg#
_MOLAR_MASS = {"SiO2": 60.08, "TiO2": 79.87, "Al2O3": 101.96, "Fe2O3": 159.69,  # Oxide molar-mass dict (g/mol): SiO2/TiO2/Al2O3/Fe2O3
               "FeO": 71.85, "MnO": 70.94, "MgO": 40.30, "CaO": 56.08,  # Continued: FeO/MnO/MgO/CaO molar masses
               "Na2O": 61.98, "K2O": 94.20, "P2O5": 141.94}  # Continued: Na2O/K2O/P2O5 molar masses (11 oxides in total)


def _table_tool(name: str, description: str, required: tuple[str, ...],  # Table-driven classification tool factory: name/description/required columns
                compute_row: Callable, summarize: Callable) -> ToolSpec:  # compute_row: per-sample row computation function; summarize: summary function
    """Classification/index tool factory: required columns present -> per-sample compute_row -> summary digest."""

    def _applicable(state: Any) -> tuple[bool, str]:  # Define the inner data-condition check function
        for u in state.get("data_units", []):  # Iterate over all DataUnits in the state
            if u.data_modality != "numeric":  # Skip non-numeric tables
                continue  # Continue to the next data unit
            syms = {_norm_element(str(h)) for h in u.raw_data.get("header", [])}  # Normalize headers into a set of element/oxide symbols
            if all(s in syms for s in required):  # Check whether all required columns are present
                return True, ""  # All present: return applicable
        return False, f"需要同时含 {'/'.join(required)} 列的数值表"  # Missing columns: not applicable, stating the required columns

    def _run(state: Any, call: ToolCall) -> dict[str, Any]:  # Define the inner run function
        columns, meta = _load_table(state, call)  # Load the numeric table: full element-column dict and sample-id metadata
        per_sample: dict[str, Any] = {}  # Initialize the per-sample result dict
        for i, sid in enumerate(meta["ids"]):  # Compute per sample (row-value dict = all element columns of the table)
            row_vals = {s: (columns[s][i] if s in columns else None) for s in columns}  # Assemble the value dict of all columns for this row (missing as None)
            per_sample[sid] = compute_row(row_vals, i)  # Call compute_row to get this sample's classification/index value
        valid = [v for v in per_sample.values() if v is not None]  # Collect non-null results for summarization
        return {  # Return the result payload dict
            "observation": summarize(valid) if valid else f"{description}：无可计算数据",  # Summary digest when valid results exist; otherwise report nothing computable
            "per_sample": per_sample, "supports": [f"classification_{name}"]}  # Attach per-sample results and evidence tag (and close the dict)

    return ToolSpec(name=name, tool_family="rock_classification", description=description,  # Assemble the ToolSpec: in the rock_classification family
                    required_data=required,  # Declare required data columns
                    capabilities=("rock_classification", "diagram_axis"),  # Capability tags: rock classification + diagram axis
                    applicable=_applicable, run=_run)  # Bind the applicability check and run function


def _tas_class(sio2: float | None) -> str | None:  # Define the four-level SiO2 classification function
    """Classify into four levels by SiO2 wt% (generic boundaries: 45/52/63/69)."""
    if sio2 is None:  # SiO2 missing: cannot classify
        return None  # Return None
    if sio2 < 45:  # SiO2<45%: ultrabasic
        return "超基性岩类"  # Return "ultrabasic rock class"
    if sio2 < 52:  # 45<=SiO2<52%: basic
        return "基性岩类"  # Return "basic rock class"
    if sio2 < 63:  # 52<=SiO2<63%: intermediate
        return "中性岩类"  # Return "intermediate rock class"
    if sio2 < 69:  # 63<=SiO2<69%: intermediate-acid
        return "中酸性岩类"  # Return "intermediate-acid rock class"
    return "酸性岩类"  # SiO2>=69%: return "acid rock class"


def _feot(v: dict) -> float | None:  # Define the total-iron conversion function FeOt
    """FeOt = FeO + 0.8998*Fe2O3 (missing FeO defaults to 0; return None if both are missing)."""
    feo, fe2o3 = v.get("FeO") or 0, v.get("Fe2O3") or 0  # Read FeO and Fe2O3 (missing counts as 0)
    if "FeO" not in v and "Fe2O3" not in v:  # When neither key exists
        return None  # Return None (nothing to convert)
    return feo + 0.8998 * fe2o3  # FeOt = FeO + 0.8998*Fe2O3 (Fe2O3 converted to FeO)


def build_classification_tools() -> list[ToolSpec]:  # Define the rock classification tool family constructor
    """Build the rock classification/index/diagram-axis tool family (7 classifiers + 4 weathering indices + 12 diagram axes)."""
    tools: list[ToolSpec] = []  # Initialize the tool list

    # 1. TAS rock type classification (four-level SiO2 + total-alkali coordinates)
    def _tas_row(v: dict, i: int) -> Any:  # Define the TAS per-sample row computation closure
        sio2 = v.get("SiO2")  # Read this sample's SiO2 content
        cls = _tas_class(sio2)  # Classify by the four SiO2 levels
        return {"rock_class": cls, "sio2": sio2,  # Return the dict of class and SiO2 (total alkali on the continued line)
                "na2o_k2o": (round(v["Na2O"] + v["K2O"], 2)  # Compute total alkali Na2O+K2O rounded to 2 decimals
                             if None not in (v["Na2O"], v["K2O"]) else None)} if cls else None  # Either alkali missing -> None; classification failure returns the whole row as None

    def _tas_summary(samples: list) -> str:  # Define the TAS summary closure
        from collections import Counter  # Local import of Counter
        dist = Counter(s["rock_class"] for s in samples)  # Count samples per rock type
        return "TAS 岩石类型分布: " + ", ".join(f"{k} {v} 件" for k, v in dist.most_common())  # Join "class N samples" summary text in descending count order

    tools.append(_table_tool(  # Append the TAS classification tool to the list
        "tas_classification", "TAS 图解岩石类型分类（SiO2 45/52/63/69 界线）+ 全碱坐标",  # Tool name and description (with the 45/52/63/69 boundaries)
        ("SiO2", "Na2O", "K2O"), _tas_row, _tas_summary))  # Three required oxide columns; bind the row and summary functions

    # 2. Total-alkali coordinate (TAS vertical axis)
    tools.append(_table_tool(  # Append the total-alkali tool
        "calculate_total_alkali", "计算全碱 Na2O+K2O（TAS 图解纵轴坐标）",  # Tool name and description: TAS vertical-axis coordinate
        ("Na2O", "K2O"),  # Required columns: Na2O and K2O
        lambda v, i: round(v["Na2O"] + v["K2O"], 2) if None not in (v["Na2O"], v["K2O"]) else None,  # Row lambda: total-alkali sum to 2 decimals (None if either missing)
        lambda vs: f"全碱 Na2O+K2O：均值 {sum(vs)/len(vs):.2f}%，范围 {min(vs):.2f}~{max(vs):.2f}%"))  # Summary lambda: mean and min~max range text

    # 3. Shand alumina-saturation index (molar ratio: >1.1 peraluminous / 1.0~1.1 metaluminous / <1.0 peralkaline)
    def _acnk_row(v: dict, i: int) -> Any:  # Define the A/CNK per-sample row computation closure
        mol = {o: (v.get(o) / _MOLAR_MASS[o] if v.get(o) is not None else None)  # Convert each oxide wt% to moles via molar mass (missing as None)
               for o in ("Al2O3", "CaO", "Na2O", "K2O")}  # Iterate the four oxides involved in A/CNK
        if mol["Al2O3"] is None or None in (mol["CaO"], mol["Na2O"], mol["K2O"]):  # Al2O3 or any auxiliary oxide missing
            return None  # Return None (not computable)
        acnk = mol["Al2O3"] / (mol["CaO"] + mol["Na2O"] + mol["K2O"])  # A/CNK = Al2O3 moles / (CaO+Na2O+K2O) moles
        acnk = round(acnk, 3)  # Round to 3 decimals
        return {"a_cnk": acnk,  # Return the dict with the A/CNK value (class on the continued line)
                "class": ("过铝质" if acnk > 1.1 else "准铝质" if acnk >= 1.0 else "过碱性")}  # Three-way: >1.1 peraluminous / >=1.0 metaluminous / <1.0 peralkaline

    def _acnk_summary(samples: list) -> str:  # Define the A/CNK summary closure
        from collections import Counter  # Local import of Counter
        dist = Counter(s["class"] for s in samples)  # Count samples per class
        return "Shand 铝饱和分类: " + ", ".join(f"{k} {v} 件" for k, v in dist.most_common())  # Join the Shand classification summary text in descending count order

    tools.append(_table_tool(  # Append the Shand alumina-saturation classification tool
        "acnk_shand_classification", "Shand 铝饱和指数分类（A/CNK 摩尔比，>1.1 过铝质）",  # Tool name and description (A/CNK molar-ratio basis)
        ("Al2O3", "CaO", "Na2O", "K2O"), _acnk_row, _acnk_summary))  # Four required oxide columns; bind the row and summary functions

    # 4. Mg# (molar: 100*Mg/(Mg+Fe2+), Fe2+ = FeO + 0.8998*Fe2O3)
    def _mg_row(v: dict, i: int) -> Any:  # Define the Mg# per-sample row computation closure
        feot_w = _feot(v)  # First compute total iron FeOt wt%
        if v.get("MgO") is None or feot_w is None:  # MgO missing or FeOt not computable
            return None  # Return None
        mg_mol = v["MgO"] / _MOLAR_MASS["MgO"]  # Convert MgO wt% to moles via molar mass
        fe_mol = feot_w / _MOLAR_MASS["FeO"]  # Convert total iron to Fe2+ moles using the FeO molar mass
        return round(100 * mg_mol / (mg_mol + fe_mol), 2)  # Mg# = 100*Mg/(Mg+Fe2+) (molar), rounded to 2 decimals

    tools.append(_table_tool(  # Append the Mg# tool
        "calculate_mg_number", "Mg# =100×Mg/(Mg+Fe2+)（摩尔；Fe2+=FeO+0.8998Fe2O3）",  # Tool name and description (with formula)
        ("MgO", "Fe2O3"), _mg_row,  # Required columns MgO/Fe2O3; bind the row computation function
        lambda vs: f"Mg#：均值 {sum(vs)/len(vs):.1f}，范围 {min(vs):.1f}~{max(vs):.1f}"))  # Summary lambda: mean and min~max range text

    # 5. Fe# (Frost iron index, weight basis)
    tools.append(_table_tool(  # Append the Fe# iron-index tool
        "calculate_fe_number", "Fe# =FeOt/(FeOt+MgO)×100（FeOt=FeO+0.9Fe2O3）",  # Tool name and description (with formula)
        ("MgO", "Fe2O3"),  # Required columns MgO and Fe2O3
        lambda v, i: (round(100 * _feot(v) / (_feot(v) + v["MgO"]), 2)  # Row computation: FeOt/(FeOt+MgO)*100 rounded to 2 decimals
                      if _feot(v) is not None and v.get("MgO") is not None else None),  # Return None when FeOt or MgO is missing
        lambda vs: f"Fe#：均值 {sum(vs)/len(vs):.1f}，范围 {min(vs):.1f}~{max(vs):.1f}"))  # Summary lambda: mean and min~max range text

    # 6. Solidification index SI (higher = less differentiated)
    tools.append(_table_tool(  # Append the solidification index SI tool
        "calculate_solidification_index", "固结指数 SI=100·MgO/(MgO+FeOt+Na2O+K2O)",  # Tool name and description (with formula)
        ("MgO", "Na2O", "K2O", "Fe2O3"),  # Required columns: the four oxides MgO/Na2O/K2O/Fe2O3
        lambda v, i: (round(100 * v["MgO"] /  # Row computation: numerator 100*MgO
                            (v["MgO"] + _feot(v) + v["Na2O"] + v["K2O"]), 2)  # Denominator = MgO+FeOt+Na2O+K2O, rounded to 2 decimals
                      if None not in (v["MgO"], v["Na2O"], v["K2O"]) and _feot(v) else None),  # Return None when a key oxide is missing or FeOt is 0
        lambda vs: f"SI：均值 {sum(vs)/len(vs):.1f}，范围 {min(vs):.1f}~{max(vs):.1f}"))  # Summary lambda: mean and min~max range text

    # 7. Mafic index M (= FeOt+MgO+TiO2)
    tools.append(_table_tool(  # Append the mafic index M tool
        "calculate_mafic_index", "镁铁指数 M=FeOt+MgO+TiO2",  # Tool name and description (with formula)
        ("MgO", "Fe2O3", "TiO2"),  # Required columns: the three oxides MgO/Fe2O3/TiO2
        lambda v, i: (round(_feot(v) + v["MgO"] + v["TiO2"], 2)  # Row computation: FeOt+MgO+TiO2 rounded to 2 decimals
                      if None not in (v.get("MgO"), v.get("TiO2")) and _feot(v) is not None  # Condition: MgO/TiO2 present and FeOt computable
                      else None),  # Return None if missing
        lambda vs: f"M 值：均值 {sum(vs)/len(vs):.2f}，范围 {min(vs):.2f}~{max(vs):.2f}"))  # Summary lambda: mean and min~max range text

    tools.extend(_build_axis_tools())  # Append the discrimination-diagram axis tool family (12 tools)
    tools.extend(_build_index_tools())  # Append the weathering/alteration index tool family (4 tools)
    return tools  # Return the complete classification tool family list


# Discrimination-diagram axis table: (tool name, diagram source, X-axis element, Y-axis element)
# Coordinates are deterministic factual outputs; threshold interpretation is done by the reasoning layer with the literature.
_AXES: list[tuple[str, str, str, str, str]] = [  # Define the diagram axis list: (name, source, X axis, Y axis, extra ratio) five-tuples
    ("sry_y", "Sr/Y–Y（Defant & Drummond 1990，埃达克质判别）", "Y", "Sr", "Sr/Y"),  # Sr/Y-Y diagram (adakite discrimination): X=Y, Y=Sr, extra ratio Sr/Y
    ("k2o_sio2", "K2O–SiO2（Peccerillo & Taylor 1976，钾质系列）", "SiO2", "K2O", ""),  # K2O-SiO2 diagram (potassic series): X=SiO2, Y=K2O, no extra ratio
    ("thyb_nbyb", "Th/Yb–Nb/Yb（Pearce 1983，构造环境）", "Nb", "Th", ""),  # Th/Yb-Nb/Yb diagram (tectonic setting): X=Nb, Y=Th, no extra ratio
    ("ti_zr", "Ti–Zr（Pearce，构造环境）", "Zr", "Ti", ""),  # Ti-Zr diagram (tectonic setting): X=Zr, Y=Ti, no extra ratio
    ("zry_tiy", "Zr/Y–Ti/Y（Pearce & Gale 1977）", "Ti", "Zr", "Ti/Y"),  # Zr/Y-Ti/Y diagram: X=Ti, Y=Zr, extra ratio Ti/Y
    ("nby_zry", "Nb/Y–Zr/Y（Floyd & Winchester，碱度/构造）", "Zr", "Nb", "Nb/Y"),  # Nb/Y-Zr/Y diagram (alkalinity/tectonics): X=Zr, Y=Nb, extra ratio Nb/Y
    ("v_ti", "V–Ti（Shervais 1982，MORB/弧/板内）", "Ti", "V", ""),  # V-Ti diagram (MORB/arc/within-plate): X=Ti, Y=V, no extra ratio
    ("feotmgo_sio2", "FeO*/MgO–SiO2（Miyashiro，拉斑/钙碱系列）", "SiO2", "FeO", "FeO/MgO"),  # FeO*/MgO-SiO2 diagram (tholeiitic/calc-alkaline): X=SiO2, Y=FeO, extra ratio FeO/MgO
    ("mgno_sio2", "Mg#–SiO2（分异-源区联判）", "SiO2", "MgO", ""),  # Mg#-SiO2 diagram: X=SiO2, Y=MgO, no extra ratio
    ("al2o3_tio2", "Al2O3–TiO2（源区/分异）", "TiO2", "Al2O3", ""),  # Al2O3-TiO2 diagram (source/differentiation): X=TiO2, Y=Al2O3, no extra ratio
    ("layb_yb", "La/Yb–Yb（REE 分馏形态）", "Yb", "La", "La/Yb"),  # La/Yb-Yb diagram (REE fractionation): X=Yb, Y=La, extra ratio La/Yb
    ("sr_ba", "Sr–Ba（流体演化）", "Ba", "Sr", ""),  # Sr-Ba diagram (fluid evolution): X=Ba, Y=Sr, no extra ratio
]


def _build_axis_tools() -> list[ToolSpec]:  # Define the diagram-axis tool family constructor
    """Build the discrimination-diagram axis calculators (one tool per diagram, outputting per-sample axis coordinates)."""
    def _axis_tool(name: str, diagram: str, x_el: str, y_el: str, extra_ratio: str) -> ToolSpec:  # Single-diagram tool factory (diagram name / two axis elements / extra ratio)
        required = (x_el, y_el) + ((extra_ratio.split("/")[0], extra_ratio.split("/")[1])  # Required columns = X and Y elements; an extra ratio adds its numerator/denominator columns
                                   if extra_ratio else ())  # Without an extra ratio the required columns are just the two axis elements

        def _applicable(state: Any) -> tuple[bool, str]:  # Define the inner data-condition check function
            for u in state.get("data_units", []):  # Iterate over all DataUnits in the state
                if u.data_modality != "numeric":  # Skip non-numeric tables
                    continue  # Continue to the next data unit
                syms = {_norm_element(str(h)) for h in u.raw_data.get("header", [])}  # Normalize headers into a set of element symbols
                if all(s in syms for s in required):  # Check whether the required elements are all present
                    return True, ""  # All present: return applicable
            return False, f"需要同时含 {'/'.join(required)} 列的数值表"  # Missing columns: not applicable, stating the required columns

        def _run(state: Any, call: ToolCall) -> dict[str, Any]:  # Define the inner run function
            columns, meta = _load_table(state, call)  # Load the numeric table: element-column dict and sample-id metadata
            coordinates = {}  # Initialize the per-sample coordinate dict
            for i, sid in enumerate(meta["ids"]):  # Iterate each sample (i is the row number)
                x = columns.get(x_el, [None])[i]  # Read the X-axis element value (missing column falls back to [None])
                y = columns.get(y_el, [None])[i]  # Read the Y-axis element value (missing column falls back to [None])
                if x is not None and y is not None:  # Record the coordinate only when both axis values are present
                    coordinates[sid] = {"x": round(x, 4), "y": round(y, 4)}  # Record this sample's (x, y), each rounded to 4 decimals
            return {"observation": f"{diagram} 坐标：{len(coordinates)} 个样品已计算轴值"  # First part of the observation text: number of samples with computed coordinates
                                   f"（X={x_el}，Y={y_el}{f'，{extra_ratio}' if extra_ratio else ''}）",  # Continued: state both axis elements and the extra ratio (shown if present)
                    "coordinates": coordinates,  # Attach the per-sample coordinate dict
                    "parameters": {"diagram": diagram}}  # Attach the diagram source parameter

        return ToolSpec(  # Assemble this diagram tool's ToolSpec
            name=f"diagram_{name}_axis", tool_family="diagram_axis",  # Tool name = diagram_<name>_axis, in the diagram_axis family
            description=f"计算 {diagram} 图解坐标轴值（X={x_el}，Y={y_el}）",  # Description states the diagram source and both axis elements
            required_data=required, capabilities=("diagram_axis",),  # Required columns and capability tags
            applicable=_applicable, run=_run)  # Bind the applicability check and run function

    return [_axis_tool(*row) for row in _AXES]  # Unpack each of the 12 _AXES rows to generate the tool list


# Weathering/alteration index table: (tool name, description, molar denominator terms, numerator terms, oxide set)
# All are molar ratios x100; CaO is not corrected for silicate/carbonate (approximate basis, noted with the results).
def _build_index_tools() -> list[ToolSpec]:  # Define the weathering/alteration index tool family constructor
    """Build the weathering/alteration index tool family (CIA/PIA/CIW/CCPI). CIA/PIA/CIW use the molar-ratio
    basis (CaO not corrected for silicates, an approximation); CCPI uses Ohmoto's classic weight-based definition."""
    def _index_tool(name: str, description: str, num: tuple[str, ...],  # Single-index tool factory: name/description/numerator terms
                    den: tuple[str, ...], molar: bool = True) -> ToolSpec:  # Parameter den: denominator oxide tuple; molar=False means weight basis
        required = tuple(dict.fromkeys((*num, *den)))  # Deduplicate while preserving order

        def _row(v: dict, i: int) -> Any:  # Define the per-sample row computation closure
            if molar:  # Molar basis: wt% -> mol
                mol = {o: (v.get(o) / _MOLAR_MASS[o] if v.get(o) is not None else None)  # Convert each oxide wt% to moles via molar mass (missing as None)
                       for o in required}  # Iterate all required oxides
                if any(x is None for x in mol.values()):  # Any mole value missing
                    return None  # Return None
                num_v = sum(mol[o] for o in num)  # Sum the numerator oxide moles
                den_v = sum(mol[o] for o in den)  # Sum the denominator oxide moles
            else:      # Weight basis (classic CCPI definition)
                if any(v.get(o) is None for o in required):  # Any oxide missing under the weight basis
                    return None  # Return None
                num_v = sum(v[o] for o in num)  # Sum the numerator wt% values
                den_v = sum(v[o] for o in den)  # Sum the denominator wt% values
            return round(100 * num_v / den_v, 2) if den_v else None  # Index = 100*numerator/denominator (None if denominator is 0), rounded to 2 decimals

        def _summary(vs: list) -> str:  # Define the summary closure
            return f"{name}：均值 {sum(vs)/len(vs):.1f}，范围 {min(vs):.1f}~{max(vs):.1f}（n={len(vs)}）"  # Join the mean/range/sample-count summary text

        def _run(state: Any, call: ToolCall) -> dict[str, Any]:  # Define the inner run function
            columns, meta = _load_table(state, call)  # Load the numeric table: element-column dict and sample-id metadata
            per_sample: dict[str, Any] = {}  # Initialize the per-sample result dict
            for i, sid in enumerate(meta["ids"]):  # Compute per sample (row-value dict = all element columns of the table)
                row_vals = {s: (columns[s][i] if s in columns else None) for s in columns}  # Assemble the value dict of all columns for this row (missing as None)
                per_sample[sid] = _row(row_vals, i)  # Call _row to compute this sample's index
            valid = [x for x in per_sample.values() if x is not None]  # Collect non-null index values
            return {"observation": _summary(valid) if valid else f"{description}：无可计算数据",  # Summary text when values exist; otherwise report no computable data
                    "per_sample": per_sample, "supports": [f"classification_{name}"]}  # Attach per-sample results and evidence tag

        return ToolSpec(  # Assemble this index tool's ToolSpec
            name=f"calculate_{name}", tool_family="weathering_index",  # Tool name = calculate_<name>, in the weathering_index family
            description=description, required_data=required,  # Pass in the description and deduplicated required columns
            capabilities=("weathering_index", "alteration_index"),  # Capability tags: weathering index + alteration index
            applicable=lambda s, _r=required: _index_applicable(s, _r), run=_run)  # Fix the required columns via a default argument for the applicability check; bind the run function

    def _index_applicable(state: Any, required: tuple[str, ...]) -> tuple[bool, str]:  # Define the data-condition check shared by the index family
        for u in state.get("data_units", []):  # Iterate over all DataUnits in the state
            if u.data_modality != "numeric":  # Skip non-numeric tables
                continue  # Continue to the next data unit
            syms = {_norm_element(str(h)) for h in u.raw_data.get("header", [])}  # Normalize headers into a set of element/oxide symbols
            if all(s in syms for s in required):  # Check whether the required columns are all present
                return True, ""  # All present: return applicable
        return False, f"需要同时含 {'/'.join(required)} 列的数值表"  # Missing columns: not applicable, stating the required columns

    return [  # Return the 4 weathering/alteration index tools
        _index_tool(  # Build the 1st: CIA chemical index of alteration
            "cia", "CIA 化学蚀变指数 =Al2O3/(Al2O3+CaO+Na2O+K2O)×100（摩尔；CaO 未校正）",  # Tool name and description (with formula and the uncorrected-CaO basis)
            ("Al2O3",), ("Al2O3", "CaO", "Na2O", "K2O")),  # Numerator = Al2O3, denominator = Al2O3+CaO+Na2O+K2O
        _index_tool(  # Build the 2nd: PIA plagioclase index of alteration
            "pia", "PIA 斜长石蚀变指数 =Al2O3/(Al2O3+CaO+Na2O)×100（摩尔）",  # Tool name and description (with formula)
            ("Al2O3",), ("Al2O3", "CaO", "Na2O")),  # Numerator = Al2O3, denominator = Al2O3+CaO+Na2O
        _index_tool(  # Build the 3rd: CIW chemical index of weathering
            "ciw", "CIW 化学风化指数 =Al2O3/(Al2O3+CaO+Na2O)×100（摩尔，Harnois 1988）",  # Tool name and description (Harnois 1988)
            ("Al2O3",), ("Al2O3", "CaO", "Na2O")),  # Numerator = Al2O3, denominator = Al2O3+CaO+Na2O
        _index_tool(  # Build the 4th: CCPI index
            "ccpi", "CCPI 碳酸盐-绿泥石-绢云母指数 =(FeO+MgO)/(FeO+MgO+Na2O+K2O)×100（质量，大本）",  # Tool name and description (classic weight-based definition)
            ("FeO", "MgO"), ("FeO", "MgO", "Na2O", "K2O"), molar=False),  # Numerator = FeO+MgO, denominator = four oxides, weight basis instead
    ]


# ── Discrimination-diagram plotting: field-basemap rendering + sample membership statistics (reuses the diagram library's geometry and interpretation engine) ──
def build_diagram_plot_tools() -> list[ToolSpec]:  # Factory function building the discrimination-diagram plotting tool family
    """Build the discrimination-diagram plotting tool: render the field basemap via the diagram library and tally sample region membership."""

    def _applicable(state: Any) -> tuple[bool, str]:  # Data-condition check: an available numeric table and a non-empty diagram library
        ok_table = any(u.data_modality == "numeric" and u.data_status != "error"  # An available numeric table exists (expression spans lines)
                       for u in state.get("data_units", []))  # Iterate data units
        from oregenesis.diagram.boundaries import DEFAULT_LIBRARY  # Lazy import of the diagram library
        return (ok_table and len(DEFAULT_LIBRARY.list()) > 0,  # Applicable only when both conditions hold
                "需要可用数值 DataUnit 且图解库非空")  # Reason text when not applicable

    def _run(state: Any, call: ToolCall) -> dict[str, Any]:  # Run function: compute coordinates -> interpret -> render -> tally
        from collections import Counter  # Local import of the counter
        from oregenesis.diagram.boundaries import DEFAULT_LIBRARY  # Lazy import of the diagram library
        from oregenesis.diagram.interpreter import (DiagramInterpreter,  # Interpretation engine and expression resolver (spans lines)
                                                    resolve_expression)
        from oregenesis.diagram.plotting import plot_diagram  # Renderer
        columns, meta = _load_table(state, call)  # Load the numeric table: element-column dict and sample ids
        _p = call.parameters or {}  # Read call parameters
        wanted = str(_p.get("diagram") or _p.get("diagram_id") or _p.get("diagram_axis") or "").strip()  # Read the diagram ID
        # (Parameter alias compatibility: H-volume smoke 2026-10-19 diagnosis — the model often guesses diagram_axis from capability tags,
        #  while the old code only accepted diagram; the silent miss fell through to auto-picking the first diagram in library order, mismatching the requested diagram with no warning)
        library_ids = [s.diagram_id for s in DEFAULT_LIBRARY.list()]  # All diagram IDs in the library
        if wanted and wanted not in library_ids:  # A non-existent diagram was requested
            raise ValueError(f"图解库不含 {wanted!r}（可用: {', '.join(library_ids)}）")  # Raise a truthful error listing the available IDs
        interp = DiagramInterpreter()  # Interpreter instance
        chosen, points = None, []  # Chosen diagram / point buffer
        candidates = [wanted] if wanted else library_ids  # Try only the requested one if given; otherwise auto-pick in library order
        for did in candidates:  # Trial-compute coordinates for each candidate diagram
            spec = DEFAULT_LIBRARY.get(did)  # Get the diagram spec
            xax, yax = spec.axis("x"), spec.axis("y")  # Get both axes
            pts: list[tuple[float, float, str]] = []  # Coordinate buffer
            for i, sid in enumerate(meta["ids"]):  # Per sample, resolve coordinates from the axis expressions
                ctx = {k: (v[i] if i < len(v) else None) for k, v in columns.items()}  # Element-value context for this sample
                try:  # Skip as missing on expression-resolution failure
                    x = resolve_expression(ctx, xax.expression)  # Resolve x
                    y = resolve_expression(ctx, yax.expression)  # Resolve y
                except Exception:  # noqa: BLE001 —— element missing from context
                    continue  # Skip this sample
                if x is None or y is None:  # Either axis unresolvable
                    continue  # Skip this sample
                pts.append((float(x), float(y), str(sid)))  # Record the point
            if pts:  # This diagram yields coordinates
                chosen = did  # Choose this diagram
                points = pts  # Record the points
                break  # Stop auto-picking
        if chosen is None or not points:  # No diagram yields coordinates
            raise ValueError("当前数据的元素列不足以计算任何已登记图解的坐标")  # Raise a truthful insufficient-data error
        spec = DEFAULT_LIBRARY.get(chosen)  # Get the chosen diagram spec
        region_names = {r.region_id: r.name_cn for r in spec.regions}  # Region ID -> Chinese-name mapping
        counts: Counter = Counter()  # Region membership counter
        for x, y, sid in points:  # Interpret region membership per point
            try:  # Per-point tolerance for out-of-domain/missing values
                reading = interp.read_coordinates(chosen, x, y, sample_id=sid)  # Interpret
                counts[region_names.get(getattr(reading, "region", ""), "图外/未定")] += 1  # Count
            except Exception:  # noqa: BLE001 —— failed interpretation counted as undetermined
                counts["图外/未定"] += 1  # Count
        save_dir = str((state.get("task") or {}).get("output_dir") or "output") + "/figures"  # Figure output directory
        path = plot_diagram(chosen, points, save_dir, prefix="diagram",  # Render basemap + points (spans lines)
                            title_extra=f"n={len(points)}")  # Title carries the sample count
        if path is None:  # Rendering returned None (no diagram/no points)
            raise ValueError("判别图解渲染失败")  # The registry converts the error into an error Observation
        dist = "；".join(f"{k} {v} 点" for k, v in counts.most_common())  # Membership distribution digest
        return {  # Unified payload (dict spans multiple lines)
            "observation": (f"{spec.display_name} 投点完成（{chosen}，fidelity={spec.fidelity}）："
                            f"共 {len(points)} 点；区域归属——{dist or '无'}"),  # Observation: composition and membership statistics
            "artifacts": {"diagram": path},  # Figure artifact (goes into the report's figure-artifacts section)
            "supports": [f"diagram_plotted:{chosen}"],  # Support tag (naming the diagram used)
            "parameters": {"diagram": chosen, "n_points": len(points)},  # Parameter echo
            "limitations": [f"图解边界数字化保真度 {spec.fidelity}（{spec.verification_status}），"
                            "区域判读供方向性参考", "区域归属为几何判定，非地质成因结论"],  # Limitations stated truthfully
        }

    return [ToolSpec(  # Assemble the ToolSpec (spans lines)
        name="plot_discrimination_diagram", tool_family="diagram_axis",  # Tool name and family (same family as the diagram-axis tools)
        description=(f"绘制判别图解底图并投点，参数 diagram=图解ID（可用: {', '.join(chr(34) + s.diagram_id + chr(34) for s in __import__('oregenesis.diagram.boundaries', fromlist=['DEFAULT_LIBRARY']).DEFAULT_LIBRARY.list())}），"
                     "省略则按库序自动挑选；返回区域归属统计"),  # Menu note: state the parameter name and list available diagrams (H-volume smoke 2026-10-19: the model once guessed diagram_axis and silently plotted the wrong diagram)
        required_data=(),  # Data condition judged dynamically by applicable
        capabilities=("diagram_plot", "data_visualization"),  # Capability tags
        applicable=_applicable, run=_run)]  # Bind the check and the run function
