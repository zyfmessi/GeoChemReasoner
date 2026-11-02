"""Geochemical Anomaly & Prospectivity Tool Family.

The main tool geochemical_prospectivity_mapping takes a "geochemical DataUnit with
coordinates" through the full chain — data validation -> preprocessing -> anomaly
identification -> spatial interpolation -> multi-element fusion -> prospectivity
prediction -> zonation -> mapping — and returns a ToolResult; all specialized
algorithms are delegated to platform (the numerical layer); this file only
orchestrates and assembles — the LLM decides, the algorithms compute.
"""

from __future__ import annotations  # Enable deferred evaluation of type annotations (PEP 563)

import os  # Import os (for creating output directories)
import random  # Import random (seeded, reproducible deposit holdout sampling for experiment-based selection)
from pathlib import Path  # Import Path (for resolving map output paths)
from typing import Any  # Import the Any type annotation

from oregenesis.tools.registry import ToolCall, ToolSpec  # Import the ToolCall and ToolSpec dataclasses

# Coordinate column aliases (X = easting/longitude, Y = northing/latitude; paired detection avoids confusion with element Y)
_X_ALIASES = {"x", "easting", "xcoord", "coordx", "经度", "longitude", "lon", "lng"}  # X coordinate column alias set (easting/longitude)
_Y_ALIASES = {"y", "northing", "ycoord", "coordy", "纬度", "latitude", "lat"}  # Y coordinate column alias set (northing/latitude)
_ID_NAMES = {"sample_id", "样品id", "样品编号", "编号", "id"}  # Sample-id column name alias set
# Non-element column keywords (skipped when assembling SamplePoints)
_NON_ELEMENT = {"loq", "分析批号", "备注", "note", "remarks"}  # Non-element column keyword set


def _pick_table(state: Any) -> tuple[dict[str, Any] | None, list[list[str]], list[str]]:  # Define the table picker: returns the (DataUnit, header, rows) triple
    """Pick the first numeric table in the State with paired coordinate columns: returns (the DataUnit's raw table, header, rows)."""
    from oregenesis.understanding import _norm_element  # Reuse the element-column recognition rules

    for u in state.get("data_units", []):  # Iterate over all DataUnits in the state
        if u.data_modality != "numeric":  # Skip non-numeric data units
            continue  # Continue to the next data unit
        header = [str(h) for h in u.raw_data.get("header", [])]  # Read the header and normalize it to a list of strings
        lower = [h.strip().lower() for h in header]  # Trim and lowercase headers (to match coordinate aliases)
        xi = next((i for i, h in enumerate(lower) if h in _X_ALIASES), -1)  # Find the X coordinate column index (-1 if absent)
        yi = next((i for i, h in enumerate(lower) if h in _Y_ALIASES), -1)  # Find the Y coordinate column index (-1 if absent)
        if xi == -1 or yi == -1:  # X or Y coordinate column missing
            continue  # Table unqualified; check the next one
        # Every column besides the coordinates that parses as an element/variable becomes a geochemical variable
        element_cols = [i for i, h in enumerate(header)  # Collect indices of non-coordinate columns parseable as elements
                        if i not in (xi, yi) and _norm_element(h) is not None]  # Condition: not one of the two coordinate columns and the name normalizes to an element
        if not element_cols:  # No geochemical variable columns at all
            continue  # Table unqualified; check the next one
        rows = [list(map(str, r)) for r in u.raw_data.get("rows", [])]  # Read the data rows and stringify every cell
        return u, header, rows  # Return the first qualified table's (DataUnit, header, rows)
    return None, [], []  # No qualified table: return (None, empty, empty)


def _applicable(state: Any) -> tuple[bool, str]:  # Define the tool's data-condition check (used by the Router)
    """Data-condition check (the Router's Data-aware criterion): paired coordinate columns + >=8 data rows."""
    u, header, rows = _pick_table(state)  # Try to pick one numeric table with coordinates
    if u is None:  # No qualified table
        return False, "需要带成对坐标列（X/Y、经度/纬度）的地球化学数值表"  # Return not applicable with the reason (paired coordinate columns missing)
    if len(rows) < 8:  # Fewer than 8 rows with coordinates
        return False, f"带坐标样品仅 {len(rows)} 行（需 ≥8）"  # Return not applicable, reporting the actual row count
    return True, ""  # All conditions met: return applicable


def _build_points(header: list[str], rows: list[list[str]]) -> list:  # Define the table-rows -> SamplePoint assembler
    """Table rows -> list of SamplePoints (coordinates/sample id/element columns assembled; cells failing numeric parsing are skipped)."""
    from oregenesis.platform.models import SamplePoint  # Lightweight sample point from the numerical layer
    from oregenesis.understanding import _norm_element  # Local import of the element-name normalization function

    lower = [h.strip().lower() for h in header]  # Trim and lowercase headers (normalization)
    xi = next(i for i, h in enumerate(lower) if h in _X_ALIASES)  # Locate the X coordinate column index
    yi = next(i for i, h in enumerate(lower) if h in _Y_ALIASES)  # Locate the Y coordinate column index
    idc = next((i for i, h in enumerate(lower) if h in _ID_NAMES), None)  # Locate the sample-id column index (None if absent)
    is_geo = lower[xi] in {"经度", "longitude", "lon", "lng"}  # Lon/lat -> planarized inside the algorithm
    element_cols = {i: _norm_element(header[i]) for i in range(len(header))  # Build the column-index -> normalized element name dict (coordinates excluded)
                    if i not in (xi, yi) and _norm_element(header[i]) is not None}  # Condition: not a coordinate column and parseable as an element
    points: list = []  # Initialize the sample point list
    for n, row in enumerate(rows, 1):  # Iterate data rows (n counts samples from 1)
        try:  # Skip rows whose coordinates fail to parse (never fabricate coordinates)
            x, y = float(row[xi]), float(row[yi])  # Convert this row's X and Y cells to float
        except (IndexError, TypeError, ValueError):  # Catch out-of-range/blank/non-numeric errors
            continue  # Skip this row (never fabricate coordinates)
        p = SamplePoint(sample_id=row[idc].strip() if idc is not None and idc < len(row)  # Create the sample point: id taken from the sample-id column and trimmed
                        and row[idc].strip() else f"P{n:03d}",  # Missing ids fall back to sequential P001-style numbering
                        coord_x=x, coord_y=y, coord_is_geo=is_geo)  # Write X/Y coordinates and the lon/lat flag
        for i, elem in element_cols.items():  # All element values go into trace_elements (the pipeline reads them under the merged convention)
            try:  # Try parsing this element's cell value
                p.trace_elements[elem] = float(row[i])  # Convert the content to float and write it into the point's trace_elements
            except (IndexError, TypeError, ValueError):  # Catch out-of-range/blank/non-numeric errors
                continue  # Skip this missing cell
        points.append(p)  # Append this sample point to the list
    return points  # Return the full sample point list


def _grid_percentiles(prob_grid, grid_def, pts: list[tuple[float, float]],  # Define the holdout scoring function (parameters span lines)
                      lon0: float, lat0: float, cos_lat: float) -> list[float]:
    """Drop lon/lat points onto the probability grid and read score percentiles (1 = highest score on the grid; out-of-domain/NaN counts as 0) — the holdout scoring for experiment-based selection."""
    import numpy as _np  # Local import of the numerical library

    flat = _np.asarray(prob_grid, dtype=float).reshape(-1)  # Flatten the grid
    valid = flat[~_np.isnan(flat)]  # Valid score set (out-of-domain cells are NaN)
    if valid.size == 0 or not pts:  # No valid scores or no points
        return [0.0] * len(pts)  # Record all zeros
    valid_sorted = _np.sort(valid)  # Ascending copy for binary-search positioning
    out: list[float] = []  # Percentile result buffer
    for lon, lat in pts:  # Iterate the holdout-exam deposits
        xk = (lon - lon0) * 111.32 * cos_lat  # Longitude -> planar km (same formula as the per-cell export)
        yk = (lat - lat0) * 110.57  # Latitude -> planar km
        ri, ci = grid_def.to_cell(xk, yk)  # Planar coordinates -> grid row/col (row/col clamped inside the grid)
        v = prob_grid[ri, ci]  # This cell's score
        if v != v:  # NaN (out-of-domain cell)
            out.append(0.0)  # Record 0
            continue  # Next point
        out.append(float(_np.searchsorted(valid_sorted, v, side="right")) / float(valid.size))  # The fraction <= v is the percentile
    return out  # Return the per-point percentiles


def _run_mapping(state: Any, call: ToolCall) -> dict[str, Any]:  # Define the main tool's run function: orchestrates the full prediction pipeline
    """Main tool run: assemble sample points -> two-stage pipeline -> mapping -> structured result payload."""
    from oregenesis.platform.prospectivity import pipeline as _pipe   # Numerical-layer main orchestration
    from oregenesis.platform.prospectivity import plotting as _plot   # Numerical-layer mapping

    _, header, rows = _pick_table(state)  # Pick the table (drop the DataUnit reference, keep header and rows)
    points = _build_points(header, rows)  # Assemble the table rows into sample points
    if len(points) < 8:  # Fewer than 8 samples with valid coordinates
        raise ValueError(f"有效坐标样品仅 {len(points)} 个（需 ≥8）")  # Raise ValueError explaining the insufficient sample count
    params = call.parameters or {}  # Read the call parameter dict (None falls back to an empty dict)
    elements = params.get("elements")                 # Target elements (None = auto-select by coverage)
    deposits = params.get("deposits")                 # Known deposits (absent -> unsupervised route)

    # ── Experiment-based selection (data-driven replacement of heuristic rules): hold out N deposits as a "quiz" and trial-run each candidate method,
    #    ranking by the hit rate of holdout deposit score percentiles; select_only=true delivers only the selection table without formal mapping,
    #    otherwise it runs in one go: winning method + all deposits for formal mapping (the prior fusion_method parameter yields to the experiment winner).
    sel_holdout = int(params.get("selection_holdout") or 0)  # Number of quiz deposits (0 = experiment-based selection off)
    sel_only = bool(params.get("select_only"))  # Selection table only (stage 1 of the two-stage mode)
    selection: dict | None = None  # Selection table (into the payload for AI interpretation)
    method_for_formal = params.get("fusion_method", "auto")  # Formal fusion method (without the experiment = call parameter/default)
    if sel_holdout > 0 and deposits and len(deposits) > sel_holdout:  # The experiment needs enough deposits to hold out
        pool = list(params.get("selection_candidates") or
                    ["weights_of_evidence", "information_value", "logistic", "random_forest"])  # Four data-driven methods by default
        rng = random.Random(int(params.get("selection_seed", 42)))  # Fixed seed: reproducible sampling
        idx = list(range(len(deposits)))  # Deposit index pool
        rng.shuffle(idx)  # In-place shuffle
        hold_idx = set(idx[:sel_holdout])  # Indices held out for the quiz
        hold = [deposits[i] for i in sorted(hold_idx)]  # Quiz deposits
        train = [d for i, d in enumerate(deposits) if i not in hold_idx]  # Supervision deposits (the rest)
        lower_s = [h.strip().lower() for h in header]  # Header normalization (same convention as assembly)
        import numpy as _np  # Local import of the numerical library
        xs_s = _np.array([float(r[next(i for i, h in enumerate(lower_s) if h in _X_ALIASES)]) for r in rows])  # Longitude series
        ys_s = _np.array([float(r[next(i for i, h in enumerate(lower_s) if h in _Y_ALIASES)]) for r in rows])  # Latitude series
        ok_s = _np.isfinite(xs_s) & _np.isfinite(ys_s)  # Valid mask
        lon0_s = float(_np.mean(xs_s[ok_s])); lat0_s = float(_np.mean(ys_s[ok_s]))  # Planarization origin (same convention as the export)
        cos_s = abs(_np.cos(_np.deg2rad(lat0_s))) or 1e-6  # Cosine of latitude (0 falls back to 1e-6)
        hold_pts = [(float(d.get("lon", 0.0)), float(d.get("lat", 0.0))) for d in hold]  # Quiz point lon/lat
        table: list[dict] = []  # Selection table
        for m in pool:  # Trial-run each candidate method (compute-only without zoning/mapping is possible; reusing the main pipeline is safest)
            try:  # One method failing must not sink the whole experiment
                rep_m = _pipe.run_prospectivity_mapping(  # Trial run: the full chain with the training deposits
                    points, elements=elements, deposits=train, fusion_method=m,  # Supervision = deposits after holdout
                    interpolation=params.get("interpolation", "idw"),  # Interpolation matches the main call
                    zone_method=params.get("zone_method", "quantile"),  # Zonation matches the main call
                    cv_folds=0)  # Skip cross-validation in trial runs (only the formal mapping run does it)
                if rep_m.model is None or rep_m.model.prob_grid is None or rep_m.grid_def is None:  # No grid produced
                    table.append({"method": m, "error": "; ".join(rep_m.warnings[-1:]) or "无有效栅格"})  # Record an error row
                    continue  # Next method
                pcts = _grid_percentiles(rep_m.model.prob_grid, rep_m.grid_def, hold_pts, lon0_s, lat0_s, cos_s)  # Quiz scoring
                cap10 = sum(1 for p in pcts if p >= 0.9) / len(pcts)  # Hit rate: fraction of percentiles in the top 10%
                mean_pct = sum(pcts) / len(pcts)  # Mean percentile (tie-breaker)
                table.append({"method": m, "holdout_capture_at_10pct": round(cap10, 3),  # Table row: method + hit rate
                              "holdout_mean_percentile": round(mean_pct, 3),  # Row continued: mean percentile
                              "n_holdout": len(hold), "n_train": len(train)})  # Row continued: both stage sample sizes
            except Exception as e:  # noqa: BLE001
                table.append({"method": m, "error": f"{type(e).__name__}: {e}"})  # Record an exception row
        ranked = sorted([t for t in table if "holdout_capture_at_10pct" in t],  # Rank successful rows only
                        key=lambda t: (t["holdout_capture_at_10pct"], t["holdout_mean_percentile"]),  # Hit rate first, mean percentile breaks ties
                        reverse=True)  # Descending
        winner = ranked[0]["method"] if ranked else None  # Winning method (None if all fail)
        selection = {"holdout_n": len(hold), "train_n": len(train),  # Selection summary
                     "seed": int(params.get("selection_seed", 42)),  # Sampling seed (reproducible)
                     "criterion": "holdout_capture_at_10pct, tie-break holdout_mean_percentile",  # Criterion statement
                     "table": table, "winner": winner}  # Table body and winner
        if winner:  # Only a winner takes over the formal method
            method_for_formal = winner  # The data decides (overrides the prior fusion_method parameter)
        if sel_only:  # Stage 1 of the two-stage mode: selection table only
            best = ranked[0] if ranked else {}  # Head row
            return {  # Early return: no formal mapping
                "observation": (f"实验选型完成：扣 {len(hold)} 个矿点当小考、{len(train)} 个做监督，"  # First part of the observation
                                f"{len(table)} 个候选方法试画完毕，赢家 {winner} "
                                f"（holdout 圈中率 {best.get('holdout_capture_at_10pct', 'NA')}）"),  # Observation continued: winner and score
                "selection": selection,  # Full selection table
                "supports": ["model_selection_done"],  # Evidence tag: selection done
                "limitations": ["选型基于 30 个扣留矿点的小考，样本量有限，正式成图将用全部矿点"],  # Limitation statement
                "warnings": ["select_only 模式：未生成正式远景区图件"],  # Warning: no map in this call
            }

    out_root = str((state.get("task") or {}).get("output_dir") or "output")  # Output root (injectable by the runner; default "output" preserves original behavior)
    save_dir = str(Path(out_root) / "maps")           # Map artifact directory (follows the task output directory, easing evaluation isolation)
    os.makedirs(save_dir, exist_ok=True)  # Recursively create the map output directory (no error if it exists)

    report = _pipe.run_prospectivity_mapping(  # Invoke the numerical-layer orchestration to run the full prediction pipeline
        points, elements=elements, deposits=deposits,  # Pass sample points, target elements and known deposits
        fusion_method=method_for_formal,  # Fusion method: experiment winner first, otherwise call parameter/default auto
        interpolation=params.get("interpolation", "idw"),  # Spatial interpolation method (default IDW inverse-distance weighting)
        zone_method=params.get("zone_method", "quantile"),  # Prospect-zone classification method (default quantile)
        cv_folds=int(params.get("cv_folds", 5)))  # Spatial cross-validation folds (default 5; 0 = off)
    if report.model is None or not report.evidence_layers:  # No fusion model selected or no evidence layers: treat as failure
        raise ValueError("; ".join(report.warnings[-2:]) or "预测未产出有效结果")  # Raise ValueError with the last two warnings attached

    plot_paths: list[str] = []  # Initialize the map path list
    try:  # Mapping failure must not block the result (maps are artifacts, not the conclusion itself)
        report.plot_paths = _plot.generate_prospectivity_maps(  # Invoke the numerical-layer mapping module to batch-generate prediction maps
            report, None, save_dir=save_dir, prefix="prospect", dpi=150)  # Pass the report, output directory, file prefix, 150 dpi
        plot_paths = list(report.plot_paths)  # Copy the map path list
    except Exception as e:  # noqa: BLE001 —— mapping exception downgraded to a warning
        report.warnings.append(f"成图失败: {e}")  # Append the mapping exception as a warning (does not block the result)

    scores_path = None  # Per-cell score export path (benchmark adaptation; failure only downgrades to a warning)
    try:  # Export failure must not block the main result
        if (report.model is not None and report.model.prob_grid is not None  # A probability grid exists
                and report.grid_def is not None):  # and a grid definition: exportable
            import numpy as _np  # Local import of the numerical library
            lower_h = [h.strip().lower() for h in header]  # Header normalization (same convention as assembly)
            xi2 = next(i for i, h in enumerate(lower_h) if h in _X_ALIASES)  # X coordinate column index
            yi2 = next(i for i, h in enumerate(lower_h) if h in _Y_ALIASES)  # Y coordinate column index
            xs = _np.array([float(r[xi2]) for r in rows])  # Input-point longitude series (i.e. the canvas cells)
            ys = _np.array([float(r[yi2]) for r in rows])  # Input-point latitude series
            ok = _np.isfinite(xs) & _np.isfinite(ys)  # Valid-coordinate mask (same convention as planarization)
            lon0, lat0 = float(_np.mean(xs[ok])), float(_np.mean(ys[ok]))  # Planarization origin (same convention as spatial_stats)
            cos_lat = abs(_np.cos(_np.deg2rad(lat0))) or 1e-6  # Cosine of the origin latitude (0 falls back to 1e-6)
            g = report.grid_def  # Grid definition (planar km coordinates)
            pg = report.model.prob_grid  # Probability/prospectivity grid (ny,nx); row 0 is the ymax row
            out_rows: list[str] = []  # Export row buffer
            for x0, y0 in zip(xs[ok], ys[ok]):  # Iterate valid input points (canvas cells)
                xk = (x0 - lon0) * 111.32 * cos_lat  # Longitude -> planar km (same formula as spatial_stats)
                yk = (y0 - lat0) * 110.57  # Latitude -> planar km
                ri, ci = g.to_cell(xk, yk)  # Planar coordinates -> grid indices (row/col clamped inside the grid)
                s = pg[ri, ci]  # Read this cell's score
                if s == s:  # NaN != NaN, so this skips no-data cells
                    out_rows.append(f"{x0:.6f},{y0:.6f},{float(s):.6f}")  # One line of lon/lat + score
            if out_rows:  # When there are exportable scores
                scores_path = str(Path(save_dir) / "cell_scores.csv")  # On-disk path
                with open(scores_path, "w", encoding="utf-8") as f:  # Write the CSV
                    f.write("longitude,latitude,score\n")  # Header (score_prospectivity.py contract)
                    f.write("\n".join(out_rows) + "\n")  # Data rows
    except Exception as e:  # noqa: BLE001 —— export exception downgraded to a warning
        report.warnings.append(f"逐格得分导出失败: {e}")  # Append a warning (does not block the result)

    zones = [(z.name_cn, round(z.area, 2), round(z.area_ratio, 3),  # Extract zone four-tuples: name/area/area ratio/deposit capture ratio
              (None if z.capture_ratio is None else round(z.capture_ratio, 3)))  # Capture ratio is non-None only when deposits exist
             for z in (report.zones.levels if report.zones else [])]  # Iterate the zones (empty list when absent)
    auc = (report.evaluation or {}).get("auc")  # Read the evaluation AUC (None when unrecorded)
    cv = (report.evaluation or {}).get("cv") or {}  # Out-of-fold cross-validation metrics (model evaluation defers to these; in-sample AUC is reference only)
    high = zones[0][1] if zones else 0.0  # First-level (class A) area; 0 when no zones
    return {  # Begin building the result payload dict
        "observation": (f"成矿预测完成（融合 {report.model.name_cn}，{report.interpolation} 插值）："  # First part of the observation text: fusion model and interpolation
                        + (f"AUC={auc:.3f}；" if isinstance(auc, (int, float)) and auc == auc else "")  # Continued: AUC appended only when a valid number (not NaN)
                        + (f"空间{cv['k']}折CV AUC={cv['mean_auc']:.3f}±{cv['std_auc']:.3f}、"  # Continued: out-of-fold CV conclusion (evaluation defers to this)
                           if cv.get("mean_auc") is not None else "")  # Appended only when out-of-fold results exist
                        + (f"折外前10%面积捕获{cv['cv_capture_at_10pct'] * 100:.0f}%；"  # Continued: out-of-fold capture rate (exploration decision metric)
                           if cv.get("cv_capture_at_10pct") is not None else "")  # Appended only when present
                        + (f"{zones[0][0]} {high} km²" if zones else "已生成概率栅格")),  # Continued: with zones, report the class-A name and area (unsupervised runs auto-name it "异常区（无矿点验证）")
        "elements": list(report.elements[:10]),  # Attach the first 10 elements that entered fusion
        "fusion_method": report.model.method, "is_probability": report.model.is_probability,  # Attach the fusion method name and the probability-model flag
        "zones": [{"name": n, "area_km2": a, "ratio": r, "capture_ratio": cr}  # Structured zone list (name/area/ratio/deposit capture ratio)
                  for n, a, r, cr in zones],  # Iterate the four-tuples
        "model_metrics": {"auc": None if auc != auc else auc,  # Attach model metrics: AUC (NaN normalized to None)
                          "n_deposits": report.n_deposits,  # Continued: known deposit count
                          "cv": cv or None},  # Continued: out-of-fold spatial CV metrics (evaluation defers to these; None when unsupervised/off)
        "artifacts": {"maps": plot_paths, "save_dir": save_dir,  # Attach artifacts: map path list and output directory
                      "cell_scores": scores_path},  # Continued: per-cell score CSV (longitude,latitude,score)
        "parameters": {k: params.get(k) for k in  # Attach parameter echo (dict comprehension over these reproducibility-critical parameter names)
                       ("elements", "fusion_method", "interpolation", "zone_method",  # Four algorithm parameters
                        "selection_holdout", "select_only", "selection_seed")},  # Three experiment-selection parameters
        "selection": selection,  # Experiment-selection table (None when unused; holds candidate scores and the winner)
        "supports": ["prospectivity_mapping_done"],  # Attach the evidence tag: prospectivity mapping done
        "limitations": list(report.warnings[:4]) or ["空间统计关联不指明具体成矿机制"],  # Attach limitations: first 4 warnings, or the default wording when empty
        "warnings": list(report.warnings[:4]),  # Attach warnings: first 4
    }


def build_prospectivity_tool() -> ToolSpec:  # Define the main tool constructor
    """Build the geochemical anomaly-prospectivity family's main tool (complex task -> orchestrating multiple atomic algorithms)."""
    return ToolSpec(  # Return the main ToolSpec
        name="geochemical_prospectivity_mapping",  # Tool name: geochemical_prospectivity_mapping
        tool_family="geochemical_mapping",  # In the geochemical_mapping tool family
        description=("根据带坐标的地球化学数据生成异常识别与成矿远景预测图"  # First part of the description: functional overview
                     "（校验→异常识别→插值→融合→分级→成图）。"  # Continued: pipeline stages (validation->anomaly->interpolation->fusion->zoning->mapping)
                     "调用参数（均可选，不传=按默认/auto 处理）："  # Start of the parameter notes: stress optionality and defaults
                     "fusion_method 融合方法，取值 weights_of_evidence|information_value|"
                     "logistic|random_forest|knowledge_overlay|unsupervised_fusion|auto；"  # Six methods plus the auto fallback
                     "选型经验规则：已知矿点≥8 选 weights_of_evidence（可解释性最强的文献标准法）"
                     "或 random_forest（矿点充足时预测力通常更强、可解释性弱）；"  # Applicable conditions for the two supervised methods
                     "矿点 1~7 选 information_value（小样本更稳）；"  # Condition for few deposits
                     "无矿点但有外部地质证据层选 knowledge_overlay（可配 AHP 专家权重）；"  # Condition for unsupervised knowledge-driven selection
                     "纯化探无矿点选 unsupervised_fusion；拿不准用 auto（管线按矿点数自动选）。"  # auto fallback note
                     "interpolation 插值方法（默认 idw 反距离加权）；"  # Interpolation parameter note
                     "zone_method 远景区分级方法（默认 quantile 分位数法）；"  # Zonation parameter note
                     "elements 目标元素列表（默认按数据覆盖自动选）。"  # Elements parameter note
                     "有监督矿点时工具自动做 0.5° 空间 5 折交叉验证（cv_folds 可调、0 关闭），"  # Start of the internal-validation note
                     "模型评价以折外 CV 指标为准（含 CV-AUC 与折外捕获率），无需自行划分训练/验证集；"  # End of the internal-validation note
                     "实验选型：selection_holdout=N 表示扣 N 个已知矿点当小考、其余做监督，"  # Start of the selection parameters
                     "候选方法各试画一遍、按小考矿点圈中率选赢家，正式成图用赢家+全部矿点；"  # Selection mechanism note
                     "select_only=true 只返回选型表不正式成图（建议先看表再按赢家 fusion_method 正式调用）；"  # Two-stage suggestion
                     "selection_seed 抽样种子默认 42，selection_candidates 可指定候选方法列表。"),  # Seed and candidates note
        required_data=("geochemical_data", "spatial_coordinates"),  # Declare the two required data kinds: geochemical data and spatial coordinates
        capabilities=("anomaly_detection", "spatial_interpolation", "multi_element_fusion",  # First three capability tags: anomaly detection / spatial interpolation / multi-element fusion
                      "prospectivity_modeling", "map_rendering"),  # Last two capability tags: prospectivity modeling / map rendering
        output_artifacts=("anomaly_map", "prospectivity_map", "prediction_grid", "model_report"),  # Declare the four artifact kinds: anomaly map / prospectivity map / grid / report
        applicable=_applicable,  # Bind the data-condition check function
        run=_run_mapping,  # Bind the run function
    )
