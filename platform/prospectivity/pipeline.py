"""2D mineral prospectivity mapping — main orchestration of the two-stage pipeline (see the run_prospectivity_mapping example in the package-level __init__ for usage).
Stage boundary (core user requirement): Stage 1 spatialization (gridding) + anomaly extraction (evidence) produce "where
is anomalous" evidence layers carrying no mineralization semantics; Stage 2 evidence fusion + probability map +
classification (classify) answers "where may mineralize, where to prioritize follow-up". The coordinate convention matches
core.toolbox.spatial_stats exactly (collect_coordinates / element_series_for are reused directly), so the same data can be
cross-verified between the "anomaly analysis tool" and the "prospectivity mapping tool". No exceptions are raised: failures
in each stage are recorded in report.warnings; map rendering and table export are done by tool.py via plotting.
"""
from __future__ import annotations  # enable PEP 563 deferred annotation evaluation; type hints are not evaluated at runtime
import logging  # import the logging module
from typing import Any, Iterable  # import the Any and Iterable type hints
import numpy as np  # import numpy and alias it as np
from oregenesis.platform.models import SamplePoint as SampleData  # import the sample point model and alias it as SampleData
from oregenesis.platform.prospectivity.models import EvidenceLayer, ProspectivityReport  # import the evidence layer and overall report data classes
from oregenesis.platform.prospectivity import evidence as ev_mod  # import the Stage 1 anomaly extraction module aliased as ev_mod
from oregenesis.platform.prospectivity import fusion as fu_mod  # import the Stage 2 evidence fusion module aliased as fu_mod
from oregenesis.platform.prospectivity import classify as cl_mod  # import the classification/evaluation module aliased as cl_mod
from oregenesis.platform.spatial_stats import (  # import the spatial statistics convention shared with anomaly analysis (spans lines)
    collect_coordinates, collect_element_map, element_series_for)  # three functions: coordinate collection / element content mapping / single-element series
from oregenesis.platform.prospectivity.gridding import (  # import the spatialization function set (spans lines)
    fill_nan_nearest, idw_interpolate, make_grid, ordinary_kriging)  # NaN fill / IDW / grid construction / ordinary kriging

logger = logging.getLogger(__name__)  # create the module-level logger


def _select_elements(samples: list[SampleData], elements: Iterable[str] | None,  # define the participating-element selection function (parameters span lines)
                     coord_finite: np.ndarray) -> tuple[list[str], list[str]]:  # param: valid-coordinate mask; returns (element list, warnings)
    """Participating element selection: with explicit input, filter missing columns; otherwise auto-select (>= 8 valid values with >= 60% coverage among coordinated samples)."""
    warnings: list[str] = []  # initialize the warnings list
    elem_map = collect_element_map(samples)  # collect the element content map over all samples
    if elements:  # when an explicit element list is given
        chosen = [e for e in elements if e in elem_map]  # keep only elements actually present in the data
        missing = set(elements) - set(elem_map)  # set of requested elements missing from the data
        if missing: warnings.append(f"未找到元素列: {', '.join(sorted(missing))}")  # record the missing-element-columns warning
        if not chosen:  # explicit elements all missed (hallucinated column names / convention mismatch): parameter-tolerant fallback to auto selection so the tool does not fail outright
            warnings.append("显式元素全部未命中数据列，回退为按数据覆盖自动选择")  # record the fallback warning (goes into the payload for the Agent to see)
            for e, vals in elem_map.items():  # auto selection (same criteria as the auto branch below)
                vf = [v for v, f in zip(vals, coord_finite) if f and v is not None and np.isfinite(v)]  # valid contents among coordinated samples
                if len(vf) >= 8 and len(vf) / max(int(coord_finite.sum()), 1) >= 0.6:  # >= 8 valid values and >= 60% coverage
                    chosen.append(e)  # selected
    else:
        chosen = []  # initialize the auto-selected list
        for e, vals in elem_map.items():  # check availability element by element
            vf = [v for v, f in zip(vals, coord_finite) if f and v is not None and np.isfinite(v)]  # collect this element's valid contents among coordinated samples
            if len(vf) >= 8 and len(vf) / max(int(coord_finite.sum()), 1) >= 0.6:  # require >= 8 valid values and >= 60% of coordinated samples
                chosen.append(e)  # add to the auto-selected elements
    chosen = sorted(set(chosen))  # deduplicate and sort by element name
    if not chosen: warnings.append("没有可用的元素数据")  # record a warning when no elements are available
    return chosen, warnings  # return the selected elements and warnings


def _plane_proj(samples: list[SampleData], is_geo: bool) -> tuple[float, float, float]:  # define the geographic flattening-parameter function
    """Geographic flattening parameters (lon0, lat0, cos_lat), consistent with the anomaly convention (equirectangular approximation).
    The center must come from the raw sample coord_x/coord_y: collect_coordinates already returns planar km coordinates
    (mean ≈ 0); taking the center from it yields lon0=lat0=0 and deposits land thousands of km away (a hidden spatial misalignment).
    """
    if not is_geo: return 0.0, 0.0, 1.0  # non-geographic input: return neutral parameters (no shift)
    vals = [(float(s.coord_x), float(s.coord_y)) for s in samples  # collect raw lon/lat pairs of samples with coordinates (spans lines)
            if getattr(s, "coord_x", None) is not None and getattr(s, "coord_y", None) is not None]  # keep samples with both coord_x and coord_y
    good = np.isfinite(np.asarray(vals, dtype=float)).all(axis=1) if vals else np.array([], dtype=bool)  # row mask where both coordinates are finite (empty array for empty input)
    if not good.any(): return 0.0, 0.0, 1.0  # return neutral parameters when no valid coordinates
    arr = np.asarray(vals, dtype=float)  # convert coordinate pairs to an array
    lon0, lat0 = float(np.mean(arr[good, 0])), float(np.mean(arr[good, 1]))  # use the valid longitude/latitude means as the flattening origin
    return lon0, lat0, abs(np.cos(np.deg2rad(lat0))) or 1e-6  # return the origin and latitude cosine (fallback 1e-6 when zero)


def _parse_deposits(deposits: Iterable[Any] | None, lon0: float, lat0: float,  # define the deposit-input parsing function (parameters span lines)
                    cos_lat: float) -> np.ndarray:  # param: flattening datum; returns an (N,2) coordinate array
    """Parse deposit inputs → planar coordinate array (km); supports list[dict] and list[SampleData].
    dict keys support x/coord_x/lon/longitude and y/coord_y/lat/latitude; geographic deposits must be converted
    with the same (lon0, lat0) datum as the samples, otherwise they are spatially misaligned. Returns an (N,2) array (may be empty).
    """
    if not deposits:  # when the deposit input is empty
        return np.empty((0, 2), dtype=float)  # return an empty (0,2) float array
    pts: list[tuple[float, float]] = []  # initialize the planar coordinate list
    for d in deposits:  # parse deposit by deposit
        if isinstance(d, SampleData):  # when the deposit is a sample point object
            vx, vy, is_geo = d.coord_x, d.coord_y, bool(d.coord_is_geo)  # take coordinates and the geographic flag
        elif isinstance(d, dict):  # when the deposit is a dict
            vx = d.get("x", d.get("coord_x", d.get("lon", d.get("longitude"))))  # try x/coord_x/lon/longitude keys in order for the x coordinate
            vy = d.get("y", d.get("coord_y", d.get("lat", d.get("latitude"))))  # try y/coord_y/lat/latitude keys in order for the y coordinate
            _geo_keys = ("lon", "lat", "longitude", "latitude")  # geographic key set (including full-name aliases)
            is_geo = bool(d.get("is_geo", d.get("coord_is_geo", any(k in d for k in _geo_keys))))  # geographic flag (defaults to inferred from the presence of geographic keys)
        else:
            continue  # skip unsupported input types
        if vx is None or vy is None: continue  # skip when coordinates are missing
        fx, fy = float(vx), float(vy)  # convert coordinates to float
        if not (np.isfinite(fx) and np.isfinite(fy)): continue  # skip non-finite coordinates
        pts.append(((fx - lon0) * 111.32 * cos_lat, (fy - lat0) * 110.57) if is_geo else (fx, fy))  # convert geographic coordinates to planar km via the datum; planar coordinates used directly
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 2), dtype=float)  # convert to an (N,2) array and return ((0,2) when empty)


def _parse_external_layers(layers_in: Iterable[dict] | None, grid, lon0: float, lat0: float,  # define the external evidence layer input parsing function (parameters span lines)
                           cos_lat: float) -> tuple[list[EvidenceLayer], list[str]]:  # returns (evidence layer list, warnings list)
    """Parse external evidence layer inputs: each layer {"name","name_cn","type","coords","radius","weight","is_geo"}.
    "lines" polylines must be densely resampled into point strings first (no automatic thinning/densification); geographic coords are
    flattened with the same datum as the samples. A single layer's parse failure is recorded in warnings without aborting.
    """
    if not layers_in:  # when the external layer input is empty
        return [], []  # return empty lists
    out: list[EvidenceLayer] = []  # initialize the output evidence layer list
    warnings: list[str] = []  # initialize the warnings list
    for i, cfg in enumerate(layers_in):  # parse layer configs one by one
        try:  # exception guard for single-layer parsing
            lid = str(cfg.get("name", f"external_{i + 1}"))  # take the layer ID (defaults to external_N by index)
            coords = np.asarray(cfg.get("coords", []), dtype=float)  # convert point-string coordinates to a float array
            if coords.size == 0:  # when coordinates are empty
                warnings.append(f"外部证据层 {lid} 无坐标，跳过"); continue  # record the no-coordinate warning and skip
            if bool(cfg.get("is_geo", False)):  # flatten with the same datum as the samples
                coords = np.column_stack([(coords[:, 0] - lon0) * 111.32 * cos_lat,  # longitudinal flattening: lon difference × 111.32 × cos(latitude) (spans lines)
                                          (coords[:, 1] - lat0) * 110.57])  # latitudinal flattening: lat difference × 110.57
            ly = ev_mod.build_external_layer(lid, str(cfg.get("name_cn", lid)), coords, grid,  # call the evidence module to build the external distance evidence layer (spans lines)
                radius=float(cfg.get("radius", 2.0)), weight=float(cfg.get("weight", 1.0)))  # pass influence radius (default 2) and weight (default 1)
            if ly is not None: out.append(ly)  # collect the layer only when built successfully
        except Exception as e:  # catch single-layer parse failures
            warnings.append(f"外部证据层 {cfg.get('name', i)} 解析失败: {e}")  # record the parse-failure warning
    return out, warnings  # return the evidence layer list and warnings


def _build_summary(report: ProspectivityReport) -> str:  # define the report → Chinese text summary function
    """Report → Chinese text summary (overview → Stage 1 → Stage 2 → zoning → warnings), injectable into an Agent prompt."""
    lines = ["=" * 56, "二维成矿预测远景图分析报告", "=" * 56, ""]  # initialize the title, separators and blank line
    add = lines.append  # append shortcut (shortens line width only; output unchanged)
    add(f"样品 {report.n_samples} 个（带坐标 {report.n_coord_samples}），坐标系统 {report.coord_system or '-'}")  # overview: sample count, coordinated count and coordinate system
    add(f"分析元素 {len(report.elements)}: {', '.join(report.elements[:20])}"  # element line: list the first 20 analyzed elements (spans lines)
        + ("…" if len(report.elements) > 20 else ""))  # append an ellipsis when more than 20 elements
    if report.grid_def is not None:  # when a grid has been built
        g = report.grid_def  # alias the grid definition
        add(f"预测网格 {g.ny}×{g.nx}（单元 {g.cell_area():.4f} km²），插值方法 {report.interpolation}")  # output grid spec, cell area and interpolation method
    add(f"已知矿点 {report.n_deposits} 个"  # deposit line: known deposit count (spans lines)
        + ("（无监督/知识驱动模式）" if report.n_deposits == 0 else ""))  # annotate the unsupervised/knowledge-driven mode when there are zero deposits
    lines += ["", "## 阶段一：地球化学异常识别"]  # append the Stage 1 heading
    add(f"证据层 {len(report.evidence_layers)} 个"  # evidence layer statistics line opening (spans lines)
        f"（单元素 {len(report.layers_by('element'))}、多元素 {len(report.layers_by('multivariate'))}、"  # single-element and multivariate layer counts (spans lines)
        f"无监督 {len(report.layers_by('unsupervised'))}、外部 {len(report.layers_by('external'))}）")  # unsupervised and external layer counts, closing the line
    for ly in sorted(report.evidence_layers, key=lambda l: -l.n_anomaly_cells)[:5]:  # top 5 by anomaly cell count
        head = f"- {ly.name_cn}: 异常 {ly.n_anomaly_cells} 格点（{ly.area_ratio * 100:.1f}%）"  # assemble the layer entry: name / anomaly cells / area ratio
        if ly.threshold is not None: head += f"，阈值 {ly.threshold:.4g}"  # append the threshold text when present
        add(head)  # write the layer entry
    lines += ["", "## 阶段二：成矿远景预测"]  # append the Stage 2 heading
    if (m := report.model) is not None:  # when a fusion model exists (walrus expression binds m)
        add(f"融合方法: {m.name_cn}（{m.method}），输出{'成矿概率' if m.is_probability else '成矿有利度得分'}")  # output the fusion method and probability/favorability semantics
        if "cis_ratio" in m.diagnostics:  # when diagnostics include the conditional independence ratio
            add(f"条件独立性比率 {m.diagnostics['cis_ratio']:.2f}（理想≈1）"  # output the ratio line (ideal ≈ 1) (spans lines)
                f"，先验概率 {m.diagnostics.get('prior_prob', float('nan')):.5f}")  # append the prior probability (NaN default)
        if "auc" in (report.evaluation or {}) and (auc := report.evaluation["auc"]) == auc:  # filter out NaN
            add(f"模型 AUC = {auc:.3f}"  # output the AUC line (spans lines)
                + (f"，前 20% 面积捕获 {report.evaluation['capture_at_20pct'] * 100:.0f}% 矿点"  # append the top-20% area capture text (spans lines)
                   if report.evaluation.get("capture_at_20pct") is not None else ""))  # concatenate only when the metric exists
        cv = (report.evaluation or {}).get("cv") or {}  # out-of-fold cross-validation metrics (authoritative for evaluation; training-point AUC is reference only)
        if cv.get("mean_auc") is not None:  # output only when out-of-fold results exist
            add(f"空间 {cv['k']} 折交叉验证 AUC = {cv['mean_auc']:.3f}±{cv['std_auc']:.3f}"  # CV conclusion line (spans lines)
                + (f"，折外前 10% 面积捕获 {cv['cv_capture_at_10pct'] * 100:.0f}% 矿点格"  # continued: out-of-fold capture rate
                   if cv.get("cv_capture_at_10pct") is not None else "")  # concatenate only when present
                + "（折外=验证矿点未参与该折建模）")  # continued: convention note
        for nt in m.notes:  # iterate over model notes
            add(f"- {nt}")  # write each note
    if (z := report.zones) is not None and z.levels:  # when zoning results with level stats exist
        lines += ["", "## 远景区分级"]  # append the prospect zoning heading
        for lv in z.levels:  # output level by level
            row = (f"- {lv.name_cn}: {lv.area:.2f} km²（{lv.area_ratio * 100:.1f}%），"  # assemble the level row: name / area / ratio (spans lines)
                   f"概率 ≥{lv.prob_min:.3g}")  # append the probability lower bound
            if lv.capture_ratio is not None:  # when a capture ratio exists
                row += f"，捕获矿点 {lv.n_deposits} 个（{lv.capture_ratio * 100:.0f}%）"  # append captured deposit count and percentage
            add(row)  # write the level row
    if report.warnings:  # when warnings exist
        lines += ["", "注意:"]  # append the notes heading
        for w in report.warnings: add(f"- {w}")  # write each warning
    return "\n".join(lines)  # join the full text with newlines and return


def run_prospectivity_mapping(  # define the 2D prospectivity main orchestration function (parameters span multiple lines)
    samples: list[SampleData], elements: Iterable[str] | None = None,  # params: sample list and explicitly specified elements
    deposits: Iterable[Any] | None = None, external_layers: Iterable[dict] | None = None,  # params: deposit input and external evidence layer input
    fusion_method: str = "auto", interpolation: str = "idw", preprocess: str = "log",  # params: fusion method / interpolation method / preprocessing
    single_algorithms: Iterable[str] = ("median_mad", "quantile", "ca_fractal"),  # param: single-element anomaly algorithm list (three by default)
    use_multivariate: bool = True, use_unsupervised: bool = False, unsupervised_method: str = "iforest",  # params: multivariate layer switch / unsupervised switch / unsupervised method
    grid_nx: int = 100, grid_ny: int = 100, idw_power: float = 2.0, zone_method: str = "quantile",  # params: grid size / IDW power / zoning method
    zone_quantiles: tuple[float, float, float] = (0.60, 0.75, 0.90),  # param: zoning quantiles (C/B/A lower bounds)
    zone_thresholds: tuple[float, float, float] | None = None,  # param: manual zoning thresholds (optional)
    layer_weights: dict[str, float] | None = None, ahp_pairwise: np.ndarray | None = None,  # params: layer weights and the AHP pairwise matrix
    overlay_mode: str = "weighted_sum",  # param: knowledge overlay mode (weighted sum / product)
    cv_folds: int = 5,  # param: spatial cross-validation folds (0 = off; with deposits, each fold trains on the other folds and validates on itself)
) -> ProspectivityReport:  # return type: the full prospectivity report
    """Run the full 2D prospectivity pipeline and return a ProspectivityReport.
    0) coordinate check and flattening → element selection → grid building; 1) Stage 1: preprocessing + interpolation → single-element anomaly layers
    + multivariate combination layers (PCA/Mahalanobis) + optional unsupervised layer + external evidence layers; 2) nearest-value NaN fill of evidence layers →
    deposits rasterized to cells as deposit_mask; 3) fusion (auto picks the method by deposit count) → probability/favorability grid;
    4) prospect zone classification (A/B/C) + model evaluation (when deposits exist); 5) summary report + text summary.
    """
    report = ProspectivityReport(n_samples=len(samples))  # create the report object and record the total sample count
    if not samples:  # empty-sample guard
        report.warnings.append("没有样品数据"); return report  # record the warning and return early
    # 0. Coordinate preparation (reuses the anomaly convention so the two tools cross-verify)
    xa, ya, is_geo, coord_sys = collect_coordinates(samples)  # extract and flatten sample coordinates (reusing the shared convention)
    if xa is None:  # when valid coordinates are insufficient
        report.warnings.append("样品缺少平面坐标（coord_x/coord_y），无法进行成矿预测"); return report  # record the missing-coordinate warning and return
    coord_finite = np.isfinite(xa)  # compute the finite-coordinate mask
    report.n_coord_samples = int(coord_finite.sum())  # record the number of coordinated samples
    report.coord_system, report.is_geo = coord_sys, is_geo  # record the coordinate system description and geographic flag
    if int(coord_finite.sum()) < 8:  # when fewer than 8 coordinated samples
        report.warnings.append(f"带坐标样品仅 {report.n_coord_samples} 个（需 ≥8），无法空间预测")  # record the insufficient-samples warning
        return report  # return the report early
    lon0, lat0, cos_lat = _plane_proj(samples, is_geo)  # compute the deposit flattening datum (lon0, lat0, cos_lat)
    chosen, warns = _select_elements(samples, elements, coord_finite)  # select elements participating in the analysis
    report.warnings.extend(warns)  # merge element-selection warnings
    if not chosen: return report  # return early when no elements are available
    report.elements = chosen  # record the participating element list
    mx, my = xa[coord_finite], ya[coord_finite]; grid = make_grid(mx, my, nx=grid_nx, ny=grid_ny)  # prediction grid
    report.grid_def, report.interpolation = grid, interpolation  # record the grid definition and interpolation method
    # ── Sample support mask (data envelope masking): per-element cumulative true IDW interpolation support ──
    support = np.zeros((grid.ny, grid.nx), dtype=bool)  # validity-domain boolean grid (True = at least one element has interpolation support)
    # ── Stage 1 ① single element: preprocessing → interpolation → threshold anomaly → evidence layer ──
    algo_list = list(single_algorithms or ("median_mad", "quantile", "ca_fractal"))  # materialize the single-element algorithm list (defaults to the three when empty)
    elem_grids: dict[str, np.ndarray] = {}  # initialize the element → interpolated grid dict
    for elem in chosen:  # spatialize and extract anomalies element by element
        series = element_series_for(elem, samples)  # take this element's content series in sample order
        vals = np.asarray([series[i] if i < len(series) else np.nan  # assemble a content array matching the sample count (spans lines)
                           for i in range(len(samples))], dtype=float)  # fill NaN where the series is short
        v = vals[coord_finite]  # keep contents of coordinated samples only
        fin = np.isfinite(v)  # compute the finite-value mask
        if fin.sum() < 8: continue  # skip the element when fewer than 8 valid values
        # Preprocessing (after log/boxcox, thresholds and scores live in the transformed space; params already recorded)
        work, used_tf = ev_mod.preprocess_values(v, transform=preprocess)  # preprocess contents and record the transform actually used
        try:  # interpolation exception guard
            if interpolation == "ordinary_kriging":  # when ordinary kriging is specified
                g = ordinary_kriging(mx[fin], my[fin], work[fin], grid)  # ordinary kriging interpolation on valid points
            else:  # otherwise take the default IDW branch
                g = idw_interpolate(mx[fin], my[fin], work[fin], grid, power=idw_power)  # IDW interpolation on valid points (adjustable power)
        except Exception as e:  # catch interpolation failures
            report.warnings.append(f"元素 {elem} 插值失败: {e}"); continue  # record the interpolation-failure warning and skip
        elem_grids[elem] = g  # cache this element's interpolated grid for multivariate layers
        support |= np.isfinite(g)  # accumulate sample support: cells where this element's IDW truly interpolates join the validity domain
        for algo in algo_list:  # generate anomaly layers per single-element algorithm
            ly = ev_mod.element_anomaly_layer(elem, g, algorithm=algo,  # call evidence to build this algorithm's anomaly evidence layer (spans lines)
                                              use_log_hint=used_tf.startswith("log"), grid_def=grid)  # pass the log-transform hint and grid definition
            if ly is None: continue  # skip when generation fails (e.g. insufficient valid values)
            if ly.n_anomaly_cells == 0:  # zero-anomaly layers contribute nothing to fusion: warn explicitly and drop
                report.warnings.append(  # record the zero-anomaly drop warning (spans lines)
                    f"证据层 {ly.layer_id} 无任何异常格点（疑似多母体分布使阈值"  # warning text continued: suspected multi-population distribution
                    f"超过数据最大值），已剔除；该元素建议改用 C-A 分形法")  # warning text closing: suggest switching to the C-A fractal method
                continue  # drop this zero-anomaly layer
            ly.params["preprocess"] = used_tf; report.evidence_layers.append(ly)  # record the preprocessing mode and add the layer to the report
    # ② multivariate combination layers (PCA captures common enrichment / Mahalanobis captures combinatorial outliers; complementary)
    if use_multivariate and len(elem_grids) >= 3:  # build multivariate layers when enabled and >= 3 elements
        for ly in (ev_mod.pca_layer(elem_grids), ev_mod.mahalanobis_layer(elem_grids)):  # build the PCA and Mahalanobis layers in turn (spans lines)
            if ly is not None: report.evidence_layers.append(ly)  # add to the report only when built successfully
    # ③ unsupervised anomaly detection layer (optional)
    if use_unsupervised and len(elem_grids) >= 2:  # build the unsupervised layer when enabled and >= 2 elements
        uns = ev_mod.unsupervised_layer(elem_grids, method=unsupervised_method)  # train the unsupervised anomaly detection layer with the given method
        if uns is not None: report.evidence_layers.append(uns)  # add to the report only when built successfully
        else: report.warnings.append(f"无监督方法 {unsupervised_method} 不可用（需 scikit-learn）")  # on failure, record the scikit-learn-required warning
    # ④ external evidence layers (faults/intrusions etc.)
    ext_layers, ext_warns = _parse_external_layers(external_layers, grid, lon0, lat0, cos_lat)  # parse and build external evidence layers
    report.warnings.extend(ext_warns); report.evidence_layers.extend(ext_layers)  # merge external-layer warnings and collect the external layers
    if not report.evidence_layers:  # when not a single evidence layer exists
        report.warnings.append("阶段一未生成任何证据层（数据/插值/阈值均未产出异常），无法进行阶段二融合")  # record the Stage 1 failure warning
        return report  # return the report early
    # Unify evidence-layer NaNs: fill with the nearest valid value inside the sample support domain (ensures per-pixel aligned fusion);
    # restore NaN outside the domain — filled values in unsupported areas are extrapolation artifacts and must not enter fusion/zoning/mapping
    domain = support  # validity domain = cells where at least one element has interpolation support
    if domain.any():  # warn and mask only when any support exists (with none at all, fusion itself will fail)
        report.warnings.append(  # masking warning into the report (also into the tool payload's warnings/limitations)
            f"采样域掩膜：有效支撑 {int(domain.sum())}/{grid.nx * grid.ny} 格"
            f"（至少一个元素有插值支撑），域外无采样支撑，不参与融合/分级/成图")  # masking statistics warning
    for ly in report.evidence_layers:  # unify NaNs layer by layer
        if ly.grid is not None:  # continuous raster layer
            g, _ = fill_nan_nearest(ly.grid)  # fill with the nearest valid value inside the domain
            g[~domain] = np.nan  # restore NaN outside the domain (artifacts must not leak)
            ly.grid = g  # write the layer grid back
        if ly.binary is not None:  # binary raster layer
            b, _ = fill_nan_nearest(ly.binary)  # fill with the nearest valid value inside the domain
            b[~domain] = np.nan  # restore NaN outside the domain
            ly.binary = b  # write the layer's binary grid back
    # Rasterize deposits into cells (outside the grid = coordinate system/datum mismatch: silent clamping would fake deposit clustering; drop them to protect the statistics)
    dep_xy = _parse_deposits(deposits, lon0, lat0, cos_lat); n_outside = 0  # parse deposit planar coordinates and zero the outside counter
    report.n_deposits = int(dep_xy.shape[0]); deposit_mask = np.zeros(grid.ny * grid.nx, dtype=bool)  # record the deposit count and create the all-False cell mask
    inside_xy: list[tuple[float, float]] = []  # planar coordinates of deposits inside the survey area (for spatial fold-split cross-validation)
    for px, py in dep_xy:  # rasterize deposit by deposit
        if not (grid.xmin <= px <= grid.xmax and grid.ymin <= py <= grid.ymax):  # test whether the deposit falls outside the survey grid
            n_outside += 1; continue  # increment the outside counter and skip
        r, c = grid.to_cell(px, py)  # convert world coordinates to raster row/column indices
        deposit_mask[r * grid.nx + c] = True  # mark the flattened mask cell as deposit-bearing
        inside_xy.append((float(px), float(py)))  # record the in-grid deposit coordinate
    if n_outside:  # when outside deposits exist
        report.warnings.append(  # record the outside-deposit drop warning (spans lines)
            f"{n_outside}/{report.n_deposits} 个矿点坐标落在测区网格外"  # warning text continued: outside-deposit statistics
            "（请检查矿点坐标系与样品是否一致），已剔除")  # warning text closing: prompt to verify the coordinate system
        report.n_deposits -= n_outside  # subtract the outside deposits from the total
    report.deposit_cells = [(int(r), int(c)) for r, c in  # record the deposit cell (row,col) list (spans lines)
                            np.argwhere(deposit_mask.reshape(grid.ny, grid.nx))]  # take row/column indices from nonzero mask positions
    # ── Stage 2: AHP-derived weights (optional) → evidence fusion ──
    weights = layer_weights  # layer weights start from explicit user input
    if ahp_pairwise is not None and len(report.evidence_layers) >= 2:  # when an AHP pairwise matrix is given and there are >= 2 evidence layers
        try:  # AHP computation exception guard
            A = np.asarray(ahp_pairwise, dtype=float)  # convert the pairwise matrix to a float array
            # Order check: the pairwise matrix must cover all evidence layers, otherwise zip truncation silently mismatches weights
            if A.ndim != 2 or A.shape[0] != len(report.evidence_layers):  # when the matrix is not 2-D or its order mismatches the layer count
                raise ValueError(  # raise the order-mismatch error (spans lines)
                    f"判断矩阵为 {A.shape} 阶，与证据层数 {len(report.evidence_layers)} 不一致"  # error text continued: matrix order vs layer count
                    "（矩阵顺序须与证据层顺序一致，见报告证据层列表）")  # error text closing: order must match
            w_vec, cr = fu_mod.ahp_weights(A)  # compute the AHP weight vector and consistency ratio CR
            if cr > 0.1: report.warnings.append(f"AHP 判断矩阵一致性比率 CR={cr:.3f} > 0.1，权重可靠性存疑")  # warn that weights are questionable when CR > 0.1
            weights = {ly.layer_id: float(wi) for ly, wi in zip(report.evidence_layers, w_vec)}  # map the AHP-normalized weights by layer ID
            report.warnings.append(f"已使用 AHP 导出权重（CR={cr:.3f}，覆盖 {len(weights)} 层）")  # record the note that AHP-derived weights are in use
        except Exception as e:  # catch AHP computation failures
            report.warnings.append(f"AHP 权重计算失败，回退默认权重: {e}")  # record the failure-and-fallback warning
    model, fuse_warns = fu_mod.fuse(  # call the fusion entry point to fuse the evidence (spans lines)
        report.evidence_layers, deposit_mask if report.n_deposits else None,  # pass the evidence layers and deposit mask (None without deposits)
        method=fusion_method, weights=weights, overlay_mode=overlay_mode)  # pass method / weights / overlay mode
    report.warnings.extend(fuse_warns); report.model = model  # merge fusion warnings and store the fusion model
    if model is None or model.prob_grid is None:  # when fusion failed or there is no probability grid
        report.warnings.append("阶段二融合失败，未产出成矿概率图"); return report  # record the failure warning and return
    model.prob_grid = np.asarray(model.prob_grid, dtype=float)  # convert the probability grid to a float array
    model.prob_grid[~domain] = np.nan  # force NaN outside the domain: unsupported areas yield no probability/prospect zones
    try:  # prospect zone classification (A/B/C) + model evaluation (AUC/P-A curve, when deposits exist)
        report.zones = cl_mod.classify_zones(  # run A/B/C prospect zone classification (spans lines)
            model.prob_grid, grid, method=zone_method, quantiles=tuple(zone_quantiles),  # pass probability grid / grid / method / quantiles
            thresholds=zone_thresholds, deposit_mask=deposit_mask if report.n_deposits else None,  # pass manual thresholds and the deposit mask (None without deposits)
            supervised=bool(report.n_deposits))  # pass the supervised flag: without deposits, the level name is "异常区（无矿点验证）"
    except Exception as e:  # catch zoning failures
        report.warnings.append(f"远景区分级失败: {e}")  # record the zoning-failure warning
    if report.n_deposits:  # evaluate the model only when known deposits exist
        try:  # evaluation exception guard
            report.evaluation = cl_mod.evaluate_model(model.prob_grid, deposit_mask)  # compute AUC, P-A curve and other evaluation metrics
        except Exception as e:  # catch evaluation failures
            report.warnings.append(f"模型评估失败: {e}")  # record the evaluation-failure warning
    # ── Spatial K-fold cross-validation (cv_folds>1 and enough deposit cells to split): model evaluation lives inside the tool,
    #    each fold "trains on the other folds, validates on itself"; out-of-fold metrics avoid training-point overfitting inflation; papers should cite CV ──
    if cv_folds > 1 and len(inside_xy) >= cv_folds * 3 and report.evidence_layers:  # splitting requires at least 3 deposit cells per fold
        try:  # CV exception guard: failure only degrades to no CV metrics and never blocks the main flow
            blk_map: dict[tuple[int, int], int] = {}  # 0.5° spatial block → fold index (new blocks join folds round-robin)
            fold_of: list[int] = []  # fold index of each in-grid deposit
            for px, py in inside_xy:  # assign a fold deposit by deposit
                lon_i = lon0 + px / (111.32 * cos_lat)  # back-compute longitude from planar km (inverse of the forward flattening)
                lat_i = lat0 + py / 110.57  # back-compute latitude from planar km
                blk = (int(lon_i // 0.5), int(lat_i // 0.5))  # 0.5° lon/lat block index
                fold_of.append(blk_map.setdefault(blk, len(blk_map) % cv_folds))  # blocks join folds round-robin on first appearance; deposits in a block always share a fold
            fold_aucs: list[float] = []  # out-of-fold AUC per fold
            fold_pcts: list[float] = []  # score percentiles of each fold's validation deposit cells (pooled for the out-of-fold capture rate)
            used_folds = 0  # number of folds actually run
            for k in range(cv_folds):  # hold out fold by fold
                val_idx = [i for i, f in enumerate(fold_of) if f == k]  # indices of this fold's validation deposits
                if not val_idx: continue  # skip empty folds
                train_mask = deposit_mask.copy()  # training mask = copy of all deposit cells
                val_mask = np.zeros_like(deposit_mask)  # validation mask (this fold's deposit cells)
                for i in val_idx:  # rasterize validation deposits one by one
                    r_k, c_k = grid.to_cell(inside_xy[i][0], inside_xy[i][1])  # planar coordinates → row/column
                    val_mask[r_k * grid.nx + c_k] = True  # mark the validation cell
                    train_mask[r_k * grid.nx + c_k] = False  # remove this fold's cell from training
                if not train_mask.any(): continue  # skip the fold when training is emptied (theoretically impossible)
                model_k, _ = fu_mod.fuse(report.evidence_layers, train_mask,  # rebuild the model on the other folds (spans lines)
                                         method=fusion_method, weights=weights, overlay_mode=overlay_mode)  # exactly the same configuration as the main model
                if model_k is None or model_k.prob_grid is None: continue  # skip when this fold fails to build
                s_k = np.asarray(model_k.prob_grid, dtype=float).ravel()  # flatten the out-of-fold score grid
                fin_k = np.isfinite(s_k)  # valid-cell mask (NaN outside the domain)
                if not fin_k.any(): continue  # skip when no valid cells
                auc_k = cl_mod.evaluate_auc(s_k, val_mask)  # out-of-fold rank-sum AUC (ties get average ranks)
                if auc_k == auc_k: fold_aucs.append(float(auc_k))  # collect only non-NaN values
                srt = np.sort(s_k[fin_k])  # ascending copy of valid scores (for percentile lookup)
                vm = val_mask.ravel() & fin_k  # scoreable validation cells
                for v in s_k[vm]:  # take the score percentile of each validation cell
                    fold_pcts.append(float(np.searchsorted(srt, v, side="right")) / float(srt.size))  # the fraction <= v is the percentile
                used_folds += 1  # count the fold
            if used_folds and fold_pcts:  # write metrics only when at least one fold ran
                fp = np.asarray(fold_pcts)  # pool all out-of-fold percentiles
                report.evaluation = report.evaluation or {}  # default the evaluation dict
                report.evaluation["cv"] = {  # out-of-fold evaluation (alongside in-training AUC; external citations should use CV)
                    "k": used_folds,  # folds actually run
                    "mean_auc": round(float(np.mean(fold_aucs)), 4) if fold_aucs else None,  # mean out-of-fold AUC
                    "std_auc": round(float(np.std(fold_aucs)), 4) if fold_aucs else None,  # between-fold variability
                    "fold_auc": [round(a, 4) for a in fold_aucs],  # per-fold details
                    "cv_capture_at_10pct": round(float((fp >= 0.9).mean()), 4),  # out-of-fold top-10% area capture rate
                    "cv_capture_at_20pct": round(float((fp >= 0.8).mean()), 4),  # out-of-fold top-20% area capture rate
                    "n_val_cells": int(len(fold_pcts)),  # number of deposit cells used for validation
                    "scheme": f"{cv_folds} 折空间交叉验证：0.5° 经纬度块轮流入折，逐折用其余折建模、本折验证",  # scheme description
                }  # write the out-of-fold metrics back into the evaluation dict
        except Exception as e:  # CV failure does not affect the main flow
            report.warnings.append(f"空间交叉验证失败: {e}")  # record the failure warning
    report.summary_text = _build_summary(report); return report  # generate the text summary and return the full report
