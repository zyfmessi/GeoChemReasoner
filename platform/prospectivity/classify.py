"""2D prospectivity mapping — prospect zoning and model evaluation.

Zoning semantics (industry convention of prospectivity mapping): class A = the top
probability/favorability segment (default ≥P90, exploration targets, priority
verification); class B = the second segment (default P75~P90, significant potential
areas); class C = the elevated segment (default P60~P75, further-work areas). Zoning
only reflects "relative quality"; thresholds can be adjusted to the actual mining
district via the manual mode. Model evaluation: rank-sum AUC (normalized
Mann-Whitney U), the prediction-rate-area (P-A) curve (a curve far above the diagonal
= most deposits captured within a small area), and practical exploration-decision
metrics such as the "capture rate of the top 20% area".
"""
from __future__ import annotations  # enable PEP 563 postponed annotation evaluation; type hints not evaluated at runtime

import logging  # import the logging module

import numpy as np  # import the numpy numeric library aliased as np

from oregenesis.platform.prospectivity.models import GridDef, ZoneLevel, ZoneResult  # import the grid definition / level statistics / zoning result dataclasses

logger = logging.getLogger(__name__)  # create the module-level logger

# Level Chinese names (level 1/2/3 ↔ A/B/C): supervised version (deposit-verified, may use "prospect/exploration target" wording)
_LEVEL_CN = {1: "A类远景区（找矿靶区）", 2: "B类远景区（潜力区）", 3: "C类远景区（工作区）"}  # A/B/C level Chinese-name mapping table
# Unsupervised naming: without deposits, zoning is only quantile slicing of the interpolated fusion surface; "exploration target" semantics must not be used
_LEVEL_CN_UNSUPERVISED = {1: "A类地球化学异常区（无矿点验证）", 2: "B类地球化学异常区（无矿点验证）",  # unsupervised naming (spanning lines)
                          3: "C类地球化学异常区（无矿点验证）"}  # unsupervised naming continued


def _jenks_breaks(values: np.ndarray, n_classes: int = 4) -> list[float] | None:  # Jenks natural-breaks computation
    """Jenks natural breaks (1D k-means approximation): cluster into n_classes classes; midpoints of adjacent class centers are the breaks.
    sklearn is lazily imported; returns None on missing library / convergence failure
    (upstream falls back to the quantile method).
    """
    try:  # attempt KMeans clustering
        from sklearn.cluster import KMeans  # lazy import
        km = KMeans(n_clusters=n_classes, n_init=10, random_state=42).fit(values.reshape(-1, 1))  # KMeans clustering of 1D probability values into 4 classes (10 initializations)
        centers = np.sort(km.cluster_centers_.ravel())  # cluster centers from low to high
        return [float((centers[i] + centers[i + 1]) / 2) for i in range(n_classes - 1)]  # midpoints of adjacent class centers as the n−1 breaks
    except Exception as e:  # catch missing library or clustering failure
        logger.info("Jenks 断点计算失败，退回分位数分级: %s", e)  # log the fallback to quantile zoning
        return None  # return None; upstream falls back to the quantile method


def classify_zones(prob_grid: np.ndarray, grid_def: GridDef, method: str = "quantile",  # main prospect zoning function (parameters span multiple lines)
                   quantiles: tuple[float, float, float] = (0.60, 0.75, 0.90),  # parameter: C/B/A lower-bound quantiles (default 0.60/0.75/0.90)
                   thresholds: tuple[float, float, float] | None = None,  # parameter: manual threshold triple (optional)
                   deposit_mask: np.ndarray | None = None,  # parameter: deposit mask (optional)
                   supervised: bool = True) -> ZoneResult:  # parameter: supervised flag (pass False with no deposits → level names use anomaly-area wording)
    """Prospect zoning: valid cells of prob_grid are classified into A/B/C levels with statistics.
    quantile = quantiles (quantiles are the C/B/A lower bounds); manual = explicit
    thresholds=[t_c,t_b,t_a] (use when the probability semantics are clear); jenks =
    natural breaks (data-driven, more reasonable for multimodal distributions). zone_grid:
    0 = non-prospect, 1 = A, 2 = B, 3 = C.
    """
    prob = np.asarray(prob_grid, dtype=float)  # probability raster to a float array
    fin = np.isfinite(prob)  # compute the valid-cell mask
    vals = prob[fin]  # take the valid cells' probability values
    zone = np.zeros(prob.shape, dtype=int)  # initialize the zoning raster to all 0 (non-prospect)
    result = ZoneResult(n_valid_cells=int(fin.sum()))  # create the zoning result and record the valid cell count
    result.method = method  # record the zoning method ID
    result.name_cn = {"quantile": "分位数分级", "manual": "手动阈值分级",  # method → Chinese-name mapping start (spanning lines)
                      "jenks": "Jenks 自然断点分级"}.get(method, method)  # complete the mapping and take the Chinese name (unknown methods keep the original name)
    if vals.size < 50:  # fewer than 50 valid cells
        result.notes.append("有效像元过少，未分级")  # record the not-zoned note
        return result  # return the empty zoning result early
    if method == "manual" and thresholds is not None:  # manual mode with thresholds provided
        t_c, t_b, t_a = (float(t) for t in thresholds)  # unpack the three manual thresholds C/B/A
    elif method == "jenks":  # enter Jenks natural-breaks mode
        brks = _jenks_breaks(vals, n_classes=4)  # cluster the probability values into 4 classes for breaks
        if brks is None:  # missing library / convergence failure → fall back to quantiles
            method, result.method, result.name_cn = "quantile", "quantile", "分位数分级(降级)"  # degrade the method and name to quantile
            t_c, t_b, t_a = (float(np.quantile(vals, q)) for q in quantiles)  # re-take the thresholds from the C/B/A quantiles
        else:
            t_c, t_b, t_a = brks[0], brks[1], brks[2]  # take the first three breaks as the C/B/A lower bounds
    else:  # quantile (default, also the fallback for an invalid method)
        method = "quantile"  # force the method to quantile (fallback)
        t_c, t_b, t_a = (float(np.quantile(vals, q)) for q in quantiles)  # take the thresholds from the C/B/A quantiles
    # Tie protection: highly discrete probabilities (e.g. many zero probabilities from
    # random forest) degrade the quantiles (t_c==t_b==t_a, class A would swallow all
    # valid cells); re-take the quantiles on non-zero probabilities — zero-probability
    # cells naturally fall into the non-prospect class
    if not (t_c < t_b < t_a):  # thresholds not strictly increasing (tie degradation)
        pos = vals[vals > 1e-12]  # take the non-zero (>1e-12) probability subset
        if pos.size >= 50:  # at least 50 non-zero cells
            t_c, t_b, t_a = (float(np.quantile(pos, q)) for q in quantiles)  # re-take the C/B/A quantiles on non-zero probabilities
            result.notes.append("概率并列严重（大量零值），已对非零概率重取分位数")  # record the note that quantiles were re-taken on non-zero probabilities
        else:
            result.notes.append("概率并列严重且非零像元不足，分级仅供参考")  # record the note of insufficient non-zero cells, for reference only
    # Ultimate fallback (rank zoning): if still tied, split by the rank percentile of
    # probability among valid cells — never degrades
    if not (t_c < t_b < t_a):  # still tied/degenerate: enter the rank-zoning fallback
        frac_flat = (np.argsort(np.argsort(vals)).astype(float)  # double argsort of valid-cell probabilities for ranks (derivation spans lines)
                     / max(len(vals) - 1, 1))  # normalize to [0,1] rank percentiles (length = valid cell count)
        frac = np.full(prob.shape, np.nan)  # all-NaN raster (out-of-domain/invalid cells excluded from rank zoning)
        frac[fin] = frac_flat  # backfill rank percentiles to valid cells only (vals and fin correspond one-to-one)
        zone[fin & (frac >= quantiles[0]) & (frac < quantiles[1])] = 3  # cells with rank in [P60,P75) assigned class C
        zone[fin & (frac >= quantiles[1]) & (frac < quantiles[2])] = 2  # cells with rank in [P75,P90) assigned class B
        zone[fin & (frac >= quantiles[2])] = 1  # cells with rank ≥P90 assigned class A
        result.zone_grid = zone  # save the rank-zoning raster
        result.notes.append("概率高度并列，已按秩百分位分级（P60/P75/P90 切分）")  # record the note of rank-percentile zoning
        _fill_levels(result, zone, grid_def, deposit_mask,  # fill per-level statistics (parameters span lines)
                     prob_bounds={1: (quantiles[2], 1.0), 2: (quantiles[1], quantiles[2]),  # rank probability intervals of the A/B levels (spanning lines)
                                  3: (quantiles[0], quantiles[1])},  # rank probability interval of the C level
                     bounds_label="秩区间", supervised=supervised)  # interval labeled as rank interval
        return result  # return the rank-zoning result
    zone[fin & (prob >= t_c) & (prob < t_b)] = 3  # class C
    zone[fin & (prob >= t_b) & (prob < t_a)] = 2  # class B
    zone[fin & (prob >= t_a)] = 1  # class A (same-cell override: higher level wins)
    result.zone_grid = zone  # save the zoning raster
    _fill_levels(result, zone, grid_def, deposit_mask,  # fill per-level statistics (parameters span lines)
                 prob_bounds={1: (t_a, 1.0), 2: (t_b, t_a), 3: (t_c, t_b)},  # A/B/C level probability intervals (t_a~1 / t_b~t_a / t_c~t_b)
                 supervised=supervised)  # pass the supervised flag (switches level naming)
    result.notes.append("分级下界 C/B/A = " + "/".join(f"{t:.3g}" for t in (t_c, t_b, t_a)))  # record the three lower-bound note
    return result  # return the zoning result


def _fill_levels(result: ZoneResult, zone: np.ndarray, grid_def: GridDef,  # zoning statistics fill function (parameters span multiple lines)
                 deposit_mask: np.ndarray | None, prob_bounds: dict[int, tuple[float, float]],  # parameters: deposit mask and per-level probability interval dict
                 bounds_label: str = "", supervised: bool = True) -> None:  # parameters: interval label (non-empty means rank zoning) and supervised flag
    """Fill zoning statistics: per-level cell count/area (km²)/ratio and deposit capture; non-empty bounds_label means rank-zoning mode."""
    cell_area = grid_def.cell_area()  # take the cell area (km²)
    names = _LEVEL_CN if supervised else _LEVEL_CN_UNSUPERVISED  # level naming table: supervised = prospect / unsupervised = anomaly area
    n_valid = max(result.n_valid_cells, 1)  # valid cell count (floored at 1 against division by zero)
    n_dep_total = int(np.asarray(deposit_mask, dtype=bool).sum()) if deposit_mask is not None else None  # total deposit count (None without a mask)
    dm = np.asarray(deposit_mask, dtype=bool) if deposit_mask is not None else None  # deposit mask as bool (None without a mask)
    for lv in (1, 2, 3):  # statistics per A/B/C level
        m = zone == lv  # boolean mask of this level's cells
        n_cells = int(m.sum())  # count this level's cells
        lo, hi = prob_bounds[lv]  # take this level's probability lower/upper bounds
        lvl = ZoneLevel(level=lv, name_cn=names[lv], prob_min=lo, prob_max=hi,  # build the level statistics object (parameters span lines)
                        n_cells=n_cells, area=n_cells * cell_area, area_ratio=n_cells / n_valid)  # fill cell count/area/area ratio
        if bounds_label:  # rank zoning: each level's prob interval displayed as a rank interval
            lvl.name_cn += f"（{bounds_label} {lo:.0%}~{hi:.0%}）" if lv != 1 else f"（{bounds_label} ≥{lo:.0%}）"  # append the rank-interval label to the name (class A shows ≥ lower bound)
        if deposit_mask is not None and n_dep_total:  # compute capture statistics when deposit data exists
            lvl.n_deposits = int((m.ravel() & dm).sum())  # deposit count within this level
            lvl.capture_ratio = lvl.n_deposits / n_dep_total  # capture ratio = this level's deposits / total deposits
        result.levels.append(lvl)  # append this level's statistics to the result


def evaluate_auc(score: np.ndarray, deposit_mask: np.ndarray) -> float:  # rank-sum AUC computation
    """Rank-sum AUC: area under the ROC curve = normalized Mann-Whitney U statistic (no ROC traversal needed)."""
    s = np.asarray(score, dtype=float)  # scores to a float array
    d = np.asarray(deposit_mask, dtype=bool)  # deposit mask to bool
    fin = np.isfinite(s)  # compute the finite-value mask
    s, d = s[fin], d[fin]  # keep only valid samples' scores and labels
    n_pos, n_neg = int(d.sum()), int((~d).sum())  # count positive/negative samples
    if n_pos == 0 or n_neg == 0: return float("nan")  # return NaN when either class is missing
    order = np.argsort(s, kind="mergesort")  # stable mergesort for rank-order indices
    ranks = np.empty(len(s), dtype=float)  # initialize the rank array
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)  # write ranks 1..n by sorted position
    sorted_s = s[order]  # take the sorted scores
    i = 0  # tie-segment scan start index
    while i < len(sorted_s):  # ties take average ranks (ties are common after probability discretization; otherwise AUC is underestimated)
        j = i  # segment-end cursor starts at the segment start
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:  # scan forward through the equal-valued tie segment
            j += 1  # extend the segment-end index
        if j > i:  # this segment has ties
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0  # tied entries uniformly take the average rank
        i = j + 1  # jump to the next segment start
    rank_sum = float(ranks[d].sum())  # sum the ranks of positive samples
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)  # normalized U statistic gives the AUC


def prediction_area_curve(score: np.ndarray, deposit_mask: np.ndarray,  # prediction-rate-area (P-A) curve function (parameters span lines)
                          n_points: int = 40) -> tuple[np.ndarray, np.ndarray]:  # parameter: curve sample point count (default 40)
    """Prediction-rate-area (P-A) curve: the relation between cumulative area fraction x and captured deposit fraction y
    when sorting by probability descending. An ideal model's curve lies far above the
    diagonal (most deposits captured within a small area); returns n_points evenly
    spaced sample points.
    """
    s = np.asarray(score, dtype=float)  # scores to a float array
    d = np.asarray(deposit_mask, dtype=bool)  # deposit mask to bool
    fin = np.isfinite(s)  # compute the finite-value mask
    s, d = s[fin], d[fin]  # keep only valid samples
    n_dep = int(d.sum())  # count total deposits
    if n_dep == 0 or s.size == 0: return np.array([]), np.array([])  # return empty arrays when there are no deposits or no samples
    order = np.argsort(s)[::-1]  # indices sorted by score descending
    d_sorted = d[order]  # deposit flags reordered by descending score
    cum_area = np.arange(1, s.size + 1) / s.size  # cumulative area fraction series (1..n)/n
    cum_dep = np.cumsum(d_sorted) / n_dep  # cumulative deposit-capture fraction series
    idx = np.unique(np.clip(np.linspace(0, s.size - 1, n_points).astype(int), 0, s.size - 1))  # take 40 evenly spaced indices (clipped and deduplicated)
    return cum_area[idx], cum_dep[idx]  # return the sampled (cumulative area, cumulative capture) series


def evaluate_model(prob_grid: np.ndarray, deposit_mask: np.ndarray) -> dict:  # model evaluation main entry
    """Model evaluation main entry: AUC + P-A curve + "capture rate of the top 20% area" (a direct exploration-decision metric)."""
    # unify by flattening: prob_grid may be (ny,nx), deposit_mask may already be flat (ny*nx,)
    s = np.asarray(prob_grid, dtype=float).ravel()  # flatten the probability raster (or vector)
    d = np.asarray(deposit_mask, dtype=bool).ravel()  # flatten the deposit mask as bool
    fin = np.isfinite(s)  # compute the finite-value mask
    s, d = s[fin], d[fin]  # keep only valid samples
    out: dict = {"auc": float("nan"), "n_pos": int(d.sum()), "n_valid": int(s.size)}  # initialize the result: AUC/positive count/valid cell count
    if d.sum() == 0 or (~d).sum() == 0: return out  # missing positive or negative class: return the initial result directly
    out["auc"] = round(evaluate_auc(s, d), 4)  # compute and record the AUC (4 decimals)
    pa_x, pa_y = prediction_area_curve(s, d)  # compute the P-A curve sample points
    out["pa_area_frac"] = [round(float(v), 4) for v in pa_x]  # record the cumulative area fraction series
    out["pa_capture_frac"] = [round(float(v), 4) for v in pa_y]  # record the cumulative capture fraction series
    if pa_x.size:  # capture rate of the top 20% area: y at x=0.2 on the P-A curve
        k = int(np.searchsorted(pa_x, 0.20))  # binary search for the position index of x=0.2
        out["capture_at_20pct"] = round(float(pa_y[min(k, pa_y.size - 1)]), 4)  # record the top-20%-area capture rate (4 decimals)
    return out  # return the evaluation result dict
