"""2D mineral prospectivity mapping — Stage 1: geochemical anomaly identification (spatialized raster → evidence layers).

Anomalies are delineated on spatialized rasters and uniformly wrapped as EvidenceLayer (grid: continuous anomaly intensity
0~1 for weighted overlay / ML features; binary: binary evidence 1=anomaly for weights-of-evidence etc.) as the standard
Stage 2 fusion input. Stage boundary (this package's core design constraint): this module only answers "where is
anomalous"; EvidenceLayer carries no mineralization probability semantics — "where may mineralize" must be answered by
Stage 2 evidence fusion. Algorithm families: single-element median_mad / mean_2sigma / quantile / ca_fractal (C-A fractal),
multi-element pca_score / mahalanobis, unsupervised iforest / lof / ocsvm (sklearn lazily imported; missing library
degrades to returning None), and external evidence point strings → distance field → buffered classification.
"""
from __future__ import annotations  # enable PEP 563 deferred annotation evaluation; type hints are not evaluated at runtime

import logging  # import the logging module

import numpy as np  # import numpy and alias it as np

from oregenesis.platform.prospectivity.gridding import distance_grid  # import the distance-grid function distance_grid
from oregenesis.platform.prospectivity.models import EvidenceLayer, GridDef  # import the evidence layer and grid definition data classes

logger = logging.getLogger(__name__)  # create the module-level logger

# Algorithm ID → Chinese name (written into EvidenceLayer.name_cn / statistics tables)
ALGO_CN = {  # define the algorithm ID → Chinese name mapping dict
    "median_mad": "中位数+k×MAD", "mean_2sigma": "均值+k×σ（迭代）",  # entries: median+k×MAD and iterative mean+k×σ threshold methods
    "quantile": "高百分位", "ca_fractal": "含量-面积(C-A)分形",  # entries: high percentile and concentration-area (C-A) fractal methods
    "pca_score": "主成分组合异常", "mahalanobis": "稳健马氏距离",  # entries: PCA and robust Mahalanobis distance multivariate methods
    "iforest": "孤立森林", "lof": "局部离群因子", "ocsvm": "单类支持向量机",  # entries: isolation forest / local outlier factor / one-class SVM
    "distance": "距离缓冲",  # entry: distance buffering for external evidence
}


def _lyr(layer_id, name_cn, kind, algorithm_id, elements, grid, binary, threshold,  # define the unified evidence layer construction helper (parameters span lines)
         **extra) -> EvidenceLayer:  # other keyword arguments passed through as-is; returns an evidence layer
    """Unified EvidenceLayer construction: anomaly cell count / area ratio computed centrally (NaN-safe); the rest passed through."""
    return EvidenceLayer(  # construct and return the evidence layer instance
        layer_id=layer_id, name_cn=name_cn, kind=kind, algorithm_id=algorithm_id, elements=elements,  # pass through layer ID / Chinese name / kind / algorithm / elements
        grid=grid, binary=binary, threshold=threshold, n_anomaly_cells=int(np.nansum(binary)),  # pass through the raster and count anomaly cells treating NaN as 0
        area_ratio=float(np.nansum(binary) / max(np.isfinite(binary).sum(), 1)), **extra)  # anomaly area ratio = anomaly cells / valid cells (floor 1 guards division by zero)


def preprocess_values(vals: np.ndarray, transform: str = "log") -> tuple[np.ndarray, str]:  # define the content preprocessing function: log/boxcox/robust_z/none
    """Content preprocessing (NaN-safe): log/boxcox/robust_z/none → (processed array, actual transform name).
    Auto-degrades when conditions are not met; pipeline calls it per element before interpolation (log/boxcox) and uses robust_z before stacking.
    """
    a = np.asarray(vals, dtype=float); fin = np.isfinite(a)  # convert input to a float array and compute the finite mask
    if fin.sum() < 3: return a, "none"  # too few valid values: return as-is
    work = a.copy()  # copy the array as the working buffer
    if transform == "log":  # enter the log-transform branch
        if np.any(work[fin] <= 0):  # shift non-positive values above the minimum positive before logging (below-detection-limit values are often 0)
            shift = float(np.nanmin(work[fin][work[fin] > 0])) if np.any(work[fin] > 0) else 1e-6  # shift amount = minimum positive value (1e-6 when no positives)
            work = np.where(fin, work + shift, work)  # shift all finite values so everything is positive
            return np.where(fin, np.log10(np.maximum(work, 1e-12)), np.nan), "log(+shift)"  # take the base-10 log (floor 1e-12), tagged log(+shift)
        return np.where(fin, np.log10(np.maximum(work, 1e-12)), np.nan), "log10"  # all positive: take log10 directly and return
    if transform == "boxcox":  # enter the Box-Cox branch
        try:  # try Box-Cox
            from scipy.stats import boxcox  # lazy import
            if np.any(work[fin] <= 0):  # when non-positive values exist
                work = np.where(fin, work - float(np.nanmin(work[fin])) + 1e-6, work)  # shift everything to min + 1e-6 to satisfy positivity
            fitted, lam = boxcox(work[fin])  # Box-Cox the finite values to get transformed values and λ
            out = np.full_like(work, np.nan); out[fin] = fitted  # initialize all-NaN and backfill the transformed values
            return out, f"boxcox(λ={lam:.3f})"  # return the result tagged with the λ value
        except Exception as e:  # scipy missing / fit failure → degrade to log
            logger.info("Box-Cox 失败，降级 log10: %s", e)  # log the Box-Cox failure and degradation
            return preprocess_values(a, "log")  # recursively fall back to the log transform
    if transform == "robust_z":  # median + 1.4826·MAD robust standardization
        med = float(np.nanmedian(work[fin])); mad = float(np.nanmedian(np.abs(work[fin] - med))) * 1.4826  # compute the median and the 1.4826×MAD robust scale
        if mad < 1e-12:  # when the robust scale is near zero
            mad = float(np.nanstd(work[fin])) or 1.0  # fall back to the finite-value std (1 when 0)
        return np.where(fin, (work - med) / mad, np.nan), "robust_z"  # robust z standardization (NaN at invalid positions) and return
    return work, "none"  # unknown transform name: return as-is


def robust_z(col: np.ndarray) -> np.ndarray:  # define the robust z-score function
    """Robust z-score (median + 1.4826·MAD scaling): per-element dimensional unification before multi-element stacking."""
    a = np.asarray(col, dtype=float)  # convert input to a float array
    fin = np.isfinite(a)  # compute the finite mask
    if fin.sum() < 3: return np.zeros_like(a)  # fewer than 3 valid values: return all zeros
    med = float(np.median(a[fin])); mad = float(np.median(np.abs(a[fin] - med))) * 1.4826  # compute the median and the 1.4826×MAD scale
    return (a - med) / (mad if mad > 1e-12 else (float(np.std(a[fin])) or 1.0))  # subtract the median and divide by the scale (fall back to std/1 when the scale fails)


def _normalize01(v: np.ndarray) -> np.ndarray:  # define the min-max normalization helper
    """min-max normalization to [0,1] (NaN-safe; all-equal input returns a constant 0.5 surface)."""
    a = np.asarray(v, dtype=float)  # convert input to a float array
    fin = np.isfinite(a)  # compute the finite mask
    if fin.sum() == 0: return a  # all NaN: return as-is
    lo, hi = float(np.nanmin(a[fin])), float(np.nanmax(a[fin]))  # take the min/max of finite values
    if hi - lo < 1e-12:  # all equal → constant 0.5
        out = np.full_like(a, 0.5); out[~fin] = np.nan; return out  # return the constant 0.5 surface (invalid positions stay NaN)
    return np.where(fin, (a - lo) / (hi - lo), np.nan)  # linearly normalize finite values to [0,1]


def _to_full(ok: np.ndarray, values: np.ndarray) -> np.ndarray:  # define the helper that back-fills valid-pixel results to full length
    """Spread the 1-D result on valid pixels back to the full-length (ny*nx,) array; invalid pixels get NaN."""
    full = np.full(ok.size, np.nan); full[ok] = values  # initialize all-NaN and fill values at valid pixels
    return full  # return the full-length 1-D array


def _ca_fractal_threshold(grid_vals: np.ndarray) -> dict | None:  # define the C-A fractal break-point threshold estimator
    """Raster C-A fractal break-point threshold: A(ρ≥v) ∝ v^(-α); a two-segment fit of the log-log cumulative area curve picks the break point."""
    gv = grid_vals[np.isfinite(grid_vals)]  # take the raster's finite pixels
    if gv.size < 100: return None  # fewer than 100 valid pixels: return None
    gv_sorted = gv[np.argsort(gv)[::-1]]; total = gv_sorted.size  # descending: the top i cells = the set with concentration >= the i-th largest value
    log_v = np.log10(np.maximum(gv_sorted, np.finfo(float).eps))  # log10 of descending concentrations (floor guards log(0))
    log_s = np.log10(np.arange(1, total + 1) / total)  # log10 of the cumulative area fraction (i-th largest / total)
    max_area_idx = max(int(total * 0.40), 10)  # geologic constraint: anomaly area <= 40%
    cand = np.unique(np.clip(  # generate candidate break-point indices and deduplicate (spans lines)
        np.linspace(max(3, int(total * 0.005)), max_area_idx, num=80).astype(int),  # 80 candidate indices evenly spaced within the 0.5%~40% area range
        3, max_area_idx))  # clamp candidates to the [3, 40% area] range
    best = None  # initialize the best break-point record
    for bi in cand:  # break-point enumeration: two-segment linear least squares; smallest SSE wins
        if bi < 5 or total - bi < 5:  # skip a candidate when either segment has fewer than 5 samples
            continue  # move to the next candidate
        with np.errstate(all="ignore"):  # temporarily suppress numpy numerical warnings
            b1 = np.polyfit(log_v[:bi], log_s[:bi], 1); b2 = np.polyfit(log_v[bi:], log_s[bi:], 1)  # fit a degree-1 least-squares polynomial on each side of the break point
            e1 = float(np.sum((log_s[:bi] - np.polyval(b1, log_v[:bi])) ** 2))  # compute the high-concentration segment SSE
            e2 = float(np.sum((log_s[bi:] - np.polyval(b2, log_v[bi:])) ** 2))  # compute the low-concentration segment SSE
        if np.isfinite(e1 + e2) and (best is None or e1 + e2 < best[0]):  # update the best when the total SSE is finite and smaller
            best = (e1 + e2, bi, float(b1[0]), float(b2[0]))  # record (total SSE, break-point index, both segment slopes)
    if best is None: return None  # no valid break point: return None
    _, bi, slope_high, slope_low = best  # unpack the best break-point index and both slopes
    return {"threshold_work": float(10 ** log_v[bi]),  # break-point concentration (working space)
            "break_index": int(bi), "slope_high": slope_high, "slope_low": slope_low,  # return the break-point index and both slopes
            "break_area_frac": float(bi / total)}  # return the cumulative area fraction at the break point


def element_anomaly_layer(element: str, grid_vals: np.ndarray, algorithm: str = "median_mad",  # define the single-element anomaly → evidence layer function (parameters span lines)
                          k: float = 2.0, quantile_q: float = 0.95, use_log_hint: bool = True,  # params: threshold multiplier k, quantile q, log-transform hint
                          grid_def: GridDef | None = None) -> EvidenceLayer | None:  # param: grid definition; returns an evidence layer or None
    """Single-element anomaly → evidence layer: background and anomaly threshold on the working raster (threshold in the working space).
    median_mad = median + k×1.4826·MAD; mean_2sigma = iterative ±kσ trimming; quantile = high percentile;
    ca_fractal = C-A fractal break point. Fewer than 100 valid values returns None.
    """
    gv = np.asarray(grid_vals, dtype=float)  # convert the working raster to a float array
    fin = np.isfinite(gv)  # compute the finite mask
    if fin.sum() < 100:  # when there are fewer than 100 valid values
        logger.info("元素 %s 栅格有效值不足（%d），跳过异常层", element, int(fin.sum())); return None  # log and return None to skip this layer
    v = gv[fin]  # take the finite values as a 1-D array
    if algorithm == "median_mad":  # enter the median + k×MAD branch
        bg = float(np.median(v))  # background = median
        mad = float(np.median(np.abs(v - bg))) * 1.4826  # robust scale = 1.4826×MAD
        thr = bg + k * (mad if mad > 1e-12 else (float(v.std()) or 1.0)); params = {"k": k}  # anomaly threshold = background + k×scale (falls back to std when degenerate); record params
    elif algorithm == "mean_2sigma":  # iteratively trim the high tail before estimating the background
        w = v.copy()  # working copy for iterative trimming
        for _ in range(6):  # at most 6 iterations
            m, s = float(w.mean()), float(w.std())  # compute the current mean and std
            if s <= 1e-12:  # when the std is near zero
                break  # end the iteration early
            kept = w[np.abs(w - m) <= k * s]  # keep only values within mean ± kσ
            if kept.size == w.size or kept.size < max(3, int(w.size * 0.5)):  # stop when nothing is trimmed or fewer than half remain
                break  # end the iteration
            w = kept  # next round on the cleaned subset
        bg, s = float(w.mean()), float(w.std()); thr = bg + k * s; params = {"k": k}  # background = cleaned mean, threshold = +k×std; record params
    elif algorithm == "quantile":  # enter the high-percentile branch
        bg = float(np.median(v)); thr = float(np.quantile(v, quantile_q)); params = {"q": quantile_q}  # background = median, threshold = given quantile; record params
    elif algorithm == "ca_fractal":  # enter the C-A fractal branch
        ca = _ca_fractal_threshold(gv)  # compute the raster C-A fractal break-point threshold
        if ca is None: return None  # return None when the break point cannot be estimated
        bg, thr = float(np.median(v)), ca["threshold_work"]  # background = median, threshold = break-point concentration
        params = {kk: ca[kk] for kk in ("break_index", "slope_high", "slope_low", "break_area_frac")}  # extract break-point index / slopes / area fraction into params
    else:
        logger.warning("未知单元素算法: %s", algorithm); return None  # warn on unknown algorithm and return None
    # Zero-anomaly degeneracy: still return the full layer with the reason annotated (an early None would lose the drop reason); pipeline filters and warns
    degenerate = not bool((v > thr).any())  # test for zero anomalies (no pixel exceeds the threshold)
    if degenerate:  # on zero-anomaly degeneracy
        logger.info("元素 %s 算法 %s 阈值 %.4g 超过数据最大值 %.4g（疑似多母体分布），零异常",  # log the zero-anomaly warning (parameters span lines)
                    element, algorithm, thr, float(v.max()))  # pass element name / algorithm / threshold and data maximum
    # Continuous-score contrast normalization: background→0, P99→1, preserving the anomaly intensity gradient for weighted overlay
    denom = max(float(np.quantile(v, 0.99)) - bg, 1e-12)  # normalization denominator = P99 − background (floor 1e-12 guards against 0)
    score = np.where(fin, np.clip((gv - bg) / denom, 0.0, 1.0), np.nan)  # continuous score = (value − background)/denominator clipped to [0,1]; NaN at invalid positions
    binary = np.where(fin, (gv > thr).astype(float), np.nan)  # binary evidence = value > threshold (1/0); NaN at invalid positions
    notes = []  # initialize the notes list
    if degenerate:  # add an explanatory note on zero anomalies
        notes.append(f"阈值 {thr:.4g} 超过数据最大值 {float(v.max()):.4g}："  # append the threshold-above-maximum explanation (spans lines)
                     "疑似双峰/多母体分布拉大离散度，建议改用 C-A 分形法")  # string continuation: suspected multi-population distribution, suggest the C-A fractal method
    if algorithm == "ca_fractal":  # C-A method: add two extra notes
        notes.append(f"C-A 拐点面积占比 {params['break_area_frac'] * 100:.1f}%")  # append the break-point area fraction note
        notes.append(f"分段斜率 {params['slope_high']:.2f}/{params['slope_low']:.2f}")  # append the two-segment slope note
    elif algorithm == "quantile":  # quantile method: add a note
        notes.append(f"阈值=P{quantile_q * 100:.0f}")  # append the threshold percentile (e.g. P95) note
    else:
        notes.append(f"k={k:g}")  # other algorithms: append the k multiplier note
    return _lyr(f"{element}_{algorithm}", f"{element} {ALGO_CN.get(algorithm, algorithm)}异常",  # construct and return the evidence layer via the helper (parameters span lines)
                "element", algorithm, [element], score, binary, float(thr),  # kind / algorithm / element list / continuous score / binary / threshold
                background=float(bg), params={**params, "use_log_hint": use_log_hint}, notes=notes)  # attach background, algorithm params (with log hint) and notes


def _stack_features(elem_grids: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[str]]:  # define the multi-element feature stacking helper
    """Stack per-pixel robust_z of multiple elements column-wise → (X valid-pixel feature matrix, valid mask, element names)."""
    names = list(elem_grids.keys())  # take the element name list
    mats, ok = [], None  # initialize the feature-column list and the common valid mask
    for name in names:  # process element by element
        z = robust_z(np.asarray(elem_grids[name], dtype=float).ravel())  # robust-z standardize the flattened raster
        fin = np.isfinite(z)  # compute this element's valid-pixel mask
        ok = fin if ok is None else (ok & fin)  # intersection of pixels valid for every element
        mats.append(z)  # collect this element's feature column
    return np.column_stack(mats)[ok], ok, names  # stack column-wise and keep commonly valid rows; return feature matrix / mask / element names


def pca_layer(elem_grids: dict[str, np.ndarray], var_ratio_min: float = 0.25) -> EvidenceLayer | None:  # define the PC1 principal-component combination anomaly layer function
    """Pixel-level PC1 combination anomaly layer: PC1 = the common multi-element enrichment direction, sign-aligned to positive correlation.
    Scores normalized to 0~1, binary takes > P90 (combination anomaly area constrained to ~10%); fewer than 3 elements
    or fewer than 100 valid pixels returns None.
    """
    X, ok, names = _stack_features(elem_grids)  # stack the multi-element robust z features
    if X.shape[0] < 100 or X.shape[1] < 3: return None  # return None when valid pixels < 100 or elements < 3
    Xc = X - X.mean(axis=0, keepdims=True)  # column-wise demeaned (centered) feature matrix
    try:  # try singular value decomposition
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # SVD the centered matrix for principal components
    except np.linalg.LinAlgError: return None  # return None when the decomposition fails
    pc1 = Xc @ Vt[0]  # PC1 score = projection of the centered matrix onto the first loading direction
    explained = float(S[0] ** 2 / max(np.sum(S ** 2), 1e-12))  # PC1 explained variance ratio = largest eigenvalue / sum of all eigenvalues
    corrs = [float(np.corrcoef(pc1, Xc[:, j])[0, 1]) for j in range(X.shape[1])]  # compute correlations between PC1 and each element
    if float(np.mean([np.sign(c) for c in corrs])) < 0:  # sign alignment: PC1 positively correlated with most elements
        pc1, corrs = -pc1, [-c for c in corrs]  # flip PC1 and correlation signs to make it overall positive
    ny, nx = next(iter(elem_grids.values())).shape  # take the (ny,nx) shape from any element raster
    norm01 = _normalize01(pc1); thr = float(np.quantile(norm01, 0.90))  # threshold and binary both in the normalized space
    score = _to_full(ok, norm01).reshape(ny, nx)  # back-fill scores to full length and reshape to an (ny,nx) raster
    binary = _to_full(ok, (norm01 > thr).astype(float)).reshape(ny, nx)  # binarized result likewise back-filled and reshaped to a raster
    return _lyr("pca_pc1", f"主成分组合异常 PC1（{'-'.join(names[:5])}{'…' if len(names) > 5 else ''}）",  # construct the PC1 evidence layer (parameters span lines)
                "multivariate", "pca_score", names, score, binary, thr,  # kind / algorithm / elements / score / binary / threshold
                params={"explained_ratio": explained,  # param: PC1 explained variance ratio (spans lines)
                        "loadings": {n: round(Vt[0][j], 3) for j, n in enumerate(names)}},  # param: per-element PC1 loadings (3 decimals)
                notes=[f"PC1 解释方差 {explained * 100:.1f}%",  # note: explained variance percentage
                       "载荷: " + ", ".join(f"{n}({Vt[0][j]:+.2f})" for j, n in enumerate(names))])  # note: per-element loading list string


def mahalanobis_layer(elem_grids: dict[str, np.ndarray]) -> EvidenceLayer | None:  # define the robust Mahalanobis distance combination anomaly layer function
    """Pixel-level robust Mahalanobis distance combination anomaly layer: a large distance = a statistically "rare" element combination (binary > P90).
    μ is the median, Σ the covariance after robust_z; complementary to pca_layer (PCA captures common enrichment,
    Mahalanobis captures combinatorial outliers).
    """
    X, ok, names = _stack_features(elem_grids)  # stack the multi-element robust z features
    if X.shape[0] < 100 or X.shape[1] < 2: return None  # return None when valid pixels < 100 or elements < 2
    mu = np.median(X, axis=0)  # robust center = per-element medians
    cov = np.cov(X, rowvar=False)  # compute the inter-element covariance matrix
    try:  # try inverting the covariance
        cov_inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-9)  # slight regularization against singularity
    except np.linalg.LinAlgError: return None  # return None when the inversion fails
    d = X - mu  # each pixel's deviation vector from the center
    maha = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, cov_inv, d), 0.0))  # Mahalanobis distance = √(dᵀΣ⁻¹d) (batched einsum, negatives clipped to 0)
    ny, nx = next(iter(elem_grids.values())).shape  # take the (ny,nx) shape from any element raster
    score = _to_full(ok, _normalize01(maha)).reshape(ny, nx)  # back-fill normalized distances and reshape to a raster
    thr = float(np.quantile(maha, 0.90))  # threshold in the raw distance space
    binary = _to_full(ok, (maha > thr).astype(float)).reshape(ny, nx)  # pixels above the P90 threshold marked 1 and back-filled to a raster
    return _lyr("mahalanobis", "稳健马氏距离组合异常", "multivariate", "mahalanobis",  # construct the Mahalanobis evidence layer (parameters span lines)
                names, score, binary, float(thr),  # elements / score / binary / threshold (raw distance space)
                notes=["中位数重心 + 稳健协方差", "距离大 = 元素组合统计稀有"])  # notes: two method highlights


def unsupervised_layer(elem_grids: dict[str, np.ndarray], method: str = "iforest",  # define the unsupervised anomaly detection layer function (parameters span lines)
                       contamination: float = 0.10) -> EvidenceLayer | None:  # param: contamination (expected anomaly fraction, default 0.10)
    """Unsupervised anomaly detection layer (iforest/lof/ocsvm): rare multi-element combination identification when deposit labels are missing.
    It detects statistical outliers, not necessarily mineralization-related; Stage 2 must screen them with geological evidence;
    sklearn is lazily imported — when unavailable, return None and let pipeline record a warning.
    """
    X, ok, names = _stack_features(elem_grids)  # stack the multi-element robust z features
    if X.shape[0] < 100 or X.shape[1] < 2: return None  # return None when valid pixels < 100 or elements < 2
    try:  # try training the model
        if method == "iforest":  # enter the isolation forest branch
            from sklearn.ensemble import IsolationForest  # lazy import
            model = IsolationForest(n_estimators=200, contamination=contamination,  # train a 200-tree isolation forest (parameters span lines)
                                    random_state=42, n_jobs=-1).fit(X)  # fixed random seed, all-core parallel fit, with the contamination set
            raw = -model.score_samples(X)  # larger = more anomalous
        elif method == "lof":  # enter the local outlier factor branch
            from sklearn.neighbors import LocalOutlierFactor  # lazy import the LOF class
            model = LocalOutlierFactor(n_neighbors=min(20, max(5, X.shape[0] // 50)),  # neighbor count 5~20 by sample size (spans lines)
                                       novelty=True, contamination=contamination).fit(X)  # novelty-mode fit, with the contamination set
            raw = -model.score_samples(X)  # negate the score so larger = more anomalous
        elif method == "ocsvm":  # enter the one-class SVM branch
            from sklearn.svm import OneClassSVM  # lazy import OneClassSVM
            model = OneClassSVM(nu=contamination, gamma="scale").fit(X)  # train a one-class SVM with nu=contamination and auto-scaled gamma
            raw = -model.decision_function(X)  # negate the decision function so larger = more anomalous
        else: logger.warning("未知无监督方法: %s", method); return None  # warn on unknown method and return None
    except ImportError as e:  # catch the ImportError when sklearn is missing
        logger.warning("sklearn 不可用，跳过无监督方法 %s: %s", method, e); return None  # log a warning and return None so the caller degrades
    ny, nx = next(iter(elem_grids.values())).shape  # take the (ny,nx) shape from any element raster
    norm = _normalize01(raw); thr = float(np.quantile(norm, 1.0 - contamination))  # normalize scores and threshold at the (1 − contamination) quantile
    score = _to_full(ok, norm).reshape(ny, nx)  # back-fill scores to full length and reshape to a raster
    binary = _to_full(ok, (norm > thr).astype(float)).reshape(ny, nx)  # binarized result back-filled and reshaped to a raster
    return _lyr(method, f"{ALGO_CN.get(method, method)}多元素组合异常", "unsupervised",  # construct the unsupervised evidence layer (parameters span lines)
                method, names, score, binary, thr, params={"contamination": contamination},  # algorithm / elements / score / binary / threshold plus the contamination param
                notes=["统计离群分数，需结合地质证据筛选（阶段二融合）"])  # note: statistical outliers need Stage 2 screening with geological evidence


def build_external_layer(layer_id: str, name_cn: str, coords, grid: GridDef,  # define the external point-string evidence layer builder (parameters span lines)
                         radius: float = 2.0, weight: float = 1.0) -> EvidenceLayer | None:  # params: influence radius (default 2 km) and suggested weight (default 1)
    """External point-string evidence layer (faults/intrusions etc.; resample linear structures densely into point strings first).
    grid = linear distance decay max(0, 1 - d/radius); binary = d < radius (1 inside the buffer band).
    """
    pts = np.asarray(coords, dtype=float)  # convert point-string coordinates to a float array
    if pts.ndim != 2 or pts.shape[0] < 2:  # guard: not 2-D or fewer than 2 points
        logger.info("外部证据层 %s 点串为空，跳过", layer_id); return None  # log and return None
    d = distance_grid(pts, grid)  # compute the raster of nearest distances from each cell to the point string
    score = np.clip(1.0 - d / max(radius, 1e-9), 0.0, 1.0)  # linear decay score = max(0, 1 − d/radius)
    binary = (d < radius).astype(float)  # whole-area binary (no NaN)
    return _lyr(layer_id, name_cn or layer_id, "external", "distance", [], score, binary,  # construct the external evidence layer (name defaults to the ID; parameters span lines)
                float(radius), weight_hint=float(weight),  # threshold records the influence radius; attach the suggested weight
                params={"radius": radius, "n_points": int(pts.shape[0])},  # params: influence radius and point count of the point string
                notes=[f"影响半径 {radius:g}，距离线性衰减"])  # note: influence radius and linear decay explanation
