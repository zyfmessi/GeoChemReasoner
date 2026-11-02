"""2D prospectivity mapping — stage two: multi-source evidence fusion (evidence layers → mineralization probability map).

Empirical method-selection rules (fuse's auto): deposits ≥8 → data-driven (weights of
evidence by default, random forest optional when sklearn is available); no deposits but
external geological evidence layers → knowledge-driven overlay (equal weights or
AHP/expert weights); no deposits and no geological evidence → unsupervised anomaly-score
fusion (pure geochemical mode). Probability semantics convention (is_probability):
True = weights of evidence/logistic/random forest output the "probability of deposit
occurrence given the evidence"; False = knowledge overlay/information value/unsupervised
fusion output a normalized "mineralization favorability score", usable for relative
ranking and grading but not a frequency probability. sklearn is lazily imported: without
the library, logistic falls back to a hand-written IRLS, and random_forest returns None
for upstream degradation.
"""
from __future__ import annotations  # enable PEP 563 postponed annotation evaluation; type hints not evaluated at runtime

import logging  # import the logging module

import numpy as np  # import the numpy numeric library aliased as np

from oregenesis.platform.prospectivity.models import EvidenceLayer, FusionModel  # import the evidence layer and fusion model dataclasses

logger = logging.getLogger(__name__)  # create the module-level logger

# Fusion method → Chinese name
METHOD_CN = {  # define the fusion method ID → Chinese name mapping
    "weights_of_evidence": "证据权法", "information_value": "信息量法",  # entries: weights of evidence and information value
    "logistic": "逻辑回归", "random_forest": "随机森林",  # entries: logistic regression and random forest
    "knowledge_overlay": "知识驱动叠加", "unsupervised_fusion": "无监督异常融合",  # entries: knowledge-driven overlay and unsupervised anomaly fusion
}

# AHP random consistency index RI (order → value)
_AHP_RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}  # AHP random consistency index RI table (orders 1~10)


def ahp_weights(pairwise: np.ndarray) -> tuple[np.ndarray, float]:  # compute weights from an AHP pairwise matrix
    """AHP pairwise matrix → (normalized weights, consistency ratio CR). Weights are the normalized eigenvector of the largest eigenvalue.
    CR = CI/RI, CI=(λmax-n)/(n-1), CR<0.1 means the judgment matrix is usable; clamped
    non-negative (with perfect consistency, numerical noise makes CI slightly negative,
    and exposing a negative consistency ratio would only be confusing).
    """
    A = np.asarray(pairwise, dtype=float); n = A.shape[0]  # pairwise matrix to float and take order n
    if A.shape != (n, n) or n < 2:  # validate non-square matrix or order below 2
        raise ValueError("AHP 判断矩阵须为 ≥2 阶方阵")  # raise matrix-format error
    eigvals, eigvecs = np.linalg.eig(A)  # eigendecomposition for eigenvalues and eigenvectors
    k = int(np.argmax(eigvals.real)); lam_max = float(eigvals[k].real)  # take the largest real eigenvalue λmax and its index
    w = np.abs(eigvecs[:, k].real); w = w / w.sum()  # take the absolute value of the eigenvector and normalize it into weights
    ci = (lam_max - n) / (n - 1) if n > 1 else 0.0  # consistency index CI=(λmax−n)/(n−1)
    ri = _AHP_RI.get(n, 1.49)  # look up RI for this order (order-10 value beyond the table)
    return w, float(max(0.0, ci / ri) if ri > 0 else 0.0)  # return weights and consistency ratio CR=CI/RI (negatives clamped to 0)


def _feature_stack(layers: list[EvidenceLayer]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # helper stacking features over the common valid area
    """Intersection of cells where all layers' grids are finite → (X feature matrix, B binary matrix, valid mask); fusion happens only on the common valid area."""
    grids = [np.asarray(ly.grid, dtype=float).ravel() for ly in layers]  # flatten each layer's raster into a 1D array
    valid = np.ones(grids[0].shape[0], dtype=bool)  # initialize the all-True valid mask
    for g in grids:  # check finiteness layer by layer
        valid &= np.isfinite(g)  # intersect masks: keep only cells valid in every layer
    X = np.column_stack(grids)  # stack each layer's continuous scores by column into feature matrix X
    B = np.column_stack([np.nan_to_num(np.asarray(ly.binary, dtype=float).ravel(), nan=0.0) for ly in layers])  # stack each layer's binary evidence into B (NaN as 0)
    return X[valid], B[valid], valid  # return X/B over the common valid area and the mask


def fuse_knowledge_overlay(layers: list[EvidenceLayer], weights: dict[str, float] | None = None,  # knowledge-driven overlay fusion function (parameters span lines)
                           mode: str = "weighted_sum") -> FusionModel:  # parameter: overlay mode (weighted sum / product)
    """Knowledge-driven overlay (the classic method when the metallogenic model is clear but deposits are few).
    weighted_sum: P=Σw_i·s_i/Σw_i (linear weighting, default, robust to zero values);
    product: P=Πs_i^{w_i} (exponential overlay with a one-vote-veto character,
    suited to "necessary condition" style metallogenic models).
    """
    ids = [ly.layer_id for ly in layers]  # take the ID list of the evidence layers
    w = np.asarray([float((weights or {}).get(i, 1.0)) for i in ids], dtype=float)  # take the weight per layer (default 1 when unspecified)
    w = w / max(w.sum(), 1e-12)  # normalize weights (sum floored at 1e-12 against division by zero)
    shape = np.asarray(layers[0].grid).shape  # take the first layer's raster shape
    S = np.column_stack([np.nan_to_num(np.asarray(ly.grid, dtype=float).ravel(), nan=0.0) for ly in layers])  # stack layer scores (NaN as 0 in the computation)
    if mode == "product":  # enter the product (exponential) overlay branch
        Sp = np.clip(S, 0.0, 1.0) * 0.95 + 0.05  # smooth to [0.05,1]: prevent a single 0 layer zeroing the whole map (suppress, not veto)
        p = np.exp(np.sum(w[None, :] * np.log(Sp), axis=1))  # exponential overlay P=exp(Σ w·ln s)=Π s^w
    else:
        p = (S * w[None, :]).sum(axis=1)  # weighted-sum branch: P=Σ w·s
    any_valid = np.zeros(p.shape, dtype=bool)  # validity: output wherever any participating layer is valid (missing layers treated as 0)
    for ly in layers:  # accumulate validity layer by layer
        any_valid |= np.isfinite(np.asarray(ly.grid, dtype=float).ravel())  # cells valid in any layer are marked valid
    prob = np.full(p.shape, np.nan); prob[any_valid] = p[any_valid]  # initialize full-length NaN and backfill valid cells
    return FusionModel(  # build and return the fusion model (parameters span lines)
        method="knowledge_overlay", name_cn=f"知识驱动{'乘积' if mode == 'product' else '加权'}叠加",  # method name and Chinese name (product/weighted switches with mode)
        prob_grid=prob.reshape(shape), is_probability=False,  # probability raster reshaped back to original form; semantics: not a probability
        layer_weights={i: float(wi) for i, wi in zip(ids, w)}, diagnostics={"overlay_mode": mode},  # layer → normalized weight mapping and overlay-mode diagnostics
        notes=["无矿点数据驱动，结果为成矿有利度得分（非频率概率）"] if mode != "product"  # weighted-sum mode note: mineralization favorability score
        else ["乘积叠加具有一票否决特性，适合必要条件型成矿模式"])  # product mode note: one-vote-veto character


def fuse_weights_of_evidence(layers: list[EvidenceLayer], deposit_mask: np.ndarray) -> FusionModel | None:  # weights-of-evidence fusion function (parameters span lines)
    """Weights of evidence (Bonham-Carter): spatial association between binary evidence and deposits → posterior probability map.
    W+=ln[P(B|D)/P(B|D̄)] (deposit posterior odds increase when the evidence is present),
    W-=ln[P(B̄|D)/P(B̄|D̄)]; contrast C=W+-W- (larger |C| means stronger prospecting
    indication); logit=prior+Σ(W+|W-), posterior probability=sigmoid(logit).
    Conditional-independence overall test: Σ P_post/N1 ≈ 1 is good; deviation >20% warns;
    all layers must have binary.
    """
    flat_dep = np.asarray(deposit_mask, dtype=bool).ravel()  # deposit mask to bool and flatten
    _, B, valid = _feature_stack(layers)  # take the common-valid-area binary matrix and mask
    dep = flat_dep[valid]; N = valid.sum(); N1 = int(dep.sum())  # take valid-area deposit flags / total cells / deposit count
    if N1 < 1: return None  # return None when there are no deposits
    logits = np.full(N, np.log(max(N1, 1) / max(N - N1, 1)))  # prior logit
    w_details: dict[str, dict] = {}; layer_c: dict[str, float] = {}  # initialize per-layer weight details and contrast dicts
    n_zero_warn = 0  # reset the contingency-table zero-cell warning counter
    for j, ly in enumerate(layers):  # compute W+/W− per evidence layer
        b = B[:, j].astype(bool)  # this layer's binary evidence to bool
        Nb = int(b.sum()); N1b = int((b & dep).sum())  # cell count with evidence Nb and deposit count among them N1b
        c11, c10 = N1b, N1 - N1b  # four cells of the 2×2 contingency table (any zero cell → weights not estimable, skip and count a warning)
        c01, c00 = Nb - N1b, (N - N1) - (Nb - N1b)  # evidence-present no-deposit count c01 and evidence-absent no-deposit count c00
        if min(c11, c10, c01, c00) <= 0:  # any of the four cells being 0 makes the weights non-estimable
            n_zero_warn += 1; layer_c[ly.layer_id] = 0.0  # increment the warning count and record this layer's contrast as 0
            w_details[ly.layer_id] = {"W+": None, "W-": None, "C": None, "note": "列联表含零格，权重不可估"}  # record the detail that this layer's weights are not estimable
            continue  # skip this layer and continue
        wplus = float(np.log((c11 / N1) / (c01 / (N - N1))))  # positive weight W+=ln[(c11/N1)/(c01/(N−N1))]
        wminus = float(np.log((c10 / N1) / (c00 / (N - N1))))  # negative weight W−=ln[(c10/N1)/(c00/(N−N1))]
        var_wp, var_wm = 1.0 / c11 + 1.0 / c01, 1.0 / c10 + 1.0 / c00  # weight variances
        C = wplus - wminus  # contrast C=W+−W−
        std_c = float(np.sqrt(var_wp + var_wm)); t = C / std_c if std_c > 1e-12 else 0.0  # studentized t
        layer_c[ly.layer_id] = C  # record this layer's contrast
        w_details[ly.layer_id] = {"W+": round(wplus, 4), "W-": round(wminus, 4),  # record W+/W− (4 decimals)
                                  "varW+": round(var_wp, 5), "varW-": round(var_wm, 5),  # record both weight variances (5 decimals)
                                  "C": round(C, 4), "studentized_C": round(t, 3)}  # record contrast and studentized t
        logits += np.where(b, wplus, wminus)  # accumulate W+/W− into the logit depending on evidence presence
    prob_flat = 1.0 / (1.0 + np.exp(-logits))  # posterior probability = sigmoid(logit)
    cis_ratio = float(prob_flat.sum() / max(N1, 1))  # conditional-independence test: predicted/actual deposit count (ideally ≈1)
    prob = np.full(flat_dep.size, np.nan); prob[valid] = prob_flat; notes = []  # initialize full-length NaN, backfill posterior probabilities, and create the notes list
    if n_zero_warn: notes.append(f"{n_zero_warn} 个证据层列联表含零格被跳过")  # append a warning note when layers with zero cells were skipped
    if not (0.8 <= cis_ratio <= 1.2):  # when the conditional-independence ratio strays outside [0.8,1.2]
        notes.append(f"条件独立性比率 {cis_ratio:.2f} 偏离 1，层间可能相关，后验概率仅供参考")  # append a note that layers may be correlated
    return FusionModel(  # build and return the weights-of-evidence fusion model (parameters span lines)
        method="weights_of_evidence", name_cn="证据权法成矿概率",  # method ID and Chinese name
        prob_grid=prob.reshape(np.asarray(layers[0].grid).shape), is_probability=True,  # probability raster back to original shape; semantics: true probability
        layer_weights=layer_c,  # layer → contrast C mapping as weights
        diagnostics={"per_layer": w_details, "prior_prob": N1 / N, "cis_ratio": cis_ratio}, notes=notes)  # diagnostics: per-layer details / prior probability / conditional-independence ratio, plus notes


def fuse_information_value(layers: list[EvidenceLayer], deposit_mask: np.ndarray) -> FusionModel | None:  # information-value fusion function (parameters span lines)
    """Information value method: I_i = ln[(N_i/N)/(S_i/S)] (log ratio of within-layer to whole-area deposit density), summed over layers.
    I>0 indicates favorable for mineralization, I<0 unfavorable; output is a normalized
    favorability (not a probability). Semantics are simpler and more intuitive than
    weights of evidence, and it is one of the traditional mainstream methods;
    returns None with fewer than 1 deposit.
    """
    flat_dep = np.asarray(deposit_mask, dtype=bool).ravel()  # deposit mask to bool and flatten
    _, B, valid = _feature_stack(layers)  # take the common-valid-area binary matrix and mask
    dep = flat_dep[valid]; N = int(valid.sum()); N1 = int(dep.sum())  # take valid-area deposit flags / total cells / deposit count
    if N1 < 1: return None  # return None when there are no deposits
    info_vals: dict[str, float] = {}  # initialize the layer → information value dict
    I_total = np.zeros(N)  # initialize the total information-value vector
    for j, ly in enumerate(layers):  # compute the information value per evidence layer
        b = B[:, j].astype(bool)  # this layer's binary evidence to bool
        Ni = int((b & dep).sum())  # deposit count within the layer
        Si = int(b.sum())  # cell count within the layer (area proxy)
        if Ni == 0 or Si == 0: info_vals[ly.layer_id] = 0.0; continue  # with no deposits or no evidence cells, record 0 and skip
        Ii = float(np.log((Ni / N1) / (Si / N))); info_vals[ly.layer_id] = round(Ii, 4)  # information value I=ln[(within-layer deposit share)/(within-layer area share)]
        I_total += np.where(b, Ii, 0.0)  # cells with evidence accumulate this layer's information value
    lo, hi = float(I_total.min()), float(I_total.max())  # take the min/max of the total information value
    # normalize to 0~1 (relative ordering preserved; negative information value = below prior, also mapped into the score)
    score_flat = (I_total - lo) / (hi - lo) if hi - lo > 1e-12 else np.full(N, 0.5)  # min-max normalize to 0~1 (constant 0.5 when all values are equal)
    out = np.full(flat_dep.size, np.nan); out[valid] = score_flat  # initialize full-length NaN and backfill valid-area scores
    return FusionModel(  # build and return the information-value fusion model (parameters span lines)
        method="information_value", name_cn="信息量法成矿有利度",  # method ID and Chinese name
        prob_grid=out.reshape(np.asarray(layers[0].grid).shape), is_probability=False,  # raster back to original shape; semantics: not a probability
        layer_weights=info_vals, diagnostics={"info_range": [lo, hi], "prior_prob": N1 / N},  # layer → information value mapping and range / prior probability diagnostics
        notes=["信息量归一化得分（非频率概率）；负信息量层表示不利证据"])  # note: normalized-score semantics and the meaning of negative information values


def _fit_logistic_irsl(X: np.ndarray, y: np.ndarray, max_iter: int = 50) -> np.ndarray:  # hand-written IRLS logistic regression fit (parameters span lines)
    """Hand-written IRLS (iteratively reweighted least squares) logistic regression: β←(XᵀWX)⁻¹XᵀWz, W=μ(1-μ).
    Fallback implementation when sklearn is missing (numerically stable for small
    feature counts); falls back to least squares when singular.
    """
    Xd = np.column_stack([np.ones(len(X)), X]); beta = np.zeros(Xd.shape[1])  # design matrix with an intercept column; coefficients initialized to 0
    for _ in range(max_iter):  # iterate at most max_iter rounds
        eta = Xd @ beta  # linear predictor η=Xβ
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))  # probability μ=sigmoid(η) (η clipped to ±30 against overflow)
        w = np.maximum(mu * (1 - mu), 1e-9)  # IRLS weight W=μ(1−μ) (floored at 1e-9 against 0)
        z = eta + (y - mu) / w  # working response z=η+(y−μ)/W
        try:  # try solving the normal equations directly
            beta_new = np.linalg.solve(Xd.T @ (Xd * w[:, None]), Xd.T @ (w * z))  # solve (XᵀWX)β=XᵀWz for the new coefficients
        except np.linalg.LinAlgError:  # singular → least-squares fallback
            beta_new = np.linalg.lstsq(Xd * w[:, None], w * z, rcond=None)[0]  # weighted least-squares fallback for the coefficients
        if float(np.max(np.abs(beta_new - beta))) < 1e-8: beta = beta_new; break  # max coefficient change <1e-8 means convergence; stop
        beta = beta_new  # proceed to the next round with the new coefficients
    return beta  # return the fitted coefficients β


def fuse_logistic(layers: list[EvidenceLayer], deposit_mask: np.ndarray) -> FusionModel | None:  # logistic regression fusion function (parameters span lines)
    """Logistic regression probability map: P=sigmoid(β₀+Σβᵢxᵢ), x=continuous scores of the evidence layers (retain more information than binaries).
    sklearn preferred (class_weight='balanced' eases the rare-deposit imbalance), IRLS
    when the library is missing; returns None with <2 positives or <10 negatives;
    coefficients go into layer_weights (semi-quantitative feature importance).
    """
    X, _, valid = _feature_stack(layers)  # take the common-valid-area feature matrix X
    y = np.asarray(deposit_mask, dtype=float).ravel()[valid]  # take the valid-area deposit labels y (0/1)
    if y.sum() < 2 or (1 - y).sum() < 10: return None  # return None when positives <2 or negatives <10
    coefs: list[float] | None = None; backend = "irsl"  # initialize the coefficient list and the backend flag
    try:
        from sklearn.linear_model import LogisticRegression  # lazy import
        model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42).fit(X, y)  # class-balanced weighted logistic regression fit (max 2000 iterations)
        p = model.predict_proba(X)[:, 1]; coefs = [float(c) for c in model.coef_[0]]  # take positive-class probabilities and per-feature coefficients
        backend = "sklearn"  # mark the backend as sklearn
    except Exception:  # sklearn missing / fit failed → IRLS fallback
        beta = _fit_logistic_irsl(X, y)  # hand-written IRLS coefficient fit fallback
        eta = beta[0] + X @ beta[1:]; p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))  # manually compute probability=sigmoid(β₀+Xβ) (clipped against overflow)
        coefs = [float(c) for c in beta[1:]]  # take the feature coefficients (skip the intercept)
    out = np.full(valid.size, np.nan); out[valid] = p  # initialize full-length NaN and backfill probabilities
    return FusionModel(  # build and return the logistic fusion model (parameters span lines)
        method="logistic", name_cn="逻辑回归成矿概率",  # method ID and Chinese name
        prob_grid=out.reshape(np.asarray(layers[0].grid).shape), is_probability=True,  # probability raster back to original shape; semantics: true probability
        layer_weights={ly.layer_id: c for ly, c in zip(layers, coefs)},  # layer → regression coefficient mapping (semi-quantitative importance)
        diagnostics={"backend": backend, "n_pos": int(y.sum()), "n_train": int(len(y))},  # diagnostics: backend / positive count / training sample count
        notes=["概率输出透明可解释，建议作为机器学习模型的基准对照"])  # note: recommended as the baseline comparison for machine-learning models


def fuse_random_forest(layers: list[EvidenceLayer], deposit_mask: np.ndarray,  # random forest fusion function (parameters span lines)
                       n_estimators: int = 300) -> FusionModel | None:  # parameter: number of trees (default 300)
    """Random forest probability map: cell-level evidence features → deposit binary classification → positive-class probability + feature importances.
    min_samples_leaf≈2×positive count (clipped to [5,30]) large-leaf pruning: under the
    default settings, training-cell probabilities polarize to 0/1 (overfitting) and
    degrade quantile zoning, while within-leaf frequencies of large leaves remain
    continuous and order-preserving; returns None with <3 positives or missing sklearn;
    overfitting warning below 10 positives.
    """
    try:  # try importing sklearn
        from sklearn.ensemble import RandomForestClassifier  # lazy import
    except ImportError:  # catch the import error
        logger.warning("sklearn 不可用，随机森林融合跳过")  # log the sklearn-unavailable warning
        return None  # return None for upstream degradation
    X, _, valid = _feature_stack(layers)  # take the common-valid-area feature matrix X
    y = np.asarray(deposit_mask, dtype=int).ravel()[valid]  # take the valid-area deposit labels y (integer 0/1)
    if y.sum() < 3: return None  # return None with fewer than 3 positives
    leaf = int(np.clip(2 * max(y.sum(), 1), 5, 30))  # min leaf samples = 2×positive count (clipped to 5~30) against probability polarization
    model = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced_subsample",  # train the class-balanced random forest (parameters span lines)
                                   random_state=42, n_jobs=-1, min_samples_leaf=leaf,  # fixed seed, all-core parallelism, large-leaf pruning
                                   oob_score=True).fit(X, y)  # enable out-of-bag evaluation and fit
    p = model.predict_proba(X)[:, 1]  # predict training-cell positive-class probability (within-leaf frequency)
    out = np.full(valid.size, np.nan); out[valid] = p; notes = []  # initialize full-length NaN, backfill, and create the notes list
    if y.sum() < 10: notes.append(f"正样本仅 {int(y.sum())} 个，随机森林结果易过拟合，建议与证据权法对照")  # positives <10: append the overfitting warning note
    return FusionModel(  # build and return the random-forest fusion model (parameters span lines)
        method="random_forest", name_cn="随机森林成矿概率",  # method ID and Chinese name
        prob_grid=out.reshape(np.asarray(layers[0].grid).shape), is_probability=True,  # probability raster back to original shape; semantics: true probability
        layer_weights={ly.layer_id: float(imp) for ly, imp in zip(layers, model.feature_importances_)},  # layer → feature importance mapping
        diagnostics={"n_pos": int(y.sum()), "n_train": int(len(y)),  # diagnostics: positive count / training sample count (spanning lines)
                     "prob_source": f"insample(leaf={leaf})",  # diagnostics: probability source is within-leaf frequency, plus leaf size
                     "oob_score": float(getattr(model, "oob_score_", 0.0) or 0.0)}, notes=notes)  # diagnostics: out-of-bag score and notes list


def fuse_unsupervised(layers: list[EvidenceLayer], weights: dict[str, float] | None = None) -> FusionModel:  # unsupervised fusion function (parameters span lines)
    """Unsupervised fusion: weighted average of continuous anomaly-score layers when there are no deposits (geochemical layers only, no geological evidence; distinct from knowledge overlay)."""
    return fuse_knowledge_overlay(layers, weights=weights, mode="weighted_sum")  # delegate to the knowledge overlay's weighted-sum mode


def fuse(layers: list[EvidenceLayer], deposit_mask: np.ndarray | None, method: str = "auto",  # evidence fusion main entry (parameters span multiple lines)
         weights: dict[str, float] | None = None,  # parameter: layer weight dict (optional)
         overlay_mode: str = "weighted_sum") -> tuple[FusionModel | None, list[str]]:  # parameter: overlay mode; returns (model, warnings list)
    """Evidence fusion main entry: dispatch by method; automatically degrade when a method is infeasible, explained in warnings.
    auto: deposits ≥8 → weights_of_evidence (most interpretable); 1~7 →
    information_value (slightly more stable than WofE when deposits are very few);
    no deposits + external evidence layers → knowledge_overlay; pure geochemical
    layers → unsupervised_fusion. Returns (FusionModel, warnings).
    """
    if not layers: return None, ["没有可用证据层，无法融合"]  # no evidence layers: return the warning directly
    warnings: list[str] = []  # initialize the warnings list
    n_dep = int(np.asarray(deposit_mask).sum()) if deposit_mask is not None else 0  # count deposits (0 when there is no mask)
    has_external = any(ly.kind == "external" for ly in layers)  # check whether external geological evidence layers are present
    if method == "auto":  # enter the auto method-selection branch
        if n_dep >= 8: method = "weights_of_evidence"  # deposits ≥8 → weights of evidence (most interpretable)
        elif n_dep >= 1: method = "information_value"  # deposits 1~7 → information value (more stable for small samples)
        elif has_external: method = "knowledge_overlay"  # no deposits but external layers → knowledge overlay
        else: method = "unsupervised_fusion"  # pure geochemical layers → unsupervised fusion
        logger.info("融合方法 auto → %s（矿点 %d，外部层 %s）", method, n_dep, has_external)  # log the auto-selected method and its basis
    model = None  # initialize the model as None
    if method == "weights_of_evidence" and n_dep < 1:  # weights of evidence requested but no deposits
        warnings.append("证据权法需要已知矿点，当前无矿点 → 降级为知识叠加")  # append the degradation warning
        model = fuse_knowledge_overlay(layers, weights=weights, mode=overlay_mode)  # degrade to knowledge overlay
    elif method == "weights_of_evidence":
        model = fuse_weights_of_evidence(layers, deposit_mask)  # run weights-of-evidence fusion
    elif method == "information_value" and n_dep < 1:  # information value requested but no deposits
        warnings.append("信息量法需要已知矿点，当前无矿点 → 降级为知识叠加")  # append the degradation warning
        model = fuse_knowledge_overlay(layers, weights=weights, mode=overlay_mode)  # degrade to knowledge overlay
    elif method == "information_value":
        model = fuse_information_value(layers, deposit_mask)  # run information-value fusion
    elif method == "logistic" and n_dep < 3:  # logistic requested but fewer than 3 deposits
        warnings.append("逻辑回归需要至少 3 个矿点，当前矿点不足 → 降级为证据权法/知识叠加")  # append the degradation warning
        model, w2 = fuse(layers, deposit_mask, method="auto", weights=weights, overlay_mode=overlay_mode)  # recurse with auto to reselect a feasible method
        warnings.extend(w2)  # merge warnings produced by the recursion
    elif method == "logistic":
        model = fuse_logistic(layers, deposit_mask)  # run logistic fusion
        if model is None:  # fit failed → knowledge overlay
            warnings.append("逻辑回归训练失败 → 降级为知识叠加")  # append the fit-failure degradation warning
            model = fuse_knowledge_overlay(layers, weights=weights, mode=overlay_mode)  # degrade to knowledge overlay
    elif method == "random_forest" and n_dep < 3:  # random forest requested but fewer than 3 deposits
        warnings.append("随机森林需要至少 3 个矿点 → 降级为知识叠加")  # append the degradation warning
        model = fuse_knowledge_overlay(layers, weights=weights, mode=overlay_mode)  # degrade to knowledge overlay
    elif method == "random_forest":
        model = fuse_random_forest(layers, deposit_mask)  # run random-forest fusion
        if model is None:  # sklearn missing → weights of evidence
            warnings.append("随机森林不可用（sklearn 缺失）→ 降级为证据权法")  # append the sklearn-missing degradation warning
            model, w2 = fuse(layers, deposit_mask, method="weights_of_evidence")  # recurse switching to weights of evidence
            warnings.extend(w2)  # merge warnings produced by the recursion
    elif method == "knowledge_overlay":
        model = fuse_knowledge_overlay(layers, weights=weights, mode=overlay_mode)  # run the knowledge-driven overlay
    elif method == "unsupervised_fusion":
        model = fuse_unsupervised(layers, weights=weights)  # run the unsupervised fusion
    else:
        warnings.append(f"未知融合方法 {method} → auto 处理")  # append the unknown-method warning
        return fuse(layers, deposit_mask, method="auto", weights=weights, overlay_mode=overlay_mode)  # recurse with auto as the fallback
    if model is None:  # model still None (execution failed)
        warnings.append(f"融合方法 {method} 执行失败 → 降级为知识叠加")  # append the failure degradation warning
        model = fuse_knowledge_overlay(layers, weights=weights, mode=overlay_mode)  # final fallback: run the knowledge overlay
    return model, warnings  # return the fusion model and the warnings list
