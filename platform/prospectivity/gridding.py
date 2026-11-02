"""
2D prospectivity mapping — spatialization layer (point data → continuous rasters).

make_grid builds the prediction grid from the sample-point extent (padded by a 2%
margin); idw_interpolate is inverse-distance weighting (default, robust and
parameter-free); ordinary_kriging is ordinary kriging (spherical variogram auto-fit +
nearest-neighbor subset solving); distance_grid is the nearest distance from cells to a
point set (the spatial representation of external evidence such as faults/intrusions);
fill_nan_nearest fills NaN with the nearest valid value (unified rasters for fusion).
All interpolators share the same GridDef and output (ny, nx) rasters (row order = y
from north to south); cells with no samples within the radius are NaN.
"""
from __future__ import annotations  # enable PEP 563 postponed annotation evaluation; type hints not evaluated at runtime

import logging  # import the logging module

import numpy as np  # import the numpy numeric library aliased as np
from scipy.spatial import cKDTree  # import scipy's cKDTree spatial nearest-neighbor index class

from oregenesis.platform.prospectivity.models import GridDef  # import the grid definition dataclass GridDef

logger = logging.getLogger(__name__)  # create the module-level logger


# Build the prediction grid: expand the sample-point extent outward by pad_ratio
# (default 2% margin to ease edge effects); default grid size 100×100 (1e4 cells
# balances kriging nearest-neighbor solving and map precision).
# Called once by pipeline.run_prospectivity_mapping; all rasters share the returned GridDef.
def make_grid(  # build the prediction grid from the sample-point extent
    x: np.ndarray,  # parameter: sample x coordinate array
    y: np.ndarray,  # parameter: sample y coordinate array
    nx: int = 100,  # parameter: cell count in the x direction (default 100)
    ny: int = 100,  # parameter: cell count in the y direction (default 100)
    pad_ratio: float = 0.02,  # parameter: margin expansion ratio (default 2%)
) -> GridDef:  # return type: grid definition object
    """Build the prediction grid from the sample-point extent (with an expanded margin)."""
    xa = np.asarray(x, dtype=float)  # sample x to a float numpy array
    ya = np.asarray(y, dtype=float)  # sample y to a float numpy array
    xmin, xmax = float(np.min(xa)), float(np.max(xa))  # take min/max of x as the west/east boundaries
    ymin, ymax = float(np.min(ya)), float(np.max(ya))  # take min/max of y as the south/north boundaries
    # when width or height is 0 (all points collinear), fall back to 1 against division by zero
    dx = (xmax - xmin) or 1.0  # x span 0 (collinear): fall back to 1 against division by zero
    dy = (ymax - ymin) or 1.0  # y span 0 (collinear): fall back to 1 against division by zero
    return GridDef(  # build and return the grid definition (parameters span multiple lines)
        xmin=xmin - dx * pad_ratio, xmax=xmax + dx * pad_ratio,  # expand the west/east boundaries by span×2% each to ease edge effects
        ymin=ymin - dy * pad_ratio, ymax=ymax + dy * pad_ratio,  # expand the south/north boundaries by span×2% each
        nx=nx, ny=ny,  # pass the cell counts in both directions
    )


# IDW inverse-distance-weighted interpolation: estimated cell value z = Σ(w_i·v_i)/Σ(w_i), w_i = 1/d_i^power.
# Search radius = mean nearest-neighbor distance × radius_factor (roughly a multiple of
# the mean sampling spacing); cells with no samples within the radius stay NaN (handled
# later by fill_nan_nearest or transparent map areas).
# Called per element by the pipeline (the main channel of element spatialization, default method).
def idw_interpolate(  # IDW inverse-distance-weighted interpolation function
    x: np.ndarray,  # parameter: sample x coordinates
    y: np.ndarray,  # parameter: sample y coordinates
    values: np.ndarray,  # parameter: sample values
    grid: GridDef,  # parameter: target grid definition
    power: float = 2.0,  # parameter: distance weight power (default 2)
    radius_factor: float = 3.0,  # parameter: search radius factor (default 3× mean nearest-neighbor distance)
) -> np.ndarray:  # return type: (ny,nx) raster array
    """IDW interpolation onto the grid defined by grid; returns an (ny, nx) raster."""
    xa = np.asarray(x, dtype=float)  # sample x to a float array
    ya = np.asarray(y, dtype=float)  # sample y to a float array
    va = np.asarray(values, dtype=float)  # sample values to a float array
    if xa.size < 4:  # guard: fewer than 4 valid samples
        raise ValueError("IDW 插值至少需要 4 个有效采样点")  # raise the error requiring at least 4 valid samples

    xs, ys = grid.axes()  # get the grid x/y axes
    gx, gy = np.meshgrid(xs, ys)          # (ny, nx)
    pts = np.column_stack([gx.ravel(), gy.ravel()])  # flatten grid coordinates into (N,2) estimation points

    tree = cKDTree(np.column_stack([xa, ya]))  # build a kd-tree from the sample coordinates
    # adaptive search radius: mean nearest-neighbor spacing of the samples (2nd neighbor, the 1st is itself) × factor
    if xa.size > 1:  # estimate the mean nearest-neighbor spacing when there is more than 1 sample
        d0, _ = tree.query(np.column_stack([xa, ya]), k=2)  # query each point's distance to itself and its 2nd neighbor
        mean_nn = float(np.mean(d0[:, 1])) if d0.ndim == 2 else float(np.mean(d0))  # mean of 2nd-neighbor distances as the mean sampling spacing (overall mean as fallback for a 1D result)
    else:
        mean_nn = max(grid.xmax - grid.xmin, grid.ymax - grid.ymin) / 5.0  # single point: fall back to 1/5 of the larger grid span
    radius = max(mean_nn * radius_factor, 1e-9)  # search radius = mean nearest-neighbor distance × factor (floored at 1e-9 against 0)

    idx_list = tree.query_ball_point(pts, r=radius)  # query sample indices within each cell's search radius
    out = np.full(pts.shape[0], np.nan, dtype=float)  # initialize the output as all NaN
    for i, nbr in enumerate(idx_list):  # iterate each cell's neighbor index list
        if not nbr:  # no samples within the radius
            continue  # keep NaN and move to the next cell
        nbr = np.asarray(nbr)  # neighbor index list to a numpy array
        dist = np.hypot(pts[i, 0] - xa[nbr], pts[i, 1] - ya[nbr])  # Euclidean distances from the cell to each neighbor
        near = dist < 1e-9  # flag samples nearly coincident with the cell (distance <1e-9)
        if near.any():  # coincident points exist: take the value directly
            out[i] = float(va[nbr[near][0]])  # take the coincident sample's value (distance 0 cannot be weighted)
            continue  # done; move to the next cell
        w = 1.0 / (dist ** power + 1e-12)  # inverse-distance power-p weights (denominator +1e-12 against division by zero)
        out[i] = float(np.sum(w * va[nbr]) / np.sum(w))  # weighted mean Σ(w·v)/Σw as this cell's interpolation
    return out.reshape(grid.ny, grid.nx)  # reshape the 1D result to an (ny,nx) raster


# ──────────────────────────────────────────────
#  Ordinary Kriging
# ──────────────────────────────────────────────

# Experimental variogram: bin point-pair distances into lags bins; per-bin semivariance γ(h) = mean[(v_i-v_j)²]/2.
# Only h ∈ (0, max_dist] is counted (max_dist takes half the survey-area diagonal; beyond
# that scale the semivariance is no longer reliable). Called by ordinary_kriging;
# returns (bin-center distances, semivariances) arrays.
def _experimental_variogram(  # experimental variogram computation (internal function)
    xy: np.ndarray, v: np.ndarray, n_lags: int = 12,  # parameters: coordinate array, value array, and bin count (default 12)
) -> tuple[np.ndarray, np.ndarray]:  # return type: (bin-center distances, semivariances) array pair
    tree = cKDTree(xy)  # build a kd-tree from the sample coordinates
    max_dist = float(np.max(np.hypot(xy[:, 0] - xy[:, 0].mean(), xy[:, 1] - xy[:, 1].mean())) * 1.5)  # point-pair search cap = max distance from points to the centroid ×1.5
    # all point-pair distances (sample count usually ≤ a few hundred, full enumeration acceptable)
    pairs = tree.query_pairs(max_dist, output_type="ndarray")  # enumerate all point-pair indices within the distance cap
    if pairs.size == 0:  # no point pairs at all
        return np.array([0.0]), np.array([0.0])  # return placeholder zero arrays
    d = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)  # compute each pair's Euclidean distance
    g = 0.5 * (v[pairs[:, 0]] - v[pairs[:, 1]]) ** 2  # each pair's semivariance = half the squared value difference
    # equal-width binning and averaging
    edges = np.linspace(0, d.max() * 1.001, n_lags + 1)  # generate n_lags+1 equal-width bin edges (cap ×1.001 to include the maximum)
    centers, means = [], []  # initialize the bin-center and mean-semivariance lists
    for i in range(n_lags):  # per-bin statistics
        m = (d >= edges[i]) & (d < edges[i + 1])  # boolean mask of pairs within this bin's distance interval
        if m.sum() >= 3:                      # too few pairs in the bin to be reliable; skip
            centers.append(0.5 * (edges[i] + edges[i + 1]))  # record the bin-center distance
            means.append(float(g[m].mean()))  # record the in-bin mean semivariance
    if len(centers) < 3:  # fewer than 3 valid bins: cannot fit a model
        return np.array([0.0]), np.array([0.0])  # return placeholder zero arrays
    return np.asarray(centers), np.asarray(means)  # return the bin-center and semivariance arrays


# Spherical variogram model: γ(h) = nugget + sill·(1.5·h/r − 0.5·(h/r)³), h<r;
# γ(h) = nugget + sill, h≥r. nugget = the nugget effect (microscale variability +
# measurement error), sill = the sill (total variance), r = the range (limit distance
# of spatial autocorrelation). Called by ordinary_kriging; h may be an array.
def _spherical(h: np.ndarray, nugget: float, sill: float, r: float) -> np.ndarray:  # spherical variogram model (internal function)
    h = np.asarray(h, dtype=float)  # lag distance h to a float array
    hr = h / max(r, 1e-12)  # lag distance divided by the range (denominator guarded against 0)
    return nugget + sill * np.where(hr < 1.0, 1.5 * hr - 0.5 * hr ** 3, 1.0)  # spherical model: cubic form below the range, reaching the sill beyond


# Spherical model fitting: first curve_fit (constrained nugget≥0, sill>0, r>0); on
# failure or absurd results (r exceeding 5× the max lag, etc.) fall back to
# heuristics: nugget=0, sill=sample variance, r=max lag distance. Called by
# ordinary_kriging.
def _fit_spherical(lags: np.ndarray, semivar: np.ndarray) -> tuple[float, float, float]:  # spherical model parameter fitting (internal function)
    sill0 = float(np.nanvar(semivar)) or 1.0  # initial sill = variance of the semivariances (fall back to 1 when 0)
    r0 = float(lags.max())  # initial range = max lag distance
    try:  # attempt the nonlinear least-squares fit
        from scipy.optimize import curve_fit  # lazy import of the curve_fit fitter
        popt, _ = curve_fit(  # curve-fit the spherical model (parameters span multiple lines)
            _spherical, lags, semivar,  # pass the model function and the (lag distances, semivariances) data
            p0=[0.0, sill0, r0],  # initial values: nugget 0, sill sill0, range r0
            bounds=([0.0, 1e-12, r0 * 0.05], [sill0, sill0 * 10, r0 * 3]),  # parameter bounds: nugget≥0, sill≤10×initial, range∈[0.05,3]×r0
            maxfev=2000,  # max function evaluations 2000
        )
        nugget, sill, r = (float(v) for v in popt)  # unpack the fitted parameters and convert to float
        if not all(np.isfinite([nugget, sill, r])) or r <= 0:  # non-finite parameters or non-positive range: invalid
            raise ValueError("克里金变差函数拟合结果无效")  # raise so the heuristic fallback takes over
        return nugget, sill, r  # return the fitted (nugget, sill, range)
    except Exception as e:  # catch the fit failure
        logger.info("球状变差函数拟合回退启发式参数: %s", e)  # log the fallback to heuristic parameters
        return 0.0, sill0, r0  # fallback: nugget 0, sample-variance sill, max-lag range


# Ordinary kriging interpolation: each cell takes its n_neighbors nearest samples and
# builds the OK system [K 1; 1ᵀ 0]·[w; μ] = [k; 1], K_ij=γ(d_ij),
# k_i=γ(distance from the cell to sample j). Once the weights w are solved,
# z = Σ w_i·v_i (unbiased, minimum estimation variance). Per-cell loop + a small
# linear system (≤(n_neighbors+1)²); 1e4 cells finish within seconds.
# Called by the pipeline (replaces IDW when interpolation="ordinary_kriging").
def ordinary_kriging(  # ordinary kriging interpolation function
    x: np.ndarray,  # parameter: sample x coordinates
    y: np.ndarray,  # parameter: sample y coordinates
    values: np.ndarray,  # parameter: sample values
    grid: GridDef,  # parameter: target grid definition
    n_neighbors: int = 16,  # parameter: nearest neighbors used per cell (default 16)
    n_lags: int = 12,  # parameter: variogram bin count (default 12)
) -> np.ndarray:  # return type: (ny,nx) raster array
    """Ordinary kriging interpolation (spherical variogram auto-fit); returns an (ny, nx) raster."""
    xa = np.asarray(x, dtype=float)  # sample x to a float array
    ya = np.asarray(y, dtype=float)  # sample y to a float array
    va = np.asarray(values, dtype=float)  # sample values to a float array
    xy = np.column_stack([xa, ya])  # assemble the (N,2) sample coordinate matrix
    if xy.shape[0] < 8:  # guard: fewer than 8 valid samples
        raise ValueError("普通克里金至少需要 8 个有效采样点")  # raise the minimum-sample-count error

    lags, semivar = _experimental_variogram(xy, va, n_lags=n_lags)  # compute the experimental variogram (binned semivariances)
    if lags.size < 3:  # guard: fewer than 3 valid lags
        raise ValueError("点对太少，无法估计变差函数")  # raise the too-few-pairs error
    nugget, sill, rng = _fit_spherical(lags, semivar)  # fit the three spherical-model parameters (nugget/sill/range)
    logger.info("克里金球状模型: nugget=%.4g sill=%.4g range=%.4g", nugget, sill, rng)  # log the fitted parameters

    tree = cKDTree(xy)  # build a kd-tree from the sample coordinates
    xs, ys = grid.axes()  # get the grid x/y axes
    gx, gy = np.meshgrid(xs, ys)  # generate the grid coordinate matrices (ny,nx)
    pts = np.column_stack([gx.ravel(), gy.ravel()])  # flatten into (N,2) estimation points

    k = min(n_neighbors, xy.shape[0])  # actual neighbor count = min of the parameter and the sample count
    dist, idx = tree.query(pts, k=k)  # query each cell's k-nearest-neighbor distances and indices
    if k == 1:                                # scipy returns 1D for a single neighbor; unify by adding a dimension
        dist, idx = dist[:, None], idx[:, None]  # raise the single-neighbor result to 2D, aligned with multi-neighbor

    out = np.full(pts.shape[0], np.nan, dtype=float)  # initialize the output as all NaN
    A = np.zeros((k + 1, k + 1))  # initialize the (k+1)-order kriging linear system matrix
    A[k, :] = 1.0                             # Lagrange multiplier row/column (Σw=1 constraint)
    A[:, k] = 1.0  # Lagrange multiplier column set to 1 (the other half of the unbiasedness constraint)
    A[k, k] = 0.0  # bottom-right corner of the system matrix set to 0
    for i in range(pts.shape[0]):  # solve the kriging weights cell by cell
        di = dist[i]                          # distances from the cell to its k neighbors
        if not np.isfinite(di).all():  # neighbor distances contain NaN (beyond the search range)
            continue  # keep this cell NaN and skip
        neigh = idx[i]  # take this cell's neighbor sample indices
        # K_ij = γ(distance between samples i and j)
        sub = xy[neigh]  # take the neighbor sample coordinate submatrix
        dnn = np.hypot(*(sub[:, None, :] - sub[None, :, :]).T)  # compute the pairwise neighbor distance matrix
        K = _spherical(dnn.ravel(), nugget, sill, rng).reshape(k, k)  # spherical model gives inter-neighbor semivariances, reshaped to the k×k matrix K
        kv = _spherical(di, nugget, sill, rng)  # spherical model gives the semivariance vector from the cell to each neighbor
        A[:k, :k] = K  # fill the semivariance matrix into the top-left block of the system matrix
        rhs = np.concatenate([kv, [1.0]])  # right-hand side = semivariance vector followed by 1 (unbiasedness constraint)
        try:  # try solving the linear system directly
            sol = np.linalg.solve(A, rhs)  # solve the system for the weights and the Lagrange multiplier
        except np.linalg.LinAlgError:  # matrix singular (neighbors nearly coincident)
            # K singular (neighbors nearly coincident) → minimum-norm solution fallback
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]  # least-squares fallback for the minimum-norm solution
        w = sol[:k]  # take the first k entries as the kriging weights
        out[i] = float(np.sum(w * va[neigh]))  # weighted sum Σw·v as this cell's estimate
    return out.reshape(grid.ny, grid.nx)  # reshape the 1D result to an (ny,nx) raster


# ──────────────────────────────────────────────
#  Distance raster and NaN filling
# ──────────────────────────────────────────────

# Nearest-distance raster from cells to a point set: the spatial representation of
# external evidence (densified fault-line point strings, intrusion boundary points,
# deposits, etc.) — the distance field itself is a continuous proxy of "structural
# influence strength" (closer to faults is more favorable). Called by
# evidence.build_external_layer (building external evidence layers) and the pipeline
# (deposit-to-cell placement).
def distance_grid(  # nearest-distance raster from cells to a point set
    coords: np.ndarray,  # parameter: point set (N,2) coordinate array
    grid: GridDef,  # parameter: target grid definition
) -> np.ndarray:  # return type: (ny,nx) distance raster
    """Euclidean distance from each cell to the nearest point in coords; returns an (ny, nx) raster."""
    pts_xy = np.asarray(coords, dtype=float)  # point coordinates to a float array
    if pts_xy.ndim != 2 or pts_xy.shape[0] == 0:  # guard: non-2D or empty array
        raise ValueError("距离栅格需要非空 (N,2) 坐标数组")  # raise the coordinate-array format error
    xs, ys = grid.axes()  # get the grid x/y axes
    gx, gy = np.meshgrid(xs, ys)  # generate the grid coordinate matrices
    pts = np.column_stack([gx.ravel(), gy.ravel()])  # flatten into an (N,2) cell coordinate array
    tree = cKDTree(pts_xy)  # build a kd-tree from the input point set
    d, _ = tree.query(pts, k=1)  # query the nearest-point distance per cell
    return d.reshape(grid.ny, grid.nx)  # reshape to an (ny,nx) distance raster


# Fill NaN with the nearest valid value: fusion and zoning require all evidence layers
# to be valid on the same cell set; NaN at distance-field/interpolation edges is
# extrapolated from the nearest valid cell value (fill mask returned for statistics).
# Called once by the pipeline after all evidence layers are built (may also fill
# prob_grid only).
def fill_nan_nearest(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # fill NaN with the nearest valid value
    """Nearest-neighbor NaN fill; returns (filled array, valid mask [cells valid before filling])."""
    a = np.asarray(arr, dtype=float)  # input to a float array
    valid = np.isfinite(a)  # compute the valid (finite) cell mask
    if valid.all():  # all valid: no filling needed
        return a.copy(), valid  # return the copy and the all-True mask
    ny, nx = a.shape  # take the raster row and column counts
    ys, xs = np.where(valid)  # take the row/col indices of all valid cells
    if ys.size == 0:  # no valid cells at all
        return a.copy(), valid  # return the copy and mask unchanged
    # kd-tree on valid cells (row,col); NaN cells query the nearest valid cell for its value (memory O(N))
    tree = cKDTree(np.column_stack([ys, xs]))  # build a kd-tree from valid-cell (row,col) coordinates
    qy, qx = np.where(~valid)  # take the row/col indices of all NaN cells
    _, nearest = tree.query(np.column_stack([qy, qx]), k=1)  # query the nearest valid-cell index per NaN cell
    filled = a.copy()  # copy the original array for filling
    filled[qy, qx] = a[ys[nearest], xs[nearest]]  # fill NaN positions with the nearest valid cell values
    return filled, valid  # return the filled array and the original valid mask
