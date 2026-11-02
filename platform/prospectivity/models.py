"""2D mineral prospectivity mapping — result data models (pure dataclasses, no algorithm logic).

Unified data carriers of the two-stage pipeline: stage one (spatialization + anomaly
extraction) produces EvidenceLayer → stage two (multi-source evidence fusion) produces
FusionModel → classify produces ZoneResult → pipeline aggregates into
ProspectivityReport → plotting consumes them read-only.
Raster conventions: unified (ny, nx) numpy arrays, rows = y from north to south;
grid is 0~1 continuous intensity (NaN = no data), binary is binary evidence
(1 = anomaly), and prob_grid semantics are determined by method.
"""

from __future__ import annotations  # enable PEP 563 postponed annotation evaluation; type hints not evaluated at runtime

from dataclasses import dataclass, field  # import the dataclass decorator and the field factory from dataclasses
from typing import Any  # import the Any type (for free-form field annotations)

import numpy as np  # import the numpy numeric library aliased as np


@dataclass  # dataclass decorator: auto-generates __init__/__repr__ etc.
class GridDef:  # prediction grid definition dataclass (extent + size)
    """Prediction grid definition: extent + size + cell area; all rasters share the same GridDef to guarantee per-pixel aligned fusion."""

    xmin: float  # grid west boundary (planar coordinates; lon/lat already flattened to km)
    xmax: float  # east boundary
    ymin: float  # south boundary
    ymax: float  # north boundary
    nx: int      # number of grid points (columns) in the x direction
    ny: int      # number of grid points (rows) in the y direction

    def axes(self) -> tuple[np.ndarray, np.ndarray]:  # grid axes method
        """Grid axes: xs west→east, ys south→north reversed (row 0 is the ymax row, directly usable with imshow origin="upper")."""
        return np.linspace(self.xmin, self.xmax, self.nx), np.linspace(self.ymax, self.ymin, self.ny)  # generate xs (nx evenly spaced points from xmin to xmax) and ys (ny points reversed from ymax to ymin)

    def cell_area(self) -> float:  # cell area method
        """Cell area (squared coordinate units; lon/lat already flattened to km, so area ≈ km²)."""
        return abs((self.xmax - self.xmin) / max(self.nx - 1, 1) * (self.ymax - self.ymin) / max(self.ny - 1, 1))  # area = product of cell spacings (width/(nx−1)) × (height/(ny−1)), absolute value taken

    def total_area(self) -> float:  # total grid area method
        """Total grid area (cell count × cell area, approximate)."""
        return self.nx * self.ny * self.cell_area()  # total area = total cell count × cell area (approximate)

    def extent(self) -> tuple[float, float, float, float]:  # display extent method for imshow
        """extent for imshow: (xmin, xmax, ymin, ymax)."""
        return (self.xmin, self.xmax, self.ymin, self.ymax)  # return the (xmin, xmax, ymin, ymax) tuple

    def to_cell(self, x: float, y: float) -> tuple[int, int]:  # world coordinates → raster indices (row, col) conversion method
        """World coordinates → raster indices (row, col); out-of-range values clipped to the boundary."""
        col = int(round((x - self.xmin) / max(self.xmax - self.xmin, 1e-12) * (self.nx - 1)))  # column index = round((x−xmin)/width × (nx−1))
        row = int(round((self.ymax - y) / max(self.ymax - self.ymin, 1e-12) * (self.ny - 1)))  # row index = round((ymax−y)/height × (ny−1)) (y is positive northward, hence ymax−y)
        return (min(max(row, 0), self.ny - 1), min(max(col, 0), self.nx - 1))  # clamp row/col indices to [0, ny−1]/[0, nx−1] against out-of-bounds


@dataclass  # dataclass decorator: auto-generates constructor and repr methods
class EvidenceLayer:  # evidence layer dataclass (stage one → stage two interface)
    """One evidence layer = the rasterized result of one "anomaly extraction algorithm × (element or combination)" (stage one → stage two interface).

    grid is the continuous anomaly intensity 0~1 (for weighted overlay / machine-learning
    features), binary is the binary evidence (1 = anomaly, for weights of evidence /
    information value); both derive from the same threshold.
    """

    layer_id: str        # layer ID, e.g. "Cu_median_mad" / "pca_pc1" / "iforest"
    name_cn: str         # Chinese display name, e.g. "Cu median +2MAD anomaly"
    kind: str            # category: element / multivariate / unsupervised / external
    algorithm_id: str    # algorithm ID: median_mad / mean_2sigma / quantile / ca_fractal /
                         #        pca_score / mahalanobis / iforest / lof / ocsvm / distance
    elements: list[str] = field(default_factory=list)   # elements involved (multiple for combination layers)
    grid: np.ndarray | None = None      # continuous anomaly intensity raster (ny,nx), 0~1, NaN = no data
    binary: np.ndarray | None = None    # binary evidence raster (ny,nx), 1 = anomaly, 0 = background
    threshold: float | None = None      # anomaly threshold (binary = grid > threshold)
    n_anomaly_cells: int = 0            # number of anomaly cells
    area_ratio: float = 0.0             # anomaly area ratio (anomaly / valid cells)
    background: float | None = None     # background value (original space, present for single-element layers)
    weight_hint: float = 1.0            # suggested weight for knowledge-driven overlay (external layers may be user-specified)
    params: dict[str, Any] = field(default_factory=dict)  # algorithm parameters (including C-A inflection point, etc.)
    notes: list[str] = field(default_factory=list)  # notes list (empty by default, records algorithm explanations)

    def to_row(self) -> dict[str, Any]:  # method converting to one statistics-table row
        """Convert to one row of the CSV/Markdown statistics table (Chinese field names)."""
        elems = "+".join(self.elements[:6]) + ("…" if len(self.elements) > 6 else "")  # join the first 6 element names with +; append an ellipsis when more
        return {"证据层": self.layer_id, "名称": self.name_cn, "涉及元素": elems,  # assemble the row dict: layer ID / Chinese name / elements involved — three columns
                "算法": self.algorithm_id, "异常格点": self.n_anomaly_cells,  # continued: algorithm ID and anomaly cell count columns
                "面积占比": f"{self.area_ratio * 100:.1f}%",  # continued: area ratio formatted as a one-decimal percentage
                "阈值": "-" if self.threshold is None else f"{self.threshold:.4g}"}  # continued: threshold shown as "-" when None, else 4 significant digits


@dataclass  # dataclass decorator: auto-generates constructor and repr methods
class FusionModel:  # multi-source evidence fusion model dataclass
    """Multi-source evidence fusion model: mineralization probability/favorability map + evidence layer contributions."""

    method: str                          # weights_of_evidence / information_value / logistic /
                                         # random_forest / knowledge_overlay / unsupervised_fusion
    name_cn: str                         # Chinese name
    prob_grid: np.ndarray | None = None  # mineralization probability/favorability raster (ny,nx), 0~1, NaN = no data
    is_probability: bool = False         # True = true probability (WofE/logistic/RF), False = normalized score
    layer_weights: dict[str, float] = field(default_factory=dict)  # evidence layer → weight / contrast C
    diagnostics: dict[str, Any] = field(default_factory=dict)      # AUC / conditional independence ratio / feature importances / per-layer W+ W-
    notes: list[str] = field(default_factory=list)  # notes list (empty by default, records methodological explanations)


@dataclass  # dataclass decorator: auto-generates constructor and repr methods
class ZoneLevel:  # prospectivity zone level statistics dataclass
    """Statistics of one prospectivity zone level."""

    level: int                           # 1 = class A (exploration target), 2 = class B, 3 = class C (0 = non-prospect, not listed)
    name_cn: str                         # e.g. "Class-A prospect (exploration target)"
    prob_min: float = 0.0                # lower bound of this level's probability/score
    prob_max: float = 1.0                # upper bound
    n_cells: int = 0                     # number of cells
    area: float = 0.0                    # area (km²)
    area_ratio: float = 0.0              # area ratio (/ valid area)
    n_deposits: int = 0                  # number of known deposits falling in this level (0 = no deposits)
    capture_ratio: float | None = None   # deposit capture ratio (this level / total deposits, None when no deposits)

    def to_row(self) -> dict[str, Any]:  # method converting to one statistics-table row
        """Convert to one statistics-table row; the capture ratio column is omitted when there are no deposits."""
        row = {"级别": self.name_cn, "概率区间": f"{self.prob_min:.3g} ~ {self.prob_max:.3g}",  # assemble the row dict: level Chinese name and probability range (3 significant digits)
               "格点数": self.n_cells, "面积": f"{self.area:.2f}",  # continued: cell count and area (2 decimal places)
               "面积占比": f"{self.area_ratio * 100:.1f}%"}  # continued: area ratio formatted as a one-decimal percentage
        if self.capture_ratio is not None:  # add the two deposit columns only when capture ratio data exists
            row["矿点数"], row["捕获率"] = self.n_deposits, f"{self.capture_ratio * 100:.1f}%"  # write deposit count and capture ratio percentage string in one statement
        return row  # return the statistics row dict


@dataclass  # dataclass decorator: auto-generates constructor and repr methods
class ZoneResult:  # prospectivity zoning result dataclass
    """Prospectivity zoning result: zone grid (0 = non-prospect, 1 = A, 2 = B, 3 = C) + per-level statistics."""

    zone_grid: np.ndarray | None = None  # zone grid (ny,nx): 0 = non-prospect, 1/2/3 = A/B/C
    method: str = "quantile"             # quantile / jenks / manual
    name_cn: str = "分位数分级"  # Chinese name of the zoning method (default: quantile zoning)
    levels: list[ZoneLevel] = field(default_factory=list)  # per-level statistics list (empty by default)
    n_valid_cells: int = 0               # number of valid evaluation cells
    notes: list[str] = field(default_factory=list)  # notes list (empty by default)


@dataclass  # dataclass decorator: auto-generates constructor and repr methods
class ProspectivityReport:  # dataclass for the complete report of one 2D prospectivity mapping run
    """Complete report of one 2D prospectivity mapping run (all products of both stages)."""

    n_samples: int = 0  # total number of input samples (0 by default)
    n_coord_samples: int = 0             # number of samples with coordinates
    coord_system: str = ""               # coordinate system description
    is_geo: bool = False                 # whether the input is lon/lat
    elements: list[str] = field(default_factory=list)   # elements participating in anomaly identification
    grid_def: GridDef | None = None  # prediction grid definition (None when the grid is not built)
    interpolation: str = "idw"           # idw / ordinary_kriging
    evidence_layers: list[EvidenceLayer] = field(default_factory=list)  # stage-one products
    n_deposits: int = 0                  # number of known deposits (0 = unsupervised/knowledge-driven mode)
    deposit_cells: list[tuple[int, int]] = field(default_factory=list)  # deposit cell placement (row,col)
    model: FusionModel | None = None     # stage-two product: fusion model
    zones: ZoneResult | None = None      # stage-two product: prospect zoning
    evaluation: dict[str, Any] = field(default_factory=dict)  # when deposits exist: AUC, P-A curve, etc.
    warnings: list[str] = field(default_factory=list)  # warnings list (empty by default)
    plot_paths: list[str] = field(default_factory=list)  # list of generated plot paths (empty by default)
    summary_text: str = ""               # text summary (injectable into Agent prompt / report)

    def layers_by(self, kind: str) -> list[EvidenceLayer]:  # method filtering evidence layers by kind
        """Filter evidence layers by kind (for plotting per-panel figures / summary)."""
        return [ly for ly in self.evidence_layers if ly.kind == kind]  # list comprehension filtering evidence layers whose kind matches
