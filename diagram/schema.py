"""Data contract layer: diagram specs (DiagramSpec/AxisSpec/RegionSpec/BoundarySpec),
Mode A interpretation output contract (BoundaryReading and diagram/region/boundary specs).

Design principles: (1) the first 7 keys of a Mode A output are a stable contract
(STABLE_READING_KEYS); extension keys are append-only and never renamed;
(2) Mode B always outputs interval strings + 0-1 confidence, never a bare single
point posing as exact; (3) boundary entries must carry provenance and fidelity,
and interpretations based on approximate encoding must be downgraded with a
warning. This module has no third-party dependencies and can be safely imported
by any entry point.
"""

from __future__ import annotations  # Enable deferred evaluation of type annotations (the | syntax on older Pythons)

from dataclasses import dataclass, field  # Import the dataclass decorator and the field default factory
from enum import Enum  # Import the Enum base class (defines coordinate scales)
import math  # Import math (floor/ceil/log10 for interval rounding)
from typing import Any  # Import the Any type annotation

class ScaleKind(Enum):  # Coordinate scale enum (linear/log/ternary)
    """Coordinate scale of a single axis (or of a ternary diagram as a whole).

    LOG10: take log10 before plotting; distances/intervals are computed in the
    transformed space. TERNARY: ternary diagram (A-B-C end-members normalized
    to 100%); the current library has no entries yet, so the enum keeps full
    support for image classification and future extension.
    """
    LINEAR = "linear"  # Enum member: linear scale
    LOG10 = "log10"  # Enum member: base-10 logarithmic scale
    TERNARY = "ternary"  # Enum member: ternary diagram scale

@dataclass  # Declare the class below a dataclass (auto-generates __init__/__repr__)
class AxisSpec:  # Single-axis spec dataclass
    """Single-axis spec: what quantity to plot, its unit, scale and display range.

    expression follows the plotting conventions: "Y" takes the element directly
    (ppm), "Sr/Y" divides two elements, and "Yb_N" is a chondrite-normalized
    key; axis_range is the display range. Points outside the range can still be
    interpreted, but a warning is raised and data_quality is downgraded.
    """
    key: str  # Axis key "x"/"y"; used to index the coordinate dict and JSON output
    expression: str  # Value expression (element or A/B ratio), consistent with plotting
    label: str  # Axis label (for JSON display, e.g. "Y", "Sr/Y")
    unit: str  # Unit ("ppm"/"ratio"/"wt%" etc.; ratio axes use "ratio")
    scale: ScaleKind = ScaleKind.LINEAR  # Coordinate scale
    axis_range: tuple[float, float] = (0.0, 100.0)  # Display range [min, max] (data space)

@dataclass  # Declare the class below a dataclass
class RegionSpec:  # Discrimination region dataclass
    """Discrimination region: closed polygon + Chinese name + evidence tags.

    polygon_vertices are data-space vertices (not pixels) and may extend beyond
    axis_range (clipped before testing, so "open regions" need no infinite
    vertices); when a point lies exactly on a boundary shared with an adjacent
    region, it belongs to the region with the smaller priority value.
    """
    # Used directly as the value of the output region field (e.g. "high_SrY_field", matching the contract example)
    region_id: str  # Region ID field declaration (value of the output region field)
    name_cn: str  # Chinese name (used to generate visual_conclusion)
    polygon_vertices: list[list[float]]  # Data-space vertices [[x, y], ...] (closed automatically)
    description: str = ""  # One-line geological meaning (goes into region_description)
    evidence_tags: list[str] = field(default_factory=list)  # Process evidence tags
    priority: int = 0  # Priority when boundaries coincide (smaller wins)

@dataclass  # Declare the class below a dataclass
class BoundarySpec:  # Explicit discrimination boundary dataclass
    """One explicit discrimination boundary: a polyline plus the regions it
    separates (boundaries are first-class objects).

    Region polygon edges include "plot-frame clipping edges" (not discrimination
    boundaries; treating them as boundaries would falsely report "close to the
    boundary"); boundary_distance is computed only over explicitly declared
    boundaries; diagrams without declared boundaries fall back to "all region
    polygon edges".
    """
    boundary_id: str  # Key used in boundary_distances_detail
    polyline_vertices: list[list[float]]  # Data-space polyline vertices (a straight line on a log axis needs only its two endpoints)
    separates: list[str] = field(default_factory=list)  # IDs of the separated regions (for diagnostics/matching)
    description: str = ""  # One-line description (e.g. "Sr/Y=40 线：埃达克质下界")

@dataclass  # Declare the class below a dataclass
class DiagramSpec:  # Complete spec dataclass for a single diagram
    """Complete spec for one discrimination diagram: two axes + region set +
    provenance metadata.

    fidelity/verification_status implement the "do not pretend to be exact"
    principle: entries encoded approximately (e.g. the Pearce series encoded
    from memory, status != "verified") must carry a warning during
    interpretation.
    """
    diagram_id: str  # Diagram ID (library key, snake_case)
    display_name: str  # Display name (value of the output diagram field)
    category: str  # Category (aligned with plot_config C1-C7)
    axes: list[AxisSpec]  # Two axes [x_axis, y_axis] (ternary diagrams will add a third axis later)
    regions: list[RegionSpec]  # Set of discrimination regions
    boundaries: list[BoundarySpec] = field(default_factory=list)  # Explicit boundaries (recommended to declare)
    references: list[str] = field(default_factory=list)  # Reference citations (digitization provenance)
    fidelity: float = 1.0  # Boundary digitization fidelity 0-1 (1 = exact to the original figure)
    verification_status: str = "verified"  # "verified"|"approximate_digitization_required"
    encoding_notes: str = ""  # How the boundaries were encoded (e.g. "矩形简化：Sr/Y>=40 且 Y<=18")

    def axis(self, key: str) -> AxisSpec:  # Return the axis by key ("x"/"y")
        """Return the axis by key ("x"/"y"); raises KeyError if absent."""
        for ax in self.axes:  # Iterate over this diagram's axis list
            if ax.key == key:  # Hit when the key equals the target axis key
                return ax  # Return the matching axis spec object
        raise KeyError(f"图解 {self.diagram_id} 无坐标轴 {key!r}")  # Raise KeyError when the loop ends without a match

    def validate(self) -> list[str]:  # Structural self-check method returning a problem list
        """Structural self-check: returns a problem list (empty list = pass).

        Called when the library loads; problems do not block loading (so the
        boundary library can grow incrementally) but are recorded in the
        library warnings and passed through in interpretation output.
        """
        problems: list[str] = []  # Initialize the problem list (empty = pass)
        if len(self.axes) != 2:  # Check that the number of axes is 2
            problems.append(f"{self.diagram_id}: 轴数 {len(self.axes)} != 2")  # Record the wrong axis count
            return problems  # Early return on a wrong axis count (later checks assume two axes)
        if not self.regions:  # Check that at least one region is defined
            problems.append(f"{self.diagram_id}: 未定义任何判别区域")  # Record the missing-regions problem
        if not self.references:  # Check references are non-empty (provenance is mandatory)
            problems.append(f"{self.diagram_id}: 缺少文献出处（provenance 必填）")  # Record the missing-provenance problem
        for r in self.regions:  # Region polygons need at least 3 vertices
            if len(r.polygon_vertices) < 3:  # Check the minimum polygon vertex count
                problems.append(  # Append a too-few-polygon-vertices entry
                    f"{self.diagram_id}/{r.region_id}: 多边形顶点数 "  # First half of the message (implicit string concatenation)
                    f"{len(r.polygon_vertices)} < 3"  # Second half of the message: the actual vertex count
                )
        region_ids = {r.region_id for r in self.regions}  # Collect all region IDs with a set comprehension
        boundary_ids: set[str] = set()  # Initialize an empty set for boundary ID deduplication
        for bd in self.boundaries:  # Iterate over each explicit boundary
            if len(bd.polyline_vertices) < 2:  # Check the polyline has at least 2 vertices
                problems.append(  # Append a too-few-polyline-points entry
                    f"{self.diagram_id}/{bd.boundary_id}: 边界折线点数 "  # First half of the message
                    f"{len(bd.polyline_vertices)} < 2"  # Second half of the message: the actual point count
                )
            if bd.boundary_id in boundary_ids:  # Boundary ID uniqueness
                problems.append(f"{self.diagram_id}: 边界 ID 重复 {bd.boundary_id!r}")  # Record the duplicate boundary ID
            boundary_ids.add(bd.boundary_id)  # Register the current boundary ID in the set
            for rid in bd.separates:  # Regions referenced by separates must exist
                if rid not in region_ids:  # Check the referenced region ID is registered
                    problems.append(  # Append an unknown-region-reference entry
                        f"{self.diagram_id}/{bd.boundary_id}: separates 引用 "  # First half of the message
                        f"未知区域 {rid!r}"  # Second half of the message: the unknown region ID
                    )
        if not 0.0 <= self.fidelity <= 1.0:  # Check fidelity is within 0-1
            problems.append(f"{self.diagram_id}: fidelity {self.fidelity} 越界")  # Record the out-of-range fidelity
        return problems  # Return all problems (empty list means pass)

# The 7 stable field names of the user contract: any caller may rely on these 7 keys existing with unchanged names
STABLE_READING_KEYS: tuple[str, ...] = (  # Immutable tuple of the 7 stable output keys
    "diagram", "sample_id", "coordinates", "region",  # First 4 stable keys: diagram name/sample ID/coordinates/region
    "boundary_distance", "data_quality", "visual_conclusion",  # Last 3 stable keys: boundary distance/quality/Chinese conclusion
)

# Reading status: ok=success; insufficient_data=missing coordinates, cannot interpret; out_of_range=point outside the plot
READING_STATUS_OK = "ok"  # Status constant: interpretation succeeded
READING_STATUS_INSUFFICIENT = "insufficient_data"  # Status constant: insufficient data, cannot interpret
READING_STATUS_OUT_OF_RANGE = "out_of_range"  # Status constant: point outside the diagram display range

@dataclass  # Declare the class below a dataclass
class BoundaryReading:  # Mode A interpretation output contract dataclass
    """Mode A single-sample interpretation output (7 stable contract fields +
    extended diagnostics).

    coordinates keys = axis expressions (e.g. {"Sr/Y": 55.1, "Y": 11.8}).
    boundary_distance definition: the Euclidean distance, in the plot transform
    space (log10 on log axes), from the point to the nearest boundary segment
    (2 decimal places) — contract example A17 (Y=11.8) to the Y=18 boundary =
    |log10(11.8)-log10(18)| = 0.184 -> 0.18, matching the user contract
    example. relative_boundary_distance is the companion dimensionless
    quantity (each axis divided by its characteristic scale, then the
    minimum); <0.1 means near_boundary.
    """
    status: str  # ok / insufficient_data / out_of_range
    # ---- 7 stable contract fields (required when status=ok) ----
    diagram: str = ""  # Diagram display name field (empty string by default)
    sample_id: str = ""  # Sample ID field (empty string by default)
    coordinates: dict[str, float] = field(default_factory=dict)  # Coordinates field (keys = axis expressions), empty dict by default
    region: str | None = None  # Matched region ID field; None when nothing is hit
    boundary_distance: float | None = None  # Nearest boundary distance field (None when no boundaries/infinite)
    data_quality: str = ""  # Data quality grade field (low/medium/high)
    visual_conclusion: str = ""  # Chinese visual conclusion field
    # ---- Extended diagnostic fields (append-only, never renamed) ----
    diagram_id: str = ""  # Diagram ID (library key) field
    region_name_cn: str = ""  # Matched region Chinese name field
    region_description: str = ""  # Matched region geological description field
    region_distances: dict[str, float] = field(default_factory=dict)  # Distances from the point to each region
    relative_boundary_distance: float | None = None  # Dimensionless relative distance
    near_boundary: bool = False  # relative < 0.1
    boundary_distances_detail: dict[str, float] = field(default_factory=dict)  # Distance to each boundary
    data_quality_reasons: list[str] = field(default_factory=list)  # Reasons behind the quality grade
    warnings: list[str] = field(default_factory=list)  # When non-empty, data_quality is at most medium
    meta: dict[str, Any] = field(default_factory=dict)  # Engine version/boundary library fidelity

    def to_dict(self) -> dict[str, Any]:  # Serialize into a JSON-safe dict
        """Serialize into a JSON-safe dict: the 7 contract keys first, extended keys after."""
        stable = {  # Assemble the dict of the 7 stable contract keys
            "diagram": self.diagram, "sample_id": self.sample_id,  # Write the diagram display name and sample ID
            "coordinates": dict(self.coordinates), "region": self.region,  # Write a copy of the coordinates dict and the matched region
            "boundary_distance": self.boundary_distance,  # Write the nearest boundary distance
            "data_quality": self.data_quality,  # Write the data quality grade
            "visual_conclusion": self.visual_conclusion,  # Write the Chinese conclusion
        }
        extended = {  # Assemble the dict of extended diagnostic keys
            "status": self.status, "diagram_id": self.diagram_id,  # Write the reading status and diagram ID
            "region_name_cn": self.region_name_cn,  # Write the matched region Chinese name
            "region_description": self.region_description,  # Write the region geological description
            "region_distances": dict(self.region_distances),  # Write a copy of the per-region distances
            "relative_boundary_distance": self.relative_boundary_distance,  # Write the dimensionless relative distance
            "near_boundary": self.near_boundary,  # Write the near-boundary boolean flag
            "boundary_distances_detail": dict(self.boundary_distances_detail),  # Write a copy of the per-boundary distance details
            "data_quality_reasons": list(self.data_quality_reasons),  # Write a copy of the quality grade reasons
            "warnings": list(self.warnings), "meta": dict(self.meta),  # Write copies of the warnings list and metadata
        }
        return {**stable, **extended}  # Merge by dict unpacking (stable keys first, extended keys after)

def format_interval(lo: float, hi: float, significant_figures: int = 2) -> str:  # Interval formatting function (2 significant figures by default)
    """Numeric interval -> "lo–hi" string (en dash, 2 significant figures,
    rounded outward).

    The lower bound rounds down and the upper bound rounds up, so the true
    value never falls outside the interval due to rounding:
    52.34/57.81 -> "52–58"; 0.183/0.191 -> "0.18–0.2".
    """
    if hi < lo:  # Correct the order when the bounds are swapped
        lo, hi = hi, lo  # Swap lo/hi so lo <= hi

    def _round_outward(v: float, upward: bool) -> float:  # Inner helper rounding outward
        if v == 0:  # Zero needs no rounding
            return 0.0  # Return 0.0 directly
        factor = 10 ** (significant_figures - 1 - math.floor(math.log10(abs(v))))  # Compute the power-of-10 scaling factor from the significant figures
        scaled = v * factor  # Scale the value into the integer domain at the target precision
        return (math.ceil(scaled) if upward else math.floor(scaled)) / factor  # Round up/down then scale back to the original magnitude
    lo_r = _round_outward(lo, upward=False)  # Round the lower bound down (so the true value is never excluded)
    hi_r = _round_outward(hi, upward=True)  # Round the upper bound up
    # When the endpoints coincide (interval collapsed within rounding precision), expand each outward by one minimum step to avoid "52–52"
    if lo_r == hi_r:  # Check whether the rounded endpoints coincide
        step = 10 ** math.floor(math.log10(abs(hi_r) or 1.0)) if hi_r != 0 else 1.0  # Minimum step at the current magnitude (1 if zero)
        lo_r, hi_r = lo_r - step, hi_r + step  # Expand each outward by one step to avoid a degenerate interval

    def _render(v: float) -> str:  # Inner helper rendering a single value
        return str(int(v)) if float(v).is_integer() else str(v)  # Integers drop the ".0"
    return f"{_render(lo_r)}–{_render(hi_r)}"  # Join with an en dash into the "lo–hi" interval string

@dataclass  # Declare the class below a dataclass
class DataPointEstimate:  # Estimate dataclass for a point recovered from a figure in Mode B
    """A single data point recovered from a figure in Mode B: intervals only,
    never bare values.

    x/y_value_internal are internal point estimates (means of the calibration
    transform), for tuning/QC only; the official JSON strips them by default
    (redact_internal=True) to prevent them propagating downstream as exact
    readings.
    """
    pixel: tuple[float, float]  # Pixel coordinates (marker centroid)
    intervals: dict[str, str] = field(default_factory=dict)  # Per-axis intervals (keys = axis labels)
    x_value_internal: float | None = None  # Internal point estimate (not in official output)
    y_value_internal: float | None = None  # y-axis internal point estimate field (also not in official output)
    x_half_width: float | None = None  # Internal per-axis half-interval width (data space, diagnostics)
    y_half_width: float | None = None  # y-axis internal half-interval width field (diagnostics)
    point_detection_confidence: float = 0.0  # Point detection confidence 0-1
    has_error_bar: bool = False  # Whether an error bar was detected
    error_bar_half_px: dict[str, float] = field(default_factory=dict)  # Pixel half-lengths
    region: str | None = None  # Region assignment when diagram_hint is provided

    def to_dict(self, redact_internal: bool = True) -> dict[str, Any]:  # Serializable method with optional redaction
        """Serialize; internal point estimates are included only when redact_internal=False (for debugging)."""
        d: dict[str, Any] = {  # Initialize the output dict with the four base keys
            "pixel": [round(self.pixel[0], 1), round(self.pixel[1], 1)],  # Pixel coordinates rounded to 1 decimal place
            "intervals": dict(self.intervals),  # Copy the per-axis intervals dict
            "point_detection_confidence": round(self.point_detection_confidence, 3),  # Confidence rounded to 3 decimal places
            "has_error_bar": self.has_error_bar,  # Write whether an error bar was detected
        }
        if self.error_bar_half_px:  # Emit only when error bar data exists
            d["error_bar_half_px"] = {k: round(v, 1) for k, v in self.error_bar_half_px.items()}  # Per-axis pixel half-lengths written rounded to 1 decimal place
        if self.region is not None:  # Emit only when there is a region assignment
            d["region"] = self.region  # Write the region assignment
        if not redact_internal:  # Emit internal estimates only when the caller explicitly opts out of redaction
            d["_internal"] = {  # Attach the internal debug-only key
                "x_value": self.x_value_internal, "y_value": self.y_value_internal,  # Write the x/y internal point estimates
                "x_half_width": self.x_half_width, "y_half_width": self.y_half_width,  # Write the x/y internal half-interval widths
            }
        return d  # Return the serialized dict
