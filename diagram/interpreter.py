"""Mode A: deterministic diagram interpretation engine for when raw data is
available (same input always yields the same output; no LLM/fitting/sampling).
Five steps: read the raw data (SampleData's three dicts + expression parsing)
-> identify the diagram/axes/scales (DiagramSpec from the boundary library) ->
compute point coordinates deterministically (same expression conventions as
plotting) -> call the digitized discrimination boundaries
(boundaries.DiagramLibrary; boundaries are data) -> region assignment +
boundary distance, output the contract JSON.
Data quality: low = missing/non-positive coordinates (log axes)/point outside
the display range/unverified boundary library entry; medium = alteration or
weathering without corroboration, or the point falls in the 5% edge buffer of
an axis; high = everything else. Distance to a boundary never changes quality
(quality is only about input trustworthiness); near_boundary expresses that."""

from __future__ import annotations  # Enable deferred evaluation of type annotations

import math  # Import math (inf/isinf)
from dataclasses import dataclass  # Import the dataclass decorator (used by the interpretation engine class)
from typing import TYPE_CHECKING, Any  # Import the type-checking switch and the Any annotation

from oregenesis.diagram.schema import (  # Import constants and types from the contract layer (parenthesized multi-line import)
    READING_STATUS_INSUFFICIENT, READING_STATUS_OK, READING_STATUS_OUT_OF_RANGE,  # Import the three reading status constants
    BoundaryReading, DiagramSpec, ScaleKind,  # Import the reading output contract/diagram spec/scale enum
)
from oregenesis.diagram.boundaries import (  # Import geometry and lookup facilities from the boundary library (parenthesized multi-line import)
    DEFAULT_LIBRARY, DiagramLibrary, distance_point_to_segments,  # Import the default library singleton/library class/point-to-segment-set distance
    region_boundary_segments, region_contains_point, transform_value,  # Import the region boundary segments/region containment/value transform functions
)

if TYPE_CHECKING:  # Do not import models at runtime, keeping this module independently lightweight
    from oregenesis.platform.models import SamplePoint as SampleData  # Import the sample type aliased as SampleData, for type checking only

_ENGINE_VERSION = "diagram-interpreter 0.1.0"  # For provenance in the reading output meta
_EDGE_BUFFER_RATIO = 0.05  # Axis-edge buffer ratio: inside the band counts as "edge-adjacent" (downgraded to medium)
_NEAR_BOUNDARY_THRESHOLD = 0.1  # Relative-distance near-boundary threshold (consistent with schema.BoundaryReading)
_ALTERATION_KEYWORDS = ("蚀变", "风化", "氧化", "矿化")  # A hit without corroboration downgrades to medium

def _lookup_value(sample_dict: dict[str, Any], name: str) -> float | None:  # Look up a value by name (may return None)
    """Look up a value by name (case-insensitive); returns None when missing or not convertible to float."""
    lowered = name.lower()  # Lowercase the target key name for comparison
    for k, v in sample_dict.items():  # Iterate over all key-value pairs of the value dict
        if str(k).lower() == lowered:  # Hit when the lowercased key equals the target name
            try:  # The float() conversion may fail; be tolerant
                return float(v)  # On success return the float value
            except (TypeError, ValueError):  # Value is None, a non-numeric string, etc.
                return None  # Not convertible: return None
    return None  # Name not found after the loop: return None

def resolve_expression(context: dict[str, Any], expression: str) -> float | None:  # Axis expression resolver
    """Resolve an axis expression to a number: "Y", "Sr/Y", "(Y+Nb)"; returns
    None when unresolvable.

    context is the merged value dict (trace/ree/major + derived keys
    precomputed upstream, such as the normalized value "Yb_N"). For an A/B
    ratio, either side missing or a zero denominator -> None (never guess a
    number).
    """
    expr = expression.strip()  # Strip surrounding whitespace from the expression
    if "/" in expr:  # Ratio "A/B"
        num, den = expr.split("/", 1)  # Split out numerator and denominator at the first slash
        a = _lookup_value(context, num.strip())  # Look up the numerator value
        b = _lookup_value(context, den.strip())  # Look up the denominator value
        if a is None or b is None or b == 0:  # Either side missing or zero denominator
            return None  # Return None (never fabricate a number)
        return a / b  # Return the two-element ratio
    stripped = expr.strip("() ")  # Strip parentheses and whitespace from both sides of the expression
    if "+" in stripped:  # Two-term sum (used by Y+Nb in Pearce diagrams)
        values = [_lookup_value(context, p.strip()) for p in stripped.split("+")]  # Split on "+" and look up each term
        if any(v is None for v in values):  # Any term missing a value
            return None  # Return None
        return float(sum(values))  # type: ignore[arg-type]
    return _lookup_value(context, stripped)  # Single element / pre-normalized context key

def _merge_context(sample: SampleData | dict[str, Any]) -> dict[str, Any]:  # Merge a sample's value dicts
    """Merge the sample's three element dicts and explicit coordinates into one
    value dict (trace takes precedence).

    A dict input supports {"Sr": 650.2, "Y": 11.8} or hybrid forms with a
    "trace_elements" sub-dict; a SampleData input merges major/ree/trace.
    """
    if isinstance(sample, dict):  # Branch for plain dict input
        merged: dict[str, Any] = {}  # Initialize the merged value dict
        for key in ("major_elements", "ree_data", "trace_elements"):  # Try the three sub-dict keys in turn
            sub = sample.get(key)  # Fetch the sub-dict (may be absent)
            if isinstance(sub, dict):  # Only when it really is a dict
                merged.update(sub)  # Merge it into the merged dict
        for k, v in sample.items():  # Top-level scalar keys merged in directly
            if k not in ("major_elements", "trace_elements", "ree_data",  # Exclusion test for sub-dict keys (multi-line condition)
                         "isotope_data", "mineral_chemistry"):  # Exclusion list continued: isotope and mineral chemistry sub-dicts
                merged[k] = v  # Merge the top-level scalar key-value into the merged dict
        return merged  # Return the merged value dict
    merged = {}  # SampleData object branch: initialize the merged dict
    merged.update(sample.major_elements or {})  # Merge major elements (None treated as empty)
    merged.update(sample.ree_data or {})  # Merge REE data (None treated as empty)
    merged.update(sample.trace_elements or {})  # Merge trace elements (None treated as empty)
    return merged  # Return the merged value dict

def _assess_data_quality(  # Data quality grading function (multi-line signature)
    spec: DiagramSpec,  # Arg: diagram spec
    coords: dict[str, float | None],  # Arg: resolved coordinates (may contain None)
    sample_meta: dict[str, Any],  # Arg: sample metadata dict
    in_range: bool,  # Arg: whether the point is within the display range
    warnings: list[str],  # Arg: warning output list (appended in place)
) -> tuple[str, list[str]]:  # Signature end: returns (quality grade, reasons list)
    """Data quality grading; returns (quality, reasons). Rules in the module docstring."""
    reasons: list[str] = []  # Initialize the grading reasons list
    if any(v is None for v in coords.values()):  # ---- low-grade triggers ----
        missing = [ax.label for ax, v in zip(spec.axes, coords.values()) if v is None]  # Pair axes with coordinate values, collecting missing axis labels
        reasons.append(f"坐标缺失: {missing}")  # Record the missing-coordinates reason
        return "low", reasons  # Grade low and return immediately
    if not in_range:  # Point outside the diagram display range
        reasons.append("点在图解显示范围之外")  # Record the out-of-range reason
        warnings.append("点超出图解显示范围，判读结果仅供参考")  # Append a warning
        return "low", reasons  # Grade low and return
    if spec.verification_status != "verified":  # Boundary library entry unverified
        reasons.append(  # Record the unverified reason
            f"边界库条目未经验证（{spec.verification_status}，fidelity={spec.fidelity}）"  # Message includes the verification status and fidelity
        )
        warnings.append(  # Append the approximate-encoding warning
            "该图解判别边界为近似数字化编码，判读结果仅供初筛，"  # First half of the warning message (implicit string concatenation)
            "正式使用前请用数字化流水线对原图重新标定"  # Second half of the warning message
        )
        return "low", reasons  # Grade low and return
    state_desc = str(sample_meta.get("sample_state", ""))  # ---- medium-grade triggers ----
    if any(kw in state_desc for kw in _ALTERATION_KEYWORDS):  # Sample state description hits any alteration-related keyword
        reasons.append(f"样品状态描述含蚀变/风化类关键词（{state_desc}）")  # Record the keyword-hit reason
        warnings.append("蚀变/风化样品的移动元素（如 Sr）判读需结合不活动元素交叉验证")  # Append the alteration warning
        return "medium", reasons  # Grade medium and return
    for ax, v in zip(spec.axes, coords.values()):  # Axis-edge buffer band
        assert v is not None  # Type-narrowing assertion (None already excluded above)
        lo, hi = ax.axis_range  # Unpack the axis display range bounds
        span = hi - lo  # Axis display span
        if v < lo + span * _EDGE_BUFFER_RATIO or v > hi - span * _EDGE_BUFFER_RATIO:  # A point within either 5% buffer band counts as edge-adjacent
            reasons.append(f"{ax.label}={round(v, 4)} 贴近轴显示范围边缘")  # Record the edge-adjacent reason
            return "medium", reasons  # Grade medium and return
    reasons.append("坐标齐全、位于显示范围内、边界库条目已验证")  # Record the high-grade reason
    return "high", reasons  # All other cases grade high

def _conclusion(region_name_cn: str, near_boundary: bool, region: str | None) -> str:  # Chinese conclusion template function
    """Generate the visual_conclusion Chinese template (deterministic, no LLM)."""
    if region is None:  # The point fell into no region
        return "点未落入任何已数字化判别区域"  # Return the no-hit template
    if near_boundary:  # Close to a discrimination boundary
        return f"点位于{region_name_cn}，但贴近判别边界，归属需结合其他证据"  # Return the edge-adjacent caution template
    return f"点位于{region_name_cn}，距边界较远"  # Return the regular hit template

def _transformed_point(spec: DiagramSpec, point_xy: tuple[float, float]) -> tuple[float, float]:  # Helper: data-space point -> transformed-space point
    """Data-space point -> transformed-space point (transformed per both axis
    scales; boilerplate shared across this module)."""
    x_axis, y_axis = spec.axis("x"), spec.axis("y")  # Get the diagram's x/y axis specs
    return (  # Return the transformed point tuple
        transform_value(point_xy[0], x_axis.scale),  # x component transformed per the x-axis scale
        transform_value(point_xy[1], y_axis.scale),  # y component transformed per the y-axis scale
    )

@dataclass  # Declare the class below a dataclass (fields injectable)
class DiagramInterpreter:  # Mode A deterministic diagram interpretation engine
    """Mode A interpretation engine: holds the boundary library and performs
    deterministic diagram interpretation on samples.

    Usage::

        itp = DiagramInterpreter()
        reading = itp.read(sample, "sr_y_vs_y_adakite")
        print(reading.to_dict())   # 7 contract keys + extended diagnostics
    """
    # Boundary library: process-wide singleton by default; tests can inject an independent data directory
    library: DiagramLibrary = DEFAULT_LIBRARY  # Boundary library field: process-wide singleton by default (a test double can be injected)

    def read(  # Main interpretation entry method (multi-line signature)
        self, sample: SampleData | dict[str, Any], diagram_id: str  # Args: sample object or dict, target diagram ID
    ) -> BoundaryReading:  # Signature end: returns a reading object
        """Run the five-step interpretation on a single sample, returning a
        BoundaryReading.

        No step ever raises (missing coordinates etc.) — a reading with
        status=insufficient_data is returned stating what is missing (toolbox
        principle: no evidence without sufficient data, never fabricate).
        """
        spec = self.library.get(diagram_id)  # Step 1: fetch the diagram spec from the boundary library
        x_axis, y_axis = spec.axis("x"), spec.axis("y")  # Get the diagram's x/y axis specs
        context = _merge_context(sample)  # Step 2: merge the sample's three dicts into the value context
        sample_id = self._sample_id_of(sample)  # Extract the sample ID
        coords: dict[str, float | None] = {  # Initialize the coordinates dict (keys = axis expressions)
            x_axis.expression: resolve_expression(context, x_axis.expression),  # Resolve the x-axis expression (may be None)
            y_axis.expression: resolve_expression(context, y_axis.expression),  # Resolve the y-axis expression (may be None)
        }
        if any(v is None for v in coords.values()):  # 1. Missing coordinates: incomplete reading
            missing = [ax.label for ax, v in  # List comprehension start: pair axes with coordinates
                       zip(spec.axes, coords.values()) if v is None]  # Filter axis labels whose value is None
            return BoundaryReading(  # Build the "insufficient data" reading
                status=READING_STATUS_INSUFFICIENT, diagram=spec.display_name,  # Status = insufficient data, with the diagram display name
                sample_id=sample_id, coordinates={}, data_quality="low",  # Sample ID; empty coordinates; quality graded low
                visual_conclusion=f"原始数据缺少 {missing}，无法进行确定性判读",  # Chinese conclusion stating what is missing
                diagram_id=spec.diagram_id,  # Attach the diagram ID
                data_quality_reasons=[f"坐标缺失: {missing}"],  # Quality reason: missing coordinates
                warnings=[f"原始数据缺少坐标所需量: {missing}，无法判读"],  # Warn about the missing quantities
                meta=self._meta(spec),  # Attach engine/library metadata
            )
        return self._read_from_coords(  # Coordinates complete; proceed to steps 2-5
            spec,  # Pass the diagram spec
            {x_axis.expression: float(coords[x_axis.expression]),   # type: ignore[arg-type]
             y_axis.expression: float(coords[y_axis.expression])},  # type: ignore[arg-type]
            sample_id,  # Pass the sample ID
            self._meta_of(sample),  # Pass the sample metadata dict
        )

    def _read_from_coords(  # Core interpretation driven by coordinates (multi-line signature)
        self,  # Instance reference
        spec: DiagramSpec,  # Arg: diagram spec
        coords: dict[str, float],  # Arg: resolved coordinates (guaranteed numeric)
        sample_id: str,  # Arg: sample ID
        sample_meta: dict[str, Any],  # Arg: sample metadata
    ) -> BoundaryReading:  # Signature end: returns a reading object
        """Run steps 2-5 from resolved diagram coordinates (shared by read and
        read_coordinates)."""
        x_axis, y_axis = spec.axis("x"), spec.axis("y")  # Get the diagram's x/y axis specs
        x_val = coords[x_axis.expression]  # Fetch the x coordinate value
        y_val = coords[y_axis.expression]  # Fetch the y coordinate value
        point_xy = (x_val, y_val)  # Assemble the data-space point tuple
        warnings: list[str] = []  # Initialize the warnings list
        if x_axis.scale == ScaleKind.LOG10 and x_val <= 0:  # x axis is log and received a non-positive value
            warnings.append(f"X 轴为 log10 尺度，收到非正值 {x_val}，无法放置")  # Record the x-unplaceable warning
        if y_axis.scale == ScaleKind.LOG10 and y_val <= 0:  # y axis is log and received a non-positive value
            warnings.append(f"Y 轴为 log10 尺度，收到非正值 {y_val}，无法放置")  # Record the y-unplaceable warning
        if any("无法放置" in w for w in warnings):  # Any "cannot be placed" warning exists
            return BoundaryReading(  # Build the "insufficient data" reading
                status=READING_STATUS_INSUFFICIENT, diagram=spec.display_name,  # Status = insufficient data, with the diagram display name
                sample_id=sample_id,  # Attach the sample ID
                coordinates={x_axis.expression: x_val, y_axis.expression: y_val},  # Attach the raw coordinates (eases diagnosis)
                data_quality="low",  # Quality graded low
                visual_conclusion="坐标存在非正值，无法在 log 尺度上放置",  # Chinese conclusion explaining the non-positive value problem
                diagram_id=spec.diagram_id, data_quality_reasons=["log 轴收到非正值"],  # Attach the diagram ID and quality reason
                warnings=warnings, meta=self._meta(spec),  # Attach warnings and engine metadata
            )
        # 3. Region assignment: first hit in ascending priority order (deterministic shared-boundary assignment)
        hit_region = None  # Initialize the hit region to None
        for region in sorted(spec.regions, key=lambda r: (r.priority, r.region_id)):  # Iterate regions in ascending priority (then ID) order
            if region_contains_point(region, spec, point_xy):  # Test whether the point falls in the current region
                hit_region = region  # Record the first hit region
                break  # Stop at the first hit (deterministic shared-boundary assignment)
        # 4. Boundary distance: explicit boundaries first, falling back to region polygon edges (minimum distance in transformed space)
        boundary_detail, min_dist = self._boundary_distances(spec, point_xy)  # Compute per-boundary distance details and the minimum distance
        # 5. Relative distance and near-boundary decision
        rel_dist = self._relative_distance(spec, point_xy, min_dist)  # Compute the dimensionless relative boundary distance
        near_boundary = rel_dist < _NEAR_BOUNDARY_THRESHOLD if hit_region else False  # Apply the threshold only when a region was hit
        # Range check and data quality
        in_range = (  # Combine both axes' range checks
            x_axis.axis_range[0] <= x_val <= x_axis.axis_range[1]  # x within the x-axis display range
            and y_axis.axis_range[0] <= y_val <= y_axis.axis_range[1]  # and y within the y-axis display range
        )
        quality, reasons = _assess_data_quality(spec, coords, sample_meta, in_range, warnings)  # Run data quality grading
        if hit_region is None:  # Branch for the point falling into no region
            warnings.append("点未落入任何已数字化区域（可能在图框外或区域未覆盖）")  # Append the no-hit warning
            quality = "low"  # Force quality down to low
            reasons.append("点未落入任何判别区域")  # Add the downgrade reason
        status = READING_STATUS_OK if in_range else READING_STATUS_OUT_OF_RANGE  # Reading status decided by whether the point is in range
        return BoundaryReading(  # Build the complete reading
            status=status, diagram=spec.display_name, sample_id=sample_id,  # Status/diagram display name/sample ID
            coordinates={  # Coordinates dict (keys = axis expressions)
                x_axis.expression: round(x_val, 4),  # x coordinate rounded to 4 decimals
                y_axis.expression: round(y_val, 4),  # y coordinate rounded to 4 decimals
            },
            region=hit_region.region_id if hit_region else None,  # Hit region ID (None when no hit)
            boundary_distance=round(min_dist, 2) if min_dist < math.inf else None,  # Nearest boundary distance, 2 decimals (None when infinite)
            data_quality=quality,  # Data quality grade
            visual_conclusion=_conclusion(  # Generate the Chinese conclusion
                hit_region.name_cn if hit_region else "未知区域", near_boundary,  # Pass the region Chinese name and the near-boundary flag
                hit_region.region_id if hit_region else None,  # Pass the region ID (None when no hit)
            ),
            diagram_id=spec.diagram_id,  # Attach the diagram ID
            region_name_cn=hit_region.name_cn if hit_region else "",  # Hit region Chinese name (empty string when no hit)
            region_description=hit_region.description if hit_region else "",  # Hit region description (empty string when no hit)
            region_distances={  # Per-region distance diagnostics dict
                r.region_id: round(d, 4)  # Key = region ID, value = distance rounded to 4 decimals
                for r in spec.regions  # Iterate over all diagram regions
                for d in [self._min_distance_to_region(r, spec, point_xy)]  # Minimum distance from the point to that region
                if d < math.inf  # Filter out infinite distances (region outside the plot, etc.)
            },
            relative_boundary_distance=(  # Dimensionless relative distance field
                round(rel_dist, 4) if rel_dist < math.inf else None  # 4 decimals (None when infinite)
            ),
            near_boundary=near_boundary,  # Near-boundary boolean flag
            boundary_distances_detail={k: round(v, 4) for k, v in boundary_detail.items()},  # Per-boundary distances rounded to 4 decimals
            data_quality_reasons=reasons, warnings=warnings, meta=self._meta(spec),  # Quality reasons/warnings/engine metadata
        )

    def read_coordinates(  # Interpret directly from coordinates (multi-line signature)
        self, diagram_id: str, x: float, y: float, sample_id: str = "unnamed"  # Args: diagram ID, x/y coordinates, sample ID default
    ) -> BoundaryReading:  # Signature end: returns a reading object
        """Interpret directly from diagram coordinates (minimal entry point
        without raw data; no expression parsing).

        Typical use: cross-validate region assignment for points recovered by
        Mode B digitization (substitute both interval endpoints; if they land
        in different regions -> "crosses a boundary", which must be declared
        in the conclusion).
        """
        spec = self.library.get(diagram_id)  # Fetch the diagram spec from the boundary library
        x_axis, y_axis = spec.axis("x"), spec.axis("y")  # Get the diagram's x/y axis specs
        return self._read_from_coords(  # Delegate to the core method for steps 2-5
            spec,  # Pass the diagram spec
            {x_axis.expression: float(x), y_axis.expression: float(y)},  # Build the coordinates dict from the given x/y
            sample_id,  # Pass the sample ID
            {},  # Empty metadata (no sample meta information)
        )

    def read_batch(  # Batch interpretation method (multi-line signature)
        self, samples: list[SampleData | dict[str, Any]], diagram_id: str  # Args: sample list, target diagram ID
    ) -> list[BoundaryReading]:  # Signature end: returns a list of readings
        """Batch interpretation (order matches the input, easing item-by-item reconciliation)."""
        return [self.read(s, diagram_id) for s in samples]  # Call read per sample and collect results in order
    # ---- Internal helpers ----

    @staticmethod  # Declare a static method (no instance dependency)
    def _sample_id_of(sample: SampleData | dict[str, Any]) -> str:  # Static method extracting the sample ID
        if isinstance(sample, dict):  # Dict input branch
            return str(sample.get("sample_id", "unnamed"))  # Read the sample_id key, defaulting to "unnamed"
        return sample.sample_id  # Object input: read the attribute directly

    @staticmethod  # Declare a static method
    def _meta_of(sample: SampleData | dict[str, Any]) -> dict[str, Any]:  # Static method extracting sample metadata
        if isinstance(sample, dict):  # Dict input branch
            meta = sample.get("metadata")  # First fetch the metadata sub-dict
            if isinstance(meta, dict):  # Adopt directly when it really is a dict
                return meta  # Return the metadata dict
            return {k: sample[k] for k in ("sample_state",) if k in sample}  # Fallback: pick sample_state from the top level
        return {"sample_state": sample.sample_state, "alteration": sample.alteration}  # Object input: build metadata from attributes

    def _boundary_distances(  # Distances from the point to all boundaries (multi-line signature)
        self, spec: DiagramSpec, point_xy: tuple[float, float]  # Args: diagram spec, data-space point
    ) -> tuple[dict[str, float], float]:  # Signature end: returns (details dict, minimum distance)
        """Distances from the point to all explicit boundaries in transformed
        space -> ({boundary_id: d}, minimum).

        Without explicit boundaries, falls back to "all region polygon edges
        merged" (boundary_id uses "region_edges:<region_id>").
        """
        tpt = _transformed_point(spec, point_xy)  # Transform the point into plot space
        detail: dict[str, float] = {}  # Initialize the boundary distance details dict
        min_dist = math.inf  # Initialize the minimum distance to infinity
        if spec.boundaries:  # Explicit boundaries take precedence when present
            for bd in spec.boundaries:  # Iterate over each explicit boundary
                verts = [  # Transform the polyline vertices into plot space
                    (transform_value(v[0], spec.axis("x").scale),  # Vertex x transformed per the axis scale
                     transform_value(v[1], spec.axis("y").scale))  # Vertex y transformed per the axis scale
                    for v in bd.polyline_vertices  # Iterate over all polyline vertices
                ]
                d, _ = distance_point_to_segment_list(tpt, list(zip(verts, verts[1:])))  # Pair adjacent vertices into segments and take the minimum distance
                detail[bd.boundary_id] = d  # Record the distance keyed by boundary ID
                min_dist = min(min_dist, d)  # Update the global minimum distance
        else:  # Fallback branch without explicit boundaries
            for region in spec.regions:  # Iterate over all regions
                d, _ = distance_point_to_segment_list(  # Minimum distance from the point to all edges of that region
                    tpt, region_boundary_segments(region, spec)  # Pass the transformed point and the region boundary segments
                )
                detail[f"region_edges:{region.region_id}"] = d  # Record the distance under the region_edges:ID key
                min_dist = min(min_dist, d)  # Update the global minimum distance
        return detail, min_dist  # Return the distance details and the minimum

    def _min_distance_to_region(  # Minimum distance from the point to one region (multi-line signature)
        self, region: Any, spec: DiagramSpec, point_xy: tuple[float, float]  # Args: region, diagram spec, data-space point
    ) -> float:  # Signature end: returns the distance
        """Minimum distance from a point to a single region's polygon edges
        (used by the region_distances diagnostic field)."""
        d, _ = distance_point_to_segment_list(  # Minimum distance from the point to all edges of the region
            _transformed_point(spec, point_xy), region_boundary_segments(region, spec)  # Pass the transformed point and the region boundary segments
        )
        return d  # Return the minimum distance

    def _relative_distance(  # Dimensionless relative distance computation (multi-line signature)
        self, spec: DiagramSpec, point_xy: tuple[float, float], min_dist: float  # Args: diagram spec, data-space point, absolute minimum distance
    ) -> float:  # Signature end: returns the relative distance
        """Dimensionless relative distance: the minimum Euclidean distance to
        a boundary in transformed space normalized by each axis's
        characteristic scale.

        Normalization convention: log axes divide by 1 decade, linear axes by
        range/4 — after normalization distances are axis-independent and
        comparable, serving as the unified near_boundary threshold (<0.1).
        """
        if min_dist == math.inf:  # Absolute distance infinite (no usable boundaries)
            return math.inf  # Return infinity directly
        x_axis, y_axis = spec.axis("x"), spec.axis("y")  # Get the diagram's x/y axis specs
        sx, sy = self._axis_char_scales(spec)  # Get both axes' characteristic scales
        tpt = (  # Build the point coordinates in normalized space
            transform_value(point_xy[0], x_axis.scale) / sx,  # x transformed then divided by the x characteristic scale
            transform_value(point_xy[1], y_axis.scale) / sy,  # y transformed then divided by the y characteristic scale
        )
        if spec.boundaries:  # When explicit boundaries exist
            seg_sets = []  # Initialize the segment set groups
            for bd in spec.boundaries:  # Iterate over each explicit boundary
                verts = [  # Transform and normalize the polyline vertices
                    (transform_value(v[0], x_axis.scale) / sx,  # Vertex x transformed then divided by the characteristic scale
                     transform_value(v[1], y_axis.scale) / sy)  # Vertex y transformed then divided by the characteristic scale
                    for v in bd.polyline_vertices  # Iterate over all polyline vertices
                ]
                seg_sets.append(list(zip(verts, verts[1:])))  # Pair adjacent vertices into segments and add to the groups
        else:  # Fallback branch without explicit boundaries
            seg_sets = []  # Fallback: region polygon edges (transformed into normalized space first)
            for r in spec.regions:  # Iterate over all regions
                raw_segs = region_boundary_segments(r, spec)  # Fetch the region's transformed-space boundary segments
                seg_sets.append(  # Add to the groups after normalization
                    [((a[0] / sx, a[1] / sy), (b[0] / sx, b[1] / sy)) for a, b in raw_segs]  # Divide each segment endpoint by the characteristic scales
                )
        best = math.inf  # Initialize the best relative distance to infinity
        for segs in seg_sets:  # Iterate over each segment group
            d, _ = distance_point_to_segment_list(tpt, segs)  # Minimum distance from the point to that segment group
            best = min(best, d)  # Update the global minimum relative distance
        return best  # Return the minimum boundary distance in normalized space

    @staticmethod  # Declare a static method
    def _axis_char_scales(spec: DiagramSpec) -> tuple[float, float]:  # Static method solving each axis's characteristic scale
        """Each axis's characteristic scale: 1 decade on log axes; range/4 on linear axes."""
        scales = []  # Initialize the scales list
        for ax in (spec.axis("x"), spec.axis("y")):  # Iterate over the x/y axes
            scales.append(1.0 if ax.scale == ScaleKind.LOG10  # Log axes take 1 (one decade) as the characteristic scale
                          else (ax.axis_range[1] - ax.axis_range[0]) / 4.0)  # Linear axes take a quarter of the display range
        return (scales[0], scales[1])  # Return the (x scale, y scale) tuple

    @staticmethod  # Declare a static method
    def _meta(spec: DiagramSpec) -> dict[str, Any]:  # Static method building the output meta info
        return {  # Build the provenance metadata dict
            "engine": _ENGINE_VERSION,  # Engine version
            "boundary_fidelity": spec.fidelity,  # Boundary library entry fidelity
            "verification_status": spec.verification_status,  # Boundary library entry verification status
        }

def _axis_distance(p: float, a: float, b: float) -> float:  # Utility: distance from an axis coordinate to the projected interval
    """Distance from a point's coordinate on one axis to the segment's
    projected interval [min(a,b), max(a,b)].

    Kept as a geometry utility (covered by tests); the interpretation path has
    switched to the normalized Euclidean distance.
    """
    lo, hi = (a, b) if a <= b else (b, a)  # Sort the endpoints to get the interval bounds
    if p < lo:  # Point left of the interval
        return lo - p  # Return the distance to the lower bound
    if p > hi:  # Point right of the interval
        return p - hi  # Return the distance to the upper bound
    return 0.0  # Point inside the interval: distance 0

def distance_point_to_segment_list(  # Minimum distance from a point to a segment list (multi-line signature)
    pt: tuple[float, float],  # Arg: the test point
    segments: list[tuple[tuple[float, float], tuple[float, float]]],  # Arg: segment list (two endpoints per segment)
) -> tuple[float, int]:  # Signature end: returns (minimum distance, hit index)
    """Minimum distance from a point to a segment list (same logic as
    boundaries.distance_point_to_segments; keeps the interpreter's import
    surface self-contained)."""
    return distance_point_to_segments(pt, segments)  # Delegate directly to the boundary library's implementation
