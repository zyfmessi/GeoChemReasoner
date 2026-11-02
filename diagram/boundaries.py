"""Digitized discrimination boundary library: diagram spec JSON loading +
deterministic geometry computation.

Discrimination boundaries have been digitized once into machine-readable JSON
(region polygons + explicit boundary polylines + provenance metadata); from
then on interpretation is purely deterministic geometry: scale transform
(linear/log10) -> point-in-polygon (ray casting) -> point-to-boundary-segment
distance (Euclidean in transformed space). The Mode A interpreter and the
interpretation engine (interpreter) obtain boundaries through this module.
The geometry is implemented in pure Python (performance is ample at the scale
of a few hundred vertices; no numpy is imposed on lightweight callers).
"""

from __future__ import annotations  # Enable deferred evaluation of type annotations

import json  # Import json (parses boundary library data files)
import math  # Import math (log10/hypot/inf)
from pathlib import Path  # Import Path (locates the data directory)
from typing import Any  # Import the Any type annotation

from oregenesis.diagram.schema import (  # Import spec types from the contract layer (parenthesized multi-line import)
    AxisSpec,  # Import the single-axis spec dataclass
    BoundarySpec,  # Import the explicit boundary spec dataclass
    DiagramSpec,  # Import the diagram spec dataclass
    RegionSpec,  # Import the discrimination region spec dataclass
    ScaleKind,  # Import the coordinate scale enum
)

_DATA_DIR = Path(__file__).resolve().parent / "data"  # One JSON file per diagram
_EPS = 1e-9  # Boundary tolerance (transformed-space units): edge-adjacent assignment is first-come-first-served (priority)

# Legal aliases for the scale field in JSON
_SCALE_ALIASES: dict[str, ScaleKind] = {  # Mapping from scale string aliases to enum members
    "linear": ScaleKind.LINEAR, "log10": ScaleKind.LOG10,  # Aliases linear/log10 -> the corresponding enum members
    "log": ScaleKind.LOG10, "ternary": ScaleKind.TERNARY,  # Aliases log/ternary -> the corresponding enum members
}

def transform_value(value: float, scale: ScaleKind) -> float:  # Data value -> plot transform value
    """Data value -> plot transform value: log10 on log axes, identity on
    linear axes.

    Distances/intervals/boundary matching all happen in transformed space —
    this is the key to boundary_distance matching the contract example
    (A17 -> 0.18): on a log axis the distance from 11.8 to 18 is
    log10(18/11.8) = 0.184, not 6.2.
    """
    if scale == ScaleKind.LOG10:  # Check for a log10-scale axis
        if value <= 0:  # Log axes reject non-positive values
            raise ValueError(f"log10 轴收到非正值 {value}，无法变换")  # Raise on non-positive values
        return math.log10(value)  # Take the common logarithm to complete the transform
    return float(value)  # Identity on linear axes (float() to guarantee a numeric type)

def inverse_transform_value(value: float, scale: ScaleKind) -> float:  # Plot transform value -> data value inverse transform
    """Plot transform value -> data value: 10**v on log10 axes, identity on linear axes."""
    if scale == ScaleKind.LOG10:  # Check for a log10-scale axis
        return 10.0 ** value  # Inverse transform: 10 to the power of value
    return float(value)  # Identity on linear axes

def point_in_polygon(pt: tuple[float, float], polygon: list[list[float]]) -> bool:  # Ray-casting point-in-polygon test
    """Ray casting (even-odd) test for whether a point lies inside a polygon
    (transformed-space coordinates).

    Cases where the point lies exactly on a vertex/edge are handled first by
    the caller with tolerance logic (shared boundaries assigned by
    RegionSpec.priority).
    """
    x, y = pt  # Unpack the test point's coordinates
    inside = False  # Initialize the inside flag (False = outside)
    n = len(polygon)  # Number of polygon vertices
    for i in range(n):  # Iterate over each polygon edge
        x1, y1 = polygon[i]  # Edge start vertex
        x2, y2 = polygon[(i + 1) % n]  # Edge end vertex (the last edge wraps to the first vertex)
        if (y1 > y) != (y2 > y):  # The edge can only cross when its endpoints straddle the point's horizontal line
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)  # Intersection of the edge with the horizontal line
            if x < x_cross:  # The ray crosses the edge once when the point is left of the intersection
                inside = not inside  # Each crossing flips the inside flag
    return inside  # Return the final inside/outside verdict

def distance_point_to_segment(  # Minimum distance from a point to a single segment (multi-line signature)
    pt: tuple[float, float], seg_a: tuple[float, float], seg_b: tuple[float, float]  # Args: point, segment endpoint A, endpoint B
) -> float:  # Signature end: returns the distance as a float
    """Minimum Euclidean distance from a point to a segment (transformed space)."""
    px, py = pt  # Unpack the point coordinates
    ax, ay = seg_a  # Unpack endpoint A
    bx, by = seg_b  # Unpack endpoint B
    dx, dy = bx - ax, by - ay  # Segment direction vector components
    seg_len_sq = dx * dx + dy * dy  # Squared segment length (avoids a square root)
    if seg_len_sq < _EPS:  # Degenerate segment (coincident endpoints): plain point distance
        return math.hypot(px - ax, py - ay)  # Reduces to the Euclidean distance to endpoint A
    # Project the parameter t and clamp to [0,1] -> nearest point on the segment
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))  # Clamp the projection parameter into the segment's range
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))  # Euclidean distance to the nearest point on the segment

def distance_point_to_segments(  # Minimum distance from a point to a segment set (multi-line signature)
    pt: tuple[float, float],  # Arg: the test point
    segments: list[tuple[tuple[float, float], tuple[float, float]]],  # Arg: segment list (two endpoints per segment)
) -> tuple[float, int]:  # Signature end: returns (minimum distance, hit segment index)
    """Minimum distance from a point to a segment set; returns
    (distance, hit segment index). An empty set returns (inf, -1)."""
    best_d, best_i = math.inf, -1  # Initialize the best distance to infinity and the index to -1
    for i, (a, b) in enumerate(segments):  # Iterate over the segments with indices
        d = distance_point_to_segment(pt, a, b)  # Distance from the point to the current segment
        if d < best_d:  # When closer than the current best
            best_d, best_i = d, i  # Update the best distance and hit index
    return best_d, best_i  # Return the minimum distance and hit index (inf/-1 for an empty set)

def clip_polygon_to_range(  # Clip a polygon to the axis-range rectangle (multi-line signature)
    polygon: list[list[float]],  # Arg: data-space polygon vertices
    xrange: tuple[float, float],  # Arg: x-axis display range
    yrange: tuple[float, float],  # Arg: y-axis display range
) -> list[list[float]]:  # Signature end: returns the clipped vertex list
    """Sutherland-Hodgman clipping of the polygon into the axis display-range
    rectangle.

    Vertices may take values "beyond the plot range" to express open regions
    (e.g. capped at 200 on the Sr/Y axis); clip before testing so that
    "boundary parts outside the plot" do not participate in boundary
    distances — otherwise an edge drawn outside the plot could falsely become
    the nearest boundary.
    """
    xmin, xmax = xrange  # Unpack the x range bounds
    ymin, ymax = yrange  # Unpack the y range bounds

    def clip_edge(pts: list[list[float]], inside: Any, intersect: Any) -> list[list[float]]:  # Inner helper clipping against one rectangle edge
        if not pts:  # Short-circuit on an empty vertex list
            return []  # Return an empty list
        out: list[list[float]] = []  # Initialize the output vertex list
        prev, prev_in = pts[-1], inside(pts[-1])  # Start from the last point (closed loop), with its side pre-computed
        for cur in pts:  # Iterate over each current vertex
            cur_in = inside(cur)  # Whether the current point is on the inner side
            if cur_in:  # Current point inside
                if not prev_in:  # Previous point outside -> this segment enters from outside
                    out.append(intersect(prev, cur))  # Add the frame intersection first
                out.append(cur)  # Add the current inside point
            elif prev_in:  # Previous inside, current outside -> leaving from inside
                out.append(intersect(prev, cur))  # Add only the frame intersection
            prev, prev_in = cur, cur_in  # Roll forward the previous point and its side state
        return out  # Return the clipped vertex list
    pts = [[float(p[0]), float(p[1])] for p in polygon]  # Convert all vertices to float pairs
    # Clip against the 4 rectangle edges in turn (inside test + intersection solver given as pairs)
    pts = clip_edge(pts, lambda p: p[0] >= xmin,  # Clip against the left edge (x>=xmin), keeping the inside
                    lambda a, b: [xmin, a[1] + (b[1] - a[1]) * (xmin - a[0]) / (b[0] - a[0])])  # Intersection of the segment with x=xmin
    pts = clip_edge(pts, lambda p: p[0] <= xmax,  # Clip against the right edge (x<=xmax), keeping the inside
                    lambda a, b: [xmax, a[1] + (b[1] - a[1]) * (xmax - a[0]) / (b[0] - a[0])])  # Intersection of the segment with x=xmax
    pts = clip_edge(pts, lambda p: p[1] >= ymin,  # Clip against the bottom edge (y>=ymin), keeping the inside
                    lambda a, b: [a[0] + (b[0] - a[0]) * (ymin - a[1]) / (b[1] - a[1]), ymin])  # Intersection of the segment with y=ymin
    pts = clip_edge(pts, lambda p: p[1] <= ymax,  # Clip against the top edge (y<=ymax), keeping the inside
                    lambda a, b: [a[0] + (b[0] - a[0]) * (ymax - a[1]) / (b[1] - a[1]), ymax])  # Intersection of the segment with y=ymax
    return pts  # Return the polygon vertices clipped by all four edges

def region_boundary_segments(  # Region boundary -> transformed-space segment list (multi-line signature)
    region: RegionSpec, spec: DiagramSpec  # Args: region spec and its owning diagram spec
) -> list[tuple[tuple[float, float], tuple[float, float]]]:  # Signature end: returns a segment list
    """Region boundary -> transformed-space segment list (closed polyline split
    after clipping to the axis display range).

    Unified distance convention: take the minimum over the point vs all
    boundary segments of all regions — this naturally covers both "point to a
    shared edge of two regions" and "point to the diagram's outer clipping
    edge".
    """
    x_axis, y_axis = spec.axis("x"), spec.axis("y")  # Get the diagram's x/y axis specs
    clipped = clip_polygon_to_range(  # Clip the region polygon to the display range
        region.polygon_vertices, x_axis.axis_range, y_axis.axis_range  # Args: raw vertices and both axes' display ranges
    )
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []  # Initialize the transformed-space segment list
    n = len(clipped)  # Vertex count after clipping
    for i in range(n):  # Iterate over vertices to build closed edges
        a, b = clipped[i], clipped[(i + 1) % n]  # Adjacent vertex pair (the last point wraps to the first)
        segments.append((  # Append the transformed segment
            (transform_value(a[0], x_axis.scale), transform_value(a[1], y_axis.scale)),  # Endpoint a transformed per both axis scales
            (transform_value(b[0], x_axis.scale), transform_value(b[1], y_axis.scale)),  # Endpoint b transformed per both axis scales
        ))
    return segments  # Return all boundary segments

def region_contains_point(  # Whether a data-space point falls inside a region (multi-line signature)
    region: RegionSpec, spec: DiagramSpec, point_xy: tuple[float, float]  # Args: region spec, diagram spec, data-space point
) -> bool:  # Signature end: returns a boolean verdict
    """Whether a data-space point falls inside the region (point and polygon
    vertices are first transformed into plot space).

    Ray casting is unstable for points exactly on a boundary (especially when
    a region edge coincides with the axis display range, e.g. the right edge
    x=100 of the Pearce diagram) — after a strict outside verdict, do one
    edge-adjacent recheck: the point counts as inside if its distance to any
    edge is < 1e-6 (numerically coincident).
    """
    x_axis, y_axis = spec.axis("x"), spec.axis("y")  # Get the diagram's x/y axis specs
    tpt = (  # Build the transformed-space point tuple
        transform_value(point_xy[0], x_axis.scale),  # x component transformed per the x-axis scale
        transform_value(point_xy[1], y_axis.scale),  # y component transformed per the y-axis scale
    )
    clipped = clip_polygon_to_range(  # Clip the region polygon to the display range
        region.polygon_vertices, x_axis.axis_range, y_axis.axis_range  # Args: raw vertices and both axes' display ranges
    )
    if len(clipped) < 3:  # Fewer than 3 vertices after clipping (fully outside the plot)
        return False  # Treat the point as not contained
    tverts = [  # Transform all clipped vertices into plot space
        [transform_value(v[0], x_axis.scale), transform_value(v[1], y_axis.scale)]  # Single vertex transformed per both axis scales
        for v in clipped  # Iterate over the clipped vertices
    ]
    if point_in_polygon(tpt, tverts):  # Ray-casting test for whether the point is inside the polygon
        return True  # Inside: return True directly
    # Edge-adjacent points are treated as inside; assignment ambiguity is left to the near_boundary/downgrading mechanism
    segs = list(zip(tverts, tverts[1:] + tverts[:1]))  # Pair the vertices into a list of closed edges
    d, _ = distance_point_to_segments(tpt, segs)  # Minimum distance from the point to all edges
    return d < 1e-6  # Distance < 1e-6 means edge-adjacent, treated as inside

def _spec_from_dict(payload: dict[str, Any], source_file: Path) -> DiagramSpec:  # Parse a boundary library JSON dict into a spec object
    """Boundary library JSON dict -> DiagramSpec (raises ValueError with the file name on format mismatch)."""
    try:  # Use try to catch all format errors and attach the file name
        axes = [  # Parse the axes array into a list of axis specs
            AxisSpec(  # Build one axis spec
                key=ax["key"], expression=ax["expression"],  # Required fields: axis key and value expression
                label=ax.get("label", ax["expression"]), unit=ax.get("unit", ""),  # Optional fields: label defaults to the expression, unit defaults to empty
                scale=_SCALE_ALIASES[ax.get("scale", "linear")],  # Map the scale string to the enum via the alias table (default linear)
                axis_range=tuple(ax.get("axis_range", [0.0, 100.0])),  # type: ignore[arg-type]
            )
            for ax in payload["axes"]  # Iterate over the JSON axes array
        ]
        regions = [  # Parse the regions array into a list of region specs
            RegionSpec(  # Build one region spec
                region_id=rg["region_id"], name_cn=rg["name_cn"],  # Required fields: region ID and Chinese name
                polygon_vertices=[list(map(float, v)) for v in rg["polygon_vertices"]],  # Convert each vertex to a float list
                description=rg.get("description", ""),  # Optional field: geological description defaults to empty
                evidence_tags=list(rg.get("evidence_tags", [])),  # Optional field: evidence tags default to an empty list
                priority=int(rg.get("priority", 0)),  # Optional field: priority defaults to 0
            )
            for rg in payload["regions"]  # Iterate over the JSON regions array
        ]
        boundaries = [  # Parse the boundaries array into a list of boundary specs
            BoundarySpec(  # Build one boundary spec
                boundary_id=bd["boundary_id"],  # Required field: boundary ID
                polyline_vertices=[list(map(float, v)) for v in bd["polyline_vertices"]],  # Convert each polyline vertex to a float list
                separates=list(bd.get("separates", [])),  # Optional field: separated regions default to an empty list
                description=bd.get("description", ""),  # Optional field: description defaults to empty
            )
            for bd in payload.get("boundaries", [])  # Iterate over boundaries (may be absent, defaulting to empty)
        ]
        return DiagramSpec(  # Build and return the diagram spec
            diagram_id=payload["diagram_id"], display_name=payload["display_name"],  # Required fields: diagram ID and display name
            category=payload.get("category", ""), axes=axes, regions=regions,  # Category defaults to empty; pass in the axis and region lists
            boundaries=boundaries, references=list(payload.get("references", [])),  # Pass in the boundary list and references
            fidelity=float(payload.get("fidelity", 1.0)),  # Fidelity defaults to 1.0
            verification_status=payload.get("verification_status", "verified"),  # Verification status defaults to verified
            encoding_notes=payload.get("encoding_notes", ""),  # Encoding notes default to empty
        )
    except (KeyError, TypeError, ValueError) as exc:  # Catch the three kinds of format errors: missing key/type/value
        raise ValueError(f"边界库文件 {source_file.name} 格式错误: {exc}") from exc  # Wrap into a ValueError with the file name, keeping the cause

class DiagramLibrary:  # Digitized discrimination boundary library
    """Digitized discrimination boundary library: loads data/*.json and provides
    deterministic lookups.

    Usage (the interpreter obtains boundaries uniformly through this)::

        lib = DiagramLibrary()                 # or DEFAULT_LIBRARY
        spec = lib.get("sr_y_vs_y_adakite")    # raises KeyError when missing
        hits = lib.find_by_axis("Sr/Y", "Y")   # search by axis expressions
    """

    def __init__(self, data_dir: Path | None = None) -> None:  # Constructor (data directory injectable)
        self._specs: dict[str, DiagramSpec] = {}  # Initialize the diagram_id -> spec registry
        self.warnings: list[str] = []  # Initialize the loading warnings list
        directory = data_dir if data_dir is not None else _DATA_DIR  # Fall back to the package data directory when none is given
        if directory.is_dir():  # Attempt loading only if the directory exists
            for path in sorted(directory.glob("*.json")):  # Iterate over all JSON files sorted by name
                try:  # One file failing to parse must not affect the others
                    payload = json.loads(path.read_text(encoding="utf-8"))  # Read the file content and parse as JSON
                    spec = _spec_from_dict(payload, path)  # Parse the dict into a DiagramSpec
                except (json.JSONDecodeError, ValueError) as exc:  # Catch JSON parse errors and format errors
                    self.warnings.append(str(exc))  # Record the warning message
                    continue  # Skip the bad file and keep loading
                self.warnings.extend(spec.validate())  # Problems do not block loading; record and pass through
                self._specs[spec.diagram_id] = spec  # Register the spec keyed by diagram_id

    def get(self, diagram_id: str) -> DiagramSpec:  # Get a diagram spec by ID
        """Get a diagram spec by ID; raises KeyError when unregistered (callers should call list first)."""
        if diagram_id not in self._specs:  # Check for an unregistered ID
            raise KeyError(f"边界库中无图解 {diagram_id!r}；可用: {sorted(self._specs)}")  # Raise KeyError listing the available IDs
        return self._specs[diagram_id]  # Return the corresponding spec object

    def has(self, diagram_id: str) -> bool:  # ID existence check
        return diagram_id in self._specs  # Membership test returns the boolean

    def list(self) -> list[DiagramSpec]:  # List all diagrams
        """All registered diagrams (sorted by diagram_id)."""
        return [self._specs[k] for k in sorted(self._specs)]  # Return all specs sorted by ID

    def find_by_axis(self, y_expression: str, x_expression: str) -> list[DiagramSpec]:  # Search by both axis expressions
        """Search by axis expressions (case-insensitive), for the
        "available data -> candidate diagrams" logic."""
        y_key, x_key = y_expression.lower(), x_expression.lower()  # Lowercase both expressions for comparison
        return [  # List comprehension filtering matching specs
            spec for spec in self.list()  # Iterate over all registered specs
            if spec.axis("y").expression.lower() == y_key  # y-axis expression matches (case-insensitive)
            and spec.axis("x").expression.lower() == x_key  # and the x-axis expression matches (case-insensitive)
        ]

    def __len__(self) -> int:  # Built-in len() protocol
        return len(self._specs)  # Return the number of registered diagrams

# Process-wide default library instance: loaded once, shared globally (entries are immutable data, thread-safe)
DEFAULT_LIBRARY = DiagramLibrary()  # Create the process-wide shared default boundary library singleton at import time
