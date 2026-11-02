"""Discrimination diagram renderer: field polygon base map + sample plotting
-> PNG.

All data comes from the diagram library's DiagramSpec (axis ranges/scales/
field polygons/boundary lines/references and fidelity); everything is drawn
in data space with log/linear axes set per the spec — boundaries on log axes
are naturally straightened by matplotlib. Fields are filled with light
polygons, region Chinese names are labeled at centroids, samples are colored
by interpreted region, samples close to boundaries get a black circle
highlight; the figure footnote honestly states the digitization level and
references (the visual implementation of the "do not pretend to be exact"
principle).
"""

from __future__ import annotations  # Enable deferred evaluation of type annotations (PEP 563)

from pathlib import Path  # Output path handling
from oregenesis.diagram.boundaries import (  # Reuse library geometry (multi-line)
    DEFAULT_LIBRARY, clip_polygon_to_range)
from oregenesis.diagram.schema import ScaleKind  # Scale enum (for log/linear decisions)

# Light fill color cycle for regions (translucent matplotlib tab10); samples are outlined in the same color as their region
_FILL_CYCLE = ("#aec7e8", "#ffbb78", "#98df8a", "#c5b0d5", "#c49c94", "#f7b6d2")


def _plt():  # Lazy matplotlib import + Agg backend + Chinese fonts (same style as the platform plotting)
    """Lazily import matplotlib, switch to Agg/Chinese fonts, and return the plt module."""
    import matplotlib  # Local import: loaded only when actually plotting
    matplotlib.use("Agg")  # Plot even without a display environment
    import matplotlib.pyplot as plt  # Conventional alias
    from matplotlib import rcParams  # Global font parameters
    for font in ("SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"):  # Chinese font candidate chain
        try:  # Probe locally available fonts one by one
            from matplotlib.font_manager import FontProperties  # Font probing utility
            FontProperties(family=font).get_name()  # Resolvable means usable
            rcParams["font.sans-serif"] = [font]  # Set as the preferred Chinese font
            break  # Stop on the first hit
        except Exception:  # Probe failed: try the next
            continue  # Continue down the candidate chain
    rcParams["axes.unicode_minus"] = False  # Fix the missing minus-sign glyph problem
    return plt  # Return pyplot


def _centroid(verts: list[list[float]]) -> tuple[float, float]:  # Polygon centroid (mean approximation)
    """Centroid approximated as the vertex mean (label anchor for region
    names; good enough for labeling)."""
    xs = [v[0] for v in verts]  # Vertex x values
    ys = [v[1] for v in verts]  # Vertex y values
    return (sum(xs) / len(xs), sum(ys) / len(ys)) if verts else (0.0, 0.0)  # Mean centroid


def plot_diagram(diagram_id: str, points: list[tuple[float, float, str]],  # Main render function (multi-line signature)
                 save_dir: str, prefix: str = "diagram", dpi: int = 150,  # Args cont.: output directory/prefix/resolution
                 title_extra: str = "") -> str | None:  # Args cont.: extra title; returns the PNG path
    """Render one discrimination diagram: field base map + sample points,
    returning the PNG path.

    points are data-space coordinates [(x, y, sample_id)]; each point is
    interpreted by DiagramInterpreter for its region and colored accordingly,
    with a black circle highlight when close to a boundary. Returns None when
    there are no valid points or the diagram does not exist (the caller
    downgrades to a warning instead of aborting the run).
    """
    try:  # get raises KeyError when the library lacks this diagram
        spec = DEFAULT_LIBRARY.get(diagram_id)  # Fetch the diagram spec
    except KeyError:  # Diagram never digitized/registered
        return None  # Return None; the caller warns
    if not points:  # No points to plot
        return None  # Nothing to draw for an empty plot
    xax, yax = spec.axis("x"), spec.axis("y")  # Get both axis specs
    plt = _plt()  # Initialize the plotting backend
    fig, ax = plt.subplots(figsize=(7.2, 6.0))  # Create the canvas

    # ── Field polygon base map: clip to axis range -> light fill + Chinese name labels ──
    region_face: dict[str, str] = {}  # Region ID -> fill color (for sample coloring)
    handles: list = []  # Legend handle buffer
    from matplotlib.patches import Patch, Polygon  # Polygon patch and legend color swatch
    for i, r in enumerate(spec.regions):  # Draw region by region
        color = _FILL_CYCLE[i % len(_FILL_CYCLE)]  # Cycle through the light colors
        region_face[r.region_id] = color  # Register the region -> color mapping
        clipped = clip_polygon_to_range(r.polygon_vertices, xax.axis_range, yax.axis_range)  # Clip open regions to the axis range
        if len(clipped) < 3:  # Fewer than 3 vertices after clipping (fully outside the plot)
            continue  # Skip this region
        ax.add_patch(Polygon(clipped, closed=True, facecolor=color,  # Light-filled polygon (multi-line call)
                             edgecolor="#888888", linewidth=0.5, alpha=0.45, zorder=1))  # Semi-transparent with a thin gray edge
        cx, cy = _centroid(clipped)  # Centroid
        xs_c = [v[0] for v in clipped]; ys_c = [v[1] for v in clipped]  # Clipped vertex coordinate sets
        rotate = (max(ys_c) - min(ys_c)) > (max(xs_c) - min(xs_c))  # Rotate text vertically for tall narrow strips (height exceeds width)
        ax.text(cx, cy, r.name_cn, fontsize=7.5, ha="center", va="center",  # Region Chinese name labeled at the centroid
                rotation=90 if rotate else 0, color="#333333", zorder=3, wrap=True)  # Dark gray text on top
        handles.append(Patch(facecolor=color, edgecolor="#888888", label=r.name_cn))  # Legend handle

    # ── Explicit discrimination boundaries: dark dashed lines ──
    for bd in spec.boundaries:  # Draw boundary by boundary
        xs = [p[0] for p in bd.polyline_vertices]  # Polyline x sequence
        ys = [p[1] for p in bd.polyline_vertices]  # Polyline y sequence
        ax.plot(xs, ys, color="#333333", linewidth=1.2, linestyle="--", zorder=2)  # Dark dashed boundary line

    # ── Sample points: colored by interpreted region, black circle near boundaries ──
    from oregenesis.diagram.interpreter import DiagramInterpreter  # Local import of the interpretation engine
    interp = DiagramInterpreter()  # Interpreter instance
    n_plotted = n_out = 0  # Plotted/out-of-range point counters
    for x, y, sid in points:  # Plot sample by sample
        try:  # Interpretation of out-of-range/missing-value samples may raise; tolerate per point
            reading = interp.read_coordinates(diagram_id, float(x), float(y), sample_id=str(sid))  # Interpret the region
        except Exception:  # noqa: BLE001 — interpret failure treated as a gray point
            reading = None  # No reading result
        region = getattr(reading, "region", None) if reading is not None else None  # Get the assigned region ID
        color = region_face.get(region, "#7f7f7f")  # Region color; gray for unknown regions
        near = bool(getattr(reading, "near_boundary", None)) if reading is not None else False  # Whether close to a boundary
        in_range = (xax.axis_range[0] <= x <= xax.axis_range[1]  # x within the display range (multi-line)
                    and yax.axis_range[0] <= y <= yax.axis_range[1])  # y within the display range
        if not in_range:  # Do not plot out-of-range samples (the plot would stretch)
            n_out += 1  # Count
            continue  # Skip
        ax.scatter(x, y, s=26, color=color, edgecolors=("black" if near else "white"),  # Plot the point (black circle when near a boundary) (multi-line call)
                   linewidths=(1.1 if near else 0.5), zorder=4)  # Topmost z-order
        n_plotted += 1  # Count
    if n_out:  # Note out-of-range samples in the footnote
        handles.append(plt.Line2D([], [], linestyle="none",  # Append an out-of-range legend entry (multi-line)
                                  label=f"另有 {n_out} 个样品超出显示范围未投点"))  # Explanatory text

    # ── Axes/title/footnote ──
    ax.set_xlim(xax.axis_range); ax.set_ylim(yax.axis_range)  # Set display ranges
    if xax.scale == ScaleKind.LOG10: ax.set_xscale("log")  # x log axis
    if yax.scale == ScaleKind.LOG10: ax.set_yscale("log")  # y log axis
    ax.set_xlabel(f"{xax.label} ({xax.unit})"); ax.set_ylabel(f"{yax.label} ({yax.unit})")  # Axis labels with units
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)  # Light grid (major and minor ticks on log axes)
    if handles: ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.9)  # Legend
    foot = (f"数字化保真度 fidelity={spec.fidelity}（{spec.verification_status}）；"
            f"出处：{'; '.join(spec.references[:2])}")  # Footnote: fidelity and reference provenance
    ax.set_title(f"{spec.display_name}" + (f"——{title_extra}" if title_extra else ""), fontsize=11)  # Title
    fig.text(0.01, 0.01, foot, fontsize=6.5, color="#555555", wrap=True)  # Bottom provenance footnote
    fig.tight_layout(rect=(0, 0.03, 1, 1))  # Tight layout with room left for the footnote

    out_dir = Path(save_dir)  # Output directory
    out_dir.mkdir(parents=True, exist_ok=True)  # Create recursively
    path = out_dir / f"{prefix}_{diagram_id}.png"  # Output artifact path
    fig.savefig(path, dpi=dpi, bbox_inches="tight")  # Write to disk
    plt.close(fig)  # Release the canvas (plt is the pyplot module returned by _plt())
    return str(path)  # Return the artifact path
