"""Generic data visualization tool: one multi-panel PNG overview per numeric table.

Positioning: a "glance" tool for data understanding - histograms for single-variable distributions, a correlation heatmap for
relationships between variables, coordinate scatter for spatial extent. All plotting is done by deterministic matplotlib code;
outputs are handed to the report's "figure artifacts" section as artifacts; no geological interpretation whatsoever.
"""

from __future__ import annotations  # enable deferred evaluation of type annotations (PEP 563)

from typing import Any  # any-type annotation

from oregenesis.tools.registry import ToolCall, ToolSpec  # ToolCall and ToolSpec dataclasses
from oregenesis.tools.figure_style import artifacts_from, safe_stem, save_figure  # journal-grade styling and unified saving

# Numeric-column coverage and count caps: histogram panels take the top 8 columns by coverage, preventing panel explosion
_MAX_HIST_COLS = 8  # cap on the number of histogram panels
_MAX_CORR_COLS = 12  # column cap for the correlation heatmap (the heatmap is skipped beyond it)


def _plt(style: str = "cn"):  # lazy matplotlib import + Agg backend + style application
    """Lazily import matplotlib and switch to Agg, apply the plot style per style, and return the plt module.

    style defaults to "cn" to preserve the historical look (backward compatible); the plotting tools' runners pass it
    explicitly from the call parameters (the parameter defaults to journal; see the parsing in each run_*).
    """
    import matplotlib  # local import: load only when actually plotting
    matplotlib.use("Agg")  # plotting works even without a display
    import matplotlib.pyplot as plt  # conventional alias
    from oregenesis.tools.figure_style import apply_style  # unified style module (no dependencies, no circular import)
    apply_style(style)  # journal: Times+SimSun/inward ticks/Okabe-Ito; cn: legacy SimHei chain
    return plt  # return pyplot


def _numeric_columns(header: list[str], rows: list[list[Any]]) -> list[tuple[str, list[float]]]:  # extract numeric columns
    """Extract numericable columns: returns [(column name, value list)], descending by non-empty coverage."""
    cols: list[tuple[str, list[float]]] = []  # result buffer
    for i, h in enumerate(header):  # try to numericize each column
        vals = []  # buffer of this column's non-empty numerics
        for r in rows:  # take cells row by row
            v = r[i] if i < len(r) else None  # pad None for short rows
            try:  # failed numericization counts as missing
                f = float(v)  # try converting to float
                vals.append(f)  # collect only on success
            except (TypeError, ValueError):  # None/text/empty string
                continue  # skip as missing
        if len(vals) >= 3:  # worth plotting only with at least 3 valid values
            cols.append((str(h).strip(), vals))  # collect this column
    cols.sort(key=lambda t: -len(t[1]))  # descending by coverage (valid value count)
    return cols  # return the sorted numeric columns


def run_plot_data_overview(state: Any, call: ToolCall) -> dict[str, Any]:  # tool run function
    """Produce a multi-panel overview for one numeric table: histograms + correlation heatmap + coordinate scatter."""
    p = call.parameters or {}  # get parameters
    style = str(p.get("style") or "journal")  # plot style: journal (Times/SimSun, 300dpi) | cn (legacy SimHei, 150dpi)
    formats = str(p.get("formats") or "png")  # artifact formats: png / png+svg / png+pdf
    stem = safe_stem(p.get("out_name"), "data_overview")  # artifact name stem (customizable, prevents same-name overwrites)
    units = [u for u in state.get("data_units", [])  # take usable numeric tables (comprehension spans lines)
             if u.data_modality == "numeric" and u.data_status != "error"]  # condition: numeric modality and no error
    if not units:  # no usable numeric table
        raise ValueError("没有可用的数值 DataUnit")  # the registry converts this error into an error Observation
    u = units[0]  # take the first table (same convention as element_inventory)
    header = [str(h) for h in u.raw_data.get("header", [])]  # header
    rows = u.raw_data.get("rows", [])  # data rows
    cols = _numeric_columns(header, rows)  # extract numeric columns (descending by coverage)
    if len(cols) < 2:  # fewer than 2 numeric columns: no overview possible
        raise ValueError(f"数值列不足（仅 {len(cols)} 列），无法产出概览图")  # report insufficient data honestly

    plt = _plt(style)  # initialize the plotting backend and apply the style per parameter
    n_hist = min(_MAX_HIST_COLS, len(cols))  # number of histogram panels
    has_xy = len(header) >= 2 and {h.strip().lower() for h in header[:2]} & {"x", "lon", "longitude", "经度"}  # rough check for coordinate columns (first two columns)
    n_panel = n_hist + (1 if len(cols) <= _MAX_CORR_COLS else 0) + (1 if has_xy else 0)  # total panel count
    ncol = 3 if n_panel >= 3 else n_panel  # lay panels out in 3 columns
    nrow = (n_panel + ncol - 1) // ncol  # panel row count
    fig, axes_arr = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow))  # create the canvas
    axes_list = list(_ravel_axes(axes_arr))  # flatten the axes array (helper below)
    for ax in axes_list[n_panel:]:  # hide surplus panels
        ax.set_visible(False)  # do not show empty panels

    panel = 0  # panel cursor
    hist_texts: list[str] = []  # observation text buffer (per-column distribution summaries)
    for name, vals in cols[:n_hist]:  # draw a histogram per column
        ax = axes_list[panel]; panel += 1  # take the panel and advance the cursor
        ax.hist(vals, bins=20, color="#4C72B0", edgecolor="white")  # 20-bin histogram
        ax.set_title(f"{name} 分布（n={len(vals)}）", fontsize=9)  # title with valid count
        ax.tick_params(labelsize=7)  # tick label size
        lo, hi = min(vals), max(vals)  # value range
        hist_texts.append(f"{name}∈[{lo:.3g},{hi:.3g}]")  # observation summary: range

    if len(cols) <= _MAX_CORR_COLS and panel < len(axes_list):  # draw the correlation heatmap when the column count allows
        import numpy as _np  # local import of the numeric library
        ax = axes_list[panel]; panel += 1  # take the panel
        n = min(len(c[1]) for c in cols)  # align column lengths (take the shortest)
        mat = _np.array([c[1][:n] for c in cols])  # assemble the numeric matrix
        corr = _np.corrcoef(mat)  # Pearson correlation matrix
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)  # heatmap
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[0] for c in cols], rotation=45, fontsize=6)  # column names
        ax.set_yticks(range(len(cols))); ax.set_yticklabels([c[0] for c in cols], fontsize=6)  # row names
        ax.set_title("皮尔逊相关系数", fontsize=9)  # title
        fig.colorbar(im, ax=ax, fraction=0.046)  # colorbar
        strong = [(cols[i][0], cols[j][0], corr[i, j])  # summary of strong-correlation pairs (|r|>=0.7) (comprehension spans lines)
                  for i in range(len(cols)) for j in range(i + 1, len(cols)) if abs(corr[i, j]) >= 0.7]  # condition: absolute value above threshold
        hist_texts.append("强相关(|r|≥0.7): " + (", ".join(f"{a}-{b} r={r:.2f}" for a, b, r in strong[:5]) or "无"))  # record strong correlations

    if has_xy and panel < len(axes_list):  # draw the scatter when coordinate columns exist
        import numpy as _np  # local import
        ax = axes_list[panel]; panel += 1  # take the panel
        x_col = next((c for c in cols if c[0].strip().lower() in {"x", "lon", "longitude", "经度"}), cols[0])  # X column
        y_col = next((c for c in cols if c[0].strip().lower() in {"y", "lat", "latitude", "纬度"}), cols[1])  # Y column
        n = min(len(x_col[1]), len(y_col[1]))  # align lengths
        ax.scatter(x_col[1][:n], y_col[1][:n], s=6, alpha=0.5, color="#DD8452")  # scatter
        ax.set_title(f"{x_col[0]}–{y_col[0]} 散点", fontsize=9)  # title
        ax.tick_params(labelsize=7)  # tick label size

    fig.suptitle(f"数据概览：{u.data_origin.get('source_path', '')}", fontsize=11)  # overall title with the source
    fig.tight_layout()  # tight layout
    paths = save_figure(fig, _save_dir(state), stem, formats)  # unified saving (dpi follows style: journal 300 / cn 150)
    plt.close(fig)  # release the canvas

    observation = (f"数据概览图已生成（{n_hist} 列直方图"
                   f"{' + 相关性热图' if len(cols) <= _MAX_CORR_COLS else ''}"
                   f"{' + 坐标散点' if has_xy else ''}）："
                   + "；".join(hist_texts[:6]))  # observation text: panel composition + distribution summaries
    return {  # unified payload (same contract as the other tools)
        "observation": observation,  # observation text (goes into evidence)
        "artifacts": artifacts_from(paths),  # figure artifacts (PNG as primary key + vector extra entries)
        "supports": ["data_overview_plotted"],  # support tags
        "limitations": ["概览图仅描述分布与相关，不构成地质解释"],  # state limitations honestly
    }


def _ravel_axes(axes_arr: Any) -> list:  # flatten the axes array returned by subplots (scalars also supported)
    """Unify the subplots return value into a flat list of axes objects."""
    import numpy as _np  # local import
    arr = _np.asarray(axes_arr)  # convert to an array
    return list(arr.ravel()) if arr.ndim > 0 else [axes_arr]  # flatten multi-dimensional, wrap scalars


def _save_dir(state: Any):  # get the output directory: state first, the system temp directory as default
    """Take the output directory from the task config (the runner injects output_dir); fall back to the system temp directory."""
    import tempfile  # local import: for the default output directory
    from pathlib import Path  # local import of the path type
    out = (state.get("task") or {}).get("output_dir") or tempfile.gettempdir()  # resolve the directory
    return Path(out) / "figures"  # put everything under the figures subdirectory


def build_dataviz_tool() -> list[ToolSpec]:  # constructor: registry entry point
    """Build the generic data visualization tool family."""
    return [ToolSpec(  # single-element list (same style as the other factories)
        name="plot_data_overview", tool_family="data",  # tool name and family (data description category)
        description="对一张数值表产出概览图（直方图/相关性热图/坐标散点），用于数据认知；"
                    "可选 parameters：style=journal（Times/宋体衬线、300dpi，默认）| cn（中文黑体、150dpi），"
                    "formats=png|png+svg|png+pdf，out_name 自定义产物文件名",  # menu description
        required_data=("numeric",),  # requires numeric modality
        capabilities=("data_visualization",),  # capability tags
        applicable=lambda s: (any(u.data_modality == "numeric" and u.data_status != "error"  # applicability: a usable numeric table exists (spans lines)
                                   and len([h for h in u.raw_data.get("header", [])]) >= 2 for u in s.get("data_units", [])),  # at least two columns
                              "需要至少两列数值的可用数值 DataUnit"),  # reason when not applicable
        run=run_plot_data_overview)]  # bind the run function
