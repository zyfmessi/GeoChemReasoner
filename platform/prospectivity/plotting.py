"""2D mineral prospectivity mapping — map rendering and statistics table export.

Evidence layer panel map / prospectivity probability map / prospect zone classification map (the final
deliverable map) / P-A evaluation plot / statistics tables (CSV + Markdown). Color conventions: continuous
probability/score uses the warm YlOrRd colormap (high value = warm, matching prospectivity map-reading
intuition); zone discrete colors A=red #d7301f, B=orange #fd8d3c, C=yellow #fed976, non-prospective
light gray #f0f0f0; known deposits are black five-pointed stars (a three-layer visual encoding with the
red anomaly outlines and the probability basemap). matplotlib is lazily imported (Agg backend): pure
analysis also runs in headless environments.
"""
from __future__ import annotations  # enable PEP 563 deferred annotation evaluation; type hints are not evaluated at runtime

import csv, logging, os  # import the csv/logging/os stdlib modules

import numpy as np  # import numpy and alias it as np

from oregenesis.platform.prospectivity.models import ProspectivityReport  # import the prospectivity overall report data class

logger = logging.getLogger(__name__)  # create the module-level logger

# Chinese font candidates (same list as anomaly.plotting; the two modules are independent)
_FONT_CANDIDATES = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "sans-serif"]  # Chinese font candidate list (first hit wins)
# Zone discrete colors (A red / B orange / C yellow / non-prospective light gray); statistics table columns (shared by Markdown/CSV, column names unchanged)
_ZONE_COLORS = ["#f0f0f0", "#d7301f", "#fd8d3c", "#fed976"]  # zone discrete fills: non-prospective light gray plus A red/B orange/C yellow
_EV_KEYS = ["证据层", "名称", "涉及元素", "算法", "异常格点", "面积占比", "阈值"]  # evidence layer statistics table column list (shared by Markdown/CSV)
_ZONE_KEYS = ["级别", "概率区间", "格点数", "面积", "面积占比", "矿点数", "捕获率"]  # zone statistics table column list (shared by Markdown/CSV)


def _plt_agg():  # define the matplotlib backend and Chinese font initialization function
    """matplotlib Agg backend + Chinese font setup + lazy pyplot import (shared by all plotting functions)."""
    import matplotlib  # lazy-import the matplotlib main package
    matplotlib.use("Agg")  # switch to the headless Agg backend (renders without a display)
    from matplotlib import font_manager  # import the font manager
    for name in _FONT_CANDIDATES:  # set the first system-resident candidate as the default sans-serif font
        if name in {f.name for f in font_manager.fontManager.ttflist}:  # check whether the candidate font is installed on the system
            matplotlib.rcParams["font.sans-serif"] = [name]  # set the matched font as the default sans-serif
            break  # stop at the first hit
    matplotlib.rcParams["axes.unicode_minus"] = False  # prevent the axis minus sign from rendering as a box
    import matplotlib.pyplot as plt  # lazy-import the pyplot interface
    return plt  # return the configured pyplot module


def _save(fig, plt, save_dir: str, prefix: str, suffix: str, dpi: int) -> str:  # define the figure-saving helper (parameters span lines)
    """Save a PNG (tight bbox, given dpi), close the figure and return the path."""
    path = os.path.join(save_dir, f"{prefix}_{suffix}.png")  # assemble the output PNG file path
    fig.savefig(path, dpi=dpi, bbox_inches="tight")  # save the figure (given dpi, tight bbox)
    plt.close(fig)  # close the figure to free memory
    return path  # return the saved path


def plot_evidence_layers(report: ProspectivityReport, deposits_xy: np.ndarray | None, save_dir: str,  # define the evidence layer panel plotting function (parameters span lines)
                         prefix: str = "prospect", dpi: int = 150, max_panels: int = 12) -> str | None:  # params: filename prefix / dpi / max panel count (default 12)
    """Evidence layer panel map: one cell per layer (3-column layout), YlOrRd intensity raster + red dashed binary anomaly contours."""
    if not (layers := report.evidence_layers[:max_panels]): return None  # return None when no evidence layers (take the first max_panels layers)
    plt = _plt_agg(); n, ncol = len(layers), 3; nrow = int(np.ceil(n / ncol))  # initialize the backend and compute the row count of the 3-column layout
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow), squeeze=False)  # create nrow×ncol subplots (squeeze keeps 2-D indexing)
    extent = report.grid_def.extent()  # take the grid display extent
    for i, ly in enumerate(layers):  # draw evidence layer by evidence layer
        ax = axes[i // ncol][i % ncol]  # locate the subplot for this index
        g = ly.grid  # take this layer's continuous anomaly intensity raster
        if g is None or not np.isfinite(g).any():  # placeholder for an empty layer
            ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)  # annotate the no-data placeholder at the subplot center
            ax.set_title(ly.name_cn, fontsize=9); continue  # set only the layer-name title, then skip
        im = ax.imshow(g, extent=extent, origin="upper", aspect="equal",  # draw the 0~1 intensity heatmap (parameters span lines)
                       cmap="YlOrRd", vmin=0, vmax=1, interpolation="bilinear", zorder=1)  # YlOrRd colormap / bilinear interpolation / base layer zorder=1
        if ly.binary is not None:  # binary anomaly outline (red dashed)
            try:  # try drawing the binary anomaly contour
                ax.contour(ly.binary, levels=[0.5], colors="#d7301f", linewidths=0.8,  # draw a red dashed contour at the 0.5 level (parameters span lines)
                           linestyles="--", extent=extent, origin="upper", zorder=2)  # dashed style / extent / layer order zorder=2
            except Exception:  # catch contour drawing failures
                pass  # silently ignored; the main map is unaffected
        ax.set_title(f"{ly.name_cn}\n异常格点 {ly.n_anomaly_cells}（{ly.area_ratio * 100:.1f}%）", fontsize=9)  # subplot title: layer name and anomaly cell stats
        ax.tick_params(labelsize=7); fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)  # shrink the tick font size and add the colorbar
    for j in range(n, nrow * ncol): axes[j // ncol][j % ncol].axis("off")  # turn off the extra subplot cells
    fig.suptitle(f"阶段一：地球化学异常证据层（共 {len(report.evidence_layers)} 层，插值={report.interpolation}）",  # overall title: layer count and interpolation method (spans lines)
                 fontsize=12)  # overall title font size 12
    fig.tight_layout(rect=(0, 0, 1, 0.96))  # tight layout (leave top space for the suptitle)
    return _save(fig, plt, save_dir, prefix, "evidence_layers", dpi)  # save as the evidence_layers map and return the path


def plot_probability_map(report: ProspectivityReport, deposits_xy: np.ndarray | None, save_dir: str,  # define the prospectivity probability/favorability map function (parameters span lines)
                         prefix: str = "prospect", dpi: int = 150) -> str | None:  # params: filename prefix and dpi
    """Prospectivity probability/favorability map: warm continuous raster + colorbar + black deposit stars + data-driven contours.
    The color scale takes the 2%~98% percentiles (posterior probability magnitudes are often tiny; a fixed 0~1 makes the whole map one color); labels still show true values.
    """
    if (model := report.model) is None or model.prob_grid is None: return None  # return None when there is no fusion model or probability grid
    plt = _plt_agg(); prob, extent = model.prob_grid, report.grid_def.extent()  # initialize the backend and take the probability grid and extent
    fig, ax = plt.subplots(figsize=(9, 7.5))  # create a 9×7.5 inch canvas
    if (fin := np.isfinite(prob)).any():  # data-driven color scale; the colorbar labels true probability values
        vmin = float(np.nanpercentile(prob[fin], 2)); vmax = float(np.nanpercentile(prob[fin], 98))  # color scale range from the 2%~98% percentiles of valid values
        if vmax - vmin < 1e-12: vmax = vmin + 1e-12  # when the color span is near zero, raise the upper limit by 1e-12 to avoid degeneracy
    else:
        vmin, vmax = 0.0, 1.0  # default 0~1 color scale when there are no valid pixels
    im = ax.imshow(prob, extent=extent, origin="upper", aspect="equal", cmap="YlOrRd",  # draw the probability heatmap (parameters span lines)
                   vmin=vmin, vmax=vmax, interpolation="bilinear", zorder=1)  # data-driven vmin/vmax, bilinear interpolation, base layer
    if fin.sum() > 50:  # contours at the P75/P90/P97 percentiles labeled with true values (independent of probability magnitude)
        try:  # try drawing contours
            levels = sorted({round(float(np.nanquantile(prob[fin], q)), 6) for q in (0.75, 0.90, 0.97)})  # take the P75/P90/P97 percentiles, deduplicated and sorted as contour levels
            if len(levels) >= 2:  # draw only when there are at least two distinct levels
                cs = ax.contour(prob, levels=levels, colors="#4d4d4d",  # draw gray dotted contours (parameters span lines)
                                linewidths=[0.6, 0.8, 1.0][-len(levels):], linestyles=":",  # line width thickens as the level rises (spans lines)
                                extent=extent, origin="upper", zorder=2)  # pass the extent and layer order zorder=2
                ax.clabel(cs, fmt="%.3g", fontsize=7)  # label true probability values on the contours
        except Exception:  # catch contour drawing failures
            pass  # silently ignored
    if deposits_xy is not None and len(deposits_xy):  # known deposits
        ax.scatter(deposits_xy[:, 0], deposits_xy[:, 1], marker="*", s=140, c="black",  # draw black five-pointed deposit stars (parameters span lines)
                   edgecolors="white", linewidths=0.6, zorder=4, label=f"已知矿点（{len(deposits_xy)}）")  # white edge / top layer zorder=4 / legend label with count
        ax.legend(loc="lower left", fontsize=9, framealpha=0.9)  # show the legend in the lower left
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02,  # add the colorbar (parameters span lines)
                 label="成矿概率" if model.is_probability else "成矿有利度")  # colorbar label switches with probability/favorability semantics
    tag = "" if model.is_probability else "（有利度得分，非频率概率）"  # prepare the subtitle annotation for non-probability semantics
    ax.set_title(f"二维成矿预测{model.name_cn}{tag}\n"  # set the title: method / interpolation / layer count (spans lines)
                 f"方法={model.method}  插值={report.interpolation}  "  # title second line: method and interpolation
                 f"证据层 {len(report.evidence_layers)} 个", fontsize=12)  # title closing: evidence layer count and font size 12
    ax.set_xlabel("E (km)"); ax.set_ylabel("N (km)")  # set the east-west / north-south axis labels (km)
    fig.tight_layout()  # tight layout
    return _save(fig, plt, save_dir, prefix, "probability", dpi)  # save as the probability map and return the path


def plot_zones_map(report: ProspectivityReport, deposits_xy: np.ndarray | None, save_dir: str,  # define the prospect zone classification map function (parameters span lines)
                   prefix: str = "prospect", dpi: int = 150) -> str | None:  # params: filename prefix and dpi
    """Prospect zone classification map (this tool's final deliverable): A/B/C discrete fills + deposit stars + legend."""
    if (zones := report.zones) is None or zones.zone_grid is None: return None  # return None when there is no zoning result or zone grid
    plt = _plt_agg()  # initialize the plotting backend
    from matplotlib.colors import BoundaryNorm, ListedColormap; from matplotlib.patches import Patch  # import the discrete colormap norm and legend patch classes
    zone, extent = zones.zone_grid, report.grid_def.extent()  # take the zone grid and extent
    cmap = ListedColormap(_ZONE_COLORS); norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)  # discrete colormap and 0~3 level boundary normalization
    fig, ax = plt.subplots(figsize=(9, 7.5))  # create a 9×7.5 inch canvas
    ax.imshow(np.ma.masked_where(zone == 0, zone), extent=extent, origin="upper",  # draw the zone fills (0 values masked out) (spans lines)
              aspect="equal", cmap=cmap, norm=norm, interpolation="nearest", zorder=1, alpha=0.95)  # equal aspect / nearest-neighbor interpolation keeps discrete blocks / semi-transparent
    if report.model is not None and report.model.prob_grid is not None and zones.levels:  # when the probability grid and level stats exist
        try:  # class-A lower-bound contour (light gray dotted background reference, magnitude-independent)
            ax.contour(report.model.prob_grid, levels=[zones.levels[0].prob_min], colors="#7f7f7f",  # draw the class-A lower-bound contour (light gray reference) (spans lines)
                       linewidths=0.6, linestyles=":", extent=extent, origin="upper", zorder=2)  # dotted style / extent / layer order zorder=2
        except Exception:  # catch contour failures
            pass  # silently ignored
    if deposits_xy is not None and len(deposits_xy):  # when deposits exist
        ax.scatter(deposits_xy[:, 0], deposits_xy[:, 1], marker="*", s=150, c="black",  # draw black five-pointed deposit stars (parameters span lines)
                   edgecolors="white", linewidths=0.7, zorder=4)  # white edge / top layer zorder=4
    handles = [Patch(facecolor=_ZONE_COLORS[lv.level], edgecolor="#999999",  # legend (each level + deposits)
                     label=f"{lv.name_cn}: {lv.area:.1f} km$^2$（{lv.area_ratio * 100:.1f}%）"  # legend text: level / area / ratio (spans lines)
                           + (f"，矿点捕获 {lv.n_deposits}/{int((lv.capture_ratio or 0) * 100)}%"  # append deposit capture count and percentage (spans lines)
                              if lv.capture_ratio is not None else ""))  # concatenate only when a capture ratio exists
               for lv in zones.levels]  # generate one legend patch per level
    if deposits_xy is not None and len(deposits_xy):  # when deposits exist
        handles.append(plt.Line2D([], [], marker="*", ls="", markerfacecolor="black",  # append the deposit star legend entry (spans lines)
                                  markeredgecolor="white", markersize=13, label="已知矿点"))  # black star with white edge, labeled as known deposits
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.92)  # show the legend in the lower left
    zone_word = ("远景区分级图"  # supervised with deposits: keep the prospect-zone wording (ternary wrapped in parentheses across lines)
                 if (zones.levels and "远景区" in zones.levels[0].name_cn)
                 else "地球化学异常分区图（无矿点验证）")  # unsupervised: the title honestly reads anomaly zoning
    ax.set_title(f"二维成矿预测{zone_word}（{zones.name_cn}）\n"  # set the title: zoning method name (spans lines)
                 f"{zones.notes[-1] if zones.notes else ''}", fontsize=12)  # subtitle from the last zoning note, font size 12
    ax.set_xlabel("E (km)"); ax.set_ylabel("N (km)")  # set the east-west / north-south axis labels
    fig.tight_layout()  # tight layout
    return _save(fig, plt, save_dir, prefix, "zones", dpi)  # save as the zones map and return the path


def plot_evaluation(report: ProspectivityReport, deposits_xy: np.ndarray | None, save_dir: str,  # define the P-A evaluation plot function (parameters span lines)
                    prefix: str = "prospect", dpi: int = 150) -> str | None:  # params: filename prefix and dpi
    """Prediction-rate–area (P-A) curve + random-model diagonal + AUC and 20% capture-rate annotation (when evaluation data exists)."""
    ev = report.evaluation or {}  # take the evaluation dict (default empty)
    if not (xs := ev.get("pa_area_frac")) or not (ys := ev.get("pa_capture_frac")): return None  # return None when P-A curve data is missing
    plt = _plt_agg(); fig, ax = plt.subplots(figsize=(6.5, 5.5))  # initialize the backend and create a 6.5×5.5 inch canvas
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="随机模型")  # draw the random-model dashed diagonal reference
    ax.plot(xs, ys, "o-", color="#d7301f", ms=3, lw=1.6, label="本模型")  # draw this model's P-A curve (red circle-marker line)
    ax.axvline(0.2, color="#7f7f7f", lw=0.8, ls=":")  # draw a gray dotted vertical reference line at x=0.2
    if (cap := ev.get("capture_at_20pct")) is not None:  # when the 20% capture-rate metric exists
        ax.annotate(f"前 20% 面积捕获 {cap * 100:.0f}%", xy=(0.2, cap),  # annotate the capture-rate text and arrow position (spans lines)
                    xytext=(0.26, max(cap - 0.12, 0.05)), fontsize=9,  # text placed below-right of the point (spans lines)
                    arrowprops=dict(arrowstyle="->", lw=0.8))  # arrow style: thin arrow pointing at the data point
    ax.set_xlabel("累计面积占比（概率降序）"); ax.set_ylabel("已知矿点捕获率")  # set both axis labels
    auc = ev.get("auc"); auc_txt = f"AUC = {auc:.3f}" if auc == auc else "AUC = -"  # build the AUC text (NaN, which never equals itself, is filtered)
    ax.set_title(f"预测率-面积（P-A）曲线  {auc_txt}", fontsize=12)  # title with the AUC value
    ax.legend(loc="lower right", fontsize=9)  # show the legend in the lower right
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)  # fix both axis ranges to 0~1 (y upper limit 1.02)
    fig.tight_layout()  # tight layout
    return _save(fig, plt, save_dir, prefix, "pa_curve", dpi)  # save as the pa_curve plot and return the path


def generate_prospectivity_maps(report: ProspectivityReport, deposits_xy: np.ndarray | None,  # define the main map-generation entry function (parameters span lines)
                                save_dir: str, prefix: str = "prospect", dpi: int = 150) -> list[str]:  # params: output dir / filename prefix / dpi; returns a list of paths
    """Main map entry point: render all maps from the report (the four plotting functions share a unified signature) and return the list of PNG paths."""
    paths: list[str] = []; os.makedirs(save_dir, exist_ok=True)  # initialize the path list and ensure the output dir exists
    for fn in (plot_evidence_layers, plot_probability_map, plot_zones_map, plot_evaluation):  # call the four plotting functions in turn
        try:  # per-map exception guard
            p = fn(report, deposits_xy, save_dir, prefix, dpi)  # call the plotting function for the saved path
        except Exception as e:  # one map failing does not abort the remaining maps
            import traceback; traceback.print_exc()  # import traceback and print the exception stack
            report.warnings.append(f"成图 {getattr(fn, '__name__', fn)} 失败: {e}")  # record this map's failure warning
            continue  # skip and keep rendering the remaining maps
        if p: paths.append(p)  # collect the successfully generated paths
    return paths  # return all map paths


def export_tables(report: ProspectivityReport, save_dir: str, prefix: str = "prospect") -> tuple[str | None, str]:  # define the statistics table export function (CSV+Markdown) (spans lines)
    """Statistics table export: evidence layer table / model diagnostics table / prospect zone table in three sections + evaluation and warnings → CSV + Markdown."""
    os.makedirs(save_dir, exist_ok=True)  # ensure the output dir exists
    md: list[str] = []; add = md.append  # append shortcut (shortens line width only; output unchanged)
    add("## 一、证据层（阶段一：地球化学异常识别）")  # write the first section heading: evidence layers
    add("")  # write a blank separator line
    if report.evidence_layers:  # when evidence layers exist
        add("| " + " | ".join(_EV_KEYS) + " |")  # write the Markdown header row
        add("|" + "---|" * len(_EV_KEYS))  # write the header separator row |---|
        for ly in report.evidence_layers:  # write a row per evidence layer
            row = ly.to_row()  # take this layer's statistics row dict
            add("| " + " | ".join(str(row[k]) for k in _EV_KEYS) + " |")  # assemble the Markdown table row in column order
    else:
        add("（无）")  # write the "none" placeholder
    md += ["", "## 二、证据融合模型（阶段二：成矿远景预测）", ""]  # append the second section heading: fusion model
    model = report.model  # take the fusion model
    if model is not None:  # when a fusion model exists
        add(f"- 融合方法：{model.name_cn}（{model.method}）")  # write the fusion method line
        add(f"- 输出语义：{'成矿概率' if model.is_probability else '成矿有利度得分（非频率概率）'}")  # write the output semantics line (probability/favorability)
        for nt in model.notes: add(f"- {nt}")  # write each model note
        add("")  # write a blank separator line
        per = model.diagnostics.get("per_layer")  # take the weights-of-evidence per-layer diagnostics
        if per:  # weights-of-evidence diagnostics table (W+/W-/C/t)
            add("| 证据层 | W+ | W- | 对比度C | 学生化t |")  # write the weights-of-evidence diagnostics header
            add("|---|---|---|---|---|")  # write the diagnostics separator row
            for lid, d in per.items():  # write a diagnostics row per layer
                add(f"| {lid} | {d.get('W+', '-')} | {d.get('W-', '-')} "  # write this layer's W+/W− (default placeholders) (spans lines)
                    f"| {d.get('C', '-')} | {d.get('studentized_C', '-')} |")  # continued: contrast C and studentized t
            add("")  # write a blank separator line
        elif model.layer_weights:  # not weights-of-evidence but layer weights exist
            add("| 证据层 | 权重/系数/重要性 |")  # write the weights header
            add("|---|---|")  # write the weights separator row
            for lid, w in model.layer_weights.items(): add(f"| {lid} | {w:.4f} |")  # write a weight row per layer (4 decimals)
            add("")  # write a blank separator line
        diag_extra = {k: v for k, v in model.diagnostics.items() if k != "per_layer"}  # take the diagnostics other than the per-layer details
        if diag_extra:  # when other diagnostics exist
            add("诊断量：" + "; ".join(f"{k}={v}" for k, v in diag_extra.items()))  # assemble the diagnostics k=v list line
            add("")  # write a blank separator line
    else:
        add("（融合未执行）")  # write the fusion-not-run placeholder
    md += ["", "## 三、远景区分级", ""]  # append the third section heading: prospect zone classification
    zones = report.zones  # take the zoning result
    if zones is not None and zones.levels:  # when zoning with level stats exists
        sample_keys = [k for k in _ZONE_KEYS if k in zones.levels[0].to_row()]  # take the column names actually present in the level rows
        add("| " + " | ".join(sample_keys) + " |")  # write the zone table header
        add("|" + "---|" * len(sample_keys))  # write the zone table separator row
        for lv in zones.levels:  # write a row per level
            row = lv.to_row()  # take this level's statistics row dict
            add("| " + " | ".join(str(row[k]) for k in sample_keys) + " |")  # assemble the Markdown table row in column order
        add("")  # write a blank separator line
        for nt in zones.notes: add(f"- {nt}")  # write each zoning note
        add("")  # write a blank separator line
    else:
        add("（未分级）")  # write the not-zoned placeholder
    add("")  # write a blank separator line
    ev = report.evaluation or {}  # take the evaluation dict
    if ev.get("auc") == ev.get("auc") and ev.get("auc") is not None:  # when AUC exists and is not NaN (equal to itself)
        add(f"模型评估：AUC={ev['auc']}"  # write the model evaluation line: AUC value (spans lines)
            + (f"，前 20% 面积矿点捕获率 {ev['capture_at_20pct'] * 100:.0f}%"  # append the top-20% area capture rate (spans lines)
               if ev.get("capture_at_20pct") is not None else ""))  # concatenate only when the metric exists
        add("")  # write a blank separator line
    if report.warnings:  # when warnings exist
        add("注意:")  # write the notes heading
        for w in report.warnings: add(f"- {w}")  # write each warning
        add("")  # write a blank separator line
    csv_path = os.path.join(save_dir, f"{prefix}_statistics.csv")  # CSV: evidence layer rows + zone rows merged into one wide table
    try:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:  # open the CSV for writing as utf-8-sig (with BOM)
            writer = csv.writer(f)  # create the CSV writer
            writer.writerow(["节", *_EV_KEYS])  # write the CSV header: section name + evidence layer columns
            for ly in report.evidence_layers:  # write a CSV row per evidence layer
                writer.writerow(["证据层", *[ly.to_row()[k] for k in _EV_KEYS]])  # write the evidence section data row
            if model is not None and model.diagnostics.get("per_layer"):  # when weights-of-evidence diagnostics exist
                writer.writerow([])  # write a blank row separating the two sections
                writer.writerow(["证据权诊断", "证据层", "W+", "W-", "对比度C", "学生化t"])  # write the weights-of-evidence diagnostics section header
                for lid, d in model.diagnostics["per_layer"].items():  # write a diagnostics row per layer
                    writer.writerow(["证据权诊断", lid, d.get("W+", "-"), d.get("W-", "-"),  # write the diagnostics row: layer ID/W+/W− (spans lines)
                                     d.get("C", "-"), d.get("studentized_C", "-")])  # continued: C and studentized t
            if zones is not None and zones.levels:  # when zoning results exist
                writer.writerow([])  # write a blank row separating the two sections
                writer.writerow(["远景区分级", *_ZONE_KEYS])  # write the prospect zone section header
                for lv in zones.levels:  # write a row per level
                    r = lv.to_row()  # take this level's statistics row dict
                    writer.writerow(["远景区分级", r.get("级别"), r.get("概率区间"), r.get("格点数"),  # write the zone row: level / probability interval / cell count (spans lines)
                                     r.get("面积"), r.get("面积占比"), r.get("矿点数", "-"), r.get("捕获率", "-")])  # continued: area / ratio / deposit count / capture rate
    except Exception as e:
        logger.warning("CSV 导出失败: %s", e)  # log the CSV export failure
        csv_path = None  # set the CSV path to None to indicate it was not produced
    return csv_path, "\n".join(md)  # return the CSV path and the full Markdown text
