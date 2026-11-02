

# ── Generic plotting tool family: two-axis scatter / ternary diagram / REE pattern diagram / statistics bar chart ─────────
# Design: the user specifies column names (or ratio expressions) and the tools fetch data, transform, and plot --
# no preset boundary data needed; any column combination can make a figure; outputs go uniformly to artifacts / the report figure section.

import re  # used to split expressions on +/- signs

from typing import Any  # any-type annotation

from oregenesis.tools.registry import ToolCall, ToolSpec  # ToolCall and ToolSpec dataclasses
from oregenesis.tools.dataviz import _plt, _save_dir  # reuse the plotting backend and artifact-directory helpers
from oregenesis.tools.figure_style import artifacts_from, fig_size, safe_stem, save_figure  # journal-grade style sizes and unified saving


def _mk_tool(name: str, desc: str, kw: tuple[str, ...], runner) -> ToolSpec:  # generic plotting tool factory
    """Plotting tool factory: unified family/capability tags/applicability (a numeric table exists); keywords go straight into the spec."""
    return ToolSpec(  # assemble the spec
        name=name, tool_family="data_visualization",  # tool name / family
        description=desc, required_data=("numeric",),  # description and data requirement
        capabilities=("data_visualization",), keywords=kw,  # capability tags and task-routing keywords
        applicable=lambda s: (any(u.data_modality == "numeric" and u.data_status != "error"  # usable whenever a numeric table exists (comprehension spans lines)
                                  for u in s.get("data_units", [])),  # iterate over data units
                              "需要可用数值 DataUnit"),  # return (usable, reason) tuple
        run=runner)  # bind the run function



def _find_column(header: list[str], rows: list[list[Any]], name: str) -> tuple[list[Any], bool]:  # fetch a column by name
    """Fetch one column of raw values by name: exact match (case/whitespace-insensitive) -> element-name-normalized match.

    Returns (value list, is-numeric flag); raises ValueError listing available column names when not found.
    """
    target = name.strip().lower()  # lowercase the target column name
    idx = next((i for i, h in enumerate(header) if h.strip().lower() == target), None)  # exact match
    if idx is None:  # exact match missed: element-name-normalized match ("Sr ppm"->Sr, "Sr_ppm"->Sr)
        from oregenesis.understanding import _norm_element  # reuse the element-column recognition logic
        norm_t = _norm_element(name)  # normalize the target name (None when not an element name)
        idx = next((i for i, h in enumerate(header)  # scan columns for a normalized-name match (comprehension spans lines)
                    if norm_t and _norm_element(h) and _norm_element(h).lower() == norm_t.lower()), None)  # normalized comparison
    if idx is None:  # still not found
        raise ValueError(f"找不到列 {name!r}（可用列: {', '.join(header[:12])}…）")  # raise with a list of available columns
    vals = [r[idx] if idx < len(r) else None for r in rows]  # take the whole raw column
    numeric = True  # presume a numeric column
    try:  # probe-convert the whole column to float
        for v in vals:  # per cell
            if v is not None and str(v).strip():  # try only non-empty cells
                float(v)  # conversion test
    except (TypeError, ValueError):  # any non-numeric cell
        numeric = False  # mark as a non-numeric column
    return vals, numeric  # return the value list and numericness


def _col_floats(vals: list[Any]) -> list[float | None]:  # column values -> floats (None for missing)
    """Convert a column of raw values to float; non-numeric/empty cells become None (missing)."""
    out: list[float | None] = []  # result buffer
    for v in vals:  # per cell
        try:  # fault-tolerant conversion
            out.append(float(v))  # try converting to float and collect it
        except (TypeError, ValueError):  # non-numeric
            out.append(None)  # missing placeholder
    return out  # return the float column


def _eval_side(side: str, header: list[str], rows: list[list[Any]]) -> list[float | None]:  # evaluate one side of an expression
    """Evaluate one side of an expression: "column", "A/B" (division), or "A+B-C" (term-wise +/-, left to right).

    Element names never contain +/- characters, so splitting on + / - is safe; "A/B" is not split further.
    """
    side = side.strip()  # strip whitespace
    tokens = [s.strip() for s in re.split(r"([+\-])", side) if s.strip()]  # split into tokens on +/- and keep the operators
    total: list[float | None] | None = None  # accumulated result (None = not yet initialized)
    op = "+"  # current operator (addition by default for the first token)
    for tok in tokens:  # process token by token
        if tok in ("+", "-"):  # operator token
            op = tok  # record the operator
            continue  # the next one is an operand
        if "/" in tok:  # ratio term: A/B
            num_s, den_s = tok.split("/", 1)  # split numerator/denominator
            nums, _ = _find_column(header, rows, num_s)  # fetch the numerator column
            dens, _ = _find_column(header, rows, den_s)  # fetch the denominator column
            nf, df = _col_floats(nums), _col_floats(dens)  # floatify both sides
            vals = [(a / b if a is not None and b not in (None, 0) else None)  # divide row by row (comprehension spans lines)
                    for a, b in zip(nf, df)]  # zero/missing denominator gives None
        else:  # plain column-name term
            vals, _ = _find_column(header, rows, tok)  # fetch the column
            vals = _col_floats(vals)  # floatify
        if total is None:  # first operand: assign directly
            total = list(vals)  # initialize the accumulator
        elif op == "+":  # addition term
            total = [(a + b if a is not None and b is not None else None)  # add row by row (comprehension spans lines)
                     for a, b in zip(total, vals)]  # missingness propagates
        else:  # subtraction term
            total = [(a - b if a is not None and b is not None else None)  # subtract row by row (comprehension spans lines)
                     for a, b in zip(total, vals)]  # missingness propagates
    return total if total is not None else []  # return the evaluated series (an empty expression gives an empty list)


def _table_or_error(state: Any) -> tuple[list[str], list[list[Any]]]:  # table-fetch helper
    """Return (header, rows) of the first usable numeric table; raise if none."""
    for u in state.get("data_units", []):  # iterate over data units
        if u.data_modality == "numeric" and u.data_status != "error":  # a usable numeric table
            return [str(h) for h in u.raw_data.get("header", [])], u.raw_data.get("rows", [])  # header and rows
    raise ValueError("没有可用的数值 DataUnit")  # report honestly


def _run_plot_xy(state: Any, call: ToolCall) -> dict[str, Any]:  # two-axis scatter run function
    """Two-axis scatter plot: x/y may be column names or ratio expressions (e.g. "Sr/Y"); supports log axes and group coloring."""
    p = call.parameters or {}  # get parameters
    style = str(p.get("style") or "journal")  # plot style: journal (Times/SimSun, 300dpi) | cn (legacy SimHei, 150dpi)
    formats = str(p.get("formats") or "png")  # artifact formats: png / png+svg / png+pdf
    x_name, y_name = str(p.get("x") or ""), str(p.get("y") or "")  # both axis names
    if not x_name or not y_name:  # missing axis name
        raise ValueError('请在 parameters 里给出 x 与 y（列名或比值，如 "Sr/Y"）')  # guiding error message
    header, rows = _table_or_error(state)  # fetch the table
    xs = _eval_side(x_name, header, rows)  # evaluate the x series
    ys = _eval_side(y_name, header, rows)  # evaluate the y series
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]  # pair up (drop missing)
    if len(pts) < 3:  # too few valid points
        raise ValueError(f"有效 (x,y) 点仅 {len(pts)} 个，不足以绘图")  # report insufficient data
    plt = _plt(style)  # initialize the backend and apply the style per parameter
    fig, ax = plt.subplots(figsize=fig_size("double", height=5.4))  # create the canvas (double-column journal preset)
    color_by = str(p.get("color_by") or "")  # group-coloring column (optional)
    if color_by:  # a grouping column was given
        gvals, _ = _find_column(header, rows, color_by)  # fetch the grouping column
        groups: dict[str, list[tuple[float, float]]] = {}  # group -> point set
        for (x, y), g in zip(pts, gvals[:len(pts)]):  # pair with groups
            groups.setdefault(str(g), []).append((x, y))  # collect per group
        for gi, (gname, gpts) in enumerate(groups.items()):  # scatter per group
            ax.scatter([a for a, _ in gpts], [b for _, b in gpts], s=14, alpha=0.7,  # same-color scatter within a group
                       color=f"C{gi % 10}", label=str(gname)[:20])  # color cycle
        ax.legend(fontsize=7)  # group legend
    else:  # ungrouped
        ax.scatter([a for a, _ in pts], [b for _, b in pts], s=14, alpha=0.6, color="#4C72B0")  # single-color scatter
    if p.get("log_x"): ax.set_xscale("log")  # optional log x axis
    if p.get("log_y"): ax.set_yscale("log")  # optional log y axis
    ax.set_xlabel(x_name); ax.set_ylabel(y_name)  # axis labels
    ax.grid(True, linewidth=0.3, alpha=0.4)  # light grid
    ax.set_title(f"{y_name} vs {x_name}（n={len(pts)}）", fontsize=11)  # title with point count
    fig.tight_layout()  # tight layout
    paths = save_figure(fig, _save_dir(state),  # unified saving (out_name is customizable, preventing same-name overwrites)
                        safe_stem(p.get("out_name"), "xy_scatter"), formats)
    plt.close(fig)  # release the canvas
    xs_ok = [x for x, _ in pts]; ys_ok = [y for _, y in pts]  # valid coordinate series
    return {"observation": (f"两轴散点图已生成：{y_name} vs {x_name}，n={len(pts)}；"  # observation text (spans lines)
                            f"x∈[{min(xs_ok):.3g},{max(xs_ok):.3g}]，y∈[{min(ys_ok):.3g},{max(ys_ok):.3g}]"),  # observation cont.: ranges
            "artifacts": artifacts_from(paths),  # figure artifacts (PNG as primary key + vector extra entries)
            "supports": ["xy_scatter_plotted"],  # support tags
            "limitations": ["散点图仅展示数值关系，不构成地质解释"]}  # limitation statement


def _run_plot_ternary(state: Any, call: ToolCall) -> dict[str, Any]:  # ternary diagram run function
    """Ternary diagram: three end-member columns normalized to 100% for plotting (any three columns, e.g. F-A-M / La-Th-Sc)."""
    p = call.parameters or {}  # get parameters
    style = str(p.get("style") or "journal")  # plot style
    formats = str(p.get("formats") or "png")  # artifact formats
    top, left, right = str(p.get("top") or ""), str(p.get("left") or ""), str(p.get("right") or "")  # three end-member column names
    if not (top and left and right):  # missing end member
        raise ValueError("请在 parameters 里给出 top/left/right 三个端元列名")  # guiding error message
    header, rows = _table_or_error(state)  # fetch the table
    tvals, _ = _find_column(header, rows, top)  # top end-member column
    lvals, _ = _find_column(header, rows, left)  # left end-member column
    rvals, _ = _find_column(header, rows, right)  # right end-member column
    tf, lf, rf = _col_floats(tvals), _col_floats(lvals), _col_floats(rvals)  # floatify all three columns
    pts = []  # point buffer
    for a, b, c in zip(lf, rf, tf):  # a=left b=right c=top
        if None in (a, b, c) or a + b + c <= 0:  # missing or invalid
            continue  # skip
        s = a + b + c  # three-member sum
        a, b, c = a / s * 100, b / s * 100, c / s * 100  # normalize to 100%
        pts.append((b + c / 2, c * 0.8660254))  # equilateral-triangle transform (x, y)
    if len(pts) < 3:  # too few valid points
        raise ValueError(f"有效三端元点仅 {len(pts)} 个，不足以绘图")  # report insufficient data
    plt = _plt(style)  # initialize the backend and apply the style per parameter
    fig, ax = plt.subplots(figsize=fig_size("double", height=6.0))  # create the canvas (double-column width, nearly square)
    apex_l, apex_r, apex_t = (0.0, 0.0), (100.0, 0.0), (50.0, 86.6025)  # three apexes (left/right/top)
    tri = [apex_l, apex_r, apex_t, apex_l]  # closed apex loop
    ax.plot([v[0] for v in tri], [v[1] for v in tri], color="#333333", linewidth=1.2)  # triangle frame lines
    for t in (20, 40, 60, 80):  # three families of 20% grid lines
        ax.plot([t, (100 + t) / 2], [0, (100 - t) * 0.8660254], color="#cccccc", linewidth=0.5)  # constant-left-end-member line
        ax.plot([100 - t, (100 - t) / 2], [0, (100 - t) * 0.8660254], color="#cccccc", linewidth=0.5)  # constant-right-end-member line
        ax.plot([t / 2, t / 2 + (100 - t)], [t * 0.8660254, t * 0.8660254], color="#cccccc", linewidth=0.5)  # constant-top-end-member line
    ax.scatter([v[0] for v in pts], [v[1] for v in pts], s=14, alpha=0.6, color="#4C72B0")  # plot the points
    ax.text(50, 90, top, fontsize=10, ha="center")  # apex label: top end member
    ax.text(-2, -4, left, fontsize=10, ha="right")  # bottom-left label: left end member
    ax.text(102, -4, right, fontsize=10, ha="left")  # bottom-right label: right end member
    ax.set_xlim(-5, 105); ax.set_ylim(-6, 92)  # field of view
    ax.axis("off")  # turn off the axis frame (the ternary uses apex labels)
    ax.set_title(f"三角图 {left}-{right}-{top}（n={len(pts)}）", fontsize=11)  # title
    fig.tight_layout()  # tight layout
    paths = save_figure(fig, _save_dir(state),  # unified saving
                        safe_stem(p.get("out_name"), "ternary"), formats)
    plt.close(fig)  # release the canvas
    return {"observation": (f"三角图已生成：{left}（左）-{right}（右）-{top}（顶），"  # observation text (spans lines)
                            f"n={len(pts)}（三端元已归一化到 100%）"),  # observation cont.: point count and normalization note
            "artifacts": artifacts_from(paths),  # figure artifacts (PNG as primary key + vector extra entries)
            "supports": ["ternary_plotted"],  # support tags
            "limitations": ["端元自动归一化到 100%，原始量纲不保留"]}  # limitation statement


def _run_plot_ree_spider(state: Any, call: ToolCall) -> dict[str, Any]:  # REE pattern diagram run function
    """REE chondrite-normalized pattern diagram: all sample pattern lines overlaid, logarithmic y axis."""
    p = call.parameters or {}  # get parameters
    style = str(p.get("style") or "journal")  # plot style
    formats = str(p.get("formats") or "png")  # artifact formats
    from oregenesis.tools.ree import _CHONDRITE  # reuse the chondrite reference values (McDonough & Sun 1995)
    from oregenesis.understanding import _norm_element  # reuse the element-name recognition logic
    elements = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]  # 14-element order
    header, rows = _table_or_error(state)  # fetch the table
    col_of: dict[str, int] = {}  # element -> column index mapping
    for i, h in enumerate(header):  # recognize each column by normalized name
        sym = _norm_element(h)  # normalized column name
        if sym in _CHONDRITE and sym not in col_of:  # is an REE and not yet registered
            col_of[sym] = i  # register the column index
    if len(col_of) < 6:  # not enough REE columns
        raise ValueError(f"表中仅识别出 {len(col_of)} 个 REE 列（≥6 才能成配分模式）")  # report insufficient data honestly
    plt = _plt(style)  # initialize the backend and apply the style per parameter
    fig, ax = plt.subplots(figsize=fig_size("double", height=5.2))  # create the canvas (double-column wide)
    n_plotted = n_skip = 0  # counters for plotted/skipped samples
    for r in rows[:60]:  # plot at most 60 lines (prevents an unreadable web)
        xs, ys = [], []  # buffer for this sample's pattern line
        for rank, el in enumerate(elements):  # per REE position
            i = col_of.get(el)  # get the column index
            v = r[i] if i is not None and i < len(r) else None  # get the concentration
            try:  # numericize
                fv = float(v)  # convert to float
            except (TypeError, ValueError):  # non-numeric
                continue  # skip this point (the line breaks here)
            if fv <= 0:  # non-positive concentration (below-detection-limit placeholder)
                continue  # skip
            xs.append(rank); ys.append(fv / _CHONDRITE[el])  # normalize and collect
        if len(ys) < 5:  # too few valid points to form a pattern
            n_skip += 1  # count the skip
            continue  # next row
        ax.plot(xs, ys, linewidth=0.9, alpha=0.65)  # pattern line
        n_plotted += 1  # count
    ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--")  # chondrite reference line
    ax.set_yscale("log")  # logarithmic y axis
    ax.set_xticks(range(len(elements))); ax.set_xticklabels(elements, fontsize=8)  # REE-order ticks
    ax.set_ylabel("样品/球粒陨石（对数）"); ax.grid(True, linewidth=0.3, alpha=0.4)  # axis and grid
    ax.set_title(f"REE 球粒陨石标准化配分图（n={n_plotted}，标准化值取 McDonough & Sun 1995）", fontsize=10)  # title
    fig.tight_layout()  # tight layout
    paths = save_figure(fig, _save_dir(state),  # unified saving
                        safe_stem(p.get("out_name"), "ree_spider"), formats)
    plt.close(fig)  # release the canvas
    if n_plotted == 0:  # not a single complete pattern
        raise ValueError("无样品具备足够 REE 数据（每条模式线至少需 5 个有效值）")  # report honestly
    return {"observation": (f"REE 配分图已生成：{n_plotted} 条样品模式线（14 项球粒陨石标准化，"
                            f"跳过 {n_skip} 条数据不足者）"),  # observation text
            "artifacts": artifacts_from(paths),  # figure artifacts (PNG as primary key + vector extra entries)
            "supports": ["ree_spider_plotted"],  # support tags
            "limitations": ["配分图为标准化模式展示，异常定量化请调用 calculate_eu_anomaly 等指标工具"]}  # limitation statement


def _run_plot_category_stats(state: Any, call: ToolCall) -> dict[str, Any]:  # category statistics bar chart run function
    """Categorical statistics bar chart: frequency counting and plotting on a specified (or auto-detected) categorical column."""
    import collections  # for frequency counting
    p = call.parameters or {}  # get parameters
    style = str(p.get("style") or "journal")  # plot style
    formats = str(p.get("formats") or "png")  # artifact formats
    header, rows = _table_or_error(state)  # fetch the table
    col_name = str(p.get("column") or "")  # target categorical column
    idx = None  # column index
    if col_name:  # a column was specified
        idx = next((i for i, h in enumerate(header) if h.strip().lower() == col_name.strip().lower()), None)  # exact match
        if idx is None:  # no match
            raise ValueError(f"找不到列 {col_name!r}（可用列: {', '.join(header[:12])}…）")  # raise with a list of available columns
    else:  # not specified: auto-pick a column with a moderate number of distinct values
        best, best_n = None, 0  # best column and distinct count
        for i, h in enumerate(header):  # evaluate each column
            vals = {str(r[i]).strip() for r in rows if i < len(r) and str(r[i]).strip()}  # set of distinct values
            if best_n < len(vals) <= 30:  # distinct count between 2 and 30 (plottable)
                best, best_n = i, len(vals)  # update the best
        idx = best  # use the auto-detected column
        if idx is None:  # no categorical column available
            raise ValueError("未找到适合统计的分类列，请在 parameters 里指定 column")  # guiding error message
    counts = collections.Counter(str(r[idx]).strip() for r in rows  # frequency counting (comprehension spans lines)
                                 if idx < len(r) and str(r[idx]).strip())  # empty values not counted
    if not counts:  # all empty
        raise ValueError("目标列全为空，无可统计数据")  # raise error
    common = counts.most_common(15)  # take the top 15 categories
    plt = _plt(style)  # initialize the backend and apply the style per parameter
    fig, ax = plt.subplots(figsize=fig_size("double", height=max(3.5, 0.42 * len(common))))  # canvas height scales with category count
    labels = [k[:24] for k, _ in common][::-1]  # category names (truncated to 24 chars, reversed for horizontal bars)
    values = [v for _, v in common][::-1]  # frequencies (reversed)
    total = sum(values)  # total
    ax.barh(labels, values, color="#4C72B0", edgecolor="white")  # horizontal bars
    for y, v in enumerate(values):  # label each bar with count and share
        ax.text(v, y, f" {v}（{v / total * 100:.1f}%）", va="center", fontsize=7.5)  # count + share
    ax.tick_params(labelsize=8)  # tick label size
    ax.set_title(f"分类统计：{header[idx]}（{total} 条）", fontsize=11)  # title with column name and total
    ax.grid(True, axis="x", linewidth=0.3, alpha=0.4)  # light x-direction grid
    fig.tight_layout()  # tight layout
    paths = save_figure(fig, _save_dir(state),  # unified saving
                        safe_stem(p.get("out_name"), "category_stats"), formats)
    plt.close(fig)  # release the canvas
    top_txt = "；".join(f"{k}={v}" for k, v in common[:5])  # observation summary: top 5 categories
    return {"observation": (f"分类统计柱状图已生成：{header[idx]} 共 {total} 条、{len(counts)} 类；"  # observation text (spans lines)
                            f"前五类——{top_txt}"),  # observation cont.: top 5
            "artifacts": artifacts_from(paths),  # figure artifacts (PNG as primary key + vector extra entries)
            "supports": ["category_stats_plotted"],  # support tags
            "limitations": ["统计图为频次描述，不构成地质解释"]}  # limitation statement


def build_plot_tools() -> list[ToolSpec]:  # generic plotting tool family constructor
    """Build the generic plotting tool family (two-axis scatter / ternary / REE pattern / category statistics, on demand)."""
    param_doc = ("；可选 parameters：style=journal（Times/宋体衬线、300dpi，默认）| cn（中文黑体、150dpi），"
                 "formats=png|png+svg|png+pdf，out_name 自定义产物文件名")  # shared parameter documentation
    return [  # four plotting tools (list spans multiple lines)
        _mk_tool("plot_xy_scatter",  # two-axis scatter
                 '两轴散点图：x/y 可为元素列或比值表达式（如 "Sr/Y"），支持对数轴与分组着色'
                 "（parameters: x/y/log_x/log_y/color_by）" + param_doc,
                 ("散点", "两轴", "相关图"), _run_plot_xy),
        _mk_tool("plot_ternary",  # ternary diagram
                 "三角图：任取三端元列归一化到 100% 投点（F-A-M、La-Th-Sc 等任意组合；parameters: top/left/right）" + param_doc,
                 ("三角图", "三端元"), _run_plot_ternary),
        _mk_tool("plot_ree_spider",  # REE pattern diagram
                 "REE 球粒陨石标准化配分图（蜘蛛图）：全样品模式线叠加，需表含 REE 列" + param_doc,
                 ("配分图", "稀土配分", "蜘蛛图", "标准化模式"), _run_plot_ree_spider),
        _mk_tool("plot_category_stats",  # statistics bar chart
                 "分类统计柱状图：对分类列做频次统计（自动识别分类列，可 parameters.column 指定）" + param_doc,
                 ("柱状图", "饼图", "占比", "统计图"), _run_plot_category_stats),
    ]
