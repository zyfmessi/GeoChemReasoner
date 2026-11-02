"""Journal-grade plotting style: unified fonts/colors/ticks and figure-saving parameters, shared by the generic plotting families.

journal style (default): Times New Roman (Latin serif) + SimSun (Chinese fallback, the standard Chinese-journal
pairing), STIX math fonts, inward ticks with minor ticks, frameless legend, Okabe-Ito colorblind-friendly
color cycle, and 300dpi by default when saving;
cn style: keeps the legacy SimHei/Microsoft YaHei chain and 150dpi, so the historical look remains reproducible.
"""

from __future__ import annotations  # enable PEP 563 deferred evaluation of type annotations

import re  # for sanitizing safe file names
from pathlib import Path  # path handling
from typing import Any  # any-type annotation

# Okabe-Ito colorblind-friendly color cycle (eight colors common in journals)
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]

# Current style state and font-probe note (save_figure reads the default dpi; callers may read the note to inform the user)
_CURRENT: dict[str, Any] = {"style": "cn", "note": ""}


def _probe_font(family: str) -> bool:  # font probe function
    """Probe whether a font is registered on this machine (by querying the font_manager registry directly).

    FontProperties().get_name() must not be used for probing - when the font is missing, matplotlib silently
    falls back to the default font and returns its name, so it would always appear to "exist".
    """
    try:  # font_manager import fallback
        from matplotlib import font_manager  # the font registry
        target = family.lower()  # case-insensitive comparison
        return any(f.name.lower() == target for f in font_manager.fontManager.ttflist)  # check entry by entry
    except Exception:  # fallback for extreme environments
        return False  # treat as absent


def apply_style(style: str = "journal") -> str:  # style application function
    """Reset rcParams per style (callable repeatedly to switch); returns the style name actually applied."""
    from matplotlib import rcParams  # global style parameters
    note = ""  # reset the font-probe note
    if style == "cn":  # legacy Chinese-report style: SimHei chain + relaxed defaults
        for font in ("SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"):  # Chinese font candidate chain
            if _probe_font(font):  # probe each font for availability on this machine
                rcParams["font.family"] = "sans-serif"  # sans-serif family
                rcParams["font.sans-serif"] = [font]  # set the hit font as first choice
                break  # stop on the first hit
        rcParams["axes.unicode_minus"] = False  # fix the missing minus glyph
        _CURRENT.update(style="cn", note="")  # record the current style
        return "cn"  # return the applied style
    # journal: Times (Latin serif) + SimSun (Chinese fallback); the math font is STIX, in the same family as Times
    chain = [f for f in ("Times New Roman", "STIXGeneral", "Liberation Serif", "DejaVu Serif")  # Latin serif candidate chain
             if _probe_font(f)]  # keep only those present on this machine
    if not chain:  # fallback when all are missing
        chain = ["DejaVu Serif"]  # matplotlib's bundled font is always present
    if not _probe_font(chain[0]) or chain[0] != "Times New Roman":  # note honestly when the first choice is not Times
        note = f"本机无 Times New Roman，已退回 {chain[0]}"
    cjk = [f for f in ("SimSun", "NSimSun", "STZhongsong", "Noto Serif CJK SC")  # Chinese SimSun candidate chain
           if _probe_font(f)]  # probe for available SimSun variants
    if cjk:  # SimSun present: the concrete font-name list serves directly as font.family - matplotlib >=3.7 falls back per glyph on lists
        rcParams["font.family"] = chain + cjk  # Latin glyphs go to Times, Chinese glyphs to SimSun (the standard Chinese-journal pairing)
        rcParams["font.serif"] = chain + cjk  # serif chain kept in sync, for mathtext etc. to fetch fonts by family
    else:  # no SimSun: honestly warn that Chinese glyphs may be missing
        rcParams["font.family"] = chain  # Latin chain only
        rcParams["font.serif"] = chain  # serif chain kept in sync
        note = (note + "；" if note else "") + "未找到中文宋体，中文标签可能显示为方框"
    rcParams.update(  # journal typesetting parameter set (font families were set above per the probe results; spans lines)
        **{
            "mathtext.fontset": "stix",  # math font visually in the same family as Times
            "axes.unicode_minus": False,  # fallback for the missing minus glyph
            "axes.linewidth": 0.8, "axes.titlesize": 10, "axes.labelsize": 9,  # axis linewidth and title font sizes
            "xtick.labelsize": 8, "ytick.labelsize": 8,  # tick label sizes
            "xtick.direction": "in", "ytick.direction": "in",  # inward ticks (journal convention)
            "xtick.minor.visible": True, "ytick.minor.visible": True,  # minor ticks
            "xtick.major.width": 0.8, "ytick.major.width": 0.8,  # tick linewidths
            "legend.frameon": False, "legend.fontsize": 8,  # frameless legend
            "grid.linewidth": 0.4, "grid.alpha": 0.4,  # light grid
            "lines.linewidth": 1.2, "lines.markersize": 4,  # line width and marker size
        })
    from cycler import cycler  # color-cycle constructor
    rcParams["axes.prop_cycle"] = cycler(color=OKABE_ITO)  # colorblind-friendly color cycle
    _CURRENT.update(style="journal", note=note)  # record the style and note
    return "journal"  # return the applied style


def style_note() -> str:  # get the font note
    """The font-probe note from the most recent apply_style (for citation in observations/warnings)."""
    return str(_CURRENT.get("note", ""))  # return the note


def current_style() -> str:  # get the current style
    """The currently applied style name (journal / cn)."""
    return str(_CURRENT.get("style", "cn"))  # return the style name


def fig_size(width: str = "double", height: float | None = None, ratio: float = 0.75):  # size preset function
    """Journal size presets: single=3.5in (single-column width), double=7.2in (double-column width); height derived proportionally."""
    w = 3.5 if str(width) == "single" else 7.2  # column width
    return (w, float(height) if height else round(w * ratio, 2))  # an explicitly given height takes precedence


def save_figure(fig, out_dir, stem: str, formats: str = "png", dpi: int | None = None) -> dict[str, str]:  # unified figure-saving function
    """Save per formats (png / png+svg / png+pdf) and return {extension: path}.

    dpi defaults per style: journal=300, cn=150. Artifact names are sanitized by safe_stem;
    together with the caller's out_name parameter this prevents same-name overwrites.
    """
    out = Path(out_dir)  # convert the directory to Path
    out.mkdir(parents=True, exist_ok=True)  # create recursively
    exts = [s.strip().lower() for s in str(formats).split("+") if s.strip()] or ["png"]  # split the extensions
    exts = [e for e in exts if e in ("png", "svg", "pdf")] or ["png"]  # keep only the three supported ones
    if dpi is None:  # default dpi follows the style
        dpi = 300 if _CURRENT.get("style") == "journal" else 150  # journal 300 / Chinese report 150
    paths: dict[str, str] = {}  # artifact path table
    for ext in exts:  # save per extension
        p = out / f"{stem}.{ext}"  # target path
        fig.savefig(p, dpi=dpi, bbox_inches="tight")  # save with a tight bounding box
        paths[ext] = str(p)  # register the path
    return paths  # return the path table


def safe_stem(name: Any, default: str) -> str:  # artifact-name sanitizer
    """Sanitize the out_name parameter into a safe file-name stem (whitespace/illegal characters to underscores; the default when empty)."""
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", str(name or "").strip())  # keep letters, digits, CJK characters, and hyphens
    return s.strip("_") or default  # fall back to the default name when empty after sanitizing


def artifacts_from(paths: dict[str, str], key: str = "figure") -> dict[str, str]:  # artifact-table assembly function
    """Convert save_figure's path table into the tool payload's artifacts: PNG as the primary key, vectors as extra entries."""
    arts = {key: paths.get("png") or next(iter(paths.values()))}  # primary artifact = PNG (first entry when there is no png)
    for ext, p in paths.items():  # register per extension
        if ext != "png":  # vector artifacts as extra keys (figure_svg / figure_pdf)
            arts[f"{key}_{ext}"] = p  # for parallel citation in the report's "figure artifacts" section
    return arts  # return the artifacts dict
