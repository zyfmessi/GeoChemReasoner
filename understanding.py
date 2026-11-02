from __future__ import annotations  # enable PEP 563 deferred type-annotation evaluation
import re  # regex module: sample id recognition and header cleaning
from dataclasses import dataclass, field  # import the dataclass decorator and the field default factory
from pathlib import Path  # import the Path type to take file stems
from oregenesis.io.data_models import DataUnit  # import the standard data unit model

"""Data Understanding Layer: entity enrichment + semantic enrichment + multi-source alignment (layers 2-3 of the overall architecture).

Answers two questions and organizes scattered DataUnits into sample-aggregated bundles:
- data_entity_key  "who is this" (WHO/WHAT): sample id, analysis type, mineral, rock type;
- data_semantics   "what is it expressing": variable list and deterministic, computable
  objective descriptions — not the final genesis interpretation (that belongs to the
  evidence and reasoning layers).

The current implementation is purely rule-based (filename/header/Sheet name keywords) and
serves as the fallback foundation for LLM/VLM enrichment; rules go first to guarantee
reproducibility, and LLM enrichment can layer corrections on top.
"""

# Analysis-type keyword table (header/Sheet name/filename keywords -> analysis_type) — order is priority
_ANALYSIS_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (  # analysis-type keyword table start (tuple spans lines)
    ("mineral_chemistry", ("磷灰石", "apatite", "白云母", "muscovite", "黑云母", "biotite",  # mineral chemistry keywords (spans lines)
                           "锆石", "zircon", "石榴子石", "garnet", "辉石", "clinopyroxene")),  # mineral chemistry keywords continued
    ("fluid_inclusion", ("包裹体", "inclusion", "fluid_inclusion", "均一温度")),  # fluid inclusion keywords
    ("isotope", ("同位素", "isotope", "δ34s", "δd", "δ18o", "87sr", "143nd", "206pb")),  # isotope keywords
    ("ree", ("ree", "稀土", "la", "ce", "eu", "yb")),  # REE keywords
    ("whole_rock_major", ("sio2", "主量", "major", "al2o3", "mgo")),  # whole-rock major element keywords
    ("whole_rock_trace", ("微量", "trace", "cu", "pb", "zn", "rb", "sr", "ba", "nb", "ta")),  # whole-rock trace element keywords
)
# Rock-type keyword table (filename/text keywords -> rock_type)
_ROCK_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (  # rock-type keyword table start (tuple spans lines)
    ("granite", ("花岗岩", "granite")), ("pegmatite", ("伟晶岩", "pegmatite")),  # granite and pegmatite keyword pairs
    ("basalt", ("玄武岩", "basalt")), ("andesite", ("安山岩", "andesite")),  # basalt and andesite keyword pairs
    ("skarn", ("矽卡岩", "skarn")), ("sediment", ("沉积岩", "砂岩", "页岩", "sediment")),  # skarn and sediment keyword pairs
)
# Sample id recognition: letter+digit combinations in filenames (e.g. LYS01, sample-03, S1; suffixes such as _rock do not block the match)
_SAMPLE_ID_RE = re.compile(r"(?i)\b([a-z]{1,6}[-_]?\d{1,4})(?![a-z0-9])")  # precompiled sample id regex: letter prefix + digits, case-insensitive
# Known mineral keywords (minimal vocabulary feeding data_entity_key.mineral, extensible later)
_MINERAL_HINTS = ("磷灰石", "apatite", "白云母", "muscovite", "黑云母", "biotite",  # mineral keyword list first line (tuple spans lines)
                  "锆石", "zircon", "石榴子石", "garnet", "石英", "quartz", "方解石", "calcite")  # mineral keyword list continued

# Element symbol table (for matching headers to standard element names; covers major oxides/trace/REE)
_ELEMENT_SYMBOLS: tuple[str, ...] = (  # element symbol table start (tuple spans lines)
    "SiO2", "TiO2", "Al2O3", "Fe2O3", "FeO", "MgO", "CaO", "Na2O", "K2O", "P2O5", "MnO", "LOI",  # major oxide and LOI symbols
    "Li", "Be", "B", "F", "Sc", "V", "Cr", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",  # trace element symbols, first block
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ag", "Cd", "In", "Sn", "Sb", "Cs", "Ba", "Hf", "Ta",  # trace element symbols, second block
    "W", "Pb", "Bi", "Th", "U", "Au", "Hg", "Tl",  # trace element symbols, third block
    "Ti", "Mn", "Fe", "K",  # bare element symbols (referenced by ratio tools such as K/Rb, Ti/V, Fe/Mn; oxide-only forms would make those tools never applicable)
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",  # full list of lanthanide REE symbols
)


@dataclass  # declare as a dataclass
class SampleBundle:  # define the sample bundle class
    """Multi-source alignment product: the set of all DataUnits belonging to one sample (or one dataset)."""

    bundle_key: str                                    # alignment key (sample id or dataset name)
    units: list[DataUnit] = field(default_factory=list)  # all data units belonging to this key

    def variables(self) -> list[str]:                  # convenience: aggregate all numeric variable names (deduplicated, order preserved)
        seen: list[str] = []  # seen variable list (for order-preserving dedup)
        for u in self.units:  # walk each data unit in the bundle
            for v in u.data_semantics.get("variables", []):  # walk variable names in the semantics field of this unit
                if v not in seen:  # collect only variables not seen before
                    seen.append(v)  # append to the deduplicated result list
        return seen  # return the deduplicated variable name list


# Isotope ratio column names: [mass]element[mass]/[mass]element[mass],
# e.g. Pb207/Pb206, Pb206/U238, 87Sr/86Sr (the key numeric columns of dating tables, all previously missed)
_ISOTOPE_RATIO_RE = re.compile(  # precompiled isotope ratio regex (spans lines)
    r"^(?:\d{1,3})?([A-Za-z]{1,2}\d{0,3})\s*/\s*(?:\d{1,3})?([A-Za-z]{1,2}\d{0,3})$")  # allows leading/trailing mass numbers (87Sr/86Sr) and inner spaces


def _ratio_symbol(token: str) -> str | None:  # define the ratio numerator/denominator symbol validator
    """"Pb207"/"U238"/"Sr" -> standard element symbol; return None for anything not shaped like "symbol + optional mass number"."""
    m = re.match(r"^([A-Za-z]{1,2})\d{0,3}$", token)  # strip the trailing mass number to get the element symbol part
    if not m:  # not the symbol + mass number form
        return None  # reject outright
    return next((s for s in _ELEMENT_SYMBOLS if s.lower() == m.group(1).lower()), None)  # case-insensitive match against the standard symbol table


def _norm_element(name: str) -> str | None:  # define the header name normalizer
    """Header name -> standard element/oxide name; return None for non-element columns.

    Four-level matching: (1) exact full name (case-insensitive, covering "Cu"/"cu"/"SiO2" and
    unit-suffixed/parenthesized "Cu_ppm"/"Cu (ppm)"/"Ce [ppm]"); (2) isotope ratios
    (Pb207/Pb206, 87Sr/86Sr etc., both sides standard symbols, returning the original name
    with the 1se suffix stripped for fidelity); (3) prefix rule — a column name starting with
    an element symbol followed by a boundary character (_/-/space) maps to that symbol,
    covering lab full names such as "Nb_ppm_C_ICPOES_MS_49"; (4) none of the above -> None.
    Guardrails: reject when a digit immediately follows the prefix (avoids H2O->H, D18O->D
    misreads); content after the prefix must be separated by a boundary character (avoids
    Sample->S, Lab->La misreads).
    """
    clean = re.sub(r"[\(\（\[].*?[\)\）\]]", "", str(name))  # remove ASCII/CJK parentheses/brackets and their contents
    clean = re.sub(r"[_\s]*(ppm|ppb|ppt|wt%|wtpct|percent|%)[_\s]*$", "", clean,  # strip trailing unit suffixes such as ppm (call spans lines)
                   flags=re.IGNORECASE).strip()  # unit match is case-insensitive, then strip surrounding whitespace
    clean = re.sub(r"\s*\+?\s*1\s*(se|σ|sigma)$", "", clean,  # strip the trailing 1se/1σ suffix of dating table error columns (call spans lines)
                   flags=re.IGNORECASE).strip()  # strip the suffix and surrounding whitespace
    for sym in _ELEMENT_SYMBOLS:  # walk the standard element symbol table
        if clean.lower() == sym.lower():  # case-insensitive exact comparison with the header name
            return sym  # return the standard element symbol on a hit
    m = _ISOTOPE_RATIO_RE.match(clean)  # isotope ratio branch: Pb207/Pb206 style column names
    if m:  # when a ratio candidate exists, validate numerator and denominator
        a = _ratio_symbol(m.group(1))  # the numerator must be a standard symbol (mass number optional)
        b = _ratio_symbol(m.group(2))  # same for the denominator
        if a and b:  # accept only when both ends are valid element symbols
            return clean  # return the original name with the 1se suffix stripped (fidelity, e.g. "Pb207/Pb206")
    m = re.match(r"^([A-Za-z]{1,2})(?=[_\-\s])", clean)  # prefix rule: 1-2 leading letters followed by a boundary character
    if m:  # when a prefix candidate exists, validate the symbol and the guardrails
        sym = next((s for s in _ELEMENT_SYMBOLS if s.lower() == m.group(1).lower()), None)  # whether the prefix is a standard symbol (case-insensitive)
        if sym and not re.match(r"^\d", clean[m.end():]):  # symbol valid and the remainder does not start with a digit (blocks H2O/D18O)
            return sym  # return the standard element symbol (lab full-name column recognition)
    return None  # return None when no element matches


def summarize_table_columns(header: list[str], rows: list[list[str]], *,  # define the all-column table summarizer (signature spans lines)
                            max_cols: int = 12, budget: int = 600) -> str:  # param cont: column cap and total character budget
    """Deterministic all-column table summary: numeric columns get n/range/median/quartiles/mean, categorical columns get value counts.

    Benchmark diagnosis 2026-10: seed evidence previously carried only a "sample count x
    element variable count" profile, so the numeric columns of microthermometry (Th/Tm/salinity),
    isotopes (δ34S) and geochronology never entered the context, and all three tested models
    (qwen14b inventing numbers / qwen8b giving up / deepseek refusing) hit the same wall.
    This function compresses each column into one line fragment; numeric and low-cardinality
    categorical columns come first, ID-like columns (unique count ~= row count) and
    high-cardinality text columns record only their unique value count — control the length,
    not the facts.
    """  # function docstring: motivation (benchmark diagnosis) and scope
    pieces_num: list[str] = []  # numeric column fragments (highest display priority)
    pieces_cat: list[str] = []  # low-cardinality categorical fragments (second priority)
    pieces_other: list[str] = []  # ID/high-cardinality text fragments (last, shown only with leftover budget)
    for i, h in enumerate(header):  # walk columns (i is the column index)
        if not h:  # empty header cell
            continue  # skip
        if re.search(r"date|日期|时间|_time\b", h, re.IGNORECASE) and "time_" not in h.lower():  # date/time column (by column name; the time_ prefix excludes heating duration columns)
            vals_d = [str(r[i]).strip() for r in rows if i < len(r) and str(r[i]).strip()]  # non-empty cells
            if vals_d:  # record only when the column is non-empty
                pieces_other.append(f"{h} 日期列({len(set(vals_d))}个值)")  # record the unique value count only (date statistics carry no geoscience meaning)
            continue  # next column
        vals = [str(r[i]).strip() for r in rows if i < len(r) and str(r[i]).strip()]  # non-empty cells (short rows count as missing)
        if not vals:  # whole column empty
            continue  # skip
        nums: list[float] = []  # numeric parse buffer (sentinels removed)
        n_sentinel = 0  # sentinel count (-9999 style missing placeholders)
        for v in vals:  # try numeric conversion cell by cell
            try:  # conversion wrapped in try
                f = float(v.replace(",", ""))  # tolerate thousands separators
            except ValueError:  # text placeholders such as "<LOD"
                continue  # ignore the cell
            if f <= -1000:  # large negative = missing sentinel (-9999/-3333/-5555; genuine negative values such as δ34S are far above -1000 and not affected)
                n_sentinel += 1  # count as sentinel
                continue  # exclude from statistics
            nums.append(f)  # genuine observation goes into the buffer
        if (all(x == int(x) for x in nums) and nums and min(nums) >= 1900  # all integers and >=1900 -> suspected year/date encoding
                and (max(nums) <= 2100 or 1e6 <= max(nums) <= 29991231)):  # up to 2100 is a year, 1e6~29991231 is yyyymmdd
            pieces_other.append(f"{h} 日期/编码列({len(nums)}值)")  # mark only, no statistics (the mean carries no geoscience meaning)
            continue  # next column
        if len(nums) >= max(3, int(0.25 * len(vals))):  # enough valid values (sentinels removed, so half-missing columns do not fall into the categorical branch) -> numeric column
            s = sorted(nums)  # sorted copy: for quantile lookups
            n = len(s)  # number of valid values
            if n == 0:  # no genuine values left after sentinel removal
                pieces_other.append(f"{h} 全缺失({n_sentinel}格哨兵)")  # record the sentinel count only
                continue  # next column
            piece = (f"{h} n={n} {s[0]:.4g}~{s[-1]:.4g} 中位{s[n // 2]:.4g} "  # fragment first half: name/count/min~max/median (4 significant digits)
                     f"P25~P75 {s[int(0.25 * (n - 1))]:.4g}~{s[int(0.75 * (n - 1))]:.4g} 均值{sum(s) / n:.4g}")  # fragment second half: quartiles/mean
            if len(set(s)) <= 5 and n >= 6 and all(x == int(x) for x in s):  # <=5 unique values, enough samples, and all integers -> discrete values
                piece += "（离散整数值，注意甄别是实验设定条件还是自然观测）"  # flags furnace temperature / visually estimated percentages etc.; genuine decimal measurements (e.g. Tm_CO2) are not misflagged; clustered genuine decimal measurements are not misflagged
            if n_sentinel:  # sentinel cells were removed
                piece += f"(剔{n_sentinel}哨兵)"  # record the removed count faithfully
            if len(nums) + n_sentinel < len(vals):  # unparsed cells exist
                piece += f"（{len(vals) - len(nums) - n_sentinel}格非数值）"  # record the non-numeric cell count faithfully
            if re.search(r"lat|lon|coord", h, re.IGNORECASE):  # coordinate columns (lat/lon/coord): demote to the end of the display order
                pieces_other.append(piece)  # spatial metadata has geo-analysis value but must not crowd geochemistry columns out of the catalog budget
            else:  # regular numeric column
                pieces_num.append(piece)  # collect the numeric column
        else:  # categorical/text column
            distinct: dict[str, int] = {}  # value counts (order preserved)
            n_sent = 0  # categorical side sentinel count (strip -9999 style missings before counting values)
            for v in vals:  # count value by value
                try:  # sentinel detection: parses as float and <=-1000 means a missing placeholder
                    if float(v.replace(",", "")) <= -1000:  # large negative sentinel
                        n_sent += 1  # count as sentinel
                        continue  # keep out of the value counts
                except ValueError:  # plain text value
                    pass  # count normally
                distinct[v] = distinct.get(v, 0) + 1  # count +1
            note = f"(剔{n_sent}哨兵)" if n_sent else ""  # sentinel note
            if not distinct:  # no genuine values left after sentinel removal
                pieces_other.append(f"{h} 全缺失({n_sent}格哨兵)")  # record the sentinel count only
            elif len(distinct) == len(vals) - n_sent:  # everything else unique -> ID-like column
                pieces_other.append(f"{h} ID列({len(distinct)}个){note}")  # record the unique value count only
            elif len(distinct) <= 6:  # low-cardinality categorical column
                top = sorted(distinct.items(), key=lambda t: -t[1])  # sort by count descending
                pieces_cat.append(f"{h}: " + "/".join(f"{k}×{c}" for k, c in top) + note)  # list every value x count
            else:  # high-cardinality text column
                top = sorted(distinct.items(), key=lambda t: -t[1])[:3]  # top 3 values by count
                pieces_other.append(f"{h} {len(distinct)}个取值(最多:" + "/".join(f"{k}×{c}" for k, c in top) + f"){note}")  # record the count + Top3
    out: list[str] = []  # output fragment buffer
    used = 0  # used character count
    for piece in pieces_num + pieces_cat + pieces_other:  # join in information priority order (numeric > low-cardinality categorical > other)
        if len(out) >= max_cols or used + len(piece) > budget:  # column cap or budget reached
            out.append("…")  # truncation marker
            break  # end the walk
        out.append(piece)  # include this column fragment
        used += len(piece) + 1  # charge the budget (separator included)
    return "；".join(out)  # join with semicolons and return


def stage_table_brief(header: list[str], rows: list[list[str]], *,  # define the stage-grouped table summarizer (signature spans lines)
                      budget: int = 900, top_groups: int = 4) -> str:  # param cont: total budget and number of groups included
    """Recognize stage-style categorical columns (stage/阶段/期次) and produce per-group numeric statistics — fluid stage tables become visible at a glance.

    Deep-work diagnosis 2026-10-13: the largest single loss in task1/2/6 was the fluid
    P-T-X / stage structure, while the grouped-statistics tool proved that grouping the
    Microtherm table by Stage yields exactly the scoring anchors (Q1 281~387, Q2 174~237).
    This function produces that table automatically at the seed-evidence stage: numeric
    tables containing a stage-style column get per-group statistics, taking the top
    top_groups groups by size (the largest group is usually the main mineralization stage).
    """  # function docstring: motivation (deep-work diagnosis) and scope
    gi = None  # stage column index
    for i, h in enumerate(header):  # scan columns for a stage style name
        if h and re.search(r"stage|阶段|期次", h, re.IGNORECASE):  # Chinese or English stage naming
            gi = i  # hit
            break  # take the first matching column
    if gi is None:  # no stage style column
        return ""  # table not applicable
    groups: dict[str, list[list[str]]] = {}  # group buffer: group key -> row matrix
    for r in rows:  # assign rows to groups
        key = str(r[gi]).strip() if gi < len(r) else ""  # group key
        groups.setdefault(key or "(空)", []).append(r)  # empty keys become "(空)"
    top = sorted(groups.items(), key=lambda t: -len(t[1]))[:top_groups]  # take the top N groups by size
    lines: list[str] = []  # group fragment buffer
    for g, grows in top:  # summarize group by group
        gsum = summarize_table_columns(header, grows, max_cols=6, budget=260)  # within group summary (numeric columns first)
        if gsum:  # the group has summarizable content
            lines.append(f"[{g} n={len(grows)}] {gsum}")  # group row
    out = "；".join(lines)  # join
    return out[:budget] if len(out) > budget else out  # budget truncation


def enrich_entities(unit: DataUnit) -> DataUnit:  # define the entity enricher (fills the entity key in place)
    """Entity enrichment (fills unit.data_entity_key in place): which entity does this data describe.

    Evidence sources in priority order: table column names (a sample_id column) -> filename
    sample id pattern -> analysis type keywords -> rock type/mineral keywords. Rules fill
    only the fields they are confident about; prefer omission over guessing.
    """
    stem = Path(unit.data_origin.get("source_path", "")).stem  # source file stem (extension removed)
    stem_upper = stem.upper()  # uppercase the stem for sample id matching
    key: dict = {}  # initialize the entity key dict
    m_file = _SAMPLE_ID_RE.search(stem_upper)  # filename sample id (the modality-agnostic alignment anchor)
    if m_file:  # when the filename matches the sample id pattern
        key["sample_id"] = m_file.group(1).upper().replace("-", "").replace("_", "")  # take the sample id and normalize by removing hyphens/underscores
    if unit.data_modality == "numeric":  # numeric table units take the column name branch
        header = [str(h) for h in unit.raw_data.get("header", [])]  # take the header, cast all to string
        id_col = next((h for h in header if h.strip().lower() in  # find the first sample id column (condition spans lines)
                       ("sample_id", "样品id", "样品编号", "编号", "id")), None)  # accepted sample column names, None when absent
        if id_col:  # record when a sample id column exists
            key["sample_column"] = id_col              # in-table sample id column (one sample per row)
        elements = [e for e in (_norm_element(h) for h in header) if e]  # normalize header columns to collect element variables
        if elements:  # write when element variables are recognized
            key["variables"] = elements                # element/oxide variable list (entity level facts)
        blob = (stem + " " + " ".join(header) + " " +  # concatenate filename and header text (spans lines)
                unit.data_origin.get("worksheet", "")).lower()  # append the worksheet name and lowercase the whole blob
    else:  # non-numeric units take the text clue branch
        blob = (stem + " " + str(unit.raw_data.get("text", ""))[:400]).lower()  # filename plus the first 400 chars of text, lowercased, as the clue source
    # analysis type: stop at the first keyword hit (order is priority)
    for atype, hints in _ANALYSIS_HINTS:  # walk the analysis type keyword table in priority order
        if any(h in blob for h in hints):  # a hit when any keyword appears in the clue text
            key["analysis_type"] = atype  # record the matched analysis type
            break  # first hit wins, preserving priority
    # rock type and mineral (text/filename clues)
    for rock, hints in _ROCK_HINTS:  # walk the rock type keyword table
        if any(h in blob for h in hints):  # any rock keyword hit
            key["rock_type"] = rock  # record the rock type
            break  # first hit wins
    mineral = next((m for m in _MINERAL_HINTS if m in blob), None)  # first mineral name appearing in the text, None if none
    if mineral:  # when a mineral matches
        key["mineral"] = mineral  # record the mineral name
    unit.data_entity_key = key  # write the entity key back to the data unit (filled in place)
    return unit  # return the same unit so calls can chain


def enrich_semantics(unit: DataUnit) -> DataUnit:  # define the semantic enricher (fills the semantics field in place)
    """Semantic enrichment (fills unit.data_semantics in place): what is this data expressing.

    The rule stage emits only deterministic, computable objective descriptions (variable
    list / sample size / missing cell statistics); observation and possible_process are
    left to LLM semantic enrichment or the evidence layer — never fabricated.
    """
    if unit.data_modality == "numeric" and unit.data_status != "error":  # generate numeric semantics for numeric tables parsed without errors
        header = [str(h) for h in unit.raw_data.get("header", [])]  # header as a list of strings
        rows = unit.raw_data.get("rows", [])  # data row list
        variables = [e for e in (_norm_element(h) for h in header) if e]  # normalize the header into the element variable list
        attributes: dict[str, str] = {}  # attribute list (field value pairs of a key-value dossier; for input fidelity)
        if len(rows) == 1 and 1 < len(header) <= 40:  # single-row wide table (transposed key-value product): one attribute per column
            attributes = {header[i]: str(rows[0][i]) for i in range(len(header))  # column name -> the value in that row (comprehension spans lines)
                          if i < len(rows[0]) and str(rows[0][i]).strip()}  # skip empty values
        elif len(header) == 2 and 3 <= len(rows) <= 40:  # two-column long table (untransposed key-value dossier): one attribute per row
            attributes = {str(r[0]): str(r[1]) for r in rows  # first column -> second column (comprehension spans lines)
                          if len(r) >= 2 and r[0].strip() and r[1].strip()}  # include only when both key and value are non-empty
        unit.data_semantics = {  # overwrite the semantics field (dict spans lines)
            "source": "rule",                          # semantics source marker (rule / llm / vlm)
            "variables": variables,  # write the element/oxide variable list
            "n_samples": len(rows),                    # row count = sample count (wide table convention)
            "n_missing_cells": unit.data_quality.get("missing_values", 0),  # pass through the missing cell count from the quality field
            **({"attributes": attributes} if attributes else {}),  # write only when the attribute list exists (input fidelity for key-value dossiers)
        }
    elif unit.data_modality == "text" and not unit.data_quality.get("empty_text"):  # generate text semantics for text units with non-empty text
        text = str(unit.raw_data.get("text", ""))  # original text
        unit.data_semantics = {  # overwrite the semantics field (dict spans lines)
            "source": "rule",  # mark the semantics source as rule
            "character_count": len(text),  # count text characters
            "excerpt": text[:120],                     # objective excerpt, no interpretation
        }
    return unit  # return the same unit


def enrich_all(units: list[DataUnit]) -> list[DataUnit]:  # define the batch entity + semantic enrichment entry point
    """Batch entity + semantic enrichment (LangGraph enrich node entry), returning the same units (filled in place)."""
    return [enrich_semantics(enrich_entities(u)) for u in units]  # enrich each unit (entity first, then semantics) and collect the results


def align_by_entity(units: list[DataUnit]) -> list[SampleBundle]:  # define the entity-based alignment/aggregation function
    """Multi-source alignment: aggregate DataUnits by entity key into SampleBundles (units of the same sample/dataset grouped together).

    Alignment key priority: data_entity_key.sample_id -> filename stem (units of the same
    file aggregate naturally) -> "unsorted" (the fallback bucket for units with no identity
    clue, so nothing is lost).
    """
    bundles: dict[str, SampleBundle] = {}  # alignment key -> bundle dict
    for u in units:  # walk all data units
        key = u.data_entity_key.get("sample_id") or \
            Path(u.data_origin.get("source_path", "")).stem or "unsorted"  # fall back to the file stem when there is no sample id, then to unsorted
        if key not in bundles:  # create a new bundle the first time a key appears
            bundles[key] = SampleBundle(bundle_key=key)  # instantiate the bundle with that key
        bundles[key].units.append(u)  # attach the unit to its bundle
    return list(bundles.values())  # return the list of all bundles
