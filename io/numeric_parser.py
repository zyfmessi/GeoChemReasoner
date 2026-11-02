"""Numeric parser: Excel/CSV tables -> DataUnit.

Main task: read the file -> identify worksheets -> identify the used range -> detect
the header -> read data rows -> generate DataUnits. One Excel file may contain multiple
worksheets (e.g. whole-rock geochemistry / muscovite chemistry / apatite chemistry), so
one InputAsset can yield multiple DataUnits.

Design principle: the parser keeps only the "fact" of structured numeric data; it does
not judge what a lowered Nb/Ta means or what a negative Eu anomaly indicates — those
belong to the downstream semantic interpretation layer.
"""

from __future__ import annotations  # Enable PEP 563 deferred evaluation of type annotations

import csv  # Import the stdlib CSV row parser
from pathlib import Path  # Import the Path type to get file extensions
from typing import Any  # Import the Any type annotation

from oregenesis.io.data_models import DataUnit, IdAllocator, InputAsset  # Import the data models and the ID allocator

# When the ratio of fully empty cells in the header row exceeds this value it is treated as a "suspect header", written into the quality field for upper layers to down-weight
_HEADER_WARNING_RATIO = 0.8  # Threshold ratio of empty cells for a suspect header
_KV_HEADER_NAMES = {"field", "字段", "attribute", "属性", "parameter",  # Header keywords for the first column of key-value tables (lowercase match)
                    "参数", "item", "项目", "key", "键"}  # Key-value transposition happens only on a hit (avoids false positives on ordinary two-column tables)


def _cell_str(value: Any) -> str:  # Utility: convert a cell to a string
    """Cell -> string: None (empty cell) -> empty string; everything else via faithful str()."""
    return "" if value is None else str(value)  # Empty string for empty cells; the rest stringified as-is


def _clean_rows(rows: list[list[Any]], comment_prefix: str = "") -> tuple[list[str], list[list[str]], dict[str, int]]:  # Row-matrix cleaning: returns header / data rows / stats
    """Raw row matrix -> (header, data rows, cleaning stats).

    Identifies the used range: filters out comment lines (first cell starting with
    comment_prefix, e.g. the CSV-conventional "#") and fully empty rows; the first
    remaining row becomes the header; data rows are uniformly padded/truncated to
    the header width so rows and columns stay aligned.
    """
    def _is_data_row(r: list[Any]) -> bool:  # Nested helper: decide whether a row is a valid data row
        first = _cell_str(r[0]).strip() if r else ""  # Take the first cell stripped; empty string for empty rows
        if comment_prefix and first.startswith(comment_prefix):  # When the first cell starts with the comment prefix
            return False  # Comment line (first cell only, matching the legacy CSV import semantics)
        return any(_cell_str(c).strip() for c in r)  # A row is a data row if any cell is non-empty

    non_empty = [r for r in rows if _is_data_row(r)]  # Filter out comment lines and fully empty rows
    stats = {"empty_rows_removed": len(rows) - len(non_empty)}  # Count the removed empty and comment rows
    if not non_empty:  # Early return when everything was filtered out
        return [], [], stats  # Return an empty header / empty data plus the stats
    # Multi-row header detection: when the first row is a title/metadata line (e.g. "表4-2 x") and the real header (样品号|Co/Ni|...) lies below it,
    # the true header is the row among the first few valid rows with the highest element hits and >= 3 hits (test case library 2026-10-15).
    # Normal wide tables have the header in the first row (>= 3 hits): zero behavior change; a data row has at most 1 element name and cannot trigger falsely.
    header_idx = 0  # Index of the header row among the valid rows (default: first row)
    base_hits = _element_hits([_cell_str(c) for c in non_empty[0]])  # Element hits in the first row
    for i in range(1, min(len(non_empty), 6)):  # Scan at most 5 rows down for a more header-like row
        h_i = _element_hits([_cell_str(c) for c in non_empty[i]])  # Element hits in the candidate row
        if h_i >= 3 and h_i > base_hits:  # Switch only when clearly more header-like (prevents numeric rows / single-element rows from triggering)
            base_hits, header_idx = h_i, i  # Update the best header row
    if header_idx:  # When the header is not the first row
        stats["header_row_skipped"] = header_idx  # Record the number of skipped title/metadata rows (auditable)
    header = [_cell_str(c).strip() for c in non_empty[header_idx]]  # The identified header row, stripped
    data = []  # Initialize the data-row list
    for r in non_empty[header_idx + 1:]:  # Iterate the rows after the header
        cells = [_cell_str(c) for c in r]  # Convert the row's cells to strings uniformly
        cells += [""] * (len(header) - len(cells))   # Pad short rows with empties to stay aligned with the header
        data.append(cells[: len(header)])            # Truncate overlong rows to the header width
    stats["missing_values"] = sum(1 for r in data for c in r if not c.strip())  # Count blank cells in the data area
    return header, data, stats  # Return the (header, data rows, cleaning stats) triple


def _element_hits(names: list[str]) -> int:  # Count element hits in a header (used by the orientation vote)
    """Deduplicated count of the column names recognizable by _norm_element as element/oxide/ratio columns."""
    from oregenesis.understanding import _norm_element  # Local import of the semantic-layer normalizer (no cycle: io -> understanding is one-way)
    return len({e for e in (_norm_element(n) for n in names) if e})  # Deduplicated count (repeated "续表" continuation segments do not score twice)


def _transpose_matrix(header: list[str], rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:  # Transpose the whole matrix
    """Transpose the row matrix including the header (the header row participates; row keys become the new header), padding short columns."""
    width = max([len(header)] + [len(r) for r in rows] or [1])  # Take the full matrix's maximum width
    h = [_cell_str(c).strip() for c in header] + [""] * (width - len(header))  # Pad the header to the uniform width
    matrix = [h] + [[_cell_str(c).strip() for c in r] + [""] * (width - len(r)) for r in rows]  # Pad the whole matrix
    cols = [[matrix[i][j] for i in range(len(matrix))] for j in range(width)]  # Columns become rows (whole-matrix transposition)
    return cols[0], cols[1:]  # The original column 0 becomes the new header (row keys); the remaining columns become the new data rows


def _orient_table(header: list[str], rows: list[list[str]],  # Table orientation vote (arguments span multiple lines)
                  stats: dict[str, int]) -> tuple[list[str], list[list[str]]]:  # Argument continuation: cleaning stats (records the transposition fact)
    """Auto-correct transposed wide tables: swap orientation when the original header has almost no
    recognizable variables and the transposed orientation has significantly more.

    Background (test case library 2026-10-15 diagnosis): laboratory transposed wide tables with
    "elements as rows, samples as columns" (whole-rock major/trace elements, mineral trace-element
    "consolidated once" sheets, single-mineral analysis master tables) were parsed in the original
    orientation as "N samples x 0 variables", cutting off all quantitative and plotting tools.
    Criterion: the number of recognizable element variables in the header — if the original
    orientation already has >= 3, keep it (zero overhead for normal wide tables); otherwise swap
    only when the transposed orientation has strictly more deduplicated hits and >= 3. The
    transposition fact is written into the cleaning stats (quality.transposed) for auditability.
    """
    cur = _element_hits(header)  # Recognizable variables in the original orientation
    if cur >= 3:  # The original orientation is already a recognizable wide table
        return header, rows  # No transposition (zero-overhead fast path)
    t_header, t_rows = _transpose_matrix(header, rows)  # Transpose a copy for comparison
    if _element_hits(t_header) > cur and _element_hits(t_header) >= 3:  # The transposed orientation is significantly more recognizable
        stats["transposed"] = 1  # Record the transposition fact (merged into quality for audit)
        return t_header, t_rows  # Adopt the transposed orientation
    return header, rows  # Neither direction recognizable or a tie: keep the original orientation (no guessing)


def _table_unit(asset: InputAsset, alloc: IdAllocator, *, detail_type: str,  # Build a table unit (later arguments keyword-only)
                header: list[str], rows: list[list[str]], origin: dict[str, Any]) -> DataUnit:  # Argument continuation: header / data rows / provenance
    """One fully parsed table -> one DataUnit (with quality stats and provenance)."""
    empty_header = sum(1 for h in header if not h)  # Count empty column names in the header
    quality = {  # Build the quality dict (spans multiple lines)
        "row_count": len(rows),  # Record the data-row count
        "column_count": len(header),  # Record the column count
        "header_warning": len(header) == 0 or empty_header / max(len(header), 1) > _HEADER_WARNING_RATIO,  # Warn when the header is empty or the empty-column ratio exceeds the threshold
    }
    quality.update(origin.pop("_stats", {}))  # Merge the cleaning stats into the quality field and remove them from provenance
    return DataUnit(  # Build and return the standard data unit (arguments span multiple lines)
        data_unit_id=alloc.next("unit"),  # Allocate a unit_NNN unit ID
        data_asset_id=asset.data_unit_id,  # Back-reference the owning input asset's ID
        data_modality="numeric",  # Modality marked as numeric
        data_detail_type=detail_type,  # Write the detail type (csv_table/excel_worksheet)
        raw_data={"header": header, "rows": rows},  # Raw facts: the header and data-row matrix
        raw_data_reference=str(asset.file_path),  # Record the raw file path for tracing back
        data_origin=origin,  # Write the provenance info
        data_quality=quality,  # Write the quality stats
        data_status="success",  # Status marked as parsed successfully
    )


def _parse_csv_rows(rows: list[list[str]], asset: InputAsset, alloc: IdAllocator,  # Parse a list of CSV rows
                    ext: str, comment_prefix: str = "") -> list[DataUnit]:  # Argument continuation: extension and comment prefix
    """CSV text rows -> table DataUnit (import_string's string entry also routes here).

    Key-value table adaptation: for a two-column table whose first header cell is a keyword
    like Field/字段/属性/参数, transpose it into a "1-row wide table" (each key becomes a
    column) — key-value dossiers previously parsed into the illusion of "N samples x 0
    variables", making all quantitative tools unusable (benchmark evaluation 2026-09 diagnosis).
    """
    header, data, stats = _clean_rows(rows, comment_prefix=comment_prefix)  # Clean the row matrix into header / data / stats
    if not header:  # No valid header means the file is empty
        return []  # The whole file is empty: the caller decides whether to emit a warning unit
    if (len(header) == 2 and len(data) >= 3  # Key-value criterion 1: two columns and >= 3 data rows
            and header[0].strip().lower() in _KV_HEADER_NAMES  # Criterion 2: the first header cell hits a key-name keyword
            and all(h.strip() for h, _ in ((row[0], row[1]) for row in data))):  # Criterion 3: the key column is entirely non-empty
        header, data = [r[0].strip() for r in data], [[r[1].strip() for r in data]]  # Transpose: keys become column names, values merge into a single row
        stats["kv_transposed"] = True  # Record the transposition fact in the stats (auditable)
    header, data = _orient_table(header, data, stats)  # Orientation vote for transposed wide tables (fixes lab master tables with elements as rows)
    origin = {  # Build the provenance dict (spans multiple lines)
        "source_file": asset.data_source_id,  # Record the source ID
        "source_path": str(asset.file_path),  # Record the file path
        "file_extension": ext,  # Record the extension
        "header_row": 1,  # The header is on row 1
        "first_data_row": 2,  # Data starts at row 2
        "last_data_row": 1 + len(data),  # Compute the last data row number
        "_stats": stats,  # Stash the cleaning stats for the quality field
    }
    return [_table_unit(asset, alloc, detail_type="csv_table", header=header,  # Build a single csv_table unit (arguments span multiple lines)
                        rows=data, origin=origin)]  # Argument continuation: data rows and provenance


def _parse_csv(asset: InputAsset, alloc: IdAllocator) -> list[DataUnit]:  # Parse a CSV file
    """CSV file: prefer utf-8-sig (compatible with Excel exports carrying a BOM); on decode
    failure fall back to gbk, then latin-1 (public geological data often uses non-UTF-8
    encodings, e.g. GBK Chinese data dictionaries); hand off uniformly to the row parser;
    rows whose first cell starts with "#" are treated as comments (legacy import semantics)."""
    rows: list[list[str]] | None = None  # Buffer for the parse result
    last_err: Exception | None = None  # The last decode error (re-raised when all attempts fail)
    for enc in ("utf-8-sig", "gbk", "latin-1"):  # Decode fallback chain (latin-1 never fails; it faithfully bottoms out)
        try:  # Try each encoding in turn
            with open(asset.file_path, "r", encoding=enc, newline="") as f:  # Open with the candidate encoding
                rows = list(csv.reader(f))  # Read all rows and parse into a list-of-lists matrix
            break  # Stop once decoding succeeds
        except UnicodeDecodeError as e:  # On decode failure, move to the next encoding
            last_err = e  # Record the error and keep falling back
    if rows is None:  # All encodings failed (latin-1 should never get here)
        raise last_err or ValueError("CSV 解码失败")  # Re-raise the last error
    return _parse_csv_rows(rows, asset, alloc, ext=Path(asset.file_path).suffix.lower(),  # Hand off to the row parser with the lowercase extension (call spans multiple lines)
                           comment_prefix="#")  # Argument continuation: rows starting with "#" are treated as comments


def _read_xlsx_sheets(file_path: str) -> list[tuple[str, list[list[Any]]]]:  # Read an xlsx: return a list of (sheet name, value matrix)
    """Read .xlsx/.xlsm with openpyxl: each worksheet yields (sheet name, value matrix).

    openpyxl is imported lazily: a missing library raises a clear, actionable error carrying
    the install command instead of a bare ImportError.
    """
    try:  # Try a lazy import of openpyxl
        from openpyxl import load_workbook  # Import the xlsx reader on demand
    except ImportError as exc:  # Turn a missing library into an actionable error hint
        raise ImportError("解析 .xlsx 需要 openpyxl：pip install openpyxl") from exc  # Raise a clear error carrying the install command
    wb = load_workbook(file_path, read_only=True, data_only=True)  # data_only: take the computed value of formulas
    try:  # Ensure the workbook gets closed
        return [(ws.title, [list(row) for row in ws.iter_rows(values_only=True)])  # Yield per sheet (skip cell objects, take values)
                for ws in wb.worksheets]  # Iterate every worksheet
    finally:  # Run the cleanup regardless of success
        wb.close()  # Close the workbook to release the file handle


def _read_xls_sheets(file_path: str) -> list[tuple[str, list[list[Any]]]]:  # Read a legacy xls: output isomorphic to the xlsx reader
    """Read legacy .xls binaries with xlrd: yield (sheet name, value matrix) isomorphic to openpyxl's.

    Cell type mapping (xlrd aligned with openpyxl semantics): EMPTY/BLANK/ERROR -> None,
    DATE -> xldate_as_datetime (using the workbook datemode), BOOLEAN -> bool, others taken as-is.
    Trailing empty cells are truncated: old .xls files often have "phantom dimensions" (historical
    formatting inflating the used range); without truncation the missing-cell count is inflated
    (the same artifact as the xlsx 1.25M-missing-cells illusion; done only in this branch,
    leaving the .xlsx legacy path untouched to keep benchmark comparability).
    """
    try:  # Try a lazy import of xlrd
        import xlrd  # Import the legacy xls reader on demand
    except ImportError as exc:  # Turn a missing library into an actionable error hint
        raise ImportError("解析 .xls 需要 xlrd：pip install xlrd") from exc  # Raise a clear error carrying the install command
    book = xlrd.open_workbook(file_path)  # Open the legacy binary workbook
    sheets: list[tuple[str, list[list[Any]]]] = []  # Initialize the output (sheet name, value matrix) list
    for sh in book.sheets():  # Iterate every worksheet
        rows: list[list[Any]] = []  # The current sheet's value matrix
        for r in range(sh.nrows):  # Read row by row
            row: list[Any] = []  # The current row's value list
            for c in range(sh.ncols):  # Read column by column
                ct = sh.cell_type(r, c)  # Get the cell type code (xlrd type system)
                if ct in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK, xlrd.XL_CELL_ERROR):  # Empty/blank/error cells
                    row.append(None)  # Map uniformly to None (aligned with openpyxl values_only)
                elif ct == xlrd.XL_CELL_DATE:  # Excel date serial value
                    row.append(xlrd.xldate.xldate_as_datetime(sh.cell_value(r, c), book.datemode))  # Convert to datetime using the workbook epoch
                elif ct == xlrd.XL_CELL_BOOLEAN:  # Boolean cell
                    row.append(bool(sh.cell_value(r, c)))  # Convert to a Python bool
                else:  # Regular cells such as TEXT/NUMBER
                    row.append(sh.cell_value(r, c))  # Take the raw value (str or float)
            while row and row[-1] is None:  # Truncate trailing empty cells
                row.pop()  # Pop the trailing None (shrink phantom dimensions)
            rows.append(row)  # Append the row
        sheets.append((sh.name, rows))  # Collect this sheet
    return sheets  # Return all worksheets


def _parse_xlsx(asset: InputAsset, alloc: IdAllocator) -> list[DataUnit]:  # Parse an Excel file
    """Excel (.xlsx/.xlsm/.xls): each non-empty worksheet yields one DataUnit.

    .xlsx/.xlsm go through openpyxl and the legacy .xls goes through xlrd; both produce an
    isomorphic "value matrix" and then share the same cleaning / orientation-vote / unit-
    construction downstream (behavior identical to the old .xlsx path).
    """
    suffix = Path(asset.file_path).suffix.lower()  # Get the lowercase extension to pick the reader
    if suffix == ".xls":  # Legacy binary format branch
        sheets = _read_xls_sheets(asset.file_path)  # Read with xlrd (internally raises an actionable error when the library is missing)
    else:  # Regular .xlsx/.xlsm branch
        sheets = _read_xlsx_sheets(asset.file_path)  # Read with openpyxl
    units: list[DataUnit] = []  # Initialize the output unit list
    for ws_title, rows in sheets:  # Iterate (sheet name, value matrix)
        header, data, stats = _clean_rows(rows)  # Clean the current sheet into header / data / stats
        if not header:  # When this sheet has no valid content
            continue  # A fully empty worksheet yields no unit (other non-empty sheets exist, so this is not a "silent loss")
        header, data = _orient_table(header, data, stats)  # Orientation vote for transposed wide tables (fixes lab master tables with elements as rows)
        origin = {  # Build this sheet's provenance dict (spans multiple lines)
            "source_file": asset.data_source_id,  # Record the source ID
            "source_path": str(asset.file_path),  # Record the file path
            "file_extension": suffix,  # Record the extension (.xls vs .xlsx distinguishable)
            "worksheet": ws_title,  # Record the worksheet name
            "header_row": 1,  # The header is on row 1
            "first_data_row": 2,  # Data starts at row 2
            "last_data_row": 1 + len(data),  # Compute the last data row number
            "_stats": stats,  # Stash the cleaning stats
        }
        units.append(_table_unit(asset, alloc, detail_type="excel_worksheet",  # Build an excel_worksheet unit and append it (arguments span multiple lines)
                                 header=header, rows=data, origin=origin))  # Argument continuation: header / data / provenance
    return units  # Return all worksheet units


def parse_numeric_asset(asset: InputAsset, alloc: IdAllocator | None = None) -> list[DataUnit]:  # Numeric asset parsing entry point
    """Numeric asset parsing entry point: dispatch to CSV / XLSX / XLS by extension and return a list of table DataUnits."""
    alloc = alloc or IdAllocator()  # Create a new allocator when none is passed
    suffix = Path(asset.file_path).suffix.lower()  # Get the lowercase extension for dispatch
    if suffix == ".csv":  # CSV file branch
        return _parse_csv(asset, alloc)  # Call the CSV parser and return
    if suffix in (".xlsx", ".xlsm", ".xls"):  # Excel file branch (macro workbooks and legacy binaries included)
        return _parse_xlsx(asset, alloc)  # Call the Excel parser and return
    raise ValueError(f"数值型解析器不支持的文件类型: {suffix}（支持 .csv / .xlsx / .xls）")  # For other types raise unsupported-type directly
