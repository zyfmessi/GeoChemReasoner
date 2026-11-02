"""Unified parsing entry points: parse_asset (single-asset dispatch) and parse_assets (batch scheduling).

Upper layers do not need to know which specific parser to call; they only need:
    data_units = parse_asset(asset)

Single-file parsing (parse_asset) and batch scheduling (parse_assets) are separated: the
former answers "how is one asset parsed", the latter answers "how is a batch of assets
scheduled" (sharing one ID allocator so asset/source/unit numbering stays continuous and
unique across the whole batch).
"""

from __future__ import annotations  # Enable PEP 563 deferred evaluation of type annotations

from pathlib import Path  # Import the Path type to get file extensions

from oregenesis.io.data_models import DataUnit, IdAllocator, InputAsset  # Import the data models and the ID allocator
from oregenesis.io.image_parser import parse_image_asset  # Import the image parser
from oregenesis.io.numeric_parser import parse_numeric_asset  # Import the numeric parser
from oregenesis.io.text_parser import parse_text_asset  # Import the text parser

# Extension -> modality: numeric (Excel/CSV tables), text (plain text/Markdown), image (plate photos)
_EXTENSION_MODALITY: dict[str, str] = {  # Start of the extension mapping table (dict spans multiple lines)
    ".csv": "numeric", ".xlsx": "numeric", ".xlsm": "numeric",  # Table extensions map to the numeric modality
    ".xls": "numeric",  # Legacy binary Excel (common in historical lab data sheets) also goes through numeric parsing
    ".txt": "text", ".md": "text", ".markdown": "text",  # Text extensions map to the text modality
    ".jpg": "image", ".jpeg": "image", ".png": "image",  # Common image extensions map to the image modality
    ".tif": "image", ".tiff": "image", ".bmp": "image",  # TIFF/BMP extensions map to the image modality
}

# Modality -> parser (parse_asset's dispatch table)
_PARSERS = {  # Start of the parser dispatch table (dict spans multiple lines)
    "numeric": parse_numeric_asset,  # The numeric modality maps to the numeric parser
    "text": parse_text_asset,  # The text modality maps to the text parser
    "image": parse_image_asset,  # The image modality maps to the image parser
}


def detect_modality(file_path: str) -> str:  # Detect the modality from the file extension
    """Detect the modality from the file extension; unknown types raise immediately (listing the supported types, never guessing silently)."""
    ext = Path(file_path).suffix.lower()  # Get the lowercase extension
    modality = _EXTENSION_MODALITY.get(ext)  # Look up the modality in the mapping table
    if modality is None:  # For unknown extensions
        raise ValueError(  # Raise an unsupported-type error (call spans multiple lines)
            f"不支持的数据文件类型: {ext}（当前支持 "  # Error message, first segment (implicit string concatenation continuation)
            "Excel/CSV: .csv .xlsx .xlsm .xls；文本: .txt .md；图像: .jpg .jpeg .png .tif .tiff .bmp）")  # Error message, following segment: lists all supported types
    return modality  # Return the detected modality


def make_asset(file_path: str, data_modality: str = "", *,  # Build an input asset (later arguments keyword-only)
               alloc: IdAllocator | None = None) -> InputAsset:  # Argument continuation: optional ID allocator
    """Build an InputAsset: detect the modality automatically and allocate asset/source numbers."""
    alloc = alloc or IdAllocator()  # Create a new allocator when none is passed
    modality = data_modality or detect_modality(file_path)  # Auto-detect when the modality is not explicitly given
    return InputAsset(  # Build and return the input asset (arguments span multiple lines)
        data_unit_id=alloc.next("asset"),  # Allocate an asset_NNN asset ID
        data_source_id=alloc.next("source"),  # Allocate a source_NNN source ID
        data_modality=modality,  # Write the detected modality
        file_path=str(file_path),  # Stringify the path uniformly
    )


def parse_asset(asset: InputAsset, alloc: IdAllocator | None = None) -> list[DataUnit]:  # Single-asset parsing dispatch
    """Parse a single asset: pick the parser automatically by data_modality and uniformly return list[DataUnit]."""
    alloc = alloc or IdAllocator()  # Create a new allocator when none is passed
    parser = _PARSERS.get(asset.data_modality)  # Look up the parser in the dispatch table by modality
    if parser is None:  # For unknown modalities
        raise ValueError(f"Unsupported data modality: {asset.data_modality}")  # Raise an unsupported-modality error
    units = parser(asset, alloc)  # Call the matching parser to get data units
    for u in units:  # The parsed units' asset back-reference follows the caller-supplied asset
        u.data_asset_id = asset.data_unit_id  # Force the back-reference to the caller-supplied asset ID
    return units  # Return all parsed units


def parse_assets(assets: list[InputAsset]) -> list[DataUnit]:  # Batch parsing scheduler
    """Batch parsing: process a list of InputAssets (mixed multi-Excel/multi-text/multi-image
    input) and return a flattened DataUnit list — one InputAsset does not necessarily map
    to exactly one DataUnit.

    Regardless of whether the assets shared an allocator at construction time, the
    asset/source numbers are re-allocated here with one allocator, keeping the whole
    batch's IDs continuous and unique (callers need not care about ID management).
    """
    alloc = IdAllocator()  # Create the batch-shared ID allocator
    for asset in assets:  # Re-number uniformly, preventing ID collisions among independently built assets
        asset.data_unit_id = alloc.next("asset")  # Re-allocate the asset ID to keep the batch unique
        asset.data_source_id = alloc.next("source")  # Re-allocate the source ID
    data_units: list = []  # Initialize the aggregated unit list
    for asset in assets:  # Parse asset by asset
        data_units.extend(parse_asset(asset, alloc))  # Flatten the parse results into the aggregate list
    return data_units  # Return all data units
