"""Text parser: TXT/Markdown -> blank-line-delimited DataUnits.

Reads the raw bytes and tries utf-8-sig / utf-8 / gb18030 decoding in turn (covering
the encodings common in Chinese literature); after decoding, splits lines and divides
the text into blocks at blank lines. Each block stores the original text with its start
and end line numbers, so any downstream evidence can be traced back to "which line of
the original file" — crucial for scientific reasoning and result auditing.
"""

from __future__ import annotations  # Enable PEP 563 deferred evaluation of type annotations

from pathlib import Path  # Import the Path type to read bytes and get extensions

from oregenesis.io.data_models import DataUnit, IdAllocator, InputAsset  # Import the data models and the ID allocator

# Decode attempt order: utf-8-sig handles files with a BOM; gb18030 bottoms out Chinese Windows encodings
_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")  # Tuple of candidate encodings (tried in order)


def _read_text(file_path: str) -> tuple[str, str]:  # Read a file trying each candidate encoding
    """Try reading the file with each candidate encoding in turn; return (text, actual encoding); raise UnicodeDecodeError when all fail."""
    raw_bytes = Path(file_path).read_bytes()  # Read the raw bytes in one go
    last_error: Exception | None = None  # Track the last decode exception
    for enc in _ENCODINGS:  # Try each candidate encoding in order
        try:  # Try decoding with the current encoding
            return raw_bytes.decode(enc), enc  # Return the text and the encoding used on success
        except (UnicodeDecodeError, LookupError) as e:  # Catch decode failures and unknown encodings
            last_error = e  # Save the exception for the final raise
    raise last_error if last_error else UnicodeDecodeError("all", b"", 0, 1, "无可尝试编码")  # Raise the last exception when everything failed


def _split_blocks(lines: list[str]) -> list[tuple[int, int]]:  # Split text into blocks at blank lines
    """Split the text into blocks at blank lines; return [(start line, end line)] (1-based line numbers, inclusive)."""
    blocks: list[tuple[int, int]] = []  # Initialize the list of block line ranges
    start: int | None = None  # Start line of the current block (None means not inside a block)
    for i, line in enumerate(lines, 1):  # Iterate the lines with line numbers starting at 1
        if line.strip():  # Non-blank lines belong to some block
            if start is None:  # No block in progress yet
                start = i  # A non-blank line opens a new block
            end = i  # Update the current block's end line
        elif start is not None:  # A blank line with a block in progress
            blocks.append((start, end))  # The blank line closes the current block
            start = None  # Reset to the no-block state
    if start is not None:  # When the file ends with non-blank lines
        blocks.append((start, len(lines)))  # The end of the file closes the last block
    return blocks  # Return all block line ranges


def parse_text_asset(asset: InputAsset, alloc: IdAllocator | None = None) -> list[DataUnit]:  # Text asset parsing entry point
    """Text asset parsing entry point: decode -> split lines -> split blocks at blank lines -> one DataUnit per block.

    Empty text (a truly empty file / only whitespace and newlines) is not silently
    ignored: a DataUnit with data_status="warning" is still produced, so downstream
    systems know the file was indeed ingested and processed, just without effective
    text. Decode failures yield an error-level unit carrying the cause.
    """
    alloc = alloc or IdAllocator()  # Create a new allocator when none is passed
    ext = Path(asset.file_path).suffix.lower()  # Get the lowercase extension
    base_origin = {  # Build the shared provenance dict (spans multiple lines)
        "source_file": asset.data_source_id,  # Record the source ID
        "source_path": str(asset.file_path),  # Record the file path
        "file_extension": ext,  # Record the extension
    }

    def _unit(detail: str, raw: dict, quality: dict, status: str,  # Nested unit builder (arguments span multiple lines)
              origin_extra: dict) -> DataUnit:  # Argument continuation: extra provenance fields
        """Unified construction of block/error units: provenance fields are copied per unit, so multiple blocks never share one dict."""
        origin = dict(base_origin)  # Copy the shared provenance so blocks never share a mutable dict
        origin.update(origin_extra)  # Merge this block's own provenance fields
        return DataUnit(  # Build the text data unit (arguments span multiple lines)
            data_unit_id=alloc.next("unit"),  # Allocate a unit_NNN ID
            data_asset_id=asset.data_unit_id,  # Back-reference the owning asset's ID
            data_modality="text",  # Modality marked as text
            data_detail_type=detail,  # Write the detail type (text_block)
            raw_data=raw,  # Write the raw text content
            raw_data_reference=str(asset.file_path),  # Record the raw file path
            data_origin=origin,  # Write the provenance info
            data_quality=quality,  # Write the quality info
            data_status=status,  # Write the status marker
        )

    try:  # Try to decode and read the text
        text, encoding = _read_text(asset.file_path)  # Get the text and the encoding actually used
    except (UnicodeDecodeError, OSError) as e:  # Catch decode failures and read IO errors
        return [_unit("text_block", {}, {"decode_error": True, "error": str(e)},  # Return an error unit carrying the error message (arguments span multiple lines)
                      "error", {})]  # Argument continuation: status error, no extra provenance

    lines = text.splitlines()  # Split the text into lines
    units: list[DataUnit] = []  # Initialize the output unit list
    for start, end in _split_blocks(lines):  # Iterate each text block's line range
        block_lines = lines[start - 1 : end]          # 1-based inclusive slice
        block_text = "\n".join(block_lines)  # Re-join the block text with newlines
        units.append(_unit(  # Build a unit for this block and append it (arguments span multiple lines)
            "text_block",  # Detail type: text block
            {"text": block_text},  # Raw data: the block text
            {"empty_text": False, "character_count": len(block_text),  # Quality: non-empty marker and character count (dict spans multiple lines)
             "line_count": len(block_lines)},  # Quality: line count within the block
            "success",  # Status: parsed successfully
            {"encoding": encoding, "start_line": start, "end_line": end},  # Extra provenance: encoding and start/end lines
        ))

    if not units:  # No valid text blocks at all: still keep one warning unit, never silently drop
        units.append(_unit(  # Append the empty-text warning unit (arguments span multiple lines)
            "text_block",  # Detail type: text block
            {"text": ""},  # Raw data: empty text
            {"empty_text": True, "character_count": 0, "line_count": len(lines)},  # Quality: empty-text marker and total line count
            "warning",  # Status: warning
            {"encoding": encoding, "start_line": 1, "end_line": len(lines)},  # Extra provenance: encoding and the full-text line range
        ))
    return units  # Return all text-block units
