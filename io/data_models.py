"""Unified data models of the data input layer: InputAsset (input asset) and DataUnit (standard data unit).

Data flow: raw file -> InputAsset -> (modality parsing) -> list[DataUnit] -> downstream
entity recognition and semantic understanding. The two layers are strictly separated:
InputAsset describes "what file the user provided" (the identity card of the raw input
in the system), while DataUnit describes "what usable data this file yielded after
parsing" (the standard structure downstream Agents process uniformly).

Responsibility boundary: this layer only performs objective reading (which file / where
it is / which modality / what raw structure was parsed / how good its quality is); it
does no mineral identification, sample identity inference, or geological interpretation
— those belong to the downstream entity recognition layer (data_entity_key), the
semantic layer (data_semantics), and the evidence layer.
"""

from __future__ import annotations  # Enable PEP 563 deferred evaluation of type annotations

from dataclasses import dataclass, field  # Import the dataclass decorator and the field default factory
from typing import Any  # Import the Any type annotation


class IdAllocator:  # ID allocator class
    """In-session sequential ID allocator: stable short IDs like asset_001 / source_001 / unit_001.

    All assets parsed by a single parse_assets() call share one allocator, keeping the
    whole batch's numbering continuous and conflict-free; a single-file parse_asset()
    gets its own allocator.
    """

    def __init__(self) -> None:  # Constructor: initialize the per-prefix counters
        self._counters: dict[str, int] = {}  # prefix -> number allocated so far

    def next(self, prefix: str) -> str:  # Return the next number for a prefix
        """Allocate the next prefix_NNN number (NNN zero-padded to three digits)."""
        n = self._counters.get(prefix, 0) + 1  # Increment this prefix's counter to get the new number
        self._counters[prefix] = n  # Write it back so the next call increments
        return f"{prefix}_{n:03d}"  # Format as a prefix_001-style three-digit zero-padded number


@dataclass  # Declare as a dataclass
class InputAsset:  # Input asset class
    """Raw input data asset: one user input file corresponds to one InputAsset.

    Stores only four kinds of facts — which file (data_unit_id), where it is
    (file_path), where it comes from (data_source_id), and which modality it belongs
    to (data_modality) — plus the basic input status. No geological interpretation.
    """

    data_unit_id: str                        # The asset's own ID (asset_001 style)
    data_source_id: str                      # Source ID (source_001 style), for provenance
    data_modality: str                       # Modality: numeric / text / image
    file_path: str                           # File location on disk
    data_status: str = "registered"          # Basic input status: registered / error

    def to_dict(self) -> dict[str, Any]:  # Convert to a dict
        """Convert to a dict for JSON audit dumps."""
        return {  # Return the full-field dict (the literal spans multiple lines)
            "data_unit_id": self.data_unit_id,  # Write the asset's own ID
            "data_source_id": self.data_source_id,  # Write the source ID
            "data_modality": self.data_modality,  # Write the modality
            "file_path": self.file_path,  # Write the file path
            "data_status": self.data_status,  # Write the input status
        }


@dataclass  # Declare as a dataclass
class DataUnit:  # Standard data unit class
    """The unified output of the whole input layer: standard data units parsed from tables/text/images.

    All parsers uniformly return list[DataUnit] — one InputAsset (e.g. a multi-sheet
    Excel workbook, or text split into blocks by blank lines) can yield multiple
    DataUnits. The parsing stage fills only objective fields; data_entity_key
    ("who is this") and data_semantics ("what does this mean") are left empty for the
    downstream entity recognition and semantic understanding layers to fill — the
    separation of entity and semantics is this layer's core design constraint.
    """

    data_unit_id: str                        # Unit ID (unit_001 style, unique within the batch)
    data_asset_id: str                       # Points back to the InputAsset.data_unit_id that produced this unit
    data_modality: str                       # Modality: numeric / text / image
    data_detail_type: str                    # Detail type: csv_table / excel_worksheet / text_block / image_metadata
    raw_data: dict[str, Any] = field(default_factory=dict)          # The parse result closest to the raw source (facts)
    raw_data_reference: str = ""             # Raw data location (file path), traceable back
    data_entity_key: dict[str, Any] = field(default_factory=dict)   # "Who is this" — filled by the entity recognition layer
    data_semantics: dict[str, Any] = field(default_factory=dict)    # "What does this mean" — filled by the semantic understanding layer
    data_origin: dict[str, Any] = field(default_factory=dict)       # Provenance: file / worksheet / line numbers / encoding, etc.
    data_quality: dict[str, Any] = field(default_factory=dict)      # Quality: completeness / missing values / anomalies, with no scientific meaning
    data_status: str = "success"             # Quick status: success / warning / error

    def to_dict(self) -> dict[str, Any]:  # Convert to a dict
        """Convert to a dict for JSON audit dumps and consumption by downstream layers."""
        return {  # Return the full-field dict (the literal spans multiple lines)
            "data_unit_id": self.data_unit_id,  # Write the unit ID
            "data_asset_id": self.data_asset_id,  # Write the back-reference asset ID
            "data_modality": self.data_modality,  # Write the modality
            "data_detail_type": self.data_detail_type,  # Write the data detail type
            "raw_data": self.raw_data,  # Write the raw parse result
            "raw_data_reference": self.raw_data_reference,  # Write the raw data location
            "data_entity_key": self.data_entity_key,  # Write the entity key field
            "data_semantics": self.data_semantics,  # Write the semantics field
            "data_origin": self.data_origin,  # Write the provenance field
            "data_quality": self.data_quality,  # Write the quality field
            "data_status": self.data_status,  # Write the status field
        }
