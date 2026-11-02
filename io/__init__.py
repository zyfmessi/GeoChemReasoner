"""Data Input & Parsing Layer (data input and parsing, layer 1 of the overall architecture).

Raw file -> InputAsset -> modality parsing (numeric/text/image) -> list[DataUnit].
Responsible only for reading, structuring, quality checks, and provenance recording;
not responsible for geological interpretation.

Public entry points:
- make_asset / parse_asset / parse_assets : InputAsset construction and unified parsing
- InputAsset / DataUnit                   : input asset and standard data unit models
- detect_modality                         : file extension -> modality detection
"""

from oregenesis.io.data_models import DataUnit, IdAllocator, InputAsset  # Export the data models: standard unit / ID allocator / input asset
from oregenesis.io.parser import detect_modality, make_asset, parse_asset, parse_assets  # Export the unified parsing entry functions

__all__ = [  # Declare the package's public interface (the list spans multiple lines)
    "DataUnit", "IdAllocator", "InputAsset",  # Expose the three data model classes
    "detect_modality", "make_asset", "parse_asset", "parse_assets",  # Expose the four parsing entry functions
]
