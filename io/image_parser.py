"""Image parser: JPG/PNG/TIFF/BMP -> metadata DataUnit.

The current stage only performs basic reading and metadata extraction
(width/height/color mode/file format/EXIF); it does not judge whether the image shows
quartz, muscovite, lithium pyroxene rock, or PPL/XPL — that is left to a downstream VLM
or a specialized image model. When an image cannot be opened (corrupted / not an image
file), an error-level unit is produced instead of silently dropping it.
"""

from __future__ import annotations  # Enable PEP 563 deferred evaluation of type annotations

from pathlib import Path  # Import the Path type to get file extensions

from oregenesis.io.data_models import DataUnit, IdAllocator, InputAsset  # Import the data models and the ID allocator

# Max number of EXIF keys kept (prevents the occasional plate carrying huge metadata from bloating raw_data)
_EXIF_MAX_KEYS = 20  # Constant: upper bound on the number of EXIF keys kept


def parse_image_asset(asset: InputAsset, alloc: IdAllocator | None = None) -> list[DataUnit]:  # Image asset parsing entry point
    """Image asset parsing entry point: open with PIL -> extract size/mode/format/EXIF -> one DataUnit."""
    alloc = alloc or IdAllocator()  # Create a new allocator when none is passed
    try:  # Try a lazy import of Pillow
        from PIL import Image  # Import PIL's Image module on demand
    except ImportError as exc:  # Turn a missing library into an actionable hint
        raise ImportError("解析图像需要 Pillow：pip install Pillow") from exc  # Raise an error carrying the install command

    ext = Path(asset.file_path).suffix.lower()  # Get the lowercase file extension
    origin = {  # Build the provenance dict (spans multiple lines)
        "source_file": asset.data_source_id,  # Record the source ID
        "source_path": str(asset.file_path),  # Record the file path
        "file_extension": ext,  # Record the extension
    }
    try:  # Try to open and read the image
        with Image.open(asset.file_path) as image:  # Open the image with a context manager to ensure release
            exif = {str(k): str(v) for k, v in list(image.getexif().items())[:_EXIF_MAX_KEYS]} \
                if image.getexif() else {}  # Empty dict when there is no EXIF (conditional expression continuation)
            raw = {  # Build the raw metadata dict (spans multiple lines)
                "width": image.width,          # Width in pixels
                "height": image.height,        # Height in pixels
                "mode": image.mode,            # Color mode (RGB/L/RGBA, etc.)
                "format": image.format,        # Actual file format (JPEG/PNG/TIFF...)
            }
            if exif:  # When EXIF metadata exists
                raw["exif"] = exif  # Merge in the capped set of EXIF key-value pairs
        status, quality = "success", {"corruption": False}  # Success path: status and a no-corruption quality marker
    except Exception as e:  # Unopenable images are not silently dropped: the error unit keeps provenance and the cause
        return [DataUnit(  # Return an error-unit list (arguments span multiple lines)
            data_unit_id=alloc.next("unit"),  # Allocate the unit ID
            data_asset_id=asset.data_unit_id,  # Back-reference the owning asset
            data_modality="image",  # Modality marked as image
            data_detail_type="image_metadata",  # Detail type: image metadata
            raw_data={},  # No raw data to extract
            raw_data_reference=str(asset.file_path),  # Keep the raw file path for tracing back
            data_origin=origin,  # Write the provenance info
            data_quality={"corruption": True, "error": str(e)},  # Quality: flag corruption and record the exception message
            data_status="error",  # Status: error
        )]
    return [DataUnit(  # Return the success-unit list (arguments span multiple lines)
        data_unit_id=alloc.next("unit"),  # Allocate the unit ID
        data_asset_id=asset.data_unit_id,  # Back-reference the owning asset
        data_modality="image",  # Modality marked as image
        data_detail_type="image_metadata",  # Detail type: image metadata
        raw_data=raw,  # Write size/mode/format/EXIF
        raw_data_reference=str(asset.file_path),  # Record the raw file path
        data_origin=origin,  # Write the provenance info
        data_quality=quality,  # Write the quality markers
        data_status=status,  # Write the status
    )]
