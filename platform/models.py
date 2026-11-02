"""Lightweight sample point model of the platform numeric layer: input carrier for spatial algorithms (interpolation/anomaly/fusion/prediction).

Fields are decoupled from the reasoning engine — the tool layer assembles tabular
DataUnit row data into SamplePoint lists; platform algorithms depend only on the
fields declared here and never depend back on any reasoning structure.
"""

from __future__ import annotations  # enable PEP 563 postponed annotation evaluation; type hints not evaluated at runtime

from dataclasses import dataclass, field  # import the dataclass decorator and the field factory from dataclasses
from typing import Any  # import the Any type for free-form metadata field annotations


@dataclass  # dataclass decorator: auto-generates __init__/__repr__ etc.
class SamplePoint:  # sample point dataclass: minimal input unit for spatial numeric algorithms
    """A sample point carrying coordinates and element contents (minimal input unit for spatial numeric algorithms)."""

    sample_id: str                                   # sample ID (provenance primary key)
    coord_x: float | None = None                     # easting or longitude
    coord_y: float | None = None                     # northing or latitude
    coord_is_geo: bool = False                       # True = lon/lat (flattened to planar inside algorithms)
    major_elements: dict[str, float] = field(default_factory=dict)   # major oxides wt%
    trace_elements: dict[str, float] = field(default_factory=dict)   # trace/ore-forming elements ppm
    ree_data: dict[str, float] = field(default_factory=dict)         # rare earth elements
    metadata: dict[str, Any] = field(default_factory=dict)           # free-form metadata (deposit flags, etc.)
