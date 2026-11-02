"""Diagram interpretation subpackage entry: Mode A deterministic interpretation
(interpreter + boundary library).

The boundary library consists of machine-readable data/*.json entries
(diagram_id -> region polygons + explicit boundary polylines + reference
provenance + fidelity); interpretation is purely geometric computation, and
boundaries are data. This __init__ only re-exports, keeping the package
lightweight. Historical note: Mode B image digitization (digitizer, a one-off
data preparation tool for producing data/*.json) was removed along with the
paper's production workflow; all boundary data has been baked into the library.
"""

__version__ = "0.2.0"  # Subpackage version: evolves independently (absorbed from the platform 0.2.0 snapshot)

# Contract layer: types and constants (no third-party dependencies, always importable)
from oregenesis.diagram.schema import (  # Import types and constants from the contract module (parenthesized multi-line import)
    STABLE_READING_KEYS, AxisSpec, BoundaryReading, BoundarySpec,  # Import the stable output key tuple and the axis/boundary/reading dataclasses
    DataPointEstimate, DiagramSpec, RegionSpec, ScaleKind, format_interval,  # Import data-point estimate/diagram spec/region spec/scale enum and the interval formatting function
)
# Boundary library and Mode A engine: pure standard library, always importable
from oregenesis.diagram.boundaries import DEFAULT_LIBRARY, DiagramLibrary  # Import the process-wide default boundary library singleton and the library class
from oregenesis.diagram.interpreter import DiagramInterpreter  # Import the Mode A deterministic interpretation engine class

__all__ = [  # Declare the subpackage's public export list (only these names are exported by `from ... import *`)
    "__version__",  # Export the version string
    # Contract
    "STABLE_READING_KEYS", "AxisSpec", "BoundaryReading", "BoundarySpec",  # Export the stable key tuple and axis/boundary/reading contract classes
    "DataPointEstimate", "DiagramSpec", "RegionSpec",  # Export data-point estimate/diagram spec/region spec classes
    "ScaleKind", "format_interval",  # Export the scale enum and interval formatting function
    # Boundary library / Mode A interpretation engine
    "DEFAULT_LIBRARY", "DiagramLibrary", "DiagramInterpreter",  # Export the default library singleton/library class/interpretation engine class
]
