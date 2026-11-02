"""Mineral Prospectivity Mapping toolkit (2D prospectivity maps).

Two-stage pipeline (the stage boundary is the core design constraint of this package):

    Stage 1  geochemical anomaly identification: point data → spatialization (IDW/kriging)
             → anomaly extraction (threshold/C-A fractal/PCA/Mahalanobis distance/
             unsupervised detection/external distance field) → evidence layers
    Stage 2  mineral prospectivity prediction: evidence layers → multi-source evidence
             fusion (weights of evidence/information value/LR/RF/AHP knowledge overlay/
             unsupervised fusion) → mineralization probability map → A/B/C prospect zoning

Stage 1 only answers "where is anomalous"; "where mineralization may occur" must go
through stage-two evidence fusion — an anomaly is not mineralization, and this is the
basic discipline of geochemical exploration data interpretation. Coordinate and content
extraction conventions are shared with the anomaly subpackage, and anomaly delineation
conclusions can serve as an independent cross-check for prospectivity.

Typical usage::

    from oregenesis.platform.prospectivity import run_prospectivity_mapping
    report = run_prospectivity_mapping(samples, elements=["Cu", "Mo", "Au"])
    print(report.summary_text)  # report is pure data and batch-processable; plotting invoked by the tool layer
"""
# Main orchestration: samples + deposits + external evidence → two-stage pipeline → ProspectivityReport
from oregenesis.platform.prospectivity.pipeline import run_prospectivity_mapping  # import the two-stage pipeline main orchestration function
# Data models: report/evidence layers/fusion model/zoning result, unified data carriers of the whole package
from oregenesis.platform.prospectivity.models import (  # multi-line import from the data model module (start line)
    EvidenceLayer, FusionModel, GridDef, ProspectivityReport, ZoneLevel, ZoneResult)  # import the six dataclasses: evidence layer/fusion model/grid definition/overall report/level/zoning result
# Stage-two granular APIs: fusion method dispatch / prospect zoning / model evaluation
from oregenesis.platform.prospectivity.fusion import fuse  # import fuse, the stage-two evidence fusion dispatch entry
from oregenesis.platform.prospectivity.classify import classify_zones, evaluate_model  # import the two functions: prospect zoning and model evaluation

# Package-level __all__: exposed under `from platform.prospectivity import *`
__all__ = [  # define the package-level export list (exposed on star import)
    "run_prospectivity_mapping", "ProspectivityReport", "EvidenceLayer", "FusionModel",  # export list continued: main orchestration function and report/evidence layer/fusion model
    "ZoneResult", "ZoneLevel", "GridDef", "fuse", "classify_zones", "evaluate_model",  # export list continued: zoning result/level/grid/fusion/zoning/evaluation
]
