# Discrimination-diagram interpretation (`oregenesis.diagram`)

Deterministic reading of geochemical discrimination diagrams from raw sample values. The field boundaries are data (`data/*.json`) and the interpretation is pure geometry, so the same input always gives the same output.

## Usage

Five steps: read the raw values → identify the diagram type, axes, units and log scales → compute the point coordinates → look up the digitised field boundaries → determine the field membership and the distance to the nearest boundary.

```python
from oregenesis.diagram import DiagramInterpreter

reading = DiagramInterpreter().read(
    {"sample_id": "A17", "Sr": 650.18, "Y": 11.8},   # or a SampleData object
    "sr_y_vs_y_adakite",
)
print(reading.to_dict())
```

Output (the first seven keys are the stable contract; the conclusion text is Chinese: "the point lies in the high-Sr/Y (adakitic) field, far from the boundary"):

```json
{
  "diagram": "Sr/Y-Y discrimination diagram",
  "sample_id": "A17",
  "coordinates": {"Y": 11.8, "Sr/Y": 55.1},
  "region": "high_SrY_field",
  "boundary_distance": 0.18,
  "data_quality": "high",
  "visual_conclusion": "点位于高Sr/Y（埃达克质）区，距边界较远"
}
```

Further keys hold extended diagnostics: `status`, `diagram_id`, `region_name_cn`, `region_description`, `region_distances`, `relative_boundary_distance`, `near_boundary`, `boundary_distances_detail`, `data_quality_reasons`, `warnings` and `meta`.

`boundary_distance` is the distance from the point to the nearest field boundary, measured in plot space (decades on log10 axes, data units on linear axes). For sample A17 and the Y = 18 ppm boundary: |log10(11.8) − log10(18)| ≈ 0.18.

## Boundary library

| `diagram_id` | Diagram | Reference | Fidelity | Status |
|---|---|---|---|---|
| `sr_y_vs_y_adakite` | Sr/Y vs Y (adakite) | Defant & Drummond (1990) | 0.9 | verified |
| `tas_lebas1986` | Total alkali vs silica | Le Bas et al. (1986) | 0.6 | approximate |
| `k2o_sio2_peccerillo1976` | K2O vs SiO2 series | Peccerillo & Taylor (1976) | 0.5 | approximate |
| `thyb_nbyb_pearce1983` | Th/Yb vs Nb/Yb | Pearce (1983) | 0.4 | approximate |
| `pearce1984_rb_ynb` | Rb vs Y+Nb granite tectonic setting | Pearce, Harris & Tindle (1984) | 0.35 | approximate |
| `pearce1984_nb_y` | Nb vs Y granite tectonic setting | Pearce, Harris & Tindle (1984) | 0.3 | approximate |

Approximate boundaries (`verification_status: approximate_digitization_required`) were encoded from reproduced figures. Every reading against them is graded `data_quality: low` and carries a warning that the result is for screening only and that the boundaries should be re-calibrated against the original figure before formal use.

To add a diagram, place a new JSON file in `data/` (the format is documented in the module header of `boundaries.py`) and fill in `polygon_vertices` / `polyline_vertices`; the interpreter picks it up automatically.

The image-digitisation pipeline that originally produced the boundary files (axis OCR and data-point detection) is not part of this package; all boundary data are included in `data/`, so interpretation does not depend on it.

## Use by the agent

The registered tool `plot_discrimination_diagram` (parameter `diagram` = one of the IDs above) uses this interpreter to plot the samples of a numeric table on the chosen diagram, render the field basemap and report how many samples fall in each field, together with the boundary fidelity. The twelve `diagram_*_axis` tools compute the axis coordinates of common diagrams for the planner.
