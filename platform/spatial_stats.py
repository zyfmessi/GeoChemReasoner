# -*- coding: utf-8 -*-
"""toolbox/spatial_stats.py — extraction of sample element contents and coordinates (self-contained, numpy only).

The project-wide canonical implementation of collect_element_map / collect_coordinates /
element_series_for: the spatial-applicability check of t13 prospectivity mapping and the
platform.prospectivity pipeline share this convention, avoiding copy drift across sites.
"""
from __future__ import annotations  # enable PEP 563 postponed annotation evaluation; type hints not evaluated at runtime

from typing import TYPE_CHECKING  # import the TYPE_CHECKING constant (True only during static type checking)

import numpy as np  # import the numpy numeric library aliased as np

if TYPE_CHECKING:  # type-checking-only branch: inner imports not executed at runtime
    from oregenesis.platform.models import SamplePoint as SampleData  # import the sample point model solely for type annotations, aliased as SampleData


def collect_element_map(samples: list[SampleData]) -> dict[str, list[float | None]]:  # collect the element content mapping: aligned by sample order, missing filled with None
    """Collect contents of each element across all samples (aligned by sample order, missing as None)."""
    elements: dict[str, list[float | None]] = {}  # initialize the result dict: element name → content list
    for s in samples:  # iterate over samples
        # merge the three source dicts: major elements first, then trace, then REE
        merged = {}  # initialize the merged dict for the current sample (first occurrence of a key wins)
        for d in (s.major_elements, s.trace_elements, s.ree_data):  # iterate the three source dicts in major → trace → REE priority
            for k, v in d.items():  # iterate the dict's key-value pairs (element name, content)
                if k not in merged:  # write only when the element is not yet recorded, avoiding overrides by lower-priority dicts
                    merged[k] = v  # record this element's content value
        # append the current sample's value per element (skipped when this sample lacks the element; None padding applied uniformly later)
        for elem, val in merged.items():  # iterate the current sample's merged element values
            elements.setdefault(elem, []).append(val)  # group content values by element name via append; create the list if absent
    # pad missing values (some sample lacking some element)
    n = len(samples)  # take the total sample count for length alignment
    for elem, vals in elements.items():  # iterate each element and its content sequence in the result dict
        if len(vals) < n:  # a sequence shorter than the sample count means some samples lack this element
            elements[elem] = vals + [None] * (n - len(vals))  # pad None at the tail up to the sample count, keeping sample-order alignment
    return elements  # return the element name → content list mapping


def collect_coordinates(  # extract sample coordinates and flatten them to planar
    samples: list[SampleData],  # parameter: list of sample points
) -> tuple[np.ndarray | None, np.ndarray | None, bool, str]:  # return type: x array, y array, whether lon/lat, and coordinate system description
    """Extract sample coordinates and flatten them to planar coordinates.

    Returns:
        (x, y, is_geo, coord_system); returns (None, None, ...) when too few samples have valid coordinates
    """
    xs: list[float | None] = []  # initialize the x coordinate list (missing as None)
    ys: list[float | None] = []  # initialize the y coordinate list (missing as None)
    # is_geo: any sample flagged lon/lat → the whole batch is treated as lon/lat (corresponds to SampleData.coord_is_geo)
    is_geo = any(bool(getattr(s, "coord_is_geo", False)) for s in samples)  # if any sample is flagged lon/lat, treat the whole batch as lon/lat
    for s in samples:  # iterate over samples
        xs.append(getattr(s, "coord_x", None))  # read the sample x coordinate (None if the attribute is absent) and append
        ys.append(getattr(s, "coord_y", None))  # read the sample y coordinate (None if the attribute is absent) and append

    # full-length arrays: samples with both valid X and Y keep their values, the rest are NaN (aligned with the sample list)
    n = len(samples)  # take the total sample count, aligned with the full-length arrays
    x_full = np.full(n, np.nan, dtype=float)  # create a length-n all-NaN float array as full-length x coordinates
    y_full = np.full(n, np.nan, dtype=float)  # create a length-n all-NaN float array as full-length y coordinates
    for i, (vx, vy) in enumerate(zip(xs, ys)):  # iterate the raw x/y values pairwise by index
        if vx is None or vy is None:  # skip if either coordinate is missing (keep NaN)
            continue  # skip samples lacking coordinates
        fx, fy = float(vx), float(vy)  # convert x/y to floats
        if np.isfinite(fx) and np.isfinite(fy):  # write only when both coordinates are finite
            x_full[i], y_full[i] = fx, fy  # write the valid coordinates back to the corresponding positions of the full-length arrays
    # fewer than 4 valid coordinates → cannot support any spatial analysis; return None and let the caller warn
    if int(np.isfinite(x_full).sum()) < 4:  # count valid x coordinates; fewer than 4 cannot support spatial analysis
        return None, None, is_geo, ""  # return the empty result for the caller to warn (coordinate system description left empty)
    xa, ya = x_full, y_full  # assign the full-length arrays to the flattening input aliases

    if is_geo:  # enter the flattening branch for lon/lat input
        # equirectangular flattening: survey-area center as the origin, unit ≈ km (latitude uses the cosine of the mean latitude)
        # 1° longitude ≈ 111.32 km × cos(latitude); 1° latitude ≈ 110.57 km
        good = np.isfinite(xa)  # build a boolean mask of valid coordinates
        lon0 = float(np.mean(xa[good]))  # take the mean of valid longitudes as the flattening origin longitude
        lat0 = float(np.mean(ya[good]))  # take the mean of valid latitudes as the flattening origin latitude
        cos_lat = abs(np.cos(np.deg2rad(lat0))) or 1e-6  # absolute cosine of the origin latitude (0 falls back to 1e-6 against division by zero)
        x_full = np.where(good, (xa - lon0) * 111.32 * cos_lat, np.nan)  # longitudinal flattening: longitude difference × 111.32 × cos(latitude) converted to km; NaN kept where invalid
        y_full = np.where(good, (ya - lat0) * 110.57, np.nan)  # latitudinal flattening: latitude difference × 110.57 converted to km; NaN kept where invalid
        return x_full, y_full, True, "经纬度→平面化(km)"  # return the flattened coordinates flagged as lon/lat source
    # already projected planar coordinates (e.g. Gauss-Krüger XY) → use directly, no conversion
    return x_full, y_full, False, "平面坐标(X/Y)"  # already planar: return directly with the source flag


def element_series_for(element: str, samples: list[SampleData]) -> list[float | None]:  # extract the content sequence of one element (in sample order)
    """Content sequence of one element (in sample order): search the major/trace/REE dicts in
    order and stop at the first hit, avoiding same-name keys being overridden by
    lower-priority dicts; more precise than the tail padding of collect_element_map."""
    vals: list[float | None] = []  # initialize the content sequence (missing as None)
    for s in samples:  # iterate over samples
        v = None  # default: this sample has no value for the element
        for d in (s.major_elements, s.trace_elements, s.ree_data):  # iterate the source dicts in major → trace → REE priority
            if element in d:  # check whether the dict contains this element
                v = d[element]  # on hit, take the element content
                break  # stop at the first hit to avoid same-name key overrides by lower-priority dicts
        vals.append(v)  # append the current sample's value (None if absent)
    return vals  # return the complete content sequence
