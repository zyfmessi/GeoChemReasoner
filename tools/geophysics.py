"""Geological process calculator tool family: geothermal gradient / pressure / depth / fluid inclusion / isotope calculators.

All are formula-driven deterministic calculators: parameters arrive via ToolCall.parameters, and missing parameters
are converted by the executor into error Observations (the reasoning layer can then ask the user for them). Formulas and
constants are stated explicitly in descriptions/comments, ensuring reproducibility.
"""

from __future__ import annotations  # enable deferred evaluation of type annotations (PEP 563)

import math  # import math (exp/log for the isotope formulas)
from typing import Any, Callable  # import Any and Callable type annotations

from oregenesis.tools.registry import ToolCall, ToolSpec  # import the ToolCall and ToolSpec dataclasses

# Isotope system constants (common literature values)
_LAMBDA_RB = 1.42e-11       # 87Rb decay constant (yr⁻¹)
_LAMBDA_SM = 6.54e-12       # 147Sm decay constant (yr⁻¹)
_LAMBDA_LU = 1.867e-11      # 176Lu decay constant (yr⁻¹)
_CHUR_ND = 0.512638         # CHUR (143Nd/144Nd) present-day value
_CHUR_SMND = 0.1967         # CHUR (147Sm/144Nd)
_DM_ND = 0.513151           # depleted mantle (143Nd/144Nd) (Goldstein 1984)
_DM_SMND = 0.2137           # depleted mantle (147Sm/144Nd)
_CHUR_HF = 0.282785         # CHUR (176Hf/177Hf) present-day value
_CHUR_LUHF = 0.0336         # CHUR (176Lu/177Hf)


def _param_tool(name: str, family: str, description: str, required_params: tuple[str, ...],  # parameter-driven calculator factory: name/family/description/required params
                compute: Callable, unit: str) -> ToolSpec:  # compute: the computation function; unit: the result unit
    """Parameter-driven calculator factory: missing parameters -> error Observation (listing which parameters are missing)."""

    def _run(_state: Any, call: ToolCall) -> dict[str, Any]:  # inner run function (calculators are state-independent)
        params = call.parameters or {}  # take the call parameters dict (fall back to an empty dict on None)
        missing = [p for p in required_params if params.get(p) is None]  # list required parameters whose value is None
        if missing:  # when parameters are missing
            raise ValueError(f"缺少计算参数: {', '.join(missing)}（请在 parameters 中提供）")  # raise ValueError listing the missing parameters (the executor converts it to an error Observation)
        # smart numericization: parameters convertible to float are converted; string parameters (e.g. oxide="SiO2") are kept as-is
        coerced = {}  # initialize the type-coerced parameter dict
        for p in required_params:  # coerce each required parameter in turn
            v = params[p]  # take the parameter's raw value
            try:  # try converting to float
                coerced[p] = float(v)  # store as float when convertible
            except (TypeError, ValueError):  # catch non-convertible values
                coerced[p] = v  # string parameters (e.g. oxide="SiO2") kept as-is
        value = compute(**coerced)  # call the computation function with keyword unpacking to get the result
        return {"observation": f"{description}: {value:g} {unit}",  # return observation text: description + value (%g format) + unit
                "result_value": round(value, 4), "unit": unit,  # attach the result value (4 decimals) and unit
                "parameters": {p: float(params[p]) for p in required_params  # attach an echo of numeric parameters (filtered per parameter)
                               if str(params[p]).replace('.', '', 1).replace('-', '', 1).isdigit()},  # include only when the parameter string looks numeric (one dot and one minus allowed)
                "supports": [f"calculator_{name}"]}  # attach the evidence tag calculator_<tool name>

    return ToolSpec(  # assemble and return the calculator tool spec
        name=f"calculate_{name}", tool_family=family,  # tool name = calculate_<name>, family as passed in
        description=f"{description}；参数 {', '.join(required_params)}",  # append the required-parameter list to the description (for the planner to fill in)
        required_data=(), capabilities=("process_calculator",),  # calculators need no table data; capability tag is process calculator
        applicable=lambda s: (True, ""), run=_run)  # always applicable; bind the run function


def _oxide_of(element: str) -> str:  # element name -> main oxide name mapping function
    """Element name -> main oxide name (Fe defaults to the FeO valence state); raises ValueError for unknown elements."""
    oxide = _OXIDE_OF.get(element.strip())  # look up the element->oxide mapping table after stripping spaces
    if oxide is None:  # element not found
        raise ValueError(f"不支持的元素: {element}（支持 {', '.join(sorted(_OXIDE_OF))}）")  # raise ValueError and list the supported elements
    return oxide  # return the corresponding oxide name


def _factor_of(oxide: str) -> float:  # oxide name -> conversion factor function
    """Oxide name -> conversion factor; raises ValueError for unknown oxides."""
    factor = _OXIDE_FACTOR.get(oxide.strip())  # look up the oxide->factor mapping table after stripping spaces
    if factor is None:  # oxide not found
        raise ValueError(f"不支持的氧化物: {oxide}（支持 {', '.join(sorted(_OXIDE_FACTOR))}）")  # raise ValueError and list the supported oxides
    return factor  # return the stoichiometric conversion factor


def build_geophysics_tools() -> list[ToolSpec]:  # geological process calculator tool family constructor
    """Build the geological process calculator tool family (14 calculators)."""
    return [  # start the list of 14 calculator tools
        # ── Geotherm and pressure ──
        _param_tool(  # build #1: geothermal gradient calculator
            "geothermal_gradient", "geophysics",  # tool name and family geophysics
            "地温梯度计算（(T-地表温度)/深度）", ("temp_c", "depth_km"),  # description and required params (temperature °C, depth km)
            lambda temp_c, depth_km=1.0: (temp_c - 0.0) / depth_km, "°C/km"),  # compute: temperature / depth (depth defaults to 1 km), unit °C/km
        _param_tool(  # build #2: lithostatic pressure calculator
            "lithostatic_pressure", "geophysics",  # tool name and family geophysics
            "岩石静压力 P=ρgh（默认地壳密度 2700 kg/m³）", ("depth_km",),  # description and required params (depth km)
            lambda depth_km, density_kg_m3=2700.0: density_kg_m3 * 9.81 * depth_km * 1000 / 1e6,  # compute P=rho*g*h: density*9.81*depth*1000/1e6 gives MPa
            "MPa"),  # result unit MPa
        _param_tool(  # build #3: hydrostatic pressure calculator
            "hydrostatic_pressure", "geophysics",  # tool name and family geophysics
            "静水压力 P=ρgh（默认水密度 1000 kg/m³）", ("depth_km",),  # description and required params (depth km)
            lambda depth_km, density_kg_m3=1000.0: density_kg_m3 * 9.81 * depth_km * 1000 / 1e6,  # compute water-column P=rho*g*h (default water density 1000), converted to MPa
            "MPa"),  # result unit MPa
        _param_tool(  # build #4: depth-from-lithostatic-pressure calculator
            "depth_from_lithostatic", "geophysics",  # tool name and family geophysics
            "由岩石静压力反演深度（默认地壳密度 2700 kg/m³）", ("pressure_mpa",),  # description and required params (pressure MPa)
            lambda pressure_mpa, density_kg_m3=2700.0: pressure_mpa * 1e6 / (density_kg_m3 * 9.81 * 1000),  # compute: pressure*1e6/(density*9.81*1000) inverts to depth
            "km"),  # result unit km
        _param_tool(  # build #5: depth-from-hydrostatic-pressure calculator
            "depth_from_hydrostatic", "geophysics",  # tool name and family geophysics
            "由静水压力反演深度（默认水密度 1000 kg/m³）", ("pressure_mpa",),  # description and required params (pressure MPa)
            lambda pressure_mpa, density_kg_m3=1000.0: pressure_mpa * 1e6 / (density_kg_m3 * 9.81 * 1000),  # compute: convert pressure back to depth at water density 1000
            "km"),  # result unit km
        # ── Fluid inclusions ──
        _param_tool(  # build #6: fluid-inclusion salinity calculator
            "salinity_from_ice_tm", "fluid",  # tool name and family fluid
            "由冰点温度计算 NaCl 当量盐度（Hall et al. 1988 多项式）", ("tm_ice_c",),  # description and required params (ice-melting temperature °C)
            lambda tm_ice_c: (1.76958 * abs(tm_ice_c) - 0.042384 * abs(tm_ice_c) ** 2  # compute: the first two terms of the Hall 1988 cubic polynomial (absolute value of the ice-melting point)
                              + 5.2778e-4 * abs(tm_ice_c) ** 3), "wt% NaCl eq"),  # cont.: the cubic term; the result is salinity in wt% NaCl equivalent
        # ── Isotope calculators ──
        _param_tool(  # build #7: initial (87Sr/86Sr)i calculator
            "initial_87sr_86sr", "isotope",  # tool name and family isotope
            "初始 (87Sr/86Sr)i =实测值−(Rb/Sr)×2.8954×(e^{1.42e-11·t}−1)",   # description first part: initial Sr formula (implicit concatenation)
            ("sr87_86", "rb_sr", "age_ma"),  # required params: measured 87Sr/86Sr, Rb/Sr, age Ma
            lambda sr87_86, rb_sr, age_ma: sr87_86 - rb_sr * 2.8954 * (  # compute: measured value minus Rb/Sr*2.8954*decay correction term
                math.exp(_LAMBDA_RB * age_ma * 1e6) - 1), ""),  # cont.: e^(lambda_Rb*t)-1 (age Ma*1e6 to years); unitless
        _param_tool(  # build #8: epsilonNd(t) calculator
            "epsilon_nd_t", "isotope",  # tool name and family isotope
            "εNd(t)=((143Nd/144Nd)t/CHUR(t)−1)×10⁴（CHUR 0.512638/0.1967）",  # description text: epsilonNd(t) formula and CHUR constants
            ("nd143_144", "sm143_144"),  # required params: 143Nd/144Nd and 147Sm/144Nd
            lambda nd143_144, sm143_144, age_ma=0.0: (  # compute lambda: age defaults to 0 Ma (present day)
                (nd143_144 - sm143_144 * (math.exp(_LAMBDA_SM * age_ma * 1e6) - 1))  # numerator: back-calculate the ratio to time t via Sm/Nd decay
                / (_CHUR_ND - _CHUR_SMND * (math.exp(_LAMBDA_SM * age_ma * 1e6) - 1)) - 1) * 1e4,  # cont.: divide by CHUR(t), subtract 1, multiply by 1e4 to get epsilonNd(t)
            ""),  # epsilon values are dimensionless, no unit
        _param_tool(  # build #9: epsilonHf(t) calculator
            "epsilon_hf_t", "isotope",  # tool name and family isotope
            "εHf(t)=((176Hf/177Hf)t/CHUR(t)−1)×10⁴（CHUR 0.282785/0.0336）",  # description text: epsilonHf(t) formula and CHUR constants
            ("hf176_177", "lu177_hf"),  # required params: 176Hf/177Hf and 176Lu/177Hf
            lambda hf176_177, lu177_hf, age_ma=0.0: (  # compute lambda: age defaults to 0 Ma (present day)
                (hf176_177 - lu177_hf * (math.exp(_LAMBDA_LU * age_ma * 1e6) - 1))  # numerator: back-calculate the ratio to time t via Lu/Hf decay
                / (_CHUR_HF - _CHUR_LUHF * (math.exp(_LAMBDA_LU * age_ma * 1e6) - 1)) - 1) * 1e4,  # cont.: divide by CHUR(t), subtract 1, multiply by 1e4 to get epsilonHf(t)
            ""),  # epsilon values are dimensionless, no unit
        _param_tool(  # build #10: Nd depleted-mantle model age calculator
            "nd_model_age", "isotope",  # tool name and family isotope
            "Nd 亏损地幔模式年龄 T_DM（Goldstein 1984：0.513151/0.2137）",  # description text: T_DM formula and depleted-mantle constants
            ("nd143_144", "sm143_144"),  # required params: 143Nd/144Nd and 147Sm/144Nd
            lambda nd143_144, sm143_144: math.log(  # compute: take the natural log (the fraction is on the continuation line)
                1 + (nd143_144 - _DM_ND) / (sm143_144 - _DM_SMND)) / _LAMBDA_SM / 1e9, "Ga"),  # cont.: ln(1+(sample-DM)/(Sm/Nd-DM))/lambda_Sm/1e9 gives Ga
        _param_tool(  # build #11: Sm/Nd ratio converter
            "smnd_to_147_144", "isotope",  # tool name and family isotope
            "Sm/Nd 质量比→(147Sm/144Nd) 原子比换算（×0.6046）", ("sm_nd",),  # description and required params (Sm/Nd mass ratio)
            lambda sm_nd: sm_nd * 0.6046, ""),  # compute: mass ratio*0.6046 to the (147Sm/144Nd) atomic ratio
        _param_tool(  # build #12: Rb/Sr ratio converter
            "rbsr_to_87rb_86sr", "isotope",  # tool name and family isotope
            "Rb/Sr 质量比→(87Rb/86Sr) 原子比换算（×2.8954）", ("rb_sr",),  # description and required params (Rb/Sr mass ratio)
            lambda rb_sr: rb_sr * 2.8954, ""),  # compute: mass ratio*2.8954 to the (87Rb/86Sr) atomic ratio
        # ── Unit conversion ──
        _param_tool(  # build #13: oxide-to-element converter
            "oxide_to_element", "unit_convert",  # tool name and family unit_convert
            "氧化物质量分数→元素质量分数（化学计量换算，如 SiO2→Si ×0.4674）",  # description text: stoichiometric conversion direction and example
            ("oxide", "value"),  # required params: oxide name and mass-fraction value
            lambda oxide, value: value * _factor_of(str(oxide)), "wt%"),  # compute: content*oxide-to-element factor; unit wt%
        _param_tool(  # build #14: element-to-oxide converter
            "element_to_oxide", "unit_convert",  # tool name and family unit_convert
            "元素质量分数→氧化物质量分数（化学计量反算，如 Si→SiO2 ×2.1394）",  # description text: stoichiometric back-calculation direction and example
            ("element", "value"),  # required params: element name and mass-fraction value
            lambda element, value: value / _factor_of(_oxide_of(str(element))), "wt%"),  # compute: map to the oxide first, then divide by the factor; unit wt%
    ]


# Oxide -> element mass-fraction conversion factors (stoichiometric: cation mass as a fraction of oxide mass)
_OXIDE_FACTOR: dict[str, float] = {  # define the oxide->element conversion factor dict
    "SiO2": 0.4674, "TiO2": 0.5995, "Al2O3": 0.5293, "Fe2O3": 0.6994,  # cation mass fractions of SiO2/TiO2/Al2O3/Fe2O3
    "FeO": 0.7773, "MnO": 0.7745, "MgO": 0.6032, "CaO": 0.7147,  # cation mass fractions of FeO/MnO/MgO/CaO
    "Na2O": 0.7419, "K2O": 0.8301, "P2O5": 0.4364,  # cation mass fractions of Na2O/K2O/P2O5
}
# Element -> main oxide (for back-calculation; Fe defaults to the FeO valence state)
_OXIDE_OF: dict[str, str] = {"Si": "SiO2", "Ti": "TiO2", "Al": "Al2O3", "Fe": "FeO",  # define the element->main-oxide mapping: Si/Ti/Al/Fe (Fe as FeO)
                             "Mn": "MnO", "Mg": "MgO", "Ca": "CaO", "Na": "Na2O",  # cont.: mappings of Mn/Mg/Ca/Na
                             "K": "K2O", "P": "P2O5"}  # cont.: mappings of K/P (10 entries in total), closing the dict
