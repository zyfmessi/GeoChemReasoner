# -*- coding: utf-8 -*-
"""Task-aware Routing: inject task-routing keywords into key tools.

Keyword sources: the tool-family factories already generate some automatic keywords (element pairs/index names);
this table supplements Chinese task phrases - mapping common words in user questions to tools,
so that both the rule fallback and the LLM menu ranking can route by task relevance (blueprint §48).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type annotations only, avoiding a circular import with the package __init__
    from oregenesis.tools.registry import ToolRegistry

# tool name -> task-routing keywords (appended after the spec's existing keywords, deduplicated order-preserving)
_TASK_KEYWORDS: dict[str, tuple[str, ...]] = {
    # ── Prospectivity family (spatial evidence) ──
    "geochemical_prospectivity_mapping": ("成矿预测", "远景", "异常图", "找矿靶区", "空间", "成图",
                                          "靶区", "圈定", "潜力", "勘查"),  # supplementary target/delineation/potential/exploration category words
    # ── Ratio calculators (element pairs built in; supplement Chinese meanings) ──
    "calculate_nb_ta_ratio": ("Nb/Ta", "铌钽", "高分异"),
    "calculate_zr_hf_ratio": ("Zr/Hf", "锆铪", "高分异"),
    "calculate_k_rb_ratio": ("K/Rb", "钾铷比", "流体交代"),
    "calculate_rb_sr_ratio": ("Rb/Sr", "铷锶", "演化"),
    "calculate_sr_ba_ratio": ("Sr/Ba", "锶钡", "流体演化"),
    "calculate_th_nb_ratio": ("Th/Nb", "钍铌", "俯冲组分"),
    "calculate_sr_y_ratio": ("Sr/Y", "埃达克", "深部压力"),
    "calculate_ce_pb_ratio": ("Ce/Pb", "铈铅", "壳源混染"),
    "calculate_th_u_ratio": ("Th/U", "钍铀", "风化"),
    "calculate_cu_mo_ratio": ("Cu/Mo", "铜钼", "斑岩"),
    "calculate_ti_v_ratio": ("Ti/V", "钛钒", "构造环境"),
    "calculate_fe_mn_ratio": ("Fe/Mn", "铁锰", "热液"),
    "calculate_ni_co_ratio": ("Ni/Co", "镍钴", "幔源"),
    "calculate_la_yb_ratio": ("La/Yb", "轻重稀土", "分馏"),
    # ── REE indices ──
    "calculate_eu_anomaly": ("铕异常", "Eu", "δEu", "斜长石", "稀土"),
    "calculate_ce_anomaly": ("铈异常", "Ce", "氧化还原"),
    "calculate_la_yb_n": ("La/Yb", "轻重稀土", "稀土分馏", "稀土"),
    "calculate_gd_yb_n": ("Gd/Yb", "重稀土分馏", "石榴子石"),
    "calculate_sum_ree": ("稀土总量", "ΣREE", "稀土"),
    "calculate_lree_hree": ("轻重稀土比", "LREE/HREE", "稀土"),
    # ── Rock classification and indices ──
    "tas_classification": ("TAS", "岩石分类", "岩性", "流纹岩", "安山岩", "玄武岩", "酸性岩", "中性岩", "基性岩", "超基性"),
    "calculate_total_alkali": ("全碱", "Na2O", "K2O"),
    "acnk_shand_classification": ("铝饱和", "过铝质", "A/CNK", "Shand"),
    "calculate_mg_number": ("Mg#", "镁编号", "分异"),
    "calculate_fe_number": ("Fe#", "铁指数", "分异"),
    "calculate_solidification_index": ("固结指数", "分异"),
    "calculate_mafic_index": ("镁铁指数", "铁镁"),
    "calculate_cia": ("风化指数", "CIA", "化学蚀变"),
    "calculate_pia": ("斜长石蚀变", "PIA"),
    "calculate_ciw": ("化学风化", "CIW"),
    "calculate_ccpi": ("蚀变指数", "CCPI", "绿泥石", "绢云母"),
    # ── Geological process calculators ──
    "calculate_geothermal_gradient": ("地温梯度", "温度", "地热"),
    "calculate_lithostatic_pressure": ("岩石静压", "压力", "深度换算"),
    "calculate_hydrostatic_pressure": ("静水压力", "流体压力"),
    "calculate_depth_from_lithostatic": ("压力反演深度", "深度"),
    "calculate_depth_from_hydrostatic": ("静水压力反演", "深度"),
    "calculate_salinity_from_ice_tm": ("盐度", "冰点", "包裹体", "流体"),
    "calculate_initial_87sr_86sr": ("初始锶同位素", "87Sr/86Sr", "物源"),
    "calculate_epsilon_nd_t": ("εNd", "钕同位素", "源区"),
    "calculate_epsilon_hf_t": ("εHf", "铪同位素", "源区"),
    "calculate_nd_model_age": ("Nd 模式年龄", "地壳滞留年龄"),
    "calculate_smnd_to_147_144": ("Sm/Nd 换算", "同位素比值换算"),
    "calculate_rbsr_to_87rb_86sr": ("Rb/Sr 换算", "同位素比值换算"),
    "calculate_oxide_to_element": ("氧化物换算", "单位换算"),
    "calculate_element_to_oxide": ("元素换氧化物", "单位换算"),
    # ── Discrimination diagram axes ──
    "diagram_sry_y_axis": ("Sr/Y", "埃达克", "图解坐标"),
    "diagram_k2o_sio2_axis": ("K2O", "钾质系列", "图解坐标"),
    "diagram_thyb_nbyb_axis": ("Th/Yb", "Nb/Yb", "构造环境"),
    "diagram_ti_zr_axis": ("Ti-Zr", "构造环境"),
    "diagram_zry_tiy_axis": ("Zr/Y", "Ti/Y", "构造环境"),
    "diagram_nby_zry_axis": ("Nb/Y", "Zr/Y", "碱度"),
    "diagram_v_ti_axis": ("V-Ti", "MORB", "板内"),
    "diagram_feotmgo_sio2_axis": ("FeO*/MgO", "拉斑", "钙碱系列"),
    "diagram_mgno_sio2_axis": ("Mg#", "分异趋势"),
    "diagram_al2o3_tio2_axis": ("Al2O3/TiO2", "源区"),
    "diagram_layb_yb_axis": ("La/Yb", "REE 分馏"),
    "diagram_sr_ba_axis": ("Sr-Ba", "流体演化"),
    # ── Data and vision ──
    "element_inventory": ("盘点", "数据覆盖", "变量"),
    "table_column_stats": ("统计", "分布", "范围", "区间", "均值", "中位",  # general column statistics (2026-10 baseline diagnosis: non-element columns previously had no reachable tool)
                            "测温", "均一温度", "盐度", "包裹体", "同位素",  # entries cont.: fluid-inclusion/isotope task phrases
                            "品位", "stats", "description", "column"),  # entries cont.: English phrases
    "vlm_describe_image": ("图像", "照片", "薄片", "描述", "视觉", "VLM"),
    # ── Discrimination diagram plotting ──
    "plot_discrimination_diagram": ("图解", "投点", "判别图", "绘制判别图", "构造环境图"),
    # ── Generic plotting family ──
    "plot_xy_scatter": ("散点图", "两轴图", "X-Y", "相关图", "投点图"),
    "plot_ternary": ("三角图", "三端元", "三元图"),
    "plot_ree_spider": ("配分图", "稀土配分", "蜘蛛图", "标准化模式", "球粒陨石标准化"),
    "plot_category_stats": ("柱状图", "饼图", "占比图", "频次统计"),
    # ── Generic visualization ──
    "plot_data_overview": ("绘图", "可视化", "分布", "直方图", "散点", "相关"),
    # ── Text and narrative (LLM-augmented family) ──
    "analyze_geological_text": ("文本", "野外记录", "文献", "岩相描述", "text",  # task phrases for text semantic enrichment (bilingual Chinese-English)
                                "dossier", "genesis", "genetic", "host rock", "ore mineral",  # English entries: case dossier/genesis/host rock/ore mineral
                                "alteration", "deposit case", "成因", "矿床", "档案", "记录", "成因类型"),  # English entries cont.: alteration/deposit case + Chinese supplements
    "llm_genesis_synthesis": ("叙述", "综合解释", "成因解释", "总结", "synthesis",  # task phrases for narrative synthesis
                              "narrative", "analysis", "conclusion", "interpretation",  # English entries: narrative/analysis/conclusion/interpretation
                              "结论", "分析", "解释"),  # Chinese supplementary entries
}


def apply_task_keywords(registry: "ToolRegistry") -> None:
    """Inject the task-routing keywords into the registry (deduplicated order-preserving, idempotent)."""
    for name, kws in _TASK_KEYWORDS.items():
        spec = registry.get(name)  # a nonexistent name raises KeyError directly, exposing configuration errors explicitly
        merged = tuple(dict.fromkeys((*spec.keywords, *kws)))
        if merged != spec.keywords:
            registry._specs[name] = replace(spec, keywords=merged)
