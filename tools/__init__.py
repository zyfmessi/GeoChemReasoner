"""Tool Plane package entry: registry singleton + built-in tool family registration (currently 100+ tools).

The LLM can only choose tools from REGISTRY (blueprint design principle #1). When adding a new tool family,
registering it here makes it visible to the planner; the tool total has a lower-bound assertion in selftest.
"""

from __future__ import annotations  # enable deferred evaluation of type annotations (PEP 563)

from oregenesis.tools.classification import (build_classification_tools,  # import the rock classification / discrimination diagram tool family (spans lines)
                                               build_diagram_plot_tools)  # cont.: the discrimination diagram plotting tool family
from oregenesis.tools.dataviz import build_dataviz_tool  # import the generic data visualization tool constructor
from oregenesis.tools.dataviz_family import build_plot_tools  # import the generic plotting tool family (two-axis/ternary/pattern/bar)
from oregenesis.tools.geochem import build_data_tools, build_ratio_tools  # import the data inventory and element ratio tool family constructors
from oregenesis.tools.geophysics import build_geophysics_tools  # import the geological process calculator tool family constructor
from oregenesis.tools.prospectivity import build_prospectivity_tool  # import the prospectivity (ore-targeting) main tool constructor
from oregenesis.tools.ree import build_ree_tools  # import the REE atomic tool family constructor
from oregenesis.tools.registry import ToolCall, ToolRegistry, ToolResult, ToolSpec  # import the registry four-piece set: call/registry/result/spec
from oregenesis.tools.task_keywords import apply_task_keywords
from oregenesis.tools.vlm import (build_llm_synthesis_tools,  # import the narrative synthesis tool family (evidence -> narrative) (import spans lines)
                                  build_text_analysis_tools, build_vlm_tools)  # cont.: the text semantic enrichment and VLM visual families

REGISTRY = ToolRegistry()  # create the globally unique tool registry singleton
for _spec in [*build_data_tools(), *build_ratio_tools(), *build_ree_tools(),  # expand the data inventory, element ratio, and REE families and start iterating
              *build_classification_tools(), *build_geophysics_tools(),  # cont.: concatenate the rock classification and geological process calculator families
              *build_vlm_tools(), *build_text_analysis_tools(),  # cont.: concatenate the VLM visual and text semantic enrichment families (spans lines)
              *build_llm_synthesis_tools(), *build_dataviz_tool(),  # cont.: narrative synthesis and generic data visualization
              *build_diagram_plot_tools(), *build_plot_tools(),  # cont.: discrimination diagram plotting and the generic plotting family
              build_prospectivity_tool()]:  # cont.: concatenate prospectivity last; the loop body begins
    REGISTRY.register(_spec)  # register each tool spec ToolSpec into REGISTRY (duplicate names raise)

apply_task_keywords(REGISTRY)  # inject the task-routing keywords (the matching basis of Task-aware Routing)

__all__ = ["REGISTRY", "ToolCall", "ToolRegistry", "ToolResult", "ToolSpec"]  # declare the package-level public export names (visible on from-package import *)
