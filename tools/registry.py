"""Tool Plane registry: ToolSpec / ToolCall / ToolResult / ToolRegistry.

Design principles (tool-layer detailed design §21, §51):
- tools must be registered; the LLM can only choose from the Tool Registry and cannot invent tool names on the fly;
- tool results are uniformly ToolResult (with status/result/warnings/data lineage); arbitrary Python objects
  are never thrown at the reasoning layer;
- tool inputs come from DataUnits (data lineage traceable: ToolResult -> ToolCall -> DataUnit
  -> InputAsset -> original file);
- the LLM is the decision-maker and interpreter, tools are the calculators and executors - strictly separated responsibilities.
"""

from __future__ import annotations  # enable deferred evaluation of type annotations (PEP 563)

from dataclasses import dataclass, field  # import the dataclass decorator and the field factory from the standard library
from typing import Any, Callable  # import Any and Callable type annotations

# Applicability-check function signature: reads AgentState, returns (usable, reason when not)
ApplicabilityFn = Callable[[Any], tuple[bool, str]]  # type alias: applicability-check function (state -> usable + reason)
# Tool run function signature: reads AgentState and ToolCall, returns the result payload (goes into ToolResult.result)
RunFn = Callable[[Any, "ToolCall"], dict[str, Any]]  # type alias: tool run function (state + call -> result dict)


@dataclass(frozen=True)  # declare an immutable dataclass (fields cannot be modified after instantiation)
class ToolSpec:  # tool spec class: the basis for the LLM's tool choice
    """Unified tool description: from this the Agent knows what a tool is called / what it does / what it needs / when it is usable."""

    name: str                                        # tool name (registry primary key, used for LLM selection)
    tool_family: str                                 # tool family (e.g. geochemistry / spatial)
    description: str                                 # one-line description (goes into the planner prompt)
    required_data: tuple[str, ...] = ()              # description of the required data types
    capabilities: tuple[str, ...] = ()               # capability tags
    output_artifacts: tuple[str, ...] = ()           # output artifact types (map/grid/report...)
    applicable: ApplicabilityFn | None = None        # data-condition check (Data-aware Routing)
    run: RunFn | None = None                         # run function
    keywords: tuple[str, ...] = ()                   # task-routing keywords: scored against the user's task text (Task-aware)


@dataclass  # declare a mutable dataclass
class ToolCall:  # tool call class: a structured record of one tool request
    """One tool request: a structured call record (itself auditable data)."""

    tool_call_id: str                                # call id (toolcall_001 style)
    tool_name: str                                   # tool name (must be in the registry)
    input_data_unit_ids: list[str] = field(default_factory=list)  # input DataUnit lineage
    parameters: dict[str, Any] = field(default_factory=dict)      # algorithm parameters (the basis of reproducibility)
    purpose: str = ""                                # scientific purpose of this call (goes into the audit)


@dataclass  # declare a mutable dataclass
class ToolResult:  # tool result class: the carrier of unified output
    """Unified tool output: execution status + result payload + warnings + lineage; the carrier of Observations."""

    tool_call_id: str  # field: id of this call (corresponds to ToolCall.tool_call_id)
    tool_name: str  # field: name of the executed tool
    status: str = "success"                          # success / error
    result: dict[str, Any] = field(default_factory=dict)  # field: result payload; the default factory creates an independent empty dict (avoids shared mutable defaults)
    source_data_unit_ids: list[str] = field(default_factory=list)  # field: source data lineage - list of input DataUnit ids (empty list by default)
    warnings: list[str] = field(default_factory=list)  # field: warning message list (empty list by default)
    error: str | None = None  # field: error message text; None on success

    def to_dict(self) -> dict[str, Any]:  # instance method: convert this result object into a plain dict
        """Convert to dict, for writing back to state and persisting to the audit log."""
        return {  # start building and returning the result dict
            "tool_call_id": self.tool_call_id, "tool_name": self.tool_name,  # put in the call id and tool name key-value pairs
            "status": self.status, "result": self.result,  # put in the execution status and result payload
            "source_data_unit_ids": self.source_data_unit_ids,  # put in the input data lineage list
            "warnings": self.warnings, "error": self.error,  # put in the warnings list and the error message
        }


class ToolRegistry:  # tool registry class: the single tool catalog for registration/lookup/routing
    """Tool registry: registration/lookup/routing; the single tool catalog of the Tool Plane."""

    def __init__(self) -> None:  # constructor
        self._specs: dict[str, ToolSpec] = {}  # initialize the private registry dict: tool name -> ToolSpec

    def register(self, spec: ToolSpec) -> None:  # register method: takes one tool spec
        """Register a tool; a duplicate name is treated as a configuration error and raises directly."""
        if spec.name in self._specs:  # check whether the tool name already exists in the registry
            raise ValueError(f"工具重复注册: {spec.name}")  # duplicate registration is a configuration error; raise ValueError
        self._specs[spec.name] = spec  # store the ToolSpec into the registry keyed by tool name

    def get(self, name: str) -> ToolSpec:  # lookup-by-name method
        """Fetch a tool by name; raise when absent (rather than silently returning None) - LLM-invented tool names are intercepted here."""
        if name not in self._specs:  # check whether the tool name is absent from the registry
            raise KeyError(f"未注册的工具: {name}（可用: {', '.join(sorted(self._specs))}）")  # unregistered: raise KeyError, with all available tool names attached
        return self._specs[name]  # return the ToolSpec for that tool name

    def names(self) -> list[str]:  # method returning all tool names
        """All registered tool names."""
        return sorted(self._specs)  # return the sorted list of all registered tool names

    def specs(self) -> list[ToolSpec]:  # method returning all tool specs
        """All tool specs (for the planner prompt)."""
        return list(self._specs.values())  # return the list of all ToolSpecs in the registry

    def add_keywords(self, name: str, *kws: str) -> None:
        """Append task-routing keywords to a registered tool (dataclasses.replace preserves frozen semantics)."""
        import dataclasses
        spec = self.get(name)
        merged = tuple(dict.fromkeys((*spec.keywords, *kws)))  # merge with dedup, preserving order
        self._specs[name] = dataclasses.replace(spec, keywords=merged)

    def execute(self, call: ToolCall, state: Any) -> ToolResult:  # unified executor: takes a call record and the Agent state
        """Tool executor: existence check -> data-condition check -> run -> unified ToolResult.

        Any exception is converted into a ToolResult with status="error" (never re-raised), so that
        the ReAct loop can see the failure reason as an Observation and re-plan.
        """
        try:  # enter try: the tool existence check first
            spec = self.get(call.tool_name)  # look up the ToolSpec in the registry by the called tool name
        except KeyError as e:  # catch the KeyError of a nonexistent tool
            return ToolResult(tool_call_id=call.tool_call_id, tool_name=call.tool_name,  # build an error result: with the call id and tool name
                              status="error", error=str(e))  # status set to error, the error message is the exception text
        if spec.applicable is not None:  # Data-aware: do not run when the data condition is unmet
            ok, reason = spec.applicable(state)  # run the data-condition check, getting usable and the reason
            if not ok:  # check whether the data condition is unmet
                return ToolResult(tool_call_id=call.tool_call_id, tool_name=call.tool_name,  # build an error result: with the call id and tool name
                                  status="error",  # status set to error
                                  error=f"数据条件不满足: {reason}",  # the error message notes the unmet data condition and its specific reason
                                  warnings=[reason])  # also put the reason into the warnings list
        try:  # enter try: run the tool itself
            payload = spec.run(state, call) if spec.run else {}  # with a run function, execute (state, call) for the payload; otherwise an empty dict
            return ToolResult(tool_call_id=call.tool_call_id, tool_name=call.tool_name,  # build a success result: with the call id and tool name
                              status="success", result=payload,  # status set to success, the result is the tool-returned payload
                              source_data_unit_ids=list(call.input_data_unit_ids),  # lineage: copy the input DataUnit id list from the call
                              warnings=list(payload.get("warnings", [])))  # extract the warnings key from the payload (default empty list)
        except Exception as e:  # a tool failure is also a legitimate Observation, handed back to the reasoning layer
            return ToolResult(tool_call_id=call.tool_call_id, tool_name=call.tool_name,  # catch any exception, convert to an error result instead of re-raising
                              status="error", error=f"{type(e).__name__}: {e}")  # status set to error, the error message is the exception class name + message
