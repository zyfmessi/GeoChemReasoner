"""Control Plane - global state: AgentState (the single authoritative definition of the LangGraph State).
Nodes do not pass variables to each other directly; each returns a partial update
that LangGraph merges into the State (blueprint §12-§15). Fields with
Annotated[..., operator.add] have accumulating semantics; all other fields are
replaced wholesale.
"""

from __future__ import annotations  # Enable deferred annotation evaluation: type annotations have no runtime effect
import operator  # Import operator: its add is used in the accumulate-merge annotations of list fields
from typing import Annotated, Any, TypedDict  # Import typing utilities: Annotated attaches merge semantics, TypedDict defines the State structure, Any for arbitrary types
from oregenesis.io.data_models import DataUnit, InputAsset  # Import data models: DataUnit the standard data unit, InputAsset the raw input asset
from oregenesis.reasoning.task_types import classify_task  # Import the task-type classifier: groups task text into capability directions


class AgentState(TypedDict, total=False):  # Define the global-state TypedDict; total=False means every key is optional
    """The complete state of one run: the data changes and the state keeps changing (blueprint §39 state flow)."""  # Class docstring: summarizes the semantics of the state continuously updating during a run

    # ── Task and input ──
    task: dict[str, Any]                             # task_id / question / target
    input_assets: list[InputAsset]                   # User input files (raw assets)

    # ── Data Plane ──
    data_units: list[DataUnit]                       # Standard data units after parsing + enrichment (replaced wholesale)
    bundles: list[dict[str, Any]]                    # Multi-source alignment results (SampleBundle as dicts)

    # ── Evidence ──
    evidence: Annotated[list[dict[str, Any]], operator.add]      # Evidence ledger (accumulating)
    memory_cards: list[str]                          # Long-task memory: prompt lines of conclusion cards from prior tasks (replaced wholesale; empty when disabled by default)

    # ── Tool Plane ──
    tool_results: Annotated[list[dict[str, Any]], operator.add]  # Tool results (accumulating)

    # ── Control and audit ──
    rounds: int                                      # ReAct rounds already executed
    max_rounds: int                                  # Round limit
    current_stage: str                               # Current stage (human-readable)
    node_status: dict[str, str]                      # Per-node status (running/completed/failed)
    decisions: Annotated[list[dict[str, Any]], operator.add]     # ReasoningDecision records (accumulating)
    errors: Annotated[list[str], operator.add]       # Error records (accumulating)
    final_result: dict[str, Any] | None              # Synthesized scientific output (10 items)
    backend: Any | None                              # LLM backend (injected by the runner; previously absent from the schema, so the key was dropped by LangGraph)


def initial_state(files: list[str], task: str, *, max_rounds: int = 6,  # Define the initial-state factory: files/task/max_rounds (spans lines)
                  task_type: str = "") -> AgentState:  # Parameters continued: task_type can explicitly override the task type (empty string = auto-classified)
    """Build the initial State: register the task (with auto-classified type) and input files; give every other field an empty value."""  # Function docstring: register the task and inputs; give the remaining fields type-correct empty values
    return AgentState(  # Construct and return the initial AgentState dict (arguments span multiple lines)
        task={"task_id": "task_001",  # task field: task ID (dict stays open for continuation lines)
              "task_type": task_type or classify_task(task),  # Task type: explicit value wins; otherwise auto-classified from the task text
              "question": task, "target": "deposit_prediction_and_analysis",  # task field continued: question text and reasoning target (deposit prediction and analysis)
              "files": [str(f) for f in files]},  # task field continued: stringify each input file into files; close the task dict
        input_assets=[],  # input_assets field: initially an empty list, to be registered by the init node
        data_units=[], bundles=[],  # data_units and bundles fields: both initially empty lists
        evidence=[], tool_results=[],  # evidence/tool_results fields: both initially empty lists
        rounds=0, max_rounds=max_rounds,  # rounds reset to 0; max_rounds takes the passed round limit
        current_stage="initialized", node_status={},  # current_stage set to initialized; node_status set to an empty dict
        decisions=[], errors=[], final_result=None,  # decisions/errors set to empty lists; final_result is None for now
    )
