"""OreGenesis-Agent end-to-end run entry (new architecture: ReAct + Tool Plane + Data Plane).
Usage: from runner import OreGenesisRunner, RunnerOptions
    result = OreGenesisRunner().run(["granite.xlsx", "notes.txt"],
                                    "解释该花岗伟晶岩的形成机制")
Requires the language model online by default (ReAct planner); for ablation /
no-model scenarios explicitly pass RunnerOptions(require_llm=False), and the
planner takes the deterministic rule-based fallback.
"""
from __future__ import annotations  # Enable PEP 563 deferred type-annotation evaluation
from dataclasses import dataclass, field  # Import the dataclass decorator and the field default-factory
from pathlib import Path  # Import the path type, used to persist outputs under the output directory
from typing import Any  # Import the arbitrary type annotation
from oregenesis.reasoning.graph import build_graph  # Import the main-graph builder (LangGraph compilation entry)
from oregenesis.reasoning.state import initial_state  # Import the initial-state constructor
from oregenesis.report import render_markdown  # Import the Markdown report renderer


@dataclass  # Declared as a dataclass; auto-generates construction and other methods
class RunnerOptions:  # Define the run-options class: gate / rounds / plotting / output directory and other configuration
    """Run options (same semantics as the old version: the model gate is on by default; fallback must be declared explicitly)."""

    require_llm: bool = True                # True = raise immediately when the model is unreachable (no silent degradation)
    max_rounds: int = 6                     # ReAct round limit
    make_plots: bool = True                 # Whether the prediction tool emits plots (passed through to tool parameters)
    output_dir: str = "output"              # Report/figure output directory
    task_type: str = ""                     # Explicit task-type override (prospectivity/genesis/comprehensive; empty = auto-classified from task text)
    backend: Any = None                     # Externally injected LLM backend (for tests / counting proxies; None = built per require_llm)
    extra: dict = field(default_factory=dict)  # Extra parameters passed through to tools (e.g. deposits)
    memory_dir: str = ""                    # Long-task memory workspace directory (empty = memory off, behavior byte-identical to historical versions)
    task_label: str = ""                    # Label of this task (used in memory-card filenames and the card's source field; empty = task)


@dataclass  # Declared as a dataclass
class RunnerResult:  # Define the run-result class: final state + machine-readable result + human-readable report
    """The complete output of one run: final state + machine-readable result + human-readable Markdown report."""

    state: dict[str, Any]                            # LangGraph final state (including the audit trail)
    final_result: dict[str, Any]                     # The 10-item scientific output
    markdown: str                                    # Human-readable report

    def to_dict(self) -> dict[str, Any]:  # Define the interface method converting to a machine-readable dict
        """Machine-readable result (project snapshot: task / evidence / tools / audit)."""
        return self.final_result  # Return the scientific-output dict directly as the project snapshot


class OreGenesisRunner:  # Define the end-to-end executor class
    """End-to-end executor: files + task → main graph → scientific output."""

    def run(self,  # Define the main run method (signature spans multiple lines)
            files: list[str | Path],  # Parameter: list of input file paths (str or Path)
            task: str = "",  # Parameter: natural-language scientific task description, default empty string
            options: RunnerOptions | None = None) -> RunnerResult:  # Parameter: optional run configuration; returns the full run result
        opts = options or RunnerOptions()  # Instantiate default options when none were passed
        backend = opts.backend if opts.backend is not None else self._make_backend(opts)  # Prefer the injected backend (counting proxies etc.); otherwise build one per the gate

        app = build_graph(backend=backend if opts.require_llm else None)  # Build the main graph; when the gate is off, no model backend is injected into the graph
        state = initial_state([str(f) for f in files], task, max_rounds=opts.max_rounds,  # Build the initial state, overriding the task type as needed (spans lines)
                              task_type=opts.task_type)  # Task type: explicit option wins; otherwise auto-classified inside initial_state
        state["backend"] = backend            # Inject the backend: tools like VLM fetch it from State
        state["task"]["make_plots"] = opts.make_plots  # Write the plotting switch into the task config for tools to read
        state["task"]["output_dir"] = opts.output_dir  # Output root directory: where tool figures/tables land (default "output")
        if opts.extra:  # Extra-parameter passthrough (previously only claimed in a comment, never actually wired)
            state["task"].update(opts.extra)  # Merge extra key-values into the task config (e.g. the benchmark's known-deposits table deposits)
        # ── Long-task memory - read hook (off by default): recall prior tasks' conclusion cards into the initial context ──
        memory_store = None  # Memory-store handle (for the write hook; None = memory off)
        if opts.memory_dir:  # Switch: enter memory mode only when memory_dir is non-empty
            from oregenesis.memory import MemoryStore  # Lazy import of the memory layer (zero import cost when off)
            from oregenesis.memory import card_from_result  # Lazy import of the conclusion-card constructor
            memory_store = MemoryStore(opts.memory_dir)  # Open the workspace (the directory may not exist yet = first round)
            mounted_dirs = {str(Path(f).resolve().parent) for f in files}  # Set of directories mounted for this task (permission criterion)
            cards = memory_store.recall_cards(mounted_dirs)  # Strict-subset recall (cards referencing unmounted directories are blocked whole)
            if cards:  # Inject only when cards exist (no cards on the first round = behavior identical to no memory)
                state["evidence"] = MemoryStore.recall_evidence(cards)  # Memory entries go to the ledger head (naturally covered by the synthesis summary evidence[:40])
                state["memory_cards"] = MemoryStore.render_lines(cards)  # Prompt lines (for the planner's dedicated "prior conclusions memory" section)
        final = app.invoke(state, config={"recursion_limit": 60})  # 60-step limit guards against infinite loops
        # ── Long-task memory - write hook (off by default): copy this task's conclusions into a card for later tasks to recall ──
        if memory_store is not None:  # When memory is on
            seen_dirs = {str(Path(f).resolve().parent) for f in files}  # Directories seen by this task (recorded into the card as permission credentials)
            card = card_from_result(task_label=opts.task_label or "task",  # Card label: the evaluation script passes the task id
                                    final_result=final.get("final_result") or {},  # Final-state scientific output (ready-made fields like conclusion/key_observations)
                                    seen_dirs=seen_dirs, mounted_dirs=mounted_dirs)  # Directory permission sets
            memory_store.save_card(card)  # Write the card (re-running the same task overwrites the old card)

        result = RunnerResult(state=final,  # Assemble the run-result object (arguments span multiple lines)
                              final_result=final.get("final_result") or {},  # Take the final state's final result; fall back to an empty dict when missing
                              markdown="")  # Markdown field blanked as a placeholder for now
        result.markdown = render_markdown(result.final_result, task, base_dir=opts.output_dir)  # Render the report (figure relative paths are based on the output directory)
        self._write_outputs(result, opts, task)  # Persist the report and state snapshot to the output directory
        return result  # Return the result object containing the final state and the report

    @staticmethod  # Static-method decorator (does not depend on instance state)
    def _make_backend(opts: RunnerOptions) -> Any:  # Define the backend factory method: build the LLM backend per options
        """Build the LLM backend per gate semantics: with require_llm=True, raise immediately when unreachable."""
        if not opts.require_llm:  # Gate off: return None directly to take the rule-based fallback
            return None  # None means no LLM backend
        from oregenesis.llm.gate import require_llm  # Lazy import of the connectivity gate function
        from oregenesis.llm.base import LLMBackendConfig  # Lazy import of the backend config class
        from oregenesis.llm.openai_compatible import OpenAICompatibleBackend  # Lazy import of the OpenAI-compatible backend implementation

        require_llm()  # Gate: no work without a model (the error message guides the user to connect)
        return OpenAICompatibleBackend(LLMBackendConfig())  # Instantiate the compatible backend with default config and return it

    @staticmethod  # Static-method decorator
    def _write_outputs(result: RunnerResult, opts: RunnerOptions, task: str) -> None:  # Define the output-persistence method: write the report and state JSON
        """Persist the report and state snapshot (half of reproducible auditing; the other half is in state.audit)."""
        out = Path(opts.output_dir)  # Convert the output-directory string to a Path object
        out.mkdir(parents=True, exist_ok=True)  # Create the output directory recursively; no error if it exists
        (out / "report.md").write_text(result.markdown, encoding="utf-8")  # Write the Markdown report to report.md
        import json  # Import json inside the function; only needed when persisting
        (out / "final_state.json").write_text(  # Open final_state.json for writing (call spans multiple lines)
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str),  # Serialize the machine-readable result as 2-indent JSON, non-ASCII kept verbatim
            encoding="utf-8")  # Write the file with UTF-8 encoding
