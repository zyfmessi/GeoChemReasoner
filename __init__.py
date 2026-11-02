"""OreGenesis-Agent: deposit prediction and analysis agent (ReAct + Tool Plane architecture).

Adaptively completes prospectivity targeting (mineral prediction) and
metallogenic mechanism analysis according to the input task. Core principle:
Data → Entity → Semantics → Evidence → Process → Test → Synthesis.
The package structure maps to four planes:
- io/ + understanding.py + evidence.py     DATA PLANE (what data exists)
- reasoning/ (planner/task_types/graph)    REASONING PLANE (what is stated, what is missing, what task)
- tools/                                   TOOL PLANE (what computation and tests to execute)
- reasoning/graph.py + llm/                CONTROL PLANE (where we are, where to go next)
"""

__version__ = "1.0.0"  # Package version constant


def __getattr__(name):  # Lazy exports: keep import lightweight
    if name in ("OreGenesisRunner", "RunnerOptions"):  # Only these two runner names go through lazy export
        from oregenesis import runner  # Load the runner submodule only now, so importing the package does not pull heavy dependencies
        return getattr(runner, name)  # Fetch the named object from the runner module and return it
    raise AttributeError(name)  # Other names raise AttributeError with standard semantics
