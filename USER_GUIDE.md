# OreGenesis-Agent — User Guide

This guide describes every input, output, option and the expected behavior of
the system. For installation see the top-level
[README](../README.md#installation); for worked examples see the
[Tutorial](TUTORIAL.md); for reproducing the paper see
[Reproducibility](REPRODUCIBILITY.md).

---

## 1. Two ways to run the agent

| Entry point | When to use |
|---|---|
| CLI — `python scripts/启动矿床智能体.py ...` | One-off runs from the terminal |
| Python API — `OreGenesisRunner` / `RunnerOptions` | Embedding in your own scripts and benchmarks |
| Web demo — `python frontend/api.py` (optional) | Interactive demonstration and run replay in a browser |

Both the CLI and the API execute the identical pipeline: parse inputs →
understand the task → ReAct loop (plan → call tools → observe) → review →
synthesize report.

---

## 2. Inputs

### 2.1 Data files (`--files` / `files` argument)

One or more files of the following types; modalities may be freely mixed:

| Modality | Extensions | Typical content |
|---|---|---|
| Numeric tables | `.csv` `.xlsx` `.xlsm` `.xls` | Whole-rock major/trace elements, isotope ratios, sample sheets, deposit point tables |
| Text | `.txt` `.md` | Geological background notes, reports, literature extracts |
| Images | `.jpg` `.jpeg` `.png` `.tif` `.tiff` `.bmp` | Maps, thin-section photos, scanned diagrams (interpreted via the VLM tool when the backend is multimodal) |

An unsupported extension raises `ValueError` immediately, listing the
supported types — files are never silently ignored.

### 2.2 Task text (`--task` / `task` argument)

A free-form natural-language description of the scientific question, in
Chinese or English. The task type is auto-classified by deterministic keyword
scoring (auditable, works offline):

| Task type | Triggered by keywords such as | Capability |
|---|---|---|
| `prospectivity` | 成矿预测, 靶区, 找矿, 圈定, 潜力, 勘查, 空间分布 … | Prospectivity targeting: rank favorable cells/zones from geochemical + spatial evidence |
| `genesis` | 成因, 机制, 物质来源, 构造环境, 判别, 稀土, 同位素, 蚀变, 岩浆 … | Metallogenic mechanism analysis: source → transport → deposition with discriminating tests |
| `comprehensive` | both keyword families hit, or neither | Runs both directions |

Force a type explicitly with `RunnerOptions(task_type=...)` (values:
`prospectivity`, `genesis`, `comprehensive`).

### 2.3 Model configuration (`.env` at the project root)

See [.env.example](../.env.example). Keys (priority: environment variable >
`.env` > default):

| Key | Default | Meaning |
|---|---|---|
| `LLM_API_BASE` | *(empty — must be set)* | OpenAI-compatible endpoint (Ollama, vLLM/llama-server, or a cloud API) |
| `LLM_API_KEY` | *(empty — must be set)* | API key; any non-empty string works for local servers |
| `LLM_MODEL_NAME` | *(empty — must be set)* | Model tag as served by the endpoint |
| `LLM_MIN_INTERVAL` | `2.0` | Minimum seconds between requests (cloud rate-limit protection; use `0` locally) |
| `LLM_MAX_RETRIES` | `6` | Exponential-backoff retries on 429/5xx/network errors (respects `Retry-After`) |
| `LLM_NUM_CTX` | `32768` | Context window (tokens) requested from local Ollama models |
| `LLM_DISABLE_THINKING` | `1` | `1` = ask reasoning models to answer directly (keeps thinking text out of JSON outputs); `0` = keep thinking segments |

Inference parameters: `temperature=0.1` for stable structured output; per-call
`max_tokens` is set by each pipeline stage (planning 768, reflection ~1.5k,
mechanism analysis 8k–24k, report ~20k).

---

## 3. Options

### 3.1 CLI options (`scripts/启动矿床智能体.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--files FILE...` | *(none)* | Input data files (see §2.1) |
| `--task TEXT` | `""` | Natural-language scientific task |
| `--no-llm` | off | Disable the model gate: the planner uses the deterministic rule-based fallback (ablation baseline; fully reproducible offline) |
| `--max-rounds N` | `6` | Upper bound on ReAct think→act→observe rounds |
| `--out DIR` | `output` | Directory where the report, JSON snapshot and figures are written |

### 3.2 `RunnerOptions` (Python API — `oregenesis.runner`)

| Field | Default | Meaning |
|---|---|---|
| `require_llm` | `True` | `True`: raise `LLMUnavailableError` immediately when the model is unreachable. `False`: run the deterministic rule-based planner (no model calls) |
| `max_rounds` | `6` | ReAct round limit (the graph also has a hard 60-step recursion guard) |
| `make_plots` | `True` | Let prediction tools emit figures (pass `False` for text-only runs) |
| `output_dir` | `"output"` | Root directory for `report.md`, `final_state.json`, figures |
| `task_type` | `""` | Force task type (`prospectivity`/`genesis`/`comprehensive`); empty = auto-classify |
| `backend` | `None` | Inject a custom LLM backend (e.g. a counting proxy for tests); `None` = build from config |
| `extra` | `{}` | Extra key-values merged into the task config and passed through to tools (e.g. a known-deposits table for benchmark runs) |
| `memory_dir` | `""` | Enable cross-task conclusion memory in the given directory; empty = off (behavior identical to the historical version) |
| `task_label` | `""` | Label used in memory-card filenames; defaults to the task text |

Minimal API example:

```python
from oregenesis.runner import OreGenesisRunner, RunnerOptions

result = OreGenesisRunner().run(
    ["granite.xlsx", "notes.txt"],
    "Explain the petrogenesis and magma evolution of this granite suite.",
    options=RunnerOptions(output_dir="output/demo_run"),
)
print(result.markdown)        # the same Markdown report written to report.md
print(result.final_result["conclusion"])
```

---

## 4. Outputs

Each run writes into `output_dir` (created if missing; existing files are
overwritten — use a fresh directory per run):

### 4.1 `report.md` — human-readable report

Sections: task understanding (type + capability + original question) →
evidence ledger (numbered items with source lineage) → ReAct timeline (per
round: thought with goals/missing information, action = tool + arguments,
observation = result summary) → review findings → final conclusion → key
observations → supported processes → remaining uncertainties → recommended
further tests → figure gallery. Figure links are relative to `output_dir`.

### 4.2 `final_state.json` — machine-readable snapshot

Top-level keys of `final_result` (the "scientific output" object):

| Key | Content |
|---|---|
| `task_understanding` | `{task_type, capability, question}` |
| `conclusion` | Final synthesized conclusion (calibrated against the evidence ledger) |
| `narrative` | Analysis-process narrative |
| `key_observations` | Traceable observation facts |
| `supported_processes` | Geological process labels supported by evidence |
| `target_zones` | Prospectivity deliverable (zones/cells with scores; empty unless the mapping tool ran) |
| `mechanism_analysis` | Mechanism-discovery block (empty unless mechanism tools ran) |
| `supporting_evidence` / `conflicting_evidence` | Ledger items labeled as supporting/contradicting |
| `remaining_uncertainties` | Deduplicated limitation notes across all evidence |
| `recommended_tests` | Suggested follow-up analyses |
| `figures` | Paths of agent-produced figures (relative to `output_dir`) |
| `provenance` | Audit trail (filled by the audit node) |
| `data_inventory` | `{n_samples_datasets, tool_runs}` — datasets seen and every tool invoked |

### 4.3 Figures and tables

`*.png` figures and exported `*.csv` tables (e.g. `cell_scores.csv` with
per-cell prospectivity scores in real coordinates) land in the same
directory, produced with matplotlib (non-interactive Agg backend in tests).

---

## 5. Expected behavior

- **Model gate.** With `require_llm=True` (default), an unreachable endpoint
  raises `LLMUnavailableError` naming the address, model, cause and remedy.
  The agent never emits a "complete-looking" report without real model calls.
- **Deterministic fallback.** With `--no-llm` / `require_llm=False`, the
  planner follows deterministic rules; runs are reproducible offline and are
  used as the ablation baseline (arm A123 in the paper).
- **Rounds and termination.** The ReAct loop runs at most `max_rounds`
  (default 6); the graph has a 60-step recursion limit as a hard stop. Short
  genesis tasks typically finish in 2–4 rounds.
- **Evidence discipline.** Every tool result enters a numbered evidence
  ledger with data lineage (ToolResult → ToolCall → DataUnit → InputAsset →
  file). The reviewer checks that the conclusion is grounded; ungrounded
  claims are flagged or calibrated.
- **Task routing.** Genesis tasks drive geochemical tools (TAS, trace-element
  discrimination, REE patterns, isotopes); prospectivity tasks drive the
  evidence-weight mapping chain; `comprehensive` runs both.
- **Idempotence of memory.** With `memory_dir` set, each run writes one
  conclusion card; later runs recall only cards whose directories are all
  mounted in the current task (strict permission check). Off by default.
- **File overwriting.** `report.md` and `final_state.json` are overwritten
  per run in the same `output_dir`; use distinct directories to keep runs.

---

## 6. Web demo (optional)

```bash
cd frontend
npm install && npm run build    # only after changing src/
python api.py                   # serves dist/ + FastAPI at http://127.0.0.1:8020
```

Features: replay of any archived run (`final_state.json`) event-by-event
(task understanding → evidence ledger → per-round thought/action/observation →
review → conclusion), target-zone charts, prospectivity scatter maps, figure
gallery, and an online-run panel (upload a data folder, run the agent, view
the result). Requires the LLM to be configured; a run takes roughly 1–5
minutes. Component-level details: [frontend/README.md](../frontend/README.md).

---

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `LLMUnavailableError: 未连接到语言模型 ...` | Endpoint unreachable. Start Ollama (`ollama serve`, then `ollama pull <model>`) or check `LLM_API_BASE`/`LLM_API_KEY`/`LLM_MODEL_NAME` in `.env` |
| `ValueError: 不支持的数据文件类型: .xxx` | Unsupported input extension — convert to one of the types in §2.1 |
| HTTP 500 from a local server mid-run | Context window too large for VRAM; lower `LLM_NUM_CTX` (e.g. 16384) and retry |
| Cloud 429 rate-limit errors | Automatic: the client throttles (`LLM_MIN_INTERVAL`) and retries with backoff (`LLM_MAX_RETRIES`). Increase the interval on free tiers |
| Empty/blank model replies | Some servers truncate over-long prompts silently; lower `LLM_NUM_CTX`, or set `LLM_DISABLE_THINKING=1` |
| Chinese text garbled in the Windows console | Set `PYTHONIOENCODING=utf-8` before running |
