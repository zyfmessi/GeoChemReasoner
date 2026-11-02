# GeoChemReasoner
# OreGenesis-Agent

**OreGenesis-Agent** is an evidence-constrained ReAct agent for mineral deposit
**prospectivity targeting** (*where might ore be?*) and **metallogenic genesis
analysis** (*why and how did the mineralization form?*). Given ordinary
exploration data files — Excel/CSV geochemistry, text reports, images — the
agent plans a task-specific workflow, calls computational geoscience tools,
grounds every claim in a numbered evidence ledger, and emits a reproducible
Markdown report plus a machine-readable JSON snapshot.

Core design principle: **Data → Entity → Semantics → Evidence → Process → Test → Synthesis.**

- **110 registered geoscience tools** (geochemistry, TAS/trace-element
  discrimination diagrams, REE normalization, isotope and fluid-inclusion
  calculators, geophysics, prospectivity/evidence-weight mapping, plotting),
  selected only through a typed tool registry — the LLM plans and interprets,
  tools compute.
- **Three reliability mechanisms** over a plain ReAct loop: registry
  constraints (no invented tool calls), data gating (inputs parsed and
  validated before use), and evidence synchronization (claims must cite
  ledger evidence). All three can be switched off individually for ablation.
- **Fail-fast model gate**: the agent refuses to run without a reachable LLM
  instead of silently degrading; a fully deterministic rule-based planner is
  available via an explicit `--no-llm` flag for baselines and offline testing.

If you use this software, please cite the accompanying paper (see
[Citation](#citation)).

## Repository layout

```
Agent_New/
├── oregenesis/            # Core Python package (import as `oregenesis`)
│   ├── io/                #   Data plane: file parsers (Excel/CSV/text/image)
│   ├── understanding.py   #   Data plane: input understanding
│   ├── evidence.py        #   Data plane: evidence ledger
│   ├── reasoning/         #   Reasoning + control plane: LangGraph main graph,
│   │                      #   ReAct planner, task classifier, review node
│   ├── tools/             #   Tool plane: 110 registered tools + registry
│   ├── llm/               #   LLM backends (OpenAI-compatible / Ollama), gate
│   ├── memory.py          #   Optional cross-task conclusion memory
│   ├── report.py          #   Markdown report renderer
│   └── runner.py          #   OreGenesisRunner: files + task -> result
├── scripts/               # All benchmark build / run / scoring / plotting scripts
│   └── README.md          #   Script inventory (Chinese filenames explained)
├── docs/                  # User guide, tutorial, reproducibility statement
├── frontend/              # Optional Vue 3 + FastAPI web demo (Chinese UI)
├── figures/               # Paper figure sources and generation scripts
├── requirements.txt       # Python dependencies
├── .env.example           # LLM endpoint configuration template
└── LICENSE                # MIT license
```

> Note: the folder `core/` is an identical working copy of the `oregenesis/`
> package kept for development; `oregenesis/` is the package that is imported
> and should be treated as canonical.

## Requirements

**Software**

- Python **3.10+** (all results in the paper were produced with **3.12.7**, 64-bit)
- The Python packages in [`requirements.txt`](requirements.txt)
  (numpy, pandas, scipy, scikit-learn, matplotlib, langgraph, requests,
  openpyxl, xlrd, Pillow, PyMuPDF)
- A language-model backend, either:
  - **local** via [Ollama](https://ollama.com) or any vLLM/llama.cpp
    `llama-server` with an OpenAI-compatible endpoint, or
  - **cloud** — any OpenAI-compatible API (the paper used DeepSeek and
    Zhipu BigModel endpoints).
- Optional: Node.js ≥ 18 (only for the web demo frontend).

**Hardware**

- Running the agent against a **cloud API**: any modern computer; no GPU needed.
  RAM ≥ 8 GB is comfortable; a full run writes figures and reports of a few MB.
- Running **local models** as in the paper's ablation (Qwen3-VL-8B up to
  Qwen3.6-27B, GGUF quantized, context window 32768): a single consumer GPU
  with **≥ 16 GB VRAM** for 8B–14B models, **≥ 24 GB VRAM** for 24B–30B models.
  Smaller contexts can be configured with `LLM_NUM_CTX` (see `.env.example`).
- The deterministic test suite (`架构自检`, tool inventory, soil suite,
  no-LLM smoke tests) runs CPU-only in about 1–2 minutes.

## Installation

```bash
# 1. Clone the repository
git clone <REPOSITORY_URL>
cd Agent_New

# 2. Create and activate an environment (Python >= 3.10)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure the language-model backend
copy .env.example .env           # Windows (Git Bash: cp .env.example .env)
#   edit .env: set LLM_API_BASE, LLM_API_KEY, LLM_MODEL_NAME
```

### Setting up a model backend

The agent needs exactly one reachable OpenAI-compatible chat endpoint.
Two common setups:

```bash
# Option A — local Ollama (no API key cost)
ollama pull qwen3.6:27b
#   .env: LLM_API_BASE=http://127.0.0.1:11434   LLM_API_KEY=ollama   LLM_MODEL_NAME=qwen3.6:27b

# Option B — cloud API (e.g. DeepSeek)
#   .env: LLM_API_BASE=https://api.deepseek.com  LLM_API_KEY=sk-...  LLM_MODEL_NAME=deepseek-chat
```

If no model is reachable, every agent entry point raises
`LLMUnavailableError` immediately with the endpoint, model and cause in the
message — nothing is written "as if" a model had run. To run the deterministic
rule-based planner instead (ablation baseline / offline reproduction), pass
`--no-llm` (CLI) or `RunnerOptions(require_llm=False)` (Python API).

## Quick start

```bash
# Deterministic self-test of the whole system (no model needed, ~1-2 min)
python scripts/一键运行全部测试.py --quick

# Single genesis-analysis run against your data (LLM required)
python scripts/启动矿床智能体.py \
    --files path/to/whole_rock.xlsx path/to/geology_notes.txt \
    --task "解释该岩体的成因与岩浆演化过程" \
    --out output/demo_run
```

Every run writes to the `--out` directory:

| File | Content |
|---|---|
| `report.md` | Human-readable report: task understanding, evidence ledger, ReAct rounds, conclusion, remaining uncertainties, figures |
| `final_state.json` | Machine-readable snapshot: conclusion, key observations, evidence items with lineage, tool calls, target zones |
| `*.png`, `*.csv` | Agent-produced figures (diagrams, maps) and exported tables |

CLI options: `--files` (one or more input files), `--task` (natural-language
task), `--no-llm`, `--max-rounds N` (ReAct round cap, default 6), `--out DIR`.
See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for all options, inputs, outputs
and expected behavior, and [docs/TUTORIAL.md](docs/TUTORIAL.md) for
step-by-step walkthroughs of typical use cases.

## Reproducing the paper's results

The benchmark suites (Tasks A–E), the five-model eight-arm ablation, and the
scoring pipelines are all driven by scripts in `scripts/`. A complete mapping
from paper tables/figures to exact commands, the compute environment used,
and the data-availability statement (including the parts that cannot be
redistributed) are given in
**[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)**. In short:

```bash
# sanity path: deterministic regression over 110 tools + benchmark smoke (no model)
python scripts/一键运行全部测试.py

# full formal evaluation of Tasks D, C, A with a chosen model tag
python scripts/运行正式评测.py --tag <model-tag>
```

All benchmark *construction* scripts are included, so exam papers can be
rebuilt from the original public data sources (GTK FODD, ExploreSA Gawler,
USGS Stibnite–Yellow Pine, NDI/NVCL, NSERC-CMIC Footprints, USGS soil —
each under its own license, see the data statement).

## Documentation

| Document | Content |
|---|---|
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Inputs, outputs, all options, expected behavior, troubleshooting |
| [docs/TUTORIAL.md](docs/TUTORIAL.md) | Step-by-step tutorials for typical use cases |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Paper results ↔ script mapping, compute environment, data availability & limitations |
| [scripts/README.md](scripts/README.md) | Inventory of all 50+ evaluation scripts (Chinese filenames) |

## License

Released under the [MIT License](LICENSE). Benchmark data downloaded for
reproduction remains under its original publishers' licenses (GTK Basic
Licence, CC BY 4.0, CC0, MIT — see [LICENSE](LICENSE) third-party notice and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)).

## Citation

If this code contributes to your research, please cite the accompanying
manuscript (citation block to be added upon publication).

