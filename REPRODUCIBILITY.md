# Reproducing the Paper's Results

This document maps the manuscript's main results to the exact scripts that
produce them, records the compute environment used, and states which data can
be redistributed and which cannot (with the reasons and the workarounds).

---

## 1. Software environment used for the paper

| Item | Value |
|---|---|
| OS | Windows 10 x64 |
| Python | 3.12.7 (64-bit) |
| Key packages | langgraph 1.2.9, numpy 1.26.4, pandas 2.2.2, scipy 1.13.1, scikit-learn 1.5.1, matplotlib 3.9.2, openpyxl 3.1.5, xlrd 2.0.2, Pillow 12.3.0, PyMuPDF 1.26.3, requests 2.32.3 (see [`requirements.txt`](../requirements.txt)) |
| Local inference | Ollama / llama.cpp `llama-server`, GGUF-quantized models, context window 32768 |
| Cloud inference | OpenAI-compatible endpoints (DeepSeek, Zhipu BigModel) |
| Hardware (local arms) | Single consumer GPU; ≥ 16 GB VRAM for 8B–14B, ≥ 24 GB VRAM for 24B–30B models |
| Hardware (cloud arm) | Any modern workstation; no GPU required |

Determinism notes:

- Inference uses `temperature=0.1`; structured outputs (planning JSON) are
  stable but not bit-identical across backend versions — expect small
  numerical variation in LLM-driven scores when re-running with a different
  model build.
- Everything outside the LLM is deterministic: parsers, tool computations,
  scoring scripts, and the rule-based planner (`--no-llm`) reproduce exactly.
- Benchmark construction uses fixed random seeds (`--seed`, default 42) and
  atomic 0.5°/1° spatial block splits to prevent train/test leakage.

## 2. Sanity path (5 minutes, no downloads)

```bash
pip install -r requirements.txt
python scripts/一键运行全部测试.py
```

Runs: architecture four-plane self-test, real execution of all 110 registered
tools, the USGS soil capability suite (11 checks incl. CO-prediction
end-to-end), benchmark smoke tests over the shipped exam fixtures, and the
capability-paper smoke run. Paper correspondence: Task E row of the
supplementary table (103/110 → current 110/110 tools; soil 11/11).

## 3. Main results → scripts

All scripts live in [`scripts/`](../scripts/) (Chinese filenames; a full
inventory with English descriptions is in
[scripts/README.md](../scripts/README.md)). Standard pattern per task:
**build** the exam → **run** the agent → **score**.

### Task A — Genetic-type reasoning, 652 cases (GTK FODD, Finland)

| Paper result | Script chain |
|---|---|
| Accuracy / hierarchical score / Macro-F1 over 18 genetic types; hallucination rate 3.2%; over-confidence ECE 0.294 (strongest arm) | Build: `构建成因推理基准.py --deposits <nodd_all_deposits.csv>` → Run: `运行成因推理评测.py --split test` → Score: `成因推理评分.py --answers <genetic>/private/answers.json --predictions <pred.jsonl> --split test` |
| Paper figure: Task A exam report | `生成TaskA试卷报告.py` → `taska_report.html` |

### Task B / Task D-Gawler — Prospectivity targeting (Gawler, Australia)

| Paper result | Script chain |
|---|---|
| Agent capture@10% = 11% (random = 10%), rank-AUC 0.614; bare-LLM sliding-window ≈ random | Build: `构建Gawler成矿预测基准.py` → Run: `运行Gawler成矿预测评测.py` → Score: `成矿预测评分.py`; process scoring: `运行Gawler过程评测.py` |

### Task C — District-scale report generation (Stibnite–Yellow Pine, USA)

| Paper result | Script chain |
|---|---|
| Rubric scores (avg 50/100 strongest arm; per-dimension and failure-mode breakdown) | Build: `构建Stibnite矿区基准.py` → Run: `运行Stibnite评测.py` → Score: `Stibnite评分.py` (rubric anchors in the district's `hidden_reference/`) |

### Task D — Capability papers (NDI East Tennant, Footprints, Gawler multi-commodity)

| Paper result | Script chain |
|---|---|
| NDI lithogeochem hierarchical score, evidence-grounding rate 95.7%, TAS acidity 100%; downhole Au intervals; Footprints literature synthesis; Gawler U/Ni/Ag/Zn multi-commodity | Build: `构建能力考卷.py` → Run: `运行能力考卷.py <paper> --tag <tag>` → Score: `能力考卷评分.py all --tag <tag>` |

### Five-model × eight-arm ablation (case-library suites F/G, 18 tasks)

| Paper result (Table: arms A0–A6, A123 × 5 models) | Script chain |
|---|---|
| Full agent A0: 10–13/18 PASS, avg check-pass 0.80–0.91; bare LLM A4: 2–6/18 | A0/A1/A2/A3/A123: `案例库基准运行.py --model <model> --ablation <A0|A1|A2|A3|A12|A13|A23|A123>` (`--judge-only` re-grades existing runs) · A4 bare LLM: `案例库基准裸模型.py` · A5 pure ReAct + same tools: `案例库基准原生ReAct.py` · A6 ReAct + three constraints: `案例库基准ReAct三约束.py` → Aggregate: `汇总八臂消融表.py` |
| Paper figures: ablation charts | `生成消融八臂图.py`, `生成消融均分图.py`, `生成类型级F1合并图.py`, `生成三臂对比柱状图.py` |

Models used per column: Qwen3-VL-8B, Qwen2.5-14B, Qwen3-VL-30B, Qwen3.6-27B
(local GGUF via Ollama/llama-server) and DeepSeek-V4-Pro (cloud API).

### Formal orchestration and supplementary tables

```bash
# Formal batch evaluation (Tasks D -> C -> A), resumable, tagged:
python scripts/运行正式评测.py --tag <model-tag>          # --parts taskd|taskc|taska
# Supplementary score master table (S2):
python scripts/生成投稿附件成绩总表.py
# Results dashboards / ledgers:
python scripts/生成基准成绩看板.py && python scripts/测试结果台账.py
```

## 4. Data availability statement

### 4.1 Public benchmark data (redistributable under source licenses; download from the publishers)

| Suite | Source & license | Access |
|---|---|---|
| Task A: GTK Fennoscandian Ore Deposit Database (FODD) | Geological Survey of Finland — GTK Basic Licence | <https://hakku.gtk.fi/en/locations/?id=109> (map service: <https://gtkdata.gtk.fi/nodd/>) |
| Task B/D: ExploreSA Gawler Challenge package | Government of South Australia — CC BY 4.0 | <https://data.sa.gov.au/data/dataset/exploresa-gawler-challenge-datasets> (practical repack: <https://github.com/natbutter/gawler-exploration>) |
| Task C: USGS Stibnite–Yellow Pine data releases | U.S. Geological Survey — public domain (CC0) | <https://doi.org/10.5066/P91ACJSC> (ScienceBase; 3 releases, 29 attachments) |
| Task D: NDI / NVCL East Tennant geochemistry | Geoscience Australia / NVCL public data | via the national data portals |
| Task D: NSERC-CMIC Footprints publications | NSERC-CMIC Footprints project — public access | project publication corpus |
| Task E: USGS soil geochemistry (CONUS, A/C horizons) | U.S. Geological Survey — public domain | USGS data release |
| Reference: DARPA MinMod knowledge graph | USC ISI — MIT License | <https://github.com/usc-isi-i2/ta2-minmod-data> |

The **exam-construction pipelines** for all of the above are included in
`scripts/`, so every exam paper used in the paper can be rebuilt from the
original downloads (fixed seeds, documented splits). The private
ground-truth folders (`answers.json`, `hidden_deposits.csv`,
`scoring_rubric.md`) are scorer-side material and are excluded from model
context by design.

### 4.2 Data that cannot be redistributed (limitation statement)

1. **Case-library geochemistry for the F/G suites** (Lianyunshan lithium
   deposit, Huangjindong gold deposit: whole-rock major/trace elements,
   muscovite / apatite / zircon trace elements, zircon U–Pb ages,
   pyrite/apatite chemistry). These originate
   from the authors' collaborative research program and third-party
   restrictions; they are **not** covered by a license permitting
   redistribution and are therefore **not included** in this repository.
   Reasonable workarounds, all supported by the included code:
   - the agent runs unchanged on **any** user-supplied Excel/CSV/text/image
     dataset in the same form (see Tutorial 2); the shipped synthetic
     fixtures (`test_input/`, test suites) demonstrate the expected formats;
   - the **ablation protocol** (8 arms × N models) is fully scripted and can
     be applied to any user-defined task matrix via `案例库基准运行.py` and
     companions;
   - aggregated, non-redistributable **scores** (per-arm completion rates,
     check-pass rates, correctness counts) are reproduced in the paper's
     tables and regenerable from the run archives.
2. **Full run archives** (≈ 800 `final_state.json` replays used by the web
   demo) are several GB and are not included; regenerate them by re-running
   the evaluation scripts.
3. **Cloud model access**: the DeepSeek / BigModel API arms require the
   reader's own API credentials; results depend on the provider's serving
   stack at run time. The local-model arms avoid this dependency entirely.

### 4.3 Known run-to-run variation

The eight-arm ablation numbers in the paper were produced in the batch
manner documented in `测试案例库/基准/消融八臂汇总表.md` (same-day batches per
model, same GGUF build, thinking mode disabled uniformly). Re-runs with
different backend builds or quantizations will shift absolute scores by a
few points; the paper's qualitative ordering (A0 full agent > single-mechanism
removals > A123 rule fallback > A4 bare LLM) is robust across all five
models tested.

## 5. Where the outputs land

| Evaluation step | Output location |
|---|---|
| Per-case predictions | `按地区整合/<region>/测试结果_*/运行输出/...` (e.g. Task A `predictions.jsonl`) |
| Scores and summaries | sibling `测试结果_*` folders, plus `S2_成绩总表.xlsx` via `生成投稿附件成绩总表.py` |
| Ablation grades | `测试案例库/基准/` (per-arm score sheets; `汇总八臂消融表.md`) |
| Single agent runs | the `--out` directory (`report.md`, `final_state.json`, figures) |
