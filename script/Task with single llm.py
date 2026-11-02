# -*- coding: utf-8 -*-
"""Case-library benchmark · bare-LLM baseline: the same 18 tasks, no Agent framework,
data delivered in one shot, single-turn direct answer.

Shares 任务定义.json and the judge with 案例库基准运行.py (the Agent version); the only
difference is execution:
- Data serialization goes through the **same ingestion layer** (regularized tables after
  parse), each table truncated to budget (40 rows / 8000 chars) with truncation noted —
  the bare model sees a static snapshot and has no multi-round lookup ability;
- One chat call, direct answer (no tools, no figures, no multi-turn);
- Judging reuses the Agent version's judge_task: the pseudo final_state has empty
  tool_calls/figures, so tools_any/tool_ok/artifact fail structurally — that capability
  gap is exactly what the baseline measures;
- The report splits checks into **content checks** (text_all/text_any/number_near, fairly
  comparable with the Agent) and **execution checks** (tools_any/tool_ok/artifact),
  tallied separately.

Usage:
    MPLBACKEND=Agg python scripts/案例库基准裸模型.py --base http://localhost:11434/v1 \
        --model qwen3.6:27b --tag qwen3.6-27b [--tasks F01,G02] [--force] [--list]
Output: 测试案例库/基准/运行_<tag>_裸/<任务ID>/{直答.md, final_state.json}
      测试案例库/基准/成绩单_<tag>_裸.md + 成绩_<tag>_裸.json
"""

from __future__ import annotations  # Deferred evaluation of type annotations

import argparse  # Command-line arguments
import json  # Task definitions and score persistence
import os  # Environment variable injection (must precede importing oregenesis)
import sys  # Project-root injection
import time  # Timing
from pathlib import Path  # Path handling

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Project root (Agent_New)
if str(PROJECT_ROOT) not in sys.path:  # Avoid duplicate insertion
    sys.path.insert(0, str(PROJECT_ROOT))  # Ensure oregenesis and same-directory scripts are importable
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))  # Judge reuses the Agent-version script

BASE_DIR = PROJECT_ROOT / "测试案例库"  # Case-library root
BENCH_DIR = BASE_DIR / "基准"  # Benchmark directory
DEF_PATH = BENCH_DIR / "任务定义.json"  # Task definitions (shared with the Agent version)

_ROW_LIMIT = 40  # Max data rows shown per table (beyond this, truncate and note)
_CHAR_LIMIT = 8000  # Per-table character budget (rows cut beyond this)
_TEXT_CAP = 4000  # Character cap for text archives (geological background)

CONTENT_KINDS = {"text_all", "text_any", "number_near"}  # Content checks (fairly comparable with the Agent)
EXEC_KINDS = {"tools_any", "tool_ok", "artifact"}  # Execution checks (structurally missing in the bare model)


def _load_def() -> dict:  # Load task definitions
    return json.loads(DEF_PATH.read_text(encoding="utf-8"))  # JSON load


def _dump_table(path: Path) -> str:  # Single-table serializer
    """Excel/CSV -> regularized CSV text (same ingestion layer as the Agent), truncated to budget."""
    from oregenesis.io.parser import make_asset, parse_asset  # Same ingestion layer (incl. transposed/multi-row header fixes)
    from oregenesis.io.data_models import IdAllocator  # ID allocator
    units = parse_asset(make_asset(str(path), alloc=IdAllocator()))  # Regularized parse
    blocks: list[str] = []  # One block per worksheet
    for u in units:  # Iterate worksheet units
        header = [str(h) for h in u.raw_data.get("header", [])]  # Regularized header
        rows = u.raw_data.get("rows", [])  # Regularized data rows
        if not header or not rows:  # Skip empty tables
            continue  # Next table
        ws = u.data_origin.get("worksheet", "")  # Worksheet name
        lines = [",".join(header)]  # CSV first line
        shown = 0  # Rows shown so far
        for r in rows:  # Row by row
            if shown >= _ROW_LIMIT:  # Row limit reached
                lines.append(f"…（另有 {len(rows) - shown} 行未展示）")  # Note the truncation
                break  # Stop
            lines.append(",".join(str(c) for c in r[:len(header)]))  # Data row
            shown += 1  # Count
        block = f"[工作表 {ws or Path(path).stem}：{len(rows)} 行 × {len(header)} 列]\n" + "\n".join(lines)  # Block text
        if len(block) > _CHAR_LIMIT:  # Character budget reached
            block = block[:_CHAR_LIMIT] + f"\n…（字符超预算截断，全表 {len(rows)} 行）"  # Note the truncation
        blocks.append(block)  # Collect
    return "\n\n".join(blocks) if blocks else "（表格解析为空）"  # Join


def _dump_files(case: dict, files: list[str]) -> str:  # Task data-package serializer
    """All input files of a task -> one static data text."""
    parts: list[str] = []  # File-block buffer
    for name in files:  # File by file
        p = BASE_DIR / case["dir"] / name  # Full file path
        if not p.exists():  # Note missing files as-is
            parts.append(f"### {name}\n（文件缺失）")  # Placeholder
            continue  # Next file
        if p.suffix.lower() in (".xlsx", ".xlsm", ".csv"):  # Spreadsheet type
            parts.append(f"### {name}\n{_dump_table(p)}")  # Serialize spreadsheet
        else:  # Text type (geological background)
            txt = p.read_text(encoding="utf-8", errors="replace")  # Read full text
            if len(txt) > _TEXT_CAP:  # Truncate over budget
                txt = txt[:_TEXT_CAP] + "…（文本超预算截断）"  # Note the truncation
            parts.append(f"### {name}\n{txt}")  # Collect
    return "\n\n".join(parts)  # Join


def main() -> None:  # Main entry
    ap = argparse.ArgumentParser()  # Argument parsing
    ap.add_argument("--base", default="http://localhost:11434/v1")  # LLM endpoint
    ap.add_argument("--model", default="qwen3.6:27b")  # Model name
    ap.add_argument("--key", default="env", help="API 密钥；默认 env=不注入、由 .env/环境变量提供（云端用）")  # Key source
    ap.add_argument("--interval", default="0", help="请求最小间隔秒；云端建议 2（本地 0=不限流）")  # Throttling
    ap.add_argument("--ctx", default="32768")  # Context window
    ap.add_argument("--tag", default="")  # Run tag (defaults to the model name)
    ap.add_argument("--tasks", default="")  # Task filter
    ap.add_argument("--force", action="store_true", help="已存在结果的任务也重跑")  # Force rerun
    ap.add_argument("--suffix", default="", help="成绩文件名后缀（如 v2；重判不覆盖旧成绩单）")  # Score filename suffix
    ap.add_argument("--suite", default="", help="套卷（空=F/G 主卷）")  # Suite
    ap.add_argument("--list", action="store_true", help="只打印任务矩阵")  # List the task matrix
    a = ap.parse_args()  # Parse

    os.environ["LLM_API_BASE"] = a.base  # Endpoint
    if a.key != "env":  # Inject only when a key is passed explicitly; default env=no override, provided by .env/existing environment variables (cloud)
        os.environ["LLM_API_KEY"] = a.key  # Inject key
    os.environ["LLM_MODEL_NAME"] = a.model  # Model name
    os.environ["LLM_NUM_CTX"] = a.ctx  # Context window
    os.environ["LLM_MIN_INTERVAL"] = a.interval  # Throttle interval
    os.environ["LLM_DISABLE_THINKING"] = "1"  # Disable thinking
    TAG = (a.tag or a.model.replace(":", "-").replace("/", "-")) + "_裸"  # Tag (bare-model suffix)

    from 案例库基准运行 import judge_task, _load_def  # noqa: E402 — judge and task definitions reused (import must come after the env vars)
    definition = _load_def(a.suite or "FG")  # Task definitions (routed by suite)
    cases = definition["cases"]  # Case table
    wanted = [t.strip() for t in a.tasks.split(",") if t.strip()] if a.tasks else None  # Filter
    tasks = [t for t in definition["tasks"] if not wanted or t["id"] in wanted]  # Effective tasks
    if a.list:  # Matrix listing only
        for t in tasks:  # Task by task
            print(f"{t['id']:>4} [{t['case'][:12]:<12}] {t['capability']:<10}")  # Row
        print(f"共 {len(tasks)} 任务")  # Count
        return  # Done

    from oregenesis.llm.base import LLMBackendConfig  # Backend config
    from oregenesis.llm.openai_compatible import OpenAICompatibleBackend  # Real backend
    from oregenesis.llm.gate import require_llm  # Gate
    cfg = LLMBackendConfig()  # Config (reads the env vars above)
    require_llm(cfg)  # Liveness probe
    backend = OpenAICompatibleBackend(cfg)  # Bare-model backend (single direct answer)
    print(f"裸模型对照：{cfg.model_name} @ {cfg.api_base}（无 Agent 框架、单次直答、判卷器与 Agent 版同一套）")

    system_prompt = ("你是严谨的矿床学家与地球化学数据分析专家。只依据用户给出的数据与材料用中文回答，"
                     "不编造数据；给出带置信度的明确判断——不确定通过置信度与不确定性说明表达，"
                     "不得以\"证据不足\"拒答；涉及数值时给出具体数字与口径（中位数/均值/范围）。")  # Constraints matching the Agent version's narrative layer
    SUITE = a.suite or "FG"  # Suite name (filename prefix: 测试集FG_)
    run_root = BENCH_DIR / f"测试集{SUITE}_运行_{TAG}"  # Output root
    rows: list[dict] = []  # Score rows
    for t in tasks:  # Task by task
        out_dir = run_root / t["id"]  # Output directory
        state_path = out_dir / "final_state.json"  # Pseudo final state (judge interface)
        t0, status = time.time(), ""  # Timing and status
        if state_path.exists() and not a.force:  # Resume from checkpoint
            status = "跳过（已有结果）"  # Skip
        else:  # First run or forced rerun: direct answer -> persist -> build pseudo final state
            try:  # A single task failing must not drag down the rest
                case = cases[t["case"]]  # Case
                data_text = _dump_files(case, t["files"])  # Static data snapshot
                prompt = f"{t['task']}\n\n【数据（已规整，与真实文件一致；超长部分已截断注明）】\n{data_text}"  # User message
                out_dir.mkdir(parents=True, exist_ok=True)  # Create directory
                resp = backend.chat([{"role": "user", "content": prompt}],  # Single direct answer (wrapped across lines)
                                    system_prompt=system_prompt, temperature=0.2, max_tokens=2400)  # Long-answer budget
                answer = (resp.content or "").strip()  # Take the body
                if not answer:  # Empty reply
                    raise ValueError("裸模型返回空结果")  # Record failure
                (out_dir / "直答.md").write_text(  # Persist the direct answer (human-readable)
                    f"# 裸模型直答（{t['id']} {t['capability']}）\n\n{answer}\n", encoding="utf-8")  # Write
                fake = {"conclusion": answer[:400], "narrative": answer,  # Pseudo final state: conclusion = first 400 chars, narrative = full text
                        "tool_calls": [], "figures": [],  # Structurally missing in the bare model (no tools, no figures)
                        "review": {"passed": None, "problems": []}}  # Review not applicable
                state_path.write_text(json.dumps(fake, ensure_ascii=False, indent=1), encoding="utf-8")  # Persist final state
                status = f"直答完成 {time.time() - t0:.0f}s · {len(answer)} 字"  # Status
            except Exception as exc:  # noqa: BLE001 — log the failure and continue
                status = f"直答失败: {type(exc).__name__}: {exc}"  # Failure reason
        # ── Judging (reuses the Agent-version judge) ──
        if not state_path.exists():  # No final state (direct answer failed)
            print(f"[{t['id']}] {status}｜判卷: 无结果，记 0 分")  # Progress
            rows.append({"id": t["id"], "case": t["case"], "capability": t["capability"],  # Zero-score row
                         "pass": False, "score": 0.0, "content_rate": 0.0,  # Zero score
                         "status": status, "checks": [],  # Collect
                         "efficiency": {"rounds": 0, "tool_calls": 0, "failed_calls": 0,  # Bare model has no execution profile
                                        "fail_rate": 0.0, "repeat_decisions": 0, "auto_figures": 0}})  # Zero values, continued
            continue  # Next task
        fake = json.loads(state_path.read_text(encoding="utf-8"))  # Read the pseudo final state
        verdict = judge_task(t, fake)  # Same judge
        ok = "✅ PASS" if verdict["pass"] else "❌ FAIL"  # Marker
        # Content/execution group tallies (only content checks are fairly comparable with the Agent)
        kind_of = {c.get("desc", c.get("type")): c.get("type") for c in t["checks"]}  # desc -> type mapping
        cont = [(d, o) for d, o, _ in verdict["checks"] if kind_of.get(d) in CONTENT_KINDS]  # Content group
        exec_ = [(d, o) for d, o, _ in verdict["checks"] if kind_of.get(d) in EXEC_KINDS]  # Execution group
        cont_rate = sum(1 for _, o in cont if o) / max(len(cont), 1)  # Content-group pass rate
        print(f"[{t['id']}] {status}｜判卷: {ok} {verdict['score']:.0%}"
              f"（内容类 {sum(1 for _, o in cont if o)}/{len(cont)}，执行类 {sum(1 for _, o in exec_ if o)}/{len(exec_)}）")  # Progress
        rows.append({"id": t["id"], "case": t["case"], "capability": t["capability"],  # Score row
                     "pass": verdict["pass"], "score": verdict["score"],  # Score
                     "content_rate": cont_rate, "status": status, "checks": verdict["checks"],  # Details
                     "efficiency": verdict.get("efficiency", {})})  # Efficiency profile (v2; all zeros for the bare model)
    # ── Persist the report (with a comparison row vs the Agent version, if the same model's Agent scores exist) ──
    n_pass = sum(1 for r_ in rows if r_["pass"])  # PASS count
    avg = sum(r_["score"] for r_ in rows) / max(len(rows), 1)  # Overall average (including execution checks)
    cont_avg = sum(r_["content_rate"] for r_ in rows) / max(len(rows), 1)  # Content-group average (fair metric)
    agent_json = BENCH_DIR / f"测试集FG_成绩_v1_{a.model.replace(':', '-').replace('/', '-')}.json"  # Agent-version scores (if run; v1 metric)
    agent_line = ""  # Comparison line
    if agent_json.exists():  # Agent version exists
        try:  # Fault-tolerant read
            ag = json.loads(agent_json.read_text(encoding="utf-8"))  # Read
            agent_line = (f"\n- **与 Agent 版对照（同任务同判卷）**：Agent PASS {ag['pass_rate']:.0%}、"
                          f"平均 {ag['avg_score']:.0%}（见 测试集FG_成绩单_v1_{ag['tag']}.md）——差值即框架贡献")  # Comparison
        except Exception:  # noqa: BLE001
            pass  # Ignore
    case_label = {k: v["label"] for k, v in cases.items()}  # Case labels
    md = [f"# 案例库基准·裸模型成绩单（{a.model} · tag={TAG}）", "",  # Header
          f"- 运行时间: {time.strftime('%Y-%m-%d %H:%M')} · 任务数 {len(rows)} · "
          f"PASS {n_pass}（{n_pass / max(len(rows), 1):.0%}） · 平均检查通过率 {avg:.0%}", "",
          f"- **内容类检查平均通过率（与 Agent 公平可比）: {cont_avg:.0%}**"
          f"{agent_line}", "",
          "| 任务 | 案例 | 能力维度 | 检查通过 | 得分 | 内容类 | 结果 |", "|---|---|---|---|---|---|---|"]  # Table header
    for r_ in rows:  # One row per task
        n_total = len(r_["checks"])  # Total checks
        n_ok = round(r_["score"] * n_total)  # Passed count
        md.append(f"| {r_['id']} | {case_label.get(r_['case'], r_['case'])[:10]} | {r_['capability']} "  # Row
                  f"| {n_ok}/{n_total} | {r_['score']:.0%} | {r_['content_rate']:.0%} | {'✅' if r_['pass'] else '❌'} |")  # Row continued
    md += ["", "## 失败任务明细", ""]  # Failure details
    for r_ in rows:  # Task by task
        if r_["pass"]:  # Passed tasks are not expanded
            continue  # Skip
        md.append(f"### {r_['id']} {r_['capability']}（{r_['status']}）")  # Heading
        for desc, oki, note in r_["checks"]:  # Check by check
            if not oki:  # Failed item
                md.append(f"- ❌ {desc}｜{note}")  # Reason
        md.append("")  # Blank line
    SUF = f"_{a.suffix}" if a.suffix else ""  # Score filename suffix (re-judged v2 does not overwrite the old report)
    (BENCH_DIR / f"测试集{SUITE}_成绩单{SUF}_{TAG}.md").write_text("\n".join(md) + "\n", encoding="utf-8")  # md
    (BENCH_DIR / f"测试集{SUITE}_成绩{SUF}_{TAG}.json").write_text(  # json
        json.dumps({"model": a.model, "tag": TAG, "bare": True, "judge_version": 2, "suite": SUITE,  # Metadata
                    "time": time.strftime('%Y-%m-%d %H:%M'), "pass_rate": n_pass / max(len(rows), 1),  # Scores
                    "content_rate": cont_avg, "avg_score": avg, "rows": rows},  # Details
                   ensure_ascii=False, indent=1), encoding="utf-8")  # Write
    print(f"\n成绩单: {BENCH_DIR / f'测试集{SUITE}_成绩单{SUF}_{TAG}.md'}（PASS {n_pass}/{len(rows)}，"
          f"内容类 {cont_avg:.0%}，总平均 {avg:.0%}）")  # Wrap-up
    try:  # Archive + ledger (same as the Agent version): results are auto-filed into the batch folder and the test-results ledger refreshed
        from 测试结果台账 import sweep_and_ledger  # noqa: E402
        sweep_and_ledger()  # Archive and refresh the ledger
    except Exception as exc:  # noqa: BLE001 — a ledger failure does not affect scores
        print(f"[台账] 自动归档跳过: {exc}")  # Keep a trace


if __name__ == "__main__":  # Command-line entry
    main()  # Execute
