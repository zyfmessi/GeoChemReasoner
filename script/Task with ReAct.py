# -*- coding: utf-8 -*-
"""Case-library benchmark - native ReAct comparison arm (A5): textbook ReAct free-text loop over the same registered tools.

Comparison semantics (the seventh ablation arm; answers "what exactly is stronger than plain ReAct over the same tools"):
- Kept: the same data plane (init/parse/enrich nodes ingest and enrich), the same run implementations of the
  108 registered tools, the same round budget (6 rounds), the same grader judge_task, the same data visibility
  (the planner's state-summary function);
- Removed: the structured decision object (replaced by Thought/Action/Action Input free-text parsing), registry
  interception (fabricated tool names just get an ordinary error), the data gate (calls spec.run directly, no
  applicable check), the evidence ledger (no seed evidence, no EvidenceUnit, no review/audit nodes);
- Grading compatibility: the pseudo final_state carries audit.decisions and top-level tool_calls (all
  model-initiated requests, auto_figures=0), so judge_task grades and computes efficiency unchanged.

Usage:
    MPLBACKEND=Agg python scripts/案例库基准原生ReAct.py --base http://localhost:11434/v1 \
        --model qwen3.6:27b --tag qwen3.6-27b-ReAct [--tasks F01] [--force] [--list]
Output: 测试案例库/基准/测试集FG_运行_<tag>_ReAct/<任务ID>/{报告.md, final_state.json}
      测试案例库/基准/成绩单_<tag>_ReAct.md + 成绩_<tag>_ReAct.json
"""

from __future__ import annotations  # deferred evaluation of type annotations

import argparse  # command-line arguments
import json  # parameter parsing and score persistence
import os  # environment variable injection (must happen before importing oregenesis)
import re  # Action/Action Input parsing
import sys  # project root injection
import time  # timing
from pathlib import Path  # path handling

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # project root (Agent_New)
if str(PROJECT_ROOT) not in sys.path:  # avoid duplicate injection
    sys.path.insert(0, str(PROJECT_ROOT))  # ensure oregenesis and same-directory scripts are importable
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))  # the grader reuses the Agent-version script

BASE_DIR = PROJECT_ROOT / "测试案例库"  # case-library root
BENCH_DIR = BASE_DIR / "基准"  # benchmark directory
ROUND_MAX = 6  # round budget (matches the Agent version's --rounds default)

SYS_BASE = ("你是严谨的矿床学家与地球化学数据分析专家。只依据给出的数据与工具结果用中文回答，"
            "不编造数据；涉及数值时给出具体数字与口径（中位数/均值/范围）。")  # base persona (same caliber as the Agent version's narrative layer)

SYS_REACT = (SYS_BASE + "\n你处于 ReAct 循环中，每轮严格按以下三行格式输出：\n"
             "Thought: <一句话：当前缺什么、为什么选这个工具>\n"
             "Action: <工具名（来自可用工具列表）或 finish>\n"
             "Action Input: <JSON 对象，如 {} 或 {\"unit_ids\": []}>\n"
             "每次只调用一个工具；系统会返回 Observation。数据足以回答时用 Action: finish 结束取证。"
             "不要输出格式以外的内容。")  # ReAct loop instruction (textbook three-line format)


def _tool_catalog() -> str:  # tool catalog text (names + descriptions only — plain ReAct cannot see applicable metadata)
    from oregenesis.tools import REGISTRY  # registry singleton
    return "\n".join(f"- {n}: {s.description}" for n, s in sorted(REGISTRY._specs.items()))  # name + description lines


def _norm_artifacts(arts) -> list[str]:  # normalize the artifacts field into a path list (tolerates dict/list/str)
    out: list[str] = []  # path collection
    if isinstance(arts, dict):  # dict shape: recurse over values
        for v in arts.values():  # per value
            out.extend(_norm_artifacts(v))  # recurse
    elif isinstance(arts, list):  # list shape
        for v in arts:  # per item
            out.extend(_norm_artifacts(v))  # recurse
    elif isinstance(arts, str) and arts:  # a string is a path
        out.append(arts)  # collect
    return out  # path list


def run_react_task(backend, case: dict, t: dict, out_dir: Path) -> tuple[str, dict]:  # one-task ReAct run
    """Run the native ReAct loop for one task; returns (status description, pseudo final state)."""
    from oregenesis.reasoning.graph import make_nodes  # data-plane nodes (init/parse/enrich reused)
    from oregenesis.reasoning.state import initial_state  # initial-state factory
    from oregenesis.evidence import EvidenceLedger  # the node factory needs a ledger argument (A5 does not use one; placeholder)
    from oregenesis.tools import REGISTRY, ToolCall  # registry and tool call record
    from oregenesis.reasoning.planner import _state_summary  # the same data summary as the Agent version's planner (fair visibility)

    files = [str(BASE_DIR / case["dir"] / f) for f in t["files"]]  # full input file paths
    state = initial_state(files, t["task"], max_rounds=ROUND_MAX)  # initial state (task/rounds identical to the Agent version)
    state["backend"] = backend  # backend injection (a few tools fetch it from the state)
    out_dir.mkdir(parents=True, exist_ok=True)  # output directory
    state["task"]["output_dir"] = str(out_dir)  # where tool artifacts are written (same semantics as the Agent version)
    state["task"]["make_plots"] = True  # figure switch same as the Agent version
    nodes = make_nodes(EvidenceLedger(), backend=backend)  # node closures (the ledger is only a placeholder, never enters the loop)
    state = nodes["init"](state)  # register assets
    state = nodes["parse"](state)  # ingest into DataUnits (same data plane as the Agent version)
    state = nodes["enrich"](state)  # enrichment and conservative association (data-organization layer, not one of the three mechanisms)

    messages = [{"role": "user", "content":  # first user message: task + tool catalog + data overview (visibility matches the Agent planner)
        f"{t['task']}\n\n【可用工具】\n{_tool_catalog()}\n\n【数据概况】\n{_state_summary(state)}"}]
    decisions: list[dict] = []  # initiated-request records (the "requested by decisions" basis for tools_any grading)
    tool_calls: list[dict] = []  # call records (with status/observation/artifacts)
    figures: list[dict] = []  # artifact paths (basis for the artifact grading check)
    n_calls, final_text = 0, ""  # call counter and final conclusion
    transcript: list[str] = [f"# 任务 {t['id']}（原生 ReAct）\n\n## 任务\n{t['task']}\n"]  # human-readable transcript

    for rnd in range(1, ROUND_MAX + 1):  # ReAct loop (budget identical to the Agent version)
        resp = backend.chat(messages, system_prompt=SYS_REACT, temperature=0.1, max_tokens=1000)  # decision round
        txt = (resp.content or "").strip() or "（空回复）"  # fault tolerance
        transcript.append(f"### 轮 {rnd}\n{txt}\n")  # transcript
        messages.append({"role": "assistant", "content": txt})  # into the history
        m_act = re.search(r"Action\s*[:：]\s*(.+)", txt)  # action line
        action = m_act.group(1).strip().strip("`\"' ") if m_act else ""  # tool name
        m_in = re.search(r"Action Input\s*[:：]\s*(\{.*\})", txt, re.S)  # parameters JSON
        params: dict = {}  # default empty parameters
        if m_in:  # parse only when a parameter line exists
            try:  # a JSON parse failure is treated as empty parameters (vanilla: no structured validation retry)
                params = json.loads(m_in.group(1))  # parse
                if not isinstance(params, dict):  # non-object counts as empty parameters
                    params = {}  # reset to empty
            except Exception:  # noqa: BLE001
                params = {}  # reset to empty
        if not action or action.lower() in ("finish", "stop", "final_answer"):  # finish signal received
            break  # break out into synthesis
        n_calls += 1  # count the call
        decisions.append({"next_action": action, "parameters": params})  # decision record (grading compatibility)
        try:  # execution: no registry interception (KeyError is an ordinary error), no gate (direct spec.run)
            spec = REGISTRY.get(action)  # look up the tool
        except KeyError:  # unregistered
            obs = f"Error: 工具 '{action}' 不存在。请改用可用工具，或 Action: finish。"  # ordinary error
            tool_calls.append({"tool": action, "status": "error", "observation": obs, "artifacts": []})  # record
            messages.append({"role": "user", "content": f"Observation: {obs}"})  # feed back
            continue  # next round
        call = ToolCall(tool_call_id=f"toolcall_{n_calls:03d}", tool_name=action, parameters=params)  # call record
        try:  # execute directly (A5 semantics: no applicable gate check)
            payload = spec.run(state, call) or {}  # real deterministic computation
            obs = str(payload.get("observation") or payload)  # observation text
            arts = _norm_artifacts(payload.get("artifacts"))  # normalize artifacts
            status = "success"  # success
        except Exception as exc:  # noqa: BLE001 — execution exceptions are fed back as ordinary errors
            obs, arts, status = f"Error: {type(exc).__name__}: {exc}", [], "error"  # failure
        for p in arts:  # artifacts go into figures (for the artifact grading check)
            figures.append({"path": p})  # collect
        tool_calls.append({"tool": action, "status": status, "observation": obs, "artifacts": arts})  # record
        messages.append({"role": "user", "content": f"Observation: {obs[:2000]}"})  # feed the observation back (truncated to avoid blowing the context window)

    if not final_text:  # no conclusion produced (rounds exhausted or natural end): do one extra synthesis direct answer
        messages.append({"role": "user", "content": "请现在输出最终综合结论：包含结论判断（带置信度）、关键证据数值、替代解释与数据缺口。"})  # wrap-up instruction
        resp2 = backend.chat(messages, system_prompt=SYS_BASE, temperature=0.2, max_tokens=2400)  # synthesis direct answer
        final_text = (resp2.content or "").strip() or "（未产出结论）"  # fault tolerance
        transcript.append(f"### 综合结论\n{final_text}\n")  # transcript
    fake = {"conclusion": final_text[:400], "narrative": final_text,  # pseudo final state (grading interface per the bare-model convention)
            "tool_calls": tool_calls, "figures": figures,  # real calls and artifacts
            "audit": {"decisions": decisions, "tool_calls": tool_calls},  # data source for the efficiency profile
            "review": {"passed": None, "problems": []}}  # A5 has no review node
    (out_dir / "报告.md").write_text("\n".join(transcript), encoding="utf-8")  # write the human-readable transcript
    return f"ReAct完成（轮{len(decisions)} 调{n_calls}）", fake  # status and final state


def main() -> None:  # main entry
    ap = argparse.ArgumentParser()  # argument parsing
    ap.add_argument("--base", default="http://localhost:11434/v1")  # LLM endpoint
    ap.add_argument("--model", default="qwen3.6:27b")  # model name
    ap.add_argument("--key", default="env", help="API 密钥；默认 env=由 .env/环境变量提供（云端用）")  # key
    ap.add_argument("--interval", default="0", help="请求最小间隔秒；云端建议 2")  # throttling
    ap.add_argument("--ctx", default="32768")  # context window
    ap.add_argument("--tag", default="")  # run tag
    ap.add_argument("--tasks", default="")  # task filter
    ap.add_argument("--force", action="store_true", help="已存在结果的任务也重跑")  # force rerun
    ap.add_argument("--suite", default="", help="套卷（空=F/G 主卷）")  # suite
    ap.add_argument("--list", action="store_true", help="只打印任务矩阵")  # print matrix
    ap.add_argument("--think-budget", type=int, default=0, help="弱模型服务档：>0 时开启思考并把调用 max_tokens 抬到该值下限（推理进独立字段，正文留可解析文本）")  # weak-model service tier
    a = ap.parse_args()  # parse

    os.environ["LLM_API_BASE"] = a.base  # endpoint
    if a.key != "env":  # inject explicit key
        os.environ["LLM_API_KEY"] = a.key  # inject
    os.environ["LLM_MODEL_NAME"] = a.model  # model name
    os.environ["LLM_NUM_CTX"] = a.ctx  # window
    os.environ["LLM_MIN_INTERVAL"] = a.interval  # throttling
    os.environ.setdefault("LLM_DISABLE_THINKING", "1")  # disable thinking (overridable via the environment)
    if a.think_budget > 0:  # weak-model service tier: enable thinking + raise the max_tokens floor of all calls in this script
        os.environ["LLM_DISABLE_THINKING"] = "0"  # reasoning goes into a separate field; the body stays parseable
    TAG = (a.tag or a.model.replace(":", "-").replace("/", "-")) + "_ReAct"  # tag (A5 suffix)

    from 案例库基准运行 import judge_task, _load_def  # noqa: E402 — grader and task definition reused
    definition = _load_def(a.suite or "FG")  # task definition
    cases = definition["cases"]  # case table
    wanted = [t.strip() for t in a.tasks.split(",") if t.strip()] if a.tasks else None  # filter
    tasks = [t for t in definition["tasks"] if not wanted or t["id"] in wanted]  # effective tasks
    if a.list:  # list the matrix only
        for t in tasks:  # per task
            print(f"{t['id']:>4} [{t['case'][:12]:<12}] {t['capability']:<10}")  # row
        print(f"共 {len(tasks)} 任务")  # count
        return  # done

    from oregenesis.llm.base import LLMBackendConfig  # backend config
    from oregenesis.llm.openai_compatible import OpenAICompatibleBackend  # real backend
    from oregenesis.llm.gate import require_llm  # liveness gate
    cfg = LLMBackendConfig()  # read env vars
    require_llm(cfg)  # liveness probe
    backend = OpenAICompatibleBackend(cfg)  # backend
    if a.think_budget > 0:  # instance-level wrapper: max_tokens floor for all calls (the A5 backend is created in this script, so wrapping the instance covers everything)
        _orig_chat = backend.chat  # original method
        def _chat_tb(messages, **kw):  # wrapper
            kw["max_tokens"] = max(int(kw.get("max_tokens") or 0), int(a.think_budget))  # raise the floor
            return _orig_chat(messages, **kw)  # forward to the original method
        backend.chat = _chat_tb  # instance replacement
        print(f"[弱模型服务档] 思考开启 + max_tokens≥{a.think_budget}")  # progress
    print(f"原生 ReAct 对照臂：{cfg.model_name} @ {cfg.api_base}（自由文本循环+同套工具，无注册/门禁/账本/决策对象）")

    SUITE = a.suite or "FG"  # suite name
    run_root = BENCH_DIR / f"测试集{SUITE}_运行_{TAG}"  # output root
    rows: list[dict] = []  # score rows
    for t in tasks:  # per task
        out_dir = run_root / t["id"]  # per-task output directory
        state_path = out_dir / "final_state.json"  # final state
        t0, status = time.time(), ""  # timing
        if state_path.exists() and not a.force:  # resume from checkpoint
            status = "跳过（已有结果）"  # skip
        else:  # run
            try:  # one task failing must not break the rest
                case = cases[t["case"]]  # case
                status, fake = run_react_task(backend, case, t, out_dir)  # ReAct loop
                status += f" {time.time() - t0:.0f}s"  # elapsed time
                state_path.write_text(json.dumps(fake, ensure_ascii=False, indent=1), encoding="utf-8")  # write the final state
            except Exception as exc:  # noqa: BLE001
                status = f"运行失败: {type(exc).__name__}: {exc}"  # record the failure
        if not state_path.exists():  # no final state: score zero
            print(f"[{t['id']}] {status}｜判卷: 无结果，记 0 分")  # progress
            rows.append({"id": t["id"], "case": t["case"], "capability": t["capability"],  # zero-score row
                         "pass": False, "score": 0.0, "status": status, "checks": [],  # collect
                         "efficiency": {"rounds": 0, "tool_calls": 0, "failed_calls": 0,  # zero profile
                                        "fail_rate": 0.0, "repeat_decisions": 0, "auto_figures": 0}})  # continued
            continue  # next task
        fake = json.loads(state_path.read_text(encoding="utf-8"))  # read the final state
        verdict = judge_task(t, fake)  # the same grader
        ok = "✅ PASS" if verdict["pass"] else "❌ FAIL"  # marker
        eff = verdict["efficiency"]  # efficiency
        print(f"[{t['id']}] {status}｜判卷: {len(verdict['checks'])} 项通过 {sum(1 for _, o, _ in verdict['checks'] if o)}｜{ok} {verdict['score']:.0%}"
              f"｜轮{eff['rounds']} 调{eff['tool_calls']} 败{eff['failed_calls']} 重{eff['repeat_decisions']}")  # progress
        rows.append({"id": t["id"], "case": t["case"], "capability": t["capability"],  # score row
                     "pass": verdict["pass"], "score": verdict["score"],  # score
                     "status": status, "checks": verdict["checks"], "efficiency": eff})  # details
    # ── Score persistence ──
    n_pass = sum(1 for r_ in rows if r_["pass"])  # PASS count
    avg = sum(r_["score"] for r_ in rows) / max(len(rows), 1)  # average score
    case_label = {k: v["label"] for k, v in cases.items()}  # case labels
    md = [f"# 案例库基准·原生 ReAct 成绩单（{a.model} · tag={TAG}）", "",  # header
          "- 口径: A5 原生 ReAct 臂（同数据面/同工具/同轮数/同判卷；无决策对象、无注册拦截、无门禁、无证据账本）",  # criteria
          f"- 运行时间: {time.strftime('%Y-%m-%d %H:%M')} · 任务数 {len(rows)} · PASS {n_pass}（{n_pass / max(len(rows), 1):.0%}）"
          f" · 平均检查通过率 {avg:.0%}", "",  # overview
          "| 任务 | 案例 | 能力维度 | 检查通过 | 得分 | 轮数 | 调用 | 失败 | 结果 |", "|---|---|---|---|---|---|---|---|---|"]  # table header
    for r_ in rows:  # per row
        n_total = len(r_["checks"])  # total checks
        n_ok = round(r_["score"] * n_total)  # number passed
        eff = r_.get("efficiency") or {}  # profile
        md.append(f"| {r_['id']} | {case_label.get(r_['case'], r_['case'])[:10]} | {r_['capability']} "  # row
                  f"| {n_ok}/{n_total} | {r_['score']:.0%} | {eff.get('rounds', '-')} | {eff.get('tool_calls', '-')} "
                  f"| {eff.get('failed_calls', '-')} | {'✅' if r_['pass'] else '❌'} |")  # row continued
    md += ["", "## 失败任务明细", ""]  # failure details
    for r_ in rows:  # per task
        if r_["pass"]:  # passed ones are not expanded
            continue  # skip
        md.append(f"### {r_['id']} {r_['capability']}（{r_['status']}）")  # title
        for desc, oki, note in r_["checks"]:  # per check
            if not oki:  # failed items
                md.append(f"- ❌ {desc}｜{note}")  # reason
        md.append("")  # blank line
    (BENCH_DIR / f"测试集{SUITE}_成绩单_{TAG}.md").write_text("\n".join(md) + "\n", encoding="utf-8")  # write the md report
    (BENCH_DIR / f"测试集{SUITE}_成绩_{TAG}.json").write_text(  # write the json scores
        json.dumps({"model": a.model, "tag": TAG, "arm": "A5-native-ReAct", "suite": SUITE,  # meta info
                    "time": time.strftime('%Y-%m-%d %H:%M'), "judge_version": 2,  # criteria
                    "pass_rate": n_pass / max(len(rows), 1), "avg_score": avg, "rows": rows},  # scores
                   ensure_ascii=False, indent=1), encoding="utf-8")  # write
    print(f"\n成绩单: {BENCH_DIR / f'测试集{SUITE}_成绩单_{TAG}.md'}（PASS {n_pass}/{len(rows)}，平均 {avg:.0%}）")  # wrap up


if __name__ == "__main__":  # entry
    main()  # run
