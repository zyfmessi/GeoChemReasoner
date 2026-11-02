# -*- coding: utf-8 -*-
"""Case-library benchmark runner: runs the agent task-by-task per 测试案例库/基准/任务定义.json and auto-grades.

Each task = one independent agent run (minimal input-file subset + short task text);
grading text domain = conclusion + narrative + all tool observations (tool-derived results also count as done);
seven check types: tools_any / tool_ok / text_all / text_any / number_near / number_sane / artifact;
efficiency profile is reported separately (rounds/calls/failures/repeat decisions/auto figures) and does not affect PASS.

v2 criteria (2026-10-19): tool-type checks (tools_any/tool_ok) only count calls requested by decisions;
graph.py auto figure generation (toolcall_auto_ree) no longer earns score for the agent; artifact supports min_kb.

Usage:
    MPLBACKEND=Agg python scripts/案例库基准运行.py --base http://localhost:11434/v1 \
        --model qwen3.6:27b --tag qwen3.6-27b            # all 18 tasks
    python scripts/案例库基准运行.py --tasks F03,G02 --tag smoke   # run only a subset of tasks
    python scripts/案例库基准运行.py --tag xxx --judge-only --suffix v2   # re-grade under v2 criteria (keeps the old report)
    python scripts/案例库基准运行.py --list                       # print the task matrix only
Output: 测试案例库/基准/运行_<tag>/<任务ID>/{report.md, final_state.json}
      测试案例库/基准/成绩单_<tag>.md + 成绩_<tag>.json
"""

from __future__ import annotations  # deferred evaluation of type annotations

import argparse  # command-line arguments
import json  # task definitions and score persistence
import os  # environment variable injection (must happen before importing oregenesis)
import re  # grading regexes
import sys  # project root injection
import time  # timing
from pathlib import Path  # path handling

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # project root (Agent_New)
if str(PROJECT_ROOT) not in sys.path:  # avoid duplicate injection
    sys.path.insert(0, str(PROJECT_ROOT))  # ensure oregenesis is importable

BASE_DIR = PROJECT_ROOT / "测试案例库"  # case-library root
BENCH_DIR = BASE_DIR / "基准"  # benchmark directory


def _patch_a1(backend) -> str:  # remove the registry constraint (fabricated tool names are not rejected; the LLM generates the values directly, marked model-assisted)
    from oregenesis.tools.registry import ToolRegistry
    _orig_execute = ToolRegistry.execute
    def _execute_a1(self, call, state):
        try:  # proactive probe: is the tool name registered? (the original execute turns an unregistered KeyError into an error result; must intercept before that)
            self.get(call.tool_name)
            registered = True  # registered
        except KeyError:  # unregistered
            registered = False  # mark it
        if registered:  # registered tool: take the original execution path (real deterministic computation)
            return _orig_execute(self, call, state)  # ← fix: this path previously lacked a return and silently returned None
        if not registered:  # unregistered tool: the LLM directly produces the "tool" result itself (no deterministic computation)
            from oregenesis.tools.registry import ToolResult
            prompt = (f"任务需要执行地学计算工具 '{call.tool_name}'（参数：{call.parameters}）。"
                      "系统中不存在该工具。请直接以地球化学专家身份完成该计算，"
                      "用中文输出观测结果文本，必须包含具体数值与统计口径。")
            resp = backend.chat([{"role": "user", "content": prompt}],
                                system_prompt="你是严谨的矿床学家，直接给出计算结果与数值。",
                                temperature=0.2, max_tokens=500)
            obs = (resp.content or "").strip() or "（模型未给出结果）"
            return ToolResult(tool_call_id=call.tool_call_id, tool_name=call.tool_name,
                              status="success",
                              result={"observation": f"(model-assisted) {obs}"},
                              source_data_unit_ids=list(call.input_data_unit_ids),
                              warnings=["A1 消融：注册约束已移除，本结果由模型直接生成（model-assisted）"])
    ToolRegistry.execute = _execute_a1  # class-level patch (instance method lookup goes through the class)
    return "A1：移除注册约束（未注册工具由 LLM 直接生成数值）"


def _patch_a2() -> str:  # remove the data-condition check (calls rejected by the data gate now run the real tool as usual)
    from oregenesis.tools.registry import ToolRegistry
    _orig_execute = ToolRegistry.execute  # note: in combination arms (e.g. A123) this wraps A1's modified execute, so the stacking semantics are correct
    def _execute_a2(self, call, state):
        result = _orig_execute(self, call, state)  # original path first
        if result.status == "error" and "数据条件不满足" in (result.error or ""):  # hit a data-gate rejection
            spec = self.get(call.tool_name)  # fetch the tool spec back
            if spec.run:  # if a run function exists, force-run it anyway (missing columns still run; empty values/errors are recorded as usual)
                payload = spec.run(state, call)
                from oregenesis.tools.registry import ToolResult
                return ToolResult(tool_call_id=call.tool_call_id, tool_name=call.tool_name,
                                  status="success", result=payload,
                                  source_data_unit_ids=list(call.input_data_unit_ids),
                                  warnings=["A2 消融：数据条件检查已移除"] + list(payload.get("warnings", [])))
        return result  # not a gate rejection: return as-is
    ToolRegistry.execute = _execute_a2
    return "A2：移除数据条件检查（门禁拒绝的调用照常执行）"


def _patch_a3() -> str:  # remove evidence sync (the synthesis generator no longer sees seed/tool evidence)
    import oregenesis.tools.vlm as V
    _orig_synth = V.synthesize_analysis
    def _synth_a3(state, backend_):
        return _orig_synth({**state, "evidence": []}, backend_)  # call the original synthesis with evidence emptied
    V.synthesize_analysis = _synth_a3  # graph imports locally inside nodes, so swapping the current module attribute takes effect
    return "A3：移除证据同步（综合器只看任务文本与数据档案）"


def apply_ablation(kind: str, backend) -> str:  # runtime patches for ablation arms (benchmark-script layer; the agent package code is untouched)
    """A1=remove the registry constraint; A2=remove the data-condition check; A3=remove evidence sync.
    Combination arms A12/A13/A23/A123 stack patches in 1→2→3 order (A2 wraps A1's modified execute,
    forming the "registry/gate/evidence sync all off" execution loop — the A123 arm of paper §4.4)."""
    if kind == "A0":  # full agent: no patch
        return "A0：完整智能体（无消融补丁）"
    patches = {"1": lambda: _patch_a1(backend), "2": _patch_a2, "3": _patch_a3}  # arm → patch
    if kind == "A" + kind[1:] and kind[1:] and all(ch in patches for ch in kind[1:]):  # A1/A2/A3/A12/A13/A23/A123
        msgs = [patches[ch]() for ch in kind[1:]]  # stack in the fixed 1→2→3 order
        return "＋".join(msgs)
    raise ValueError(f"未知消融臂: {kind}（支持 A0/A1/A2/A3/A12/A13/A23/A123）")


DEF_FILES = {"FG": "测试集FG_任务定义.json"}  # suite → definition file (2026-10-19 naming convention)


def _load_def(suite: str = "FG") -> dict:  # load the task definition for the requested suite
    path = BENCH_DIR / DEF_FILES[suite]  # suite routing
    return json.loads(path.read_text(encoding="utf-8"))  # JSON load


def _efficiency(final_state: dict) -> dict:  # efficiency metric extraction (new in v2; reported separately, not counted toward PASS)
    """Extract the efficiency profile from the final state: decision rounds / tool calls / failure rate / repeat decisions / auto-generated figures.

    Background (F/G benchmark 2026-10-16 diagnosis): in G09-type failures the model is stuck on the same goal for
    4–6 rounds, retrying the same tool over and over (the anti-repeat gate blocks the third attempt); the old
    grading could not see this waste at all.
    Repeat decision = the same (next_action, parameters) fingerprint appearing >= 2 times — exactly the
    "same-parameter retry" the anti-repeat gate is meant to block; countable from the decision sequence
    (blocked calls never enter tool_calls).
    """
    audit = final_state.get("audit") or {}  # audit block
    dec = audit.get("decisions") or []  # decision sequence (each entry = one ReAct decision round)
    tc = final_state.get("tool_calls") or audit.get("tool_calls", []) or []  # tool call records
    failed = [t for t in tc if str(t.get("status", "")) != "success"]  # calls that failed execution
    seen: set = set()  # seen (action, parameters) fingerprint set
    repeat = 0  # repeat-decision counter
    for x in dec:  # fingerprint per decision
        key = (str(x.get("next_action") or ""),  # requested tool action
               json.dumps(x.get("parameters") or {}, sort_keys=True, ensure_ascii=False))  # parameters fingerprint (key order normalized)
        if key in seen:  # same fingerprint planned again
            repeat += 1  # count one wasted round
        seen.add(key)  # register the fingerprint
    requested = {str(x.get("next_action") or "") for x in dec}  # set of actions requested by decisions
    requested -= {"stop", ""}  # stop/empty are not tool calls
    has_dec = bool(dec)  # whether a decision sequence exists (needed for attribution; all-stop also counts as attributable — every call is then an auto figure)
    auto = [t for t in tc  # calls never requested by any decision
            if has_dec and str(t.get("tool") or t.get("tool_name") or "") not in requested]  # = pipeline auto figures
    n = max(len(tc), 1)  # avoid division by zero
    return {"rounds": len(dec), "tool_calls": len(tc),  # decision rounds and tool call count
            "failed_calls": len(failed), "fail_rate": round(len(failed) / n, 3),  # failure profile
            "repeat_decisions": repeat, "auto_figures": len(auto)}  # repeat decisions and auto figure count


def judge_task(task_def: dict, final_state: dict) -> dict:  # grade a single task
    """Judge one task's final_state against its checks; returns {pass, score, checks, efficiency}.

    Grading text domain = conclusion + narrative + all tool observations (tool-derived results also count as done).
    v2 criteria (2026-10-19, revised after the measured F/G diagnosis):
      1) Tool-type checks (tools_any/tool_ok) only count calls "requested by decisions" — graph.py
         auto figure generation (toolcall_auto_ree, fired as a fallback only when the decision trajectory
         never requested plot_ree_spider) no longer earns score for the agent; when the final state has no
         decision sequence, fall back to all calls (old states are not falsely penalized).
      2) New number_sane: every captured number must fall within [min,max] (specifically catches copying
         ingestion artifacts like "1258728 missing cells"; nothing captured = unverifiable, not penalized).
      3) artifact supports an optional min_kb: minimum artifact file size (blank figures earn no credit).
      4) Efficiency profile goes in the separate efficiency field (see _efficiency); does not affect PASS.
    """
    concl = str(final_state.get("conclusion") or "")  # conclusion text
    narr = str(final_state.get("narrative") or "")  # narrative text
    audit = final_state.get("audit") or {}  # audit block (tool observations live here)
    tc = final_state.get("tool_calls") or audit.get("tool_calls", []) or []  # tool call records (top level preferred)
    dec = audit.get("decisions") or []  # decision sequence (basis for telling auto figures apart)
    requested = {str(x.get("next_action") or "") for x in dec}  # actions requested by decisions
    requested -= {"stop", ""}  # exclude stop/empty
    has_dec = bool(dec)  # whether a decision sequence exists (needed for attribution: with all-stop, every call is an auto figure)
    agent_tc = [t for t in tc  # agent-initiated calls = those requested by decisions
                if (not has_dec) or str(t.get("tool") or t.get("tool_name") or "") in requested]  # fall back to all calls when there is no decision sequence
    n_auto = len(tc) - len(agent_tc)  # number of auto-figure entries removed (goes into notes for auditability)
    tool_names = [str(t.get("tool") or t.get("tool_name") or "") for t in agent_tc]  # sequence of agent-initiated tool names
    tool_status: dict[str, str] = {}  # tool name → last status
    obs: list[str] = []  # tool observation list (auto-figure observations are kept too: the content really exists and is citable)
    for t in tc:  # fault-tolerant field extraction per entry
        name = str(t.get("tool") or t.get("tool_name") or "")  # tool name (both old and new keys accepted)
        if name and ((not has_dec) or name in requested):  # record status only for agent-initiated calls
            tool_status[name] = str(t.get("status", ""))  # status (last one wins on duplicate names)
        obs.append(str(t.get("observation") or ""))  # observation text (all of it enters the text domain)
    auto_note = f"；剔除自动配图 {n_auto} 条" if n_auto else ""  # auto-figure removal note
    text = "\n".join([concl, narr] + obs)  # grading text domain: tool-derived results also count as done (lenient scope)
    figures = final_state.get("figures") or []  # figure artifacts
    fig_paths = " ".join(str(f.get("path", "")) for f in figures)  # artifact path string
    results: list[tuple[str, bool, str]] = []  # (description, passed, note)
    for c in task_def.get("checks", []):  # per check
        kind = c.get("type", "")  # check type
        ok, note = False, ""  # default: not passed
        try:  # an exception in one check must not break grading
            if kind == "tools_any":  # tool-name hit count (v2: agent-initiated calls only)
                hits = sum(1 for name in c["any"] if name in tool_names)  # hit count
                ok = hits >= c.get("min", 1)  # threshold met
                note = f"命中 {hits}/{len(c['any'])}（实际调用: {sorted(set(n for n in tool_names if n))[:6]}{auto_note}）"  # note: what was actually called
            elif kind == "tool_ok":  # the specified tool must succeed (v2: agent-initiated calls only)
                ok = tool_status.get(c["name"], "") == "success"  # status comparison
                miss = "（自动配图不计）" if n_auto and c["name"] not in tool_status else ""  # call it out when the tool only ran via auto figure
                note = f"{c['name']} 状态={tool_status.get(c['name'], '未调用')}{miss}{auto_note}"  # note
            elif kind == "text_all":  # all regexes must hit
                missing = [p for p in c["all"] if not re.search(p, text)]  # list of misses
                ok = not missing  # passes only when all hit
                note = f"未命中: {missing}" if missing else "全部命中"  # note
            elif kind == "text_any":  # any pattern hitting >= min
                hits = sum(1 for p in c["any"] if re.search(p, text))  # hit count
                ok = hits >= c.get("min", 1)  # threshold
                note = f"命中 {hits}/{len(c['any'])}"  # note
            elif kind == "number_near":  # numeric anchor with ± tolerance
                m = [float(x) for x in re.findall(c["pattern"], text)]  # capture all numbers
                ref, tol = float(c["ref"]), float(c.get("tol_pct", 5))  # anchor and tolerance
                ok = any(abs(x - ref) <= abs(ref) * tol / 100 for x in m)  # any value within tolerance
                note = f"抓到 {m[:6]} vs 锚点 {ref}±{tol}%"  # note
            elif kind == "number_sane":  # number sanity (new in v2): every captured number must fall inside the range
                m = [float(x.replace(",", "")) for x in re.findall(c["pattern"], text)]  # capture all numbers (thousands separators tolerated)
                lo, hi = float(c["min"]), float(c["max"])  # sane range
                bad = [x for x in m if x < lo or x > hi]  # out-of-range numbers (evidence of copied ingestion artifacts)
                ok = not bad  # passes only when something was captured and all values are in range; nothing captured = unverifiable, not penalized (pair with text_all to require the claim exists)
                note = (f"出界 {bad[:4]}（区间 [{lo:g},{hi:g}]）" if bad else  # out-of-range details
                        (f"抓到 {m[:6]} 全部在 [{lo:g},{hi:g}]" if m else "未抓到数值（无从核验）"))  # note
            elif kind == "artifact":  # artifact exists (v2: optional min_kb size floor)
                ok = c["name_contains"] in fig_paths  # path-string containment
                if ok and c.get("min_kb"):  # when the size floor is enabled
                    try:  # the file may not exist (figures recorded the path but the write failed)
                        size = max((os.path.getsize(p) for p in fig_paths.split()  # take the largest matching artifact's size
                                    if c["name_contains"] in p and Path(p).exists()), default=0)  # fallback 0
                        ok = size >= float(c["min_kb"]) * 1024  # size comparison
                        note = f"figures: {[Path(p).name for p in fig_paths.split()]}｜最大 {size // 1024}KB（下限 {c['min_kb']}KB）"  # note
                    except OSError as exc:  # unreadable counts as not passed
                        ok, note = False, f"产物体积核验失败: {exc}"  # exception note
                else:  # size floor not enabled: keep the old criteria
                    note = f"figures: {[Path(p).name for p in fig_paths.split()]}" if fig_paths else "无图件产物"  # note
        except Exception as exc:  # noqa: BLE001 — grading exception recorded as not passed, reason kept
            note = f"判卷异常: {exc}"  # exception note
        results.append((c.get("desc", kind), ok, note))  # collect
    n_pass = sum(1 for _, ok, _ in results if ok)  # number of passes
    return {"pass": n_pass == len(results), "score": n_pass / max(len(results), 1),  # task passes = all checks pass
            "checks": results, "efficiency": _efficiency(final_state)}  # details + efficiency profile (v2)


def main() -> None:  # main entry
    ap = argparse.ArgumentParser()  # argument parsing
    ap.add_argument("--base", default="http://localhost:11434/v1")  # LLM endpoint (defaults to local Ollama)
    ap.add_argument("--model", default="qwen3.6:27b")  # model name
    ap.add_argument("--key", default="env", help="API 密钥；默认 env=不注入、由 .env/环境变量提供（云端用）")  # key source
    ap.add_argument("--interval", default="0", help="请求最小间隔秒；云端建议 2（本地 0=不限流）")  # throttling
    ap.add_argument("--ctx", default="32768")  # context window
    ap.add_argument("--rounds", type=int, default=6)  # ReAct rounds (focused tasks are short; default 6 is enough)
    ap.add_argument("--tag", default="")  # run tag (defaults to the model name)
    ap.add_argument("--tasks", default="")  # task filter (comma-separated ids; empty = all)
    ap.add_argument("--force", action="store_true", help="已存在结果的任务也重跑")  # force rerun
    ap.add_argument("--judge-only", action="store_true", help="跳过运行，只对已有结果判卷")  # grading only
    ap.add_argument("--suite", default="", help="套卷（空=F/G 主卷），成绩目录/文件名自动加套卷前缀")  # suite
    ap.add_argument("--suffix", default="", help="成绩文件名后缀（如 v2；用于重判不覆盖旧成绩单）")  # score filename suffix
    ap.add_argument("--ablation", default="A0", choices=["A0", "A1", "A2", "A3", "A12", "A13", "A23", "A123"], help="消融臂：A0=完整智能体；A1=移除注册约束；A2=移除数据条件检查；A3=移除证据同步；A123=三机制全关")  # ablation switch
    ap.add_argument("--list", action="store_true", help="只打印任务矩阵不运行")  # print matrix
    ap.add_argument("--think-budget", type=int, default=0, help="弱模型服务档：>0 时开启思考（think=true）并把所有 LLM 调用的 max_tokens 抬到该值下限。背景：2026-10-26 诊断 qwen3-vl:8b/30b 新构建把推理漏进正文且规划器写死 max_tokens=512，思考被截断致决策永远兜底 stop；开思考让推理进独立字段、正文留干净 JSON。仅用于 8B/30B 重跑批，全臂统一以保臂间可比")  # weak-model service tier
    a = ap.parse_args()  # parse

    os.environ["LLM_API_BASE"] = a.base  # endpoint (port 11434 automatically uses Ollama native /api/chat)
    if a.key != "env":  # inject when a key is passed explicitly; default env = no override, key comes from .env/existing environment variables (cloud)
        os.environ["LLM_API_KEY"] = a.key  # inject the key
    os.environ["LLM_MODEL_NAME"] = a.model  # model name
    os.environ["LLM_NUM_CTX"] = a.ctx  # context window (backend auto-halves and retries at 500)
    os.environ["LLM_MIN_INTERVAL"] = a.interval  # throttle interval (>= 2 seconds recommended on cloud free tiers)
    os.environ.setdefault("LLM_DISABLE_THINKING", "1")  # disable thinking mode (override with env LLM_DISABLE_THINKING=0: since 2026-10-26, qwen3-vl:8b/30b on newer Ollama returns empty content with think=false, so reruns must enable thinking)
    TAG = a.tag or a.model.replace(":", "-").replace("/", "-")  # tag

    if a.think_budget > 0:  # weak-model service tier (for 8B/30B reruns only; enabled uniformly across all arms to keep them comparable)
        os.environ["LLM_DISABLE_THINKING"] = "0"  # enable thinking: reasoning goes into a separate thinking field and the body keeps clean JSON (with think=false this build leaks reasoning into the body and the 512 budget must truncate it)
        from oregenesis.llm.openai_compatible import OpenAICompatibleBackend as _TB  # class-level patch target (must be applied before the runner builds its own backend)
        if not getattr(_TB.chat, "_think_budget", False):  # guard against double wrapping
            _orig_chat = _TB.chat  # original method
            def _chat_think_budget(self, messages, **kw):  # wrapper: raise the max_tokens floor for all calls
                kw["max_tokens"] = max(int(kw.get("max_tokens") or 0), int(a.think_budget))  # thinking tokens count against the generation budget; 512 would be eaten up by thinking
                return _orig_chat(self, messages, **kw)  # forward to the original method
            _chat_think_budget._think_budget = True  # marker to prevent re-wrapping
            _TB.chat = _chat_think_budget  # class-level replacement (instance lookup goes through the class; also covers backends the runner builds itself)
        print(f"[弱模型服务档] 思考开启 + max_tokens≥{a.think_budget}")  # progress visibility

    definition = _load_def(a.suite or "FG")  # task definition (routed by suite)
    cases = definition["cases"]  # case table
    wanted = [t.strip() for t in a.tasks.split(",") if t.strip()] if a.tasks else None  # task filter
    tasks = [t for t in definition["tasks"] if not wanted or t["id"] in wanted]  # effective tasks
    if a.list:  # print the matrix only
        print(f"套卷: {a.suite or 'FG'} · 定义文件: {BENCH_DIR / DEF_FILES[a.suite or 'FG']}")  # suite info
        for t in tasks:  # per task
            print(f"{t['id']:>4} [{t['case'][:12]:<12}] {t['capability']:<10} checks={len(t['checks'])} files={len(t['files'])}")  # row
        print(f"共 {len(tasks)} 任务")  # count
        return  # done
    if not a.judge_only:  # probe the LLM only when a run is needed
        from oregenesis.llm.base import LLMBackendConfig  # backend config (env vars were written at the top)
        from oregenesis.llm.gate import require_llm  # fail-fast gate
        require_llm(LLMBackendConfig())  # liveness probe: exit immediately when unreachable

    from oregenesis.runner import OreGenesisRunner, RunnerOptions  # end-to-end runner
    if a.ablation != "A0":  # ablation arm: apply runtime patches and announce them
        from oregenesis.llm.base import LLMBackendConfig as _Cfg  # backend config
        from oregenesis.llm.openai_compatible import OpenAICompatibleBackend as _BE  # backend class
        _msg = apply_ablation(a.ablation, _BE(_Cfg()))  # apply the patches (A1 needs a backend reference)
        print(f"[消融] {_msg}")  # progress visibility
    SUITE = a.suite or "FG"  # suite name (filename prefix: 测试集FG_; different suites' scores are never mixed)
    run_root = BENCH_DIR / f"测试集{SUITE}_运行_{TAG}"  # output root for this run
    rows: list[dict] = []  # score rows
    for t in tasks:  # per task
        out_dir = run_root / t["id"]  # per-task output directory
        state_path = out_dir / "final_state.json"  # final-state file
        t0, status = time.time(), ""  # timing and status
        if a.judge_only:  # grading-only mode
            status = "判卷"  # status marker
        elif state_path.exists() and not a.force:  # result exists and no --force: resume from checkpoint
            status = "跳过（已有结果）"  # skip
        else:  # normal run
            try:  # one task failing must not break the whole round
                case = cases[t["case"]]  # case info
                files = [str(BASE_DIR / case["dir"] / f) for f in t["files"]]  # full input file paths
                out_dir.mkdir(parents=True, exist_ok=True)  # create the output directory
                OreGenesisRunner().run(  # end-to-end run (call spans lines)
                    files, t["task"], options=RunnerOptions(
                        require_llm=True, output_dir=str(out_dir), max_rounds=a.rounds))
                status = f"运行完成 {time.time() - t0:.0f}s"  # status
            except Exception as exc:  # noqa: BLE001 — record the failure and continue
                status = f"运行失败: {type(exc).__name__}: {exc}"  # failure reason
        # ── Grading (whether just run or skipped, grade whenever a final state exists) ──
        if not state_path.exists():  # no final state (run failed or first round not run)
            print(f"[{t['id']}] {status}｜判卷: 无 final_state.json，记 0 分")  # progress
            rows.append({"id": t["id"], "case": t["case"], "capability": t["capability"],  # zero-score row
                         "pass": False, "score": 0.0, "status": status, "checks": [],  # collect
                         "efficiency": {"rounds": 0, "tool_calls": 0, "failed_calls": 0,  # efficiency profile (zeros for an empty state)
                                        "fail_rate": 0.0, "repeat_decisions": 0, "auto_figures": 0}})  # continued
            continue  # next task
        final_state = json.loads(state_path.read_text(encoding="utf-8"))  # read the final state
        verdict = judge_task(t, final_state)  # grade
        ok = "✅ PASS" if verdict["pass"] else "❌ FAIL"  # result marker
        failed = [f"{desc}（{note}）" for desc, oki, note in verdict["checks"] if not oki]  # failed-check details
        eff = verdict["efficiency"]  # efficiency profile
        print(f"[{t['id']}] {status}｜判卷: {len(verdict['checks'])} 项检查通过 "
              f"{sum(1 for _, o, _ in verdict['checks'] if o)}｜{ok} {verdict['score']:.0%}"
              f"｜轮{eff['rounds']} 调{eff['tool_calls']} 败{eff['failed_calls']} 重{eff['repeat_decisions']} 自图{eff['auto_figures']}"
              + (f"｜失败: {failed}" if failed else ""))  # progress (with an efficiency quick view)
        rows.append({"id": t["id"], "case": t["case"], "capability": t["capability"],  # score row
                     "pass": verdict["pass"], "score": verdict["score"],  # score
                     "status": status, "checks": verdict["checks"],  # details
                     "efficiency": verdict["efficiency"]})  # efficiency profile (v2)
    # ── Score report persistence ──
    SUF = f"_{a.suffix}" if a.suffix else ""  # score filename suffix (re-grading with v2 keeps the old report)
    n_pass = sum(1 for r_ in rows if r_["pass"])  # PASS count
    avg = sum(r_["score"] for r_ in rows) / max(len(rows), 1)  # average score
    case_label = {k: v["label"] for k, v in cases.items()}  # case labels
    md = [f"# 案例库基准成绩单（{a.model} · tag={TAG}）", "",  # header
          "- 判卷口径: v2（2026-10-19：自动配图不计入工具类检查；效率画像单列不计 PASS）",  # criteria annotation
          f"- 运行时间: {time.strftime('%Y-%m-%d %H:%M')} · 任务数 {len(rows)} · "
          f"PASS {n_pass}（{n_pass / max(len(rows), 1):.0%}） · 平均检查通过率 {avg:.0%}", "",
          "| 任务 | 案例 | 能力维度 | 检查通过 | 得分 | 轮数 | 调用 | 失败 | 重复决策 | 结果 |",
          "|---|---|---|---|---|---|---|---|---|---|"]  # table header (with efficiency columns)
    for r_ in rows:  # one row per task
        n_total = len(r_["checks"])  # total checks
        n_ok = round(r_["score"] * n_total)  # number passed
        eff = r_.get("efficiency") or {}  # efficiency profile (tolerates old score rows)
        md.append(f"| {r_['id']} | {case_label.get(r_['case'], r_['case'])[:12]} | {r_['capability']} "
                  f"| {n_ok}/{n_total} | {r_['score']:.0%} | {eff.get('rounds', '-')} | {eff.get('tool_calls', '-')} "
                  f"| {eff.get('failed_calls', '-')} | {eff.get('repeat_decisions', '-')} "
                  f"| {'✅' if r_['pass'] else '❌'} |")  # row
    md += ["", "## 失败任务明细", ""]  # failed-tasks section title
    for r_ in rows:  # scan tasks for failures
        if r_["pass"]:  # passed tasks are not expanded
            continue  # skip
        md.append(f"### {r_['id']} {r_['capability']}（{r_['status']}）")  # task title
        for desc, oki, note in r_["checks"]:  # per check
            if not oki:  # list failed items only
                md.append(f"- ❌ {desc}｜{note}")  # failure reason (with actual tools / captured numbers)
        md.append("")  # blank line
    (BENCH_DIR / f"测试集{SUITE}_成绩单{SUF}_{TAG}.md").write_text("\n".join(md) + "\n", encoding="utf-8")  # write the md report
    (BENCH_DIR / f"测试集{SUITE}_成绩{SUF}_{TAG}.json").write_text(  # write the json scores (for later multi-model comparison)
        json.dumps({"model": a.model, "tag": TAG, "suite": SUITE, "time": time.strftime('%Y-%m-%d %H:%M'),  # meta info (with the suite marker)
                    "judge_version": 2, "pass_rate": n_pass / max(len(rows), 1), "avg_score": avg,  # criteria version + scores
                    "rows": rows},  # scores
                   ensure_ascii=False, indent=1), encoding="utf-8")  # write
    print(f"\n成绩单: {BENCH_DIR / f'测试集{SUITE}_成绩单{SUF}_{TAG}.md'}（PASS {n_pass}/{len(rows)}，平均 {avg:.0%}）")  # wrap up
    try:  # archiving + ledger (2026-10-27): results are auto-filed into batch folders and refresh 测试结果台账
        from 测试结果台账 import sweep_and_ledger  # noqa: E402
        sweep_and_ledger()  # archive and refresh the ledger
    except Exception as exc:  # noqa: BLE001 — ledger failure does not affect scores
        print(f"[台账] 自动归档跳过: {exc}")  # keep a trace


if __name__ == "__main__":  # command-line entry
    main()  # run
