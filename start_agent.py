# -*- coding: utf-8 -*-
"""Deposit Agent 启动入口：加载数据文件（文件或文件夹）与分析任务，运行智能体并输出报告。

Python 调用：
    from start_agent import start_agent
    result = start_agent(["测试案例库/黄金洞金矿"], "请计算黄铁矿的 Co/Ni 比值并判断其成因")
    print(result.markdown)

命令行：
    python start_agent.py --files 测试案例库/黄金洞金矿 --task "请计算黄铁矿的 Co/Ni 比值并判断其成因"
    python start_agent.py --files a.xlsx 数据文件夹 --task-file 任务.txt --out output/演示

模型默认读项目根目录 .env（LLM_API_BASE / LLM_MODEL_NAME / LLM_API_KEY）；
--model / --api-base / --api-key 可临时覆盖；--no-llm 不用模型、规划器走确定性规则回退。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oregenesis.io.parser import detect_modality  # noqa: E402
from oregenesis.llm.base import LLMBackendConfig  # noqa: E402
from oregenesis.llm.gate import LLMUnavailableError, require_llm  # noqa: E402
from oregenesis.llm.openai_compatible import OpenAICompatibleBackend  # noqa: E402
from oregenesis.reasoning.task_types import classify_task  # noqa: E402
from oregenesis.runner import OreGenesisRunner, RunnerOptions, RunnerResult  # noqa: E402

TASK_TYPES = ("genesis", "prospectivity", "comprehensive")


def _say(msg: str) -> None:
    """打印；GBK 控制台遇到上标等字符时替换输出，不让整次运行因打印崩掉。"""
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(msg.encode(enc, errors="replace").decode(enc))


def _supported(path: Path) -> bool:
    """扩展名是否在解析层支持之列（与 oregenesis/io/parser 同一口径）。"""
    try:
        detect_modality(str(path))
        return True
    except ValueError:
        return False


def _skippable(path: Path, base: Path) -> bool:
    """Office 锁文件（~$xxx.xlsx）、隐藏文件与隐藏目录里的文件：不是数据，进解析会报错。"""
    return any(part.startswith(("~$", ".")) for part in path.relative_to(base).parts)


def load_files(inputs: Iterable[str | Path] | str | Path) -> list[str]:
    """加载数据文件：文件直接收录，文件夹递归收集受支持格式的文件；去重保序，返回绝对路径列表。"""
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    found: list[str] = []
    seen: set[str] = set()
    for item in inputs:
        p = Path(item).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"找不到文件或文件夹：{item}")
        if p.is_dir():
            candidates = sorted(f for f in p.rglob("*")
                                if f.is_file() and not _skippable(f, p) and _supported(f))
            if not candidates:
                raise ValueError(f"文件夹里没有可用的数据文件：{item}"
                                 "（支持 xlsx/xls/xlsm/csv/txt/md/jpg/png/tif/bmp）")
        else:
            detect_modality(str(p))  # 不支持的格式由解析层抛出带支持列表的标准报错
            candidates = [p]
        for f in candidates:
            key = str(f.resolve())
            if key not in seen:
                seen.add(key)
                found.append(key)
    if not found:
        raise ValueError("没有提供任何数据文件")
    return found


def load_task(task: str = "", task_file: str | Path | None = None) -> str:
    """任务输入：优先读 task_file（UTF-8，失败回退 GBK），否则用 task 字符串。"""
    if task_file:
        tf = Path(task_file).expanduser()
        if not tf.is_file():
            raise FileNotFoundError(f"找不到任务文件：{task_file}")
        raw = tf.read_bytes()
        for enc in ("utf-8-sig", "gbk"):
            try:
                task = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"任务文件编码无法识别（请另存为 UTF-8）：{task_file}")
    text = (task or "").strip()
    if not text:
        raise ValueError("分析任务为空：请用 task 字符串或 task_file 文件给出要分析的问题")
    return text


def _build_backend(model: str, api_base: str, api_key: str) -> tuple[OpenAICompatibleBackend, LLMBackendConfig]:
    """按 .env 构造模型配置，传入的参数逐项覆盖，探活通过后返回后端。

    配置默认值在 import 时就已从环境/.env 固定，运行中改环境变量不生效，所以直接改实例字段。
    """
    cfg = LLMBackendConfig()
    if model:
        cfg.model_name = model
    if api_base:
        cfg.api_base = api_base
    if api_key:
        cfg.api_key = api_key
    require_llm(cfg)
    return OpenAICompatibleBackend(cfg), cfg


def start_agent(files: Iterable[str | Path] | str | Path, task: str = "", *,
                task_file: str | Path | None = None, task_type: str = "",
                model: str = "", api_base: str = "", api_key: str = "",
                no_llm: bool = False, max_rounds: int = 6,
                output_dir: str | Path = "") -> RunnerResult:
    """加载数据与分析任务，运行智能体，返回运行结果（report.md 与 final_state.json 同时落盘）。

    files       数据文件或文件夹（单个路径或列表；文件夹递归收集 xlsx/xls/csv/txt/md/图片）
    task        自然语言分析任务；也可改用 task_file 指定一个 txt/md 文件
    task_type   genesis / prospectivity / comprehensive；留空按任务文本自动判定
    model / api_base / api_key   临时覆盖 .env 里的模型设置
    no_llm      不用模型，规划器走确定性规则回退（消融基线）
    max_rounds  ReAct 轮数上限
    output_dir  输出目录；留空则为 output/<年月日_时分秒>，每次运行互不覆盖
    """
    file_list = load_files(files)
    task_text = load_task(task, task_file)
    if task_type and task_type not in TASK_TYPES:
        raise ValueError(f"任务类型只能是 {' / '.join(TASK_TYPES)}，收到：{task_type}")
    out = Path(output_dir) if output_dir else ROOT / "output" / time.strftime("%Y%m%d_%H%M%S")

    backend = None
    if no_llm:
        model_desc = "不使用模型（规则回退模式）"
    else:
        backend, cfg = _build_backend(model, api_base, api_key)
        model_desc = f"{cfg.model_name} @ {cfg.api_base}"

    _say("=" * 60)
    _say(f"[数据] 共 {len(file_list)} 个文件：")
    for f in file_list:
        _say(f"  - {f}")
    _say(f"[任务] {task_text}")
    _say(f"[类型] {task_type or classify_task(task_text) + '（按任务文本自动判定）'}")
    _say(f"[模型] {model_desc}")
    _say(f"[输出] {out}")
    _say("=" * 60)

    opts = RunnerOptions(require_llm=not no_llm, backend=backend, max_rounds=max_rounds,
                         output_dir=str(out), task_type=task_type)
    result = OreGenesisRunner().run(file_list, task=task_text, options=opts)

    conclusion = str(result.final_result.get("conclusion") or "").strip()
    _say("-" * 60)
    _say("[结论] " + (conclusion[:300] + ("…" if len(conclusion) > 300 else "")
                     if conclusion else "（无模型叙述，见报告中的工具结果与证据）"))
    _say(f"[报告] {out / 'report.md'}")
    _say(f"[终态] {out / 'final_state.json'}")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deposit Agent 启动入口：加载数据与分析任务并运行智能体")
    ap.add_argument("--files", nargs="+", required=True, metavar="路径",
                    help="数据文件或文件夹（可多个；文件夹递归收集 xlsx/xls/xlsm/csv/txt/md/jpg/png/tif/bmp）")
    given = ap.add_mutually_exclusive_group(required=True)
    given.add_argument("--task", help="自然语言分析任务（含空格时用引号括起来）")
    given.add_argument("--task-file", metavar="文件", help="从 txt/md 文件读取分析任务")
    ap.add_argument("--type", choices=TASK_TYPES, help="任务类型（缺省按任务文本自动判定）")
    ap.add_argument("--model", default="", help="模型名（覆盖 .env 的 LLM_MODEL_NAME）")
    ap.add_argument("--api-base", default="", help="模型服务地址（覆盖 .env 的 LLM_API_BASE）")
    ap.add_argument("--api-key", default="", help="API Key（覆盖 .env 的 LLM_API_KEY）")
    ap.add_argument("--no-llm", action="store_true", help="不用模型：规划器走确定性规则回退（消融基线）")
    ap.add_argument("--max-rounds", type=int, default=6, help="ReAct 轮数上限（默认 6）")
    ap.add_argument("--out", default="", help="输出目录（缺省 output/<年月日_时分秒>）")
    a = ap.parse_args(argv)
    try:
        start_agent(a.files, a.task or "", task_file=a.task_file, task_type=a.type or "",
                    model=a.model, api_base=a.api_base, api_key=a.api_key,
                    no_llm=a.no_llm, max_rounds=a.max_rounds, output_dir=a.out)
    except (FileNotFoundError, ValueError, LLMUnavailableError) as e:
        print(f"[启动失败] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
