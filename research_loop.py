"""
Model-agnostic autonomous experiment loop.

This script keeps the same keep/discard workflow used by autoresearch,
but it can evaluate any model family by reading experiment definitions from
`workflow.json`.

Usage examples:
  uv run research_loop.py --experiment llm --description "baseline"
  uv run research_loop.py --experiment toy_regression --description "larger hidden"
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

RESULTS_HEADER = [
    "timestamp",
    "experiment",
    "commit",
    "metric",
    "memory_gb",
    "status",
    "description",
    "command",
    "log_file",
]


@dataclass
class ExperimentConfig:
    name: str
    command: str
    metric_regex: str
    metric_direction: str
    memory_regex: Optional[str] = None


@dataclass
class RunSummary:
    metric: float
    memory_gb: float
    status: str
    log_file: str


def load_config(config_path: Path) -> Dict[str, ExperimentConfig]:
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    out: Dict[str, ExperimentConfig] = {}
    for exp in raw.get("experiments", []):
        name = exp["name"]
        out[name] = ExperimentConfig(
            name=name,
            command=exp["command"],
            metric_regex=exp["metric_regex"],
            metric_direction=exp.get("metric_direction", "lower").lower(),
            memory_regex=exp.get("memory_regex"),
        )
    return out


def ensure_results_file(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(RESULTS_HEADER)


def run(cmd: str) -> str:
    proc = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
    return proc.stdout.strip()


def git_commit_short() -> str:
    return run("git rev-parse --short HEAD")


def parse_metric(log_text: str, pattern: str) -> float:
    match = re.search(pattern, log_text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not find metric using regex: {pattern}")
    return float(match.group(1))


def parse_memory_gb(log_text: str, memory_regex: Optional[str]) -> float:
    if not memory_regex:
        return 0.0
    match = re.search(memory_regex, log_text, flags=re.MULTILINE)
    if match is None:
        return 0.0
    memory_mb = float(match.group(1))
    return memory_mb / 1024.0


def read_existing_metrics(results_path: Path, experiment_name: str) -> List[float]:
    if not results_path.exists():
        return []

    metrics: List[float] = []
    with results_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("experiment") != experiment_name:
                continue
            if row.get("status") == "crash":
                continue
            try:
                metrics.append(float(row["metric"]))
            except (TypeError, ValueError, KeyError):
                continue
    return metrics


def classify_result(new_metric: float, history: List[float], direction: str) -> str:
    if not history:
        return "keep"

    best = min(history) if direction == "lower" else max(history)
    if direction == "lower":
        return "keep" if new_metric < best else "discard"
    if direction == "higher":
        return "keep" if new_metric > best else "discard"
    raise ValueError("metric_direction must be 'lower' or 'higher'")


def append_result(
    results_path: Path,
    timestamp: str,
    experiment_name: str,
    commit: str,
    metric: float,
    memory_gb: float,
    status: str,
    description: str,
    command: str,
    log_file: str,
) -> None:
    with results_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                timestamp,
                experiment_name,
                commit,
                f"{metric:.6f}",
                f"{memory_gb:.1f}",
                status,
                description,
                command,
                log_file,
            ]
        )


def execute_experiment(exp: ExperimentConfig, logs_dir: Path) -> RunSummary:
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{exp.name}_{timestamp}.log"

    resolved_command = resolve_command(exp.command)
    print(f"Running: {resolved_command}")
    full_cmd = f"{resolved_command} > {log_file} 2>&1"
    try:
        subprocess.run(full_cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        return RunSummary(metric=0.0, memory_gb=0.0, status="crash", log_file=str(log_file))

    log_text = log_file.read_text(encoding="utf-8", errors="replace")
    metric = parse_metric(log_text, exp.metric_regex)
    memory_gb = parse_memory_gb(log_text, exp.memory_regex)
    return RunSummary(metric=metric, memory_gb=memory_gb, status="ok", log_file=str(log_file))


def resolve_command(command: str) -> str:
    """Fallback for environments where uv is not available in PATH.

    If command starts with "uv run" and uv cannot be found, translate it to
    "<current python> ..." so workflow.json remains portable.
    """
    parts = shlex.split(command)
    if len(parts) >= 3 and parts[0] == "uv" and parts[1] == "run":
        if shutil.which("uv"):
            return command
        py = shlex.quote(sys.executable)
        rest = " ".join(shlex.quote(p) for p in parts[2:])
        return f"{py} {rest}"
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Model-agnostic autoresearch loop runner")
    parser.add_argument("--config", default="workflow.json", help="Path to workflow JSON config")
    parser.add_argument("--experiment", required=True, help="Experiment name from workflow config")
    parser.add_argument("--description", default="manual run", help="Short description for results.tsv")
    parser.add_argument("--results", default="results.tsv", help="Path to TSV results log")
    parser.add_argument("--logs-dir", default="logs", help="Directory where run logs are written")
    args = parser.parse_args()

    root = Path.cwd()
    config_path = root / args.config
    results_path = root / args.results
    logs_dir = root / args.logs_dir

    experiments = load_config(config_path)
    if args.experiment not in experiments:
        print(f"Unknown experiment '{args.experiment}'. Available: {', '.join(sorted(experiments))}")
        return 1

    exp = experiments[args.experiment]
    ensure_results_file(results_path)

    commit = git_commit_short()
    timestamp = dt.datetime.now().isoformat(timespec="seconds")

    try:
        summary = execute_experiment(exp, logs_dir)
    except Exception as exc:
        print(f"Run failed before completion: {exc}")
        append_result(
            results_path=results_path,
            timestamp=timestamp,
            experiment_name=exp.name,
            commit=commit,
            metric=0.0,
            memory_gb=0.0,
            status="crash",
            description=f"{args.description} (runner error: {exc})",
            command=exp.command,
            log_file="",
        )
        return 1

    if summary.status == "crash":
        append_result(
            results_path=results_path,
            timestamp=timestamp,
            experiment_name=exp.name,
            commit=commit,
            metric=0.0,
            memory_gb=0.0,
            status="crash",
            description=args.description,
            command=exp.command,
            log_file=summary.log_file,
        )
        print("status=crash")
        print(f"log={summary.log_file}")
        return 1

    history = read_existing_metrics(results_path, exp.name)
    status = classify_result(summary.metric, history, exp.metric_direction)
    append_result(
        results_path=results_path,
        timestamp=timestamp,
        experiment_name=exp.name,
        commit=commit,
        metric=summary.metric,
        memory_gb=summary.memory_gb,
        status=status,
        description=args.description,
        command=exp.command,
        log_file=summary.log_file,
    )

    print("---")
    print(f"experiment:  {exp.name}")
    print(f"metric:      {summary.metric:.6f}")
    print(f"memory_gb:   {summary.memory_gb:.1f}")
    print(f"status:      {status}")
    print(f"log_file:    {summary.log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
