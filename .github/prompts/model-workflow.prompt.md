---
description: "Use when creating, adding, or optimizing a model workflow in autoresearch with research_loop.py and workflow.json."
---

You are working in the `autoresearch` repository.

Goal:
- Add and/or optimize a new model workflow for: <MODEL_FAMILY>
- Primary metric: <METRIC_NAME>
- Metric direction: <lower|higher>

Environment:
- Use the current Python environment only (no new dependencies unless already in `pyproject.toml`).
- Run commands with `python ...`.

Files to read first:
- `README.md`
- `program.md`
- `research_loop.py`
- `workflow.json`
- `results.tsv` (if present)
- The active trainer file for this experiment (for example `train.py`)

What to implement:
1. Create or update trainer script: `<TRAINER_FILE>.py`.
2. Ensure trainer prints final metric line compatible with regex parsing, for example: `<METRIC_NAME>: 0.123456`.
3. Add or update `workflow.json` experiment entry:
   - `name`: `<EXPERIMENT_NAME>`
   - `command`: `python <TRAINER_FILE>.py`
   - `metric_regex`: `^<METRIC_NAME>:\\s*([0-9.]+)`
   - `metric_direction`: `<lower|higher>`
   - optional `memory_regex`
4. Run baseline experiment:
   - `python research_loop.py --experiment <EXPERIMENT_NAME> --description "baseline"`
5. Verify row is appended to `results.tsv` with `status`, `metric`, and `log_file`.
6. If crash, inspect the run log, fix, and rerun until baseline succeeds.

Optimization loop:
- Propose one small change at a time.
- Commit each change before running.
- Run via `research_loop.py` with a clear description.
- Keep only `status=keep` outcomes; revert/discard otherwise.

Constraints:
- Do not modify `prepare.py` for metric/evaluation bypasses.
- Keep changes simple and maintainable.
- Keep metric extraction and output compatible with `research_loop.py`.

Output requirements:
- Brief summary of code changes.
- Exact commands run.
- Latest `results.tsv` rows for this experiment.
- Remaining blockers (if any).
