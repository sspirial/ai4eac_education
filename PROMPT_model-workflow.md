# Reusable Prompt: Model Workflow

Use this prompt with your coding agent whenever you want to add or improve a model in this repo using the same `research_loop.py` process.

---

## Prompt Template

```md
You are working in the `autoresearch` repository.

Goal:
- Add and/or optimize a new model workflow for: <MODEL_FAMILY>
- Primary metric: <METRIC_NAME>
- Metric direction: <lower|higher>

Environment:
- Use the current Python environment only (no new dependencies unless already in `pyproject.toml`).
- Run commands with `python ...`.

Files to read first:
- README.md
- program.md
- research_loop.py
- workflow.json
- results.tsv (if present)
- Any trainer file relevant to this experiment (for example `train.py`)

What to implement:
1. Create or update a trainer script: <TRAINER_FILE>.py
2. Ensure trainer prints a final metric line that matches regex parsing, for example:
   - `<METRIC_NAME>: 0.123456`
3. Add or update a `workflow.json` experiment entry:
   - `name`: <EXPERIMENT_NAME>
   - `command`: `python <TRAINER_FILE>.py`
   - `metric_regex`: `^<METRIC_NAME>:\\s*([0-9.]+)`
   - `metric_direction`: `<lower|higher>`
   - optional `memory_regex`
4. Run one baseline experiment:
   - `python research_loop.py --experiment <EXPERIMENT_NAME> --description "baseline"`
5. Verify it appends to `results.tsv` and reports `status`, `metric`, and `log_file`.
6. If it crashes, inspect log, fix, and rerun until baseline works.

Optimization loop:
- Propose one small change at a time.
- Commit each change before running.
- Run through `research_loop.py` with a clear description.
- Keep changes only when status is `keep`; revert/discard otherwise.

Constraints:
- Do not modify `prepare.py` for metric/evaluation hacking.
- Keep code simple; avoid unnecessary complexity.
- Keep all logging/metric extraction compatible with `research_loop.py`.

Output requirements:
- Briefly summarize what changed.
- Provide exact commands run.
- Provide latest `results.tsv` rows for this experiment.
- Call out remaining blockers (if any).
```

---

## Example Filled Prompt

```md
You are working in the `autoresearch` repository.

Goal:
- Add and optimize a new model workflow for: tabular classification
- Primary metric: val_auc
- Metric direction: higher

Environment:
- Use the current Python environment only (no new dependencies unless already in `pyproject.toml`).
- Run commands with `python ...`.

Files to read first:
- README.md
- program.md
- research_loop.py
- workflow.json
- results.tsv (if present)

What to implement:
1. Create trainer script: train_tabular_auc.py
2. Ensure trainer prints `val_auc: <float>` at the end.
3. Add workflow entry:
   - name: tabular_auc
   - command: python train_tabular_auc.py
   - metric_regex: ^val_auc:\\s*([0-9.]+)
   - metric_direction: higher
4. Run baseline:
   - python research_loop.py --experiment tabular_auc --description "baseline"
5. Verify results.tsv updates.
6. Fix and rerun if crash.

Optimization loop:
- Make one small change per run.
- Keep only improvements.

Constraints:
- Do not modify prepare.py.
- Keep changes simple and robust.

Output requirements:
- Summary of edits, commands run, latest result rows, blockers.
```
