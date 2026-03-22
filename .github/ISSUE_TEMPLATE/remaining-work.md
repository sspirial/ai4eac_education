---
name: Remaining Work Tracker
about: Track unfinished tasks for model-agnostic training and conda base support
title: "[Follow-up] Complete model-agnostic workflow hardening"
labels: ["enhancement", "tracking"]
assignees: []
---

## Summary

Track what is still needed to make the generalized training workflow fully production-ready across model families.

## Current Status

- Generic runner exists: `research_loop.py`
- Config-driven experiments exist: `workflow.json`
- Non-LLM smoke test passes in conda base (`toy_regression`)
- `results.tsv` logging and keep/discard behavior are working

## Remaining Tasks

### Environment

- [ ] Ensure `uv` is discoverable in conda base shell (`command -v uv` works)
- [ ] Ensure `torch` is installed and importable in conda base (`python -c "import torch"`)
- [ ] Confirm CUDA/GPU availability for LLM runs (if GPU expected)

### Workflow Validation

- [ ] Run `python research_loop.py --experiment llm --description "smoke test"`
- [ ] Verify LLM metric parsing from logs (`val_bpb`)
- [ ] Verify memory parsing from logs (`peak_vram_mb`)
- [ ] Confirm keep/discard behavior for repeated LLM runs

### Documentation

- [ ] Add explicit conda base setup steps to `README.md`
- [ ] Clarify when to use `python ...` vs `uv run ...` in docs
- [ ] Add troubleshooting section for missing `uv` and missing `torch`

### Optional Hardening

- [ ] Add schema validation for `workflow.json`
- [ ] Add a lightweight self-check command (config + parser sanity checks)
- [ ] Add CI smoke test for `toy_regression` experiment

## Acceptance Criteria

- [ ] Both `toy_regression` and `llm` experiments run through `research_loop.py` in conda base
- [ ] Metrics are parsed and appended correctly to `results.tsv`
- [ ] Team can follow README setup without manual debugging

## Notes

- Last known blocker for full LLM path was missing `torch` in base.
- Keep this issue open until LLM experiment path is confirmed in base.
