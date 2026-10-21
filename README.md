# AI Eval

[![CI](https://github.com/Dhillon001/ai-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/Dhillon001/ai-eval/actions/workflows/ci.yml)

Offline evaluation runner for LLM and AI systems: datasets, adapters, scorers, run history, and regression reports.

## Quick start

```bash
cd "/Users/harsharandeepdhillon/Desktop/AI Eval system"
python -m venv .venv && source .venv/bin/activate
pip install -e ".[openai,anthropic,dev]"

# Mock run (no API key) — uses the mock adapter
ai-eval run --suite configs/example_suite.yaml --adapter mock

# OpenAI-compatible API
export OPENAI_API_KEY=sk-...
ai-eval run --suite configs/example_suite.yaml --adapter openai --model gpt-4o-mini

# Anthropic (Claude), 8 cases in parallel
export ANTHROPIC_API_KEY=sk-ant-...
ai-eval run --suite configs/rag_suite.yaml -j 8

ai-eval report --run-id <id>
ai-eval compare --baseline <id-a> --candidate <id-b> [--fail-on-score-drop]
ai-eval list-runs
```

## Layout

- `datasets/` — JSONL case files (`input`, optional `context`, `expected`, `metadata`)
- `configs/` — Suite YAML: dataset path, scorers, adapter settings
- `src/ai_eval/` — Runner, store, adapters, scorers, CLI
- `.ai-eval/` — Local SQLite run database and HTML reports (gitignored)

## Case format (JSONL)

```json
{"id": "refund-1", "input": "Can I get a refund?", "expected": "refund", "metadata": {"tags": ["support"]}, "checks": ["refund", "policy"]}
```

## Scorers

| type | what it checks |
|---|---|
| `exact` | normalized output == `expected` |
| `checklist` | every string in `checks` appears in the output |
| `llm_judge` | LLM grades output against `metadata.judge_criteria` / `expected` (`provider: openai\|anthropic`) |
| `faithfulness` | LLM checks every claim is supported by `context` (RAG); skips cases without context |

## Compare

`compare` reports pass→fail flips (`regressions`), score drops >0.1 on cases that didn't flip,
improvements, missing/new cases, per-scorer mean deltas and per-tag pass-rate deltas.
Exits 1 on regressions (and on score drops with `--fail-on-score-drop`) so it can gate CI.

## Extending

- **Adapters**: implement `SystemAdapter` in `src/ai_eval/adapters/`
- **Scorers**: implement `Scorer` in `src/ai_eval/scorers/`
- **RAG / agents**: add fields like `context` and `tools` on cases; use white-box traces in `RunResult.trace`
