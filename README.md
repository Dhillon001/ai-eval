# AI Eval

Offline evaluation runner for LLM and AI systems: datasets, adapters, scorers, run history, and regression reports.

## Quick start

```bash
cd "/Users/harsharandeepdhillon/Desktop/AI Eval system"
python -m venv .venv && source .venv/bin/activate
pip install -e ".[openai,dev]"

# Mock run (no API key) — uses the mock adapter
ai-eval run --suite configs/example_suite.yaml --adapter mock

# OpenAI-compatible API
export OPENAI_API_KEY=sk-...
ai-eval run --suite configs/example_suite.yaml --adapter openai --model gpt-4o-mini

ai-eval report --run-id <id>
ai-eval compare --baseline <id-a> --candidate <id-b>
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

## Extending

- **Adapters**: implement `SystemAdapter` in `src/ai_eval/adapters/`
- **Scorers**: implement `Scorer` in `src/ai_eval/scorers/`
- **RAG / agents**: add fields like `context` and `tools` on cases; use white-box traces in `RunResult.trace`
