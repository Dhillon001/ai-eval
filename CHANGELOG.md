# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-28

### Added
- **Benchmark harness**: `ai-eval bench --suite S --adapter A[:MODEL] --adapter B[:MODEL] ...`
  runs one suite against several systems and prints a comparison table ranked by pass rate,
  then mean score (errors, p50/p95 latency, cost and run id per system). `--markdown` prints
  a GitHub-flavoured table and `--out PATH` writes it to a file. Every system's run is stored,
  so any two can be diffed with `compare` or `report --compare`.
- **HTTP adapter** (`http`): evaluate any JSON-over-HTTP service. Request body from a template
  with `{input}` / `{context}`, output via a dotted `response_path`, headers with `${ENV_VAR}`
  interpolation, `timeout`, `method`, and `trace_path` to capture retrieved docs or tool calls
  into the run trace. Example in `configs/http_example.yaml`.
- **Retrieval scorer** (`retrieval`): `recall@k`, `MRR` and `hit_rate` comparing
  `trace["retrieved_ids"]` with `metadata["expected_doc_ids"]`; configurable `k`, `metric` and
  pass threshold. Example in `configs/retrieval_suite.yaml`.
- **Agent trajectory scorer** (`trajectory`): compares `trace["tool_calls"]` with
  `metadata["expected_tools"]` in `exact_order`, `subset` or `any_order` mode, with optional
  argument matching. Example in `configs/agent_suite.yaml`.
- The `mock` adapter emits `retrieved_ids` / `tool_calls` from `mock_retrieved_ids` /
  `mock_tool_calls` case metadata, so retrieval and agent suites run offline.
- **Cross-machine comparison**: `ai-eval run --save-baseline PATH` exports a run as JSON;
  `ai-eval compare --baseline-file / --candidate-file` compares against exported runs.
- **CI regression gate**: a `pull_request` job evaluates the base commit and the PR head with
  the mock suite and fails on regressions via `compare --fail-on-score-drop`.
- **Reliability**: provider calls retry 429/5xx/timeouts with exponential backoff and jitter
  (`max_retries`, default 3).
- **Judge cache**: LLM judge verdicts are cached in `.ai-eval/cache/`, keyed by provider, model
  and prompt; `--no-cache` bypasses it (per scorer: `cache: false`).
- **Cost tracking**: per-model price table, overridable with `pricing:` in the suite YAML;
  `cost_usd` is recorded in run metadata and shown in the CLI summary.
- **HTML report v2**: summary cards (pass rate, mean score, errors, p50/p95 latency, cost),
  per-tag table, failing cases first with collapsible details (scores, trace, judge reasoning),
  and untrusted output escaped.
- `ai-eval report --compare A B`: side-by-side HTML diff of two runs (run ids or run files)
  highlighting regressions and improvements.
- `datasets/examples/rag_v1.jsonl` grown to 30 tagged cases (answerable, unanswerable/abstain,
  conflicting context, multi-hop).
- `CHANGELOG.md`.

### Changed
- CI uses `actions/checkout@v5` and `actions/setup-python@v6`.
- README rewritten with diagrams, an adapter table and sections on retrieval, agents,
  reports and the CI regression gate.

### Fixed
- Overriding a suite's adapter with a different one on the command line (`--adapter`) no
  longer passes the suite's adapter options to the new adapter, which used to crash e.g.
  `--adapter http` on a mock suite.

## 0.1.0 - 2026-10-21

### Added
- Offline eval runner MVP: YAML suites, JSONL datasets, `mock`, `openai` (and
  OpenAI-compatible) and `anthropic` adapters.
- Scorers: `exact`, `checklist`, `llm_judge` and RAG `faithfulness`.
- Parallel runs, a SQLite run store, `ai-eval compare`, HTML reports.
- GitHub Actions CI: tests on Python 3.10–3.12 and a mock eval gate.

[0.2.0]: https://github.com/Dhillon001/ai-eval/releases/tag/v0.2.0
