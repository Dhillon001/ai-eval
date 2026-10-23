# Roadmap — v0.2 (Oct 22–28, 2026)

Four shippable steps per day. Each step is one commit that passes `pytest` on its own.
Steps are ticked (`[x]`) in the same commit that completes them.

## Thu Oct 22 — PR regression gate
- [x] 1. Core: `ai-eval run --save-baseline PATH` / `ai-eval compare --baseline-file PATH` so runs can be compared across machines (export/import a run as JSON).
- [x] 2. Tests: round-trip export/import, compare against an imported baseline, exit codes.
- [x] 3. Wiring: CI job on `pull_request` that runs the mock suite on `main` and on the PR head, then `ai-eval compare --fail-on-score-drop`; bump `actions/checkout@v5` and `actions/setup-python@v6`.
- [x] 4. Docs: README "CI regression gate" section with the workflow snippet.

## Fri Oct 23 — HTTP adapter (evaluate any service)
- [ ] 1. Core: `adapters/http.py` — POST JSON to `url`, request body built from a template with `{input}`/`{context}`, output extracted with a dotted `response_path` (e.g. `data.answer`), optional headers (env-var interpolation like `${API_TOKEN}`), timeout.
- [ ] 2. Tests: use `httpx.MockTransport`; success, non-2xx, timeout, bad `response_path`, header env interpolation.
- [ ] 3. Wiring: register `http` in `build_adapter`; `configs/http_example.yaml`; optional `trace_path` to capture retrieved docs/tool calls into `RunResult.trace`.
- [ ] 4. Docs: README adapter table + HTTP example.

## Sat Oct 24 — Retrieval metrics
- [ ] 1. Core: `scorers/retrieval.py` — `recall@k`, `MRR`, `hit_rate`, comparing `trace["retrieved_ids"]` to `case.metadata["expected_doc_ids"]`; configurable `k` and `metric`, pass threshold.
- [ ] 2. Tests: perfect/partial/zero recall, MRR ranking, missing trace or expected ids (skip), k larger than list.
- [ ] 3. Wiring: register `retrieval` scorer; mock adapter can emit `retrieved_ids` from `case.metadata["mock_retrieved_ids"]`; `configs/retrieval_suite.yaml` + dataset.
- [ ] 4. Docs: README "Evaluating retrieval vs. generation" section.

## Sun Oct 25 — Agent trajectory scorer
- [ ] 1. Core: `scorers/trajectory.py` — compare `trace["tool_calls"]` (list of `{name, args}`) to `case.metadata["expected_tools"]`; modes `exact_order`, `subset`, `any_order`; optional arg matching; score = fraction matched.
- [ ] 2. Tests: each mode, extra/missing calls, arg mismatch, no trace (skip).
- [ ] 3. Wiring: register `trajectory`; mock adapter emits `tool_calls` from `case.metadata["mock_tool_calls"]`; `configs/agent_suite.yaml` + dataset.
- [ ] 4. Docs: README "Evaluating agents" section; extend `EvalCase` docs with `expected_tools`.

## Mon Oct 26 — Reliability and cost
- [ ] 1. Core: retry with exponential backoff + jitter for provider calls in `llm.py` (retry 429/5xx/timeouts, configurable `max_retries`).
- [ ] 2. Core: on-disk judge cache in `.ai-eval/cache/` keyed by hash of (provider, model, prompt); `--no-cache` flag.
- [ ] 3. Core: cost tracking — per-model price table (overridable in suite YAML), `cost_usd` in run metadata and CLI summary.
- [ ] 4. Tests for retries (fake client raising then succeeding), cache hit/miss, cost math; README updates.

## Tue Oct 27 — Report v2
- [ ] 1. Core: HTML report summary cards (pass rate, mean score, errors, p50/p95, cost) and per-tag table.
- [ ] 2. Core: failures-first ordering, collapsible case details (scores, trace, judge reasoning).
- [ ] 3. Core: `ai-eval report --compare A B` — side-by-side HTML diff highlighting regressions/improvements.
- [ ] 4. Tests (HTML contains expected sections, escaping of untrusted output) + README screenshot-free description.

## Wed Oct 28 — Benchmark harness and v0.2.0
- [ ] 1. Data: grow `datasets/examples/rag_v1.jsonl` to ~30 cases (answerable, unanswerable/abstain, conflicting context, multi-hop), tagged.
- [ ] 2. Core: `ai-eval bench --suite S --adapter A:model --adapter B:model` — runs several systems and prints a comparison table (markdown output option).
- [ ] 3. Tests for `bench` with mock adapters; `CHANGELOG.md` for v0.2.0; version bump in `pyproject.toml` and `__init__.py`.
- [ ] 4. Release: README "Benchmarks" section (how to run; results table to be filled from a live run with real API keys), git tag `v0.2.0` pushed.
