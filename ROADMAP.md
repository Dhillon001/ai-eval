# Roadmap — v0.3 (Nov 17 – Dec 2, 2026)

45 shippable steps, done strictly in order, one commit each. Every step must pass `pytest`
on its own and is ticked (`[x]`) in the same commit that completes it.
Tests never hit real networks or API keys — use fakes in `tests/fakes.py` / in-repo fake servers.

## MCP server evals
- [ ] 1. Core: `ai_eval/mcp_client.py` — connect to a stdio MCP server (optional `mcp` extra in pyproject), list tools, call a tool, clean shutdown; server command/args/env configured in suite YAML.
- [ ] 2. Tests: a tiny in-repo fake stdio MCP server (`tests/fixtures/fake_mcp_server.py`) exercised for list/call/error/timeout.
- [ ] 3. Core: agent loop — expose MCP tools to Claude (Anthropic tool use), run until final answer or `max_steps`; record `trace["tool_calls"]` (name, args, result, error) and the final answer as output.
- [ ] 4. Tests: agent loop driven by a fake Anthropic client that emits tool_use blocks (single call, multi-step, tool error, max_steps hit).
- [ ] 5. Wiring: register `mcp` adapter; `examples/mcp_inventory_server.py` (toy parts-inventory MCP server); `configs/mcp_suite.yaml` + 8-case dataset with `expected_tools`.
- [ ] 6. Core: `tool_use` scorer — tool precision/recall, invalid-argument rate, steps vs. optimal (efficiency), with pass thresholds.
- [ ] 7. Tests for the `tool_use` scorer.
- [ ] 8. Safety: tool allowlist/denylist, per-call timeout and max steps in config; disallowed calls recorded as errors; tests.
- [ ] 9. OpenAI function-calling support in the agent loop (provider-agnostic), with tests via the fake OpenAI client.
- [ ] 10. Docs: README "Evaluating MCP servers" section with the inventory example end to end.

## Statistical rigor
- [ ] 11. Core: `ai-eval run --repeats N` — N samples per case stored per outcome; case pass rate and consistency.
- [ ] 12. Tests for repeats (storage, summary math, concurrency with repeats).
- [ ] 13. Core: `ai_eval/stats.py` — bootstrap confidence intervals for pass rate and mean score (seeded), with tests.
- [ ] 14. Core: significance in `compare` — McNemar test for paired pass/fail flips, paired bootstrap for score deltas; report p-values and CIs.
- [ ] 15. Tests for compare significance (clear regression, noise-only change, small samples).
- [ ] 16. Core: flaky-case detection (mixed outcomes across repeats) in run summary and CLI output.
- [ ] 17. Report: CIs and flaky badges in the run and comparison HTML reports; tests.
- [ ] 18. Wiring: `compare --alpha` / `--min-effect` so only significant regressions fail; CI regression gate uses them.
- [ ] 19. Docs: README "Statistical rigor" section (why evals are noisy, how to read CIs and p-values).

## Judge calibration
- [ ] 20. Core: human label support — `human_label` (pass + optional score) on cases or a separate labels JSONL keyed by case id; loader + validation.
- [ ] 21. Core: `ai-eval calibrate --suite S --labels L` — run the judge and compute accuracy, Cohen's kappa, confusion matrix and score correlation vs. humans.
- [ ] 22. Tests for calibration metrics (perfect, chance-level, skewed classes, missing labels).
- [ ] 23. Data: 20-case labeled calibration set + `configs/calibration_suite.yaml` (mock outputs stored so it runs offline).
- [ ] 24. Report: HTML calibration report (agreement summary, confusion matrix, disagreement list).
- [ ] 25. Tests for the calibration report and CLI (`--min-kappa` exit codes).
- [ ] 26. Docs: README "Can you trust the judge?" section.

## Multi-turn conversations
- [ ] 27. Core: `EvalCase.messages` (prior conversation turns) with validation; single-turn cases unchanged.
- [ ] 28. Core: openai, anthropic, http and mock adapters send conversation history.
- [ ] 29. Tests for multi-turn adapters and backward compatibility.
- [ ] 30. Core: conversation-aware scoring — judge sees the full conversation; `conversation` judge mode for whole-dialogue criteria.
- [ ] 31. Tests + `configs/multiturn_suite.yaml` with an offline example dataset.
- [ ] 32. Docs: README "Multi-turn conversations" section.

## Test-set generation
- [ ] 33. Core: document loader + chunker (md/txt, directory walk, stable chunk ids).
- [ ] 34. Core: `ai-eval generate --docs DIR --out FILE --n N` — LLM writes question/answer/expected_doc_ids cases across categories (answerable, unanswerable, multi-hop); `--provider mock` for offline templated generation.
- [ ] 35. Tests for generation with fake clients and the mock provider.
- [ ] 36. Core: quality filters — grounding check against the source chunk, near-duplicate question removal, length limits; tests.
- [ ] 37. Example: generated dataset from the repo's own docs (mock provider) + suite config.
- [ ] 38. Docs: README "Generating test sets" section.

## Distribution and polish
- [ ] 39. Packaging: complete PyPI metadata (classifiers, URLs, license), `python -m build` + `twine check` job in CI.
- [ ] 40. Release workflow: publish to PyPI via trusted publishing on `v*` tags (documented one-time PyPI setup for the owner).
- [ ] 41. Core: `ai-eval trends --suite S` — pass rate, mean score and cost across stored runs, as a table and an HTML page.
- [ ] 42. Tests for trends.
- [ ] 43. Core: `ai-eval init` — scaffold a suite YAML, dataset and CI workflow in a new project; tests.
- [ ] 44. Docs site: mkdocs-material from README sections + GitHub Pages deploy workflow.
- [ ] 45. Release v0.3.0: CHANGELOG, version bump in pyproject and `__init__`, README "What's new", git tag `v0.3.0` pushed.

---

# Roadmap — v0.2 (Oct 22–28, 2026)

Four shippable steps per day. Each step is one commit that passes `pytest` on its own.
Steps are ticked (`[x]`) in the same commit that completes them.

## Thu Oct 22 — PR regression gate
- [x] 1. Core: `ai-eval run --save-baseline PATH` / `ai-eval compare --baseline-file PATH` so runs can be compared across machines (export/import a run as JSON).
- [x] 2. Tests: round-trip export/import, compare against an imported baseline, exit codes.
- [x] 3. Wiring: CI job on `pull_request` that runs the mock suite on `main` and on the PR head, then `ai-eval compare --fail-on-score-drop`; bump `actions/checkout@v5` and `actions/setup-python@v6`.
- [x] 4. Docs: README "CI regression gate" section with the workflow snippet.

## Fri Oct 23 — HTTP adapter (evaluate any service)
- [x] 1. Core: `adapters/http.py` — POST JSON to `url`, request body built from a template with `{input}`/`{context}`, output extracted with a dotted `response_path` (e.g. `data.answer`), optional headers (env-var interpolation like `${API_TOKEN}`), timeout.
- [x] 2. Tests: use `httpx.MockTransport`; success, non-2xx, timeout, bad `response_path`, header env interpolation.
- [x] 3. Wiring: register `http` in `build_adapter`; `configs/http_example.yaml`; optional `trace_path` to capture retrieved docs/tool calls into `RunResult.trace`.
- [x] 4. Docs: README adapter table + HTTP example.

## Sat Oct 24 — Retrieval metrics
- [x] 1. Core: `scorers/retrieval.py` — `recall@k`, `MRR`, `hit_rate`, comparing `trace["retrieved_ids"]` to `case.metadata["expected_doc_ids"]`; configurable `k` and `metric`, pass threshold.
- [x] 2. Tests: perfect/partial/zero recall, MRR ranking, missing trace or expected ids (skip), k larger than list.
- [x] 3. Wiring: register `retrieval` scorer; mock adapter can emit `retrieved_ids` from `case.metadata["mock_retrieved_ids"]`; `configs/retrieval_suite.yaml` + dataset.
- [x] 4. Docs: README "Evaluating retrieval vs. generation" section.

## Sun Oct 25 — Agent trajectory scorer
- [x] 1. Core: `scorers/trajectory.py` — compare `trace["tool_calls"]` (list of `{name, args}`) to `case.metadata["expected_tools"]`; modes `exact_order`, `subset`, `any_order`; optional arg matching; score = fraction matched.
- [x] 2. Tests: each mode, extra/missing calls, arg mismatch, no trace (skip).
- [x] 3. Wiring: register `trajectory`; mock adapter emits `tool_calls` from `case.metadata["mock_tool_calls"]`; `configs/agent_suite.yaml` + dataset.
- [x] 4. Docs: README "Evaluating agents" section; extend `EvalCase` docs with `expected_tools`.

## Mon Oct 26 — Reliability and cost
- [x] 1. Core: retry with exponential backoff + jitter for provider calls in `llm.py` (retry 429/5xx/timeouts, configurable `max_retries`).
- [x] 2. Core: on-disk judge cache in `.ai-eval/cache/` keyed by hash of (provider, model, prompt); `--no-cache` flag.
- [x] 3. Core: cost tracking — per-model price table (overridable in suite YAML), `cost_usd` in run metadata and CLI summary.
- [x] 4. Tests for retries (fake client raising then succeeding), cache hit/miss, cost math; README updates.

## Tue Oct 27 — Report v2
- [x] 1. Core: HTML report summary cards (pass rate, mean score, errors, p50/p95, cost) and per-tag table.
- [x] 2. Core: failures-first ordering, collapsible case details (scores, trace, judge reasoning).
- [x] 3. Core: `ai-eval report --compare A B` — side-by-side HTML diff highlighting regressions/improvements.
- [x] 4. Tests (HTML contains expected sections, escaping of untrusted output) + README screenshot-free description.

## Wed Oct 28 — Benchmark harness and v0.2.0
- [x] 1. Data: grow `datasets/examples/rag_v1.jsonl` to ~30 cases (answerable, unanswerable/abstain, conflicting context, multi-hop), tagged.
- [x] 2. Core: `ai-eval bench --suite S --adapter A:model --adapter B:model` — runs several systems and prints a comparison table (markdown output option).
- [x] 3. Tests for `bench` with mock adapters; `CHANGELOG.md` for v0.2.0; version bump in `pyproject.toml` and `__init__.py`.
- [x] 4. Release: README "Benchmarks" section (how to run; results table to be filled from a live run with real API keys), git tag `v0.2.0` pushed.
