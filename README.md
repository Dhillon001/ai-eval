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

| field | type | used by |
|---|---|---|
| `id` | string, required | everything — results, reports and `compare` match on it |
| `input` | string, required | the adapter (`{input}` in templates) |
| `context` | string | adapters (`{context}`), `faithfulness` |
| `expected` | string | `exact`, `llm_judge` |
| `checks` | list of strings | `checklist` |
| `metadata.tags` | list of strings | per-tag pass rates in `compare` |
| `metadata.judge_criteria` | string | `llm_judge` |
| `metadata.expected_doc_ids` | list of doc ids | `retrieval` — see [Evaluating retrieval vs. generation](#evaluating-retrieval-vs-generation) |
| `metadata.expected_tools` | list of tool names or `{name, args}` | `trajectory` — see [Evaluating agents](#evaluating-agents) |
| `metadata.mock_retrieved_ids` / `metadata.mock_tool_calls` | as above | `mock` adapter only: copied into the trace so suites run offline |

Other `metadata` keys are kept and stored with the run.

## Adapters

The adapter is the system under test. Set it with `adapter.type` in the suite YAML
(other keys under `adapter:` are its options), or override with `--adapter` / `--model`.

| type | what it calls | key options |
|---|---|---|
| `mock` | nothing — deterministic `echo:{input}` output for CI and local dev | `template` |
| `openai` | OpenAI Chat Completions, or any OpenAI-compatible server via `base_url` (`OPENAI_API_KEY`) | `model`, `base_url`, `system_prompt`, `temperature`, `max_tokens` |
| `anthropic` | Anthropic Messages API (`ANTHROPIC_API_KEY`) | `model`, `system_prompt`, `temperature`, `max_tokens` |
| `http` | any JSON-over-HTTP service — your own RAG app, agent, or gateway | `url`, `body_template`, `response_path`, `headers`, `timeout`, `method`, `trace_path` |

### HTTP adapter

Point the suite at your service and describe how to build the request and where the answer is
in the response ([`configs/http_example.yaml`](configs/http_example.yaml)):

```yaml
adapter:
  type: http
  url: http://localhost:8000/answer
  method: POST                  # default POST
  timeout: 30                   # seconds, default 30
  body_template:                # default {"input": "{input}", "context": "{context}"}
    question: "{input}"
    context: "{context}"
  response_path: data.answer    # dotted path; list indexes allowed (choices.0.message.content)
  headers:
    Authorization: "Bearer ${API_TOKEN}"
  trace_path:                   # optional: copy response fields into RunResult.trace
    retrieved_ids: data.retrieved_ids
    tool_calls: data.tool_calls
```

```bash
export API_TOKEN=...
ai-eval run --suite configs/http_example.yaml
```

For a case `{"input": "Can I get a refund?"}` this POSTs
`{"question": "Can I get a refund?", "context": null}` and, given a response like
`{"data": {"answer": "Yes, within 30 days.", "retrieved_ids": ["policy-3"]}}`, scores the output
`Yes, within 30 days.` with `trace.retrieved_ids == ["policy-3"]`.

- **Templates**: a string that is exactly `"{input}"` or `"{context}"` is replaced by the raw value
  (missing context → `null`); placeholders inside longer strings are substituted as text (missing
  context → empty string). Nested objects and lists are rendered recursively.
- **Headers**: `${NAME}` is read from the environment. If a variable is unset the case fails
  with an error and no request is sent, so secrets never live in the YAML.
- **Output**: a string at `response_path` is used as-is; any other JSON value is serialized.
- **Trace**: `trace_path` is either a mapping of `trace key → dotted path` or one dotted path to
  an object whose keys are merged in. Unresolved paths are listed in `trace.trace_missing` rather
  than failing the case. The trace always records `url`, `method`, the rendered `request`, and
  `status_code`.
- **Errors**: timeouts, connection errors, non-2xx responses (with the first 200 chars of the
  body), invalid JSON, and a missing `response_path` are recorded as case errors; the run continues.

## Scorers

| type | what it checks |
|---|---|
| `exact` | normalized output == `expected` |
| `checklist` | every string in `checks` appears in the output |
| `llm_judge` | LLM grades output against `metadata.judge_criteria` / `expected` (`provider: openai\|anthropic`) |
| `faithfulness` | LLM checks every claim is supported by `context` (RAG); skips cases without context |
| `retrieval` | ranked `trace.retrieved_ids` vs. `metadata.expected_doc_ids` — `recall@k`, `hit_rate`, `mrr`; skips cases without either |
| `trajectory` | agent `trace.tool_calls` vs. `metadata.expected_tools` — `exact_order`, `any_order`, `subset`, optional arg matching; skips cases without either |

## Evaluating retrieval vs. generation

A RAG system can fail in two places: the retriever didn't fetch the right documents, or the
generator had the right documents and still answered badly. Score them separately so a
regression points at the stage that caused it.

| stage | question | scorers | needs |
|---|---|---|---|
| retrieval | did the right docs come back, ranked high enough? | `retrieval` | `trace.retrieved_ids`, `metadata.expected_doc_ids` |
| generation | is the answer correct and grounded in the context? | `exact`, `checklist`, `llm_judge`, `faithfulness` | `expected` / `checks` / `context` |

**Retrieval cases** list the relevant doc ids per case; the system under test reports the ids
it retrieved, in rank order:

```json
{"id": "ret-final-sale", "input": "Can I return a final-sale jacket?",
 "metadata": {"expected_doc_ids": ["policy-returns", "policy-final-sale"]}}
```

The `http` adapter fills the trace from your service's response with `trace_path`
(e.g. `trace_path: {retrieved_ids: retrieval.doc_ids}`). Offline, the `mock` adapter copies
`metadata.mock_retrieved_ids` into `trace.retrieved_ids`, which is how
[`configs/retrieval_suite.yaml`](configs/retrieval_suite.yaml) runs without a service:

```yaml
scorers:
  - type: retrieval
    metric: recall@3       # every expected doc must appear in the top 3
  - type: retrieval
    metric: hit_rate
    k: 1                   # is the top-ranked doc relevant? (reported, not gated)
    threshold: 0.0
  - type: retrieval
    metric: mrr
    threshold: 0.5         # first relevant doc ranked 1st or 2nd
```

```bash
ai-eval run --suite configs/retrieval_suite.yaml
```

Metrics are computed over the top `k` retrieved ids (no `k` = the whole list; duplicate ids
are counted once):

- `recall@k` — fraction of expected ids found in the top k. Default threshold `1.0`.
- `hit_rate` — `1.0` if any expected id is in the top k, else `0.0`. Default threshold `1.0`.
- `mrr` — `1 / rank` of the first expected id in the top k, `0.0` if none. Default threshold `0.5`.

`k` can be set as an option or in the metric name (`recall@5`, `hit@1`). Each scorer is
reported as `retrieval:<metric>@<k>`, so several can sit in one suite. Case details include
`found`, `missing` and `first_relevant_rank`. Cases without `expected_doc_ids` or without
`retrieved_ids` in the trace are skipped (pass, `details.skipped`); a non-list value is a
failure. Override the field names with `trace_key` / `expected_key`.

**Both stages in one suite**: give cases `context` + `checks` and `expected_doc_ids`, then list
`retrieval` next to `checklist` / `faithfulness`. Reading the per-scorer results together:

- retrieval fails, generation fails → fix the retriever first; the generator never saw the answer.
- retrieval passes, generation fails → the problem is the prompt or model.
- retrieval fails, generation passes → the answer came from the model's own knowledge or a
  doc you didn't label; check `expected_doc_ids` and faithfulness.

`compare` reports per-scorer mean deltas, so a PR that changes chunking or embeddings shows up
as a `retrieval:*` drop rather than an unexplained answer-quality regression.

## Evaluating agents

An agent can reach the right answer by the wrong path — skipping a lookup, calling a refund
tool twice, or guessing an argument. The `trajectory` scorer checks the tool calls the agent
made against the ones each case expects, alongside the usual output scorers.

**Agent cases** list the expected calls in `metadata.expected_tools`. Each entry is a bare tool
name (`"lookup_order"`) or `{name, args}`; `args` only matter when the scorer sets
`match_args: true`:

```json
{"id": "agent-order-status", "input": "Where is my order A1001?",
 "metadata": {"expected_tools": [
   {"name": "lookup_order", "args": {"order_id": "A1001"}},
   {"name": "get_shipping_status", "args": {"tracking_id": "1Z999"}}]}}
```

The system under test reports what it actually called as `trace.tool_calls`, a list in call
order. Entries may be `{name, args}`, `{name, arguments}` (arguments may be a JSON string, as
OpenAI returns them), OpenAI-style `{"function": {...}}`, or bare names. With the `http`
adapter, map it from your service's response (`trace_path: {tool_calls: steps.tool_calls}`).
Offline, the `mock` adapter copies `metadata.mock_tool_calls` into `trace.tool_calls`, which is
how [`configs/agent_suite.yaml`](configs/agent_suite.yaml) runs without a service:

```yaml
scorers:
  - type: trajectory
    mode: subset           # every expected call must happen, with the expected args ...
    match_args: true       # ... extra calls (e.g. a policy check) are allowed
  - type: trajectory
    mode: any_order        # same set of calls, order ignored, extras penalized
    threshold: 0.5
  - type: trajectory
    mode: exact_order      # strict sequence match, args included - reported, not gated
    match_args: true
    threshold: 0.0
```

```bash
ai-eval run --suite configs/agent_suite.yaml
```

The score is the fraction of calls matched, in [0, 1]; a case passes when it reaches
`threshold` (default `1.0`).

| mode | order | extra calls | score |
|---|---|---|---|
| `exact_order` (default) | must match | penalized | longest in-order match / max(expected, actual) |
| `any_order` | ignored | penalized | matched / max(expected, actual) |
| `subset` | ignored | allowed | matched / expected |

- **Arg matching** (`match_args: true`): an expected entry with `args` matches a call whose args
  contain those key/value pairs — extra actual args (say, a `carrier` the agent added) are fine.
  Expected entries without `args` match on name alone. Repeated tools are matched one-to-one,
  so expecting `get_weather` for Paris and Tokyo needs two calls.
- **Details**: each result lists `matched`, `missing`, `extra` (reported but not penalized in
  `subset` mode) and `arg_mismatches` — expected tools that were called, but with the wrong
  args.
- **Skips and errors**: cases without `expected_tools` or without `tool_calls` in the trace are
  skipped (pass, `details.skipped`). An explicit empty `expected_tools: []` means "no tools
  should be called", so any call fails it in the strict modes. Malformed entries (no name,
  non-object args) fail the case. Override the field names with `trace_key` / `expected_key`.

Each scorer is reported as `trajectory:<mode>`, so one suite can gate on a lenient mode and
just report a strict one, as above. Pair it with `checklist` or `llm_judge` on the final answer
to tell "right tools, wrong answer" from "right answer, wrong tools".

## Compare

`compare` reports pass→fail flips (`regressions`), score drops >0.1 on cases that didn't flip,
improvements, missing/new cases, per-scorer mean deltas and per-tag pass-rate deltas.
Exits 1 on regressions (and on score drops with `--fail-on-score-drop`) so it can gate CI.

Runs can also be compared across machines: `run --save-baseline PATH` exports a run
(summary, per-case outcomes, cases) as JSON, and `compare` accepts `--baseline-file` /
`--candidate-file` in place of run ids (you can mix a file with a local run id).

```bash
ai-eval run --suite configs/example_suite.yaml --adapter mock --save-baseline baseline.json
# ...later, or on another machine...
ai-eval run --suite configs/example_suite.yaml --adapter mock --save-baseline candidate.json
ai-eval compare --baseline-file baseline.json --candidate-file candidate.json --fail-on-score-drop
```

## CI regression gate

`.github/workflows/ci.yml` has a `regression-gate` job that runs on every pull request.
It checks out the PR's base commit and head commit side by side, runs the same suite on
each with `--save-baseline`, then runs `compare --fail-on-score-drop`. The job fails if
any case flips pass→fail or any score drops by more than 0.1, and both run files are
uploaded as the `regression-gate-runs` artifact for inspection.

```yaml
  regression-gate:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
        with:
          ref: ${{ github.event.pull_request.base.sha }}
          path: base
      - uses: actions/checkout@v5
        with:
          ref: ${{ github.event.pull_request.head.sha }}
          path: head
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - name: Eval on base
        working-directory: base
        run: |
          pip install -e ".[dev]"
          ai-eval run --suite configs/example_suite.yaml --adapter mock \
            --save-baseline "$RUNNER_TEMP/baseline.json"
      - name: Eval on PR head
        working-directory: head
        run: |
          pip install -e ".[dev]"
          ai-eval run --suite configs/example_suite.yaml --adapter mock \
            --save-baseline "$RUNNER_TEMP/candidate.json"
      - name: Compare
        working-directory: head
        run: |
          ai-eval compare --fail-on-score-drop \
            --baseline-file "$RUNNER_TEMP/baseline.json" \
            --candidate-file "$RUNNER_TEMP/candidate.json"
```

Notes:

- The mock adapter keeps the gate free and deterministic. To gate on a real model, swap in
  `--adapter openai --model ...` (or a suite using `anthropic`) and expose the API key as a
  repository secret via `env:`; expect some run-to-run noise from sampling.
- Each checkout installs its own copy of the package, so a PR that changes scorers or
  adapters is evaluated with its own code while the base uses the base code.
- `compare` runs from the head checkout, so the comparison logic is the PR's version.

## Extending

- **Adapters**: implement `SystemAdapter` in `src/ai_eval/adapters/`
- **Scorers**: implement `Scorer` in `src/ai_eval/scorers/`
- **RAG / agents**: add fields like `context` and `tools` on cases; use white-box traces in `RunResult.trace`
