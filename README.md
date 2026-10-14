# evalkit

[![CI](https://github.com/omergazi34/evalkit/actions/workflows/ci.yml/badge.svg)](https://github.com/omergazi34/evalkit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%E2%80%933.14-blue)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Regression tests for LLM applications.** Describe what good output looks like in YAML. evalkit runs
the cases against one or more models in parallel, grades every answer, and **fails CI when quality
drops**.

```yaml
name: support-bot
provider: { type: anthropic, model: claude-opus-5-5, effort: low }
judge:    { type: anthropic, model: claude-opus-5-5 }
system: You are the support assistant for Kargo Express...

cases:
  - id: refund-window
    input: I received my parcel 10 days ago and haven't opened it. Can I get a refund?
    assert:
      - { type: contains, value: "14" }
      - { type: not_regex, pattern: "(?i)card number|password" }
      - type: llm_judge
        rubric: Confirms eligibility because 10 days is within the 14-day window.
```

## Why

When you change a prompt, swap a model or bump a temperature, something that used to work quietly
breaks. Most teams find out from users. evalkit gives you a test suite for that, the same way pytest
does for functions. It is built around three ideas:

- **Deterministic checks first, judges second.** Cheap, exact assertions (`contains`, `regex`,
  `json_schema`, `latency`) catch most breakage. An LLM judge handles what only a human-like reader
  can grade.
- **Treat non-determinism as a measurement.** `repeat: N` runs each case N times. Pass *rates* and
  flaky cases are reported, not hidden.
- **Built for CI.** Exit codes, JUnit XML, a Markdown job summary, baseline diffs and a ready-made
  GitHub Action.

## Real run: Llama 3 on a laptop

Output of [`examples/local-ollama.yaml`](examples/local-ollama.yaml) against `llama3` via Ollama, 7
cases × 3 repeats:

```text
evalkit · local-ollama
+-------------------------------------------------------------------------------------+
| provider |     pass rate | score |   p50 |     p95 | tokens in/out | errors | flaky |
|----------+---------------+-------+-------+---------+---------------+--------+-------|
| llama3   | 85.7% (18/21) |  0.86 | 216ms | 66499ms |       900/143 |      0 |     0 |
+-------------------------------------------------------------------------------------+

3 failing trial(s)
✗ json-extraction (llama3, repeat 0)
    json_schema: invalid JSON: Expecting value
    output: 'Here is the extracted order in JSON format: { "item": "USB-C cables", ...'
```

The suite caught a real problem. Asked for "only the JSON object", the model wrapped it in prose
every time, so any code calling `json.loads` on that output would crash. (The p95 latency is the
model's cold start on the first request.)

## Install

```bash
pip install git+https://github.com/omergazi34/evalkit
evalkit init            # writes an offline example suite to evalkit.yaml
evalkit run evalkit.yaml
```

Requires Python 3.11+. Set `ANTHROPIC_API_KEY` for Claude models. Local models through Ollama or
LM Studio need no key.

## Features

| | |
|---|---|
| **Providers** | Claude (official Anthropic SDK); any OpenAI-compatible server: Ollama, LM Studio, vLLM, OpenAI; a deterministic `mock` for offline tests |
| **Model comparison** | List several `providers:` and get a side-by-side table: same cases, same grading |
| **Assertions** | `equals` `contains` `contains_all` `contains_any` `starts_with` `regex` `json_valid` `json_schema` `length` `latency` `llm_judge` `python` |
| **Negation** | Prefix any assertion with `not_` (`not_contains`, `not_regex`, …) |
| **Data-driven cases** | A suite-level `prompt` template filled from each case's `vars`; assertion values are templated too |
| **Multi-turn** | `messages:` instead of `input:` for conversation-history cases |
| **Repeats & flakiness** | `repeat: N`, pass-rate thresholds, flaky-case detection |
| **Caching** | Content-addressed SQLite cache. Re-runs only pay for cases that changed |
| **Concurrency** | Async fan-out with a bounded semaphore (`-j`), retries with exponential backoff and jitter |
| **Reports** | Terminal table, JSON, Markdown (PR comments / job summaries), JUnit XML |
| **Regression gating** | `--baseline previous.json` fails the build when any case gets worse, even if still above threshold |
| **Fails fast on bad config** | Typos, unknown fields, undefined template variables and missing judges are reported with a path into the YAML before any model is called |

## Suite reference

```yaml
name: my-suite                # required
system: "..."                 # optional system prompt ({{vars}} allowed)
prompt: "Classify: {{text}}"  # optional template for cases without `input`
repeat: 3                     # run every case N times (default 1)
threshold: 0.9                # minimum pass rate per provider (default 1.0)

providers:                    # or `provider:` for a single one
  - id: opus                  # label in reports (default "<type>:<model>")
    type: anthropic           # anthropic | openai | mock
    model: claude-opus-5-5
    effort: low               # anthropic: low | medium | high | xhigh | max
    max_tokens: 1024
    params: {}                # passed through to the API verbatim
  - type: openai
    base_url: http://localhost:11434/v1
    model: llama3
    temperature: 0.7
    api_key_env: OPENAI_API_KEY   # optional; name of the env var holding the key

judge:                        # required if any case uses llm_judge
  type: anthropic
  model: claude-opus-5-5

defaults:
  assert:                     # prepended to every case
    - { type: latency, max_ms: 30000 }

cases:
  - id: unique-id
    input: "..."              # or `messages: [{role, content}, ...]`, or `vars` + suite `prompt`
    vars: { text: "..." }
    tags: [smoke]
    assert:
      - { type: json_schema, schema: { type: object, required: [answer] } }
      - { type: llm_judge, rubric: "...", threshold: 0.7 }
      - { type: python, function: "my_checks:is_valid" }   # module next to the suite file
```

Custom `python` assertions receive `(output: str, context: dict)` and return `bool`, `(bool, reason)`
or `{"pass": bool, "score": float, "reason": str}`.

## CLI

```text
evalkit run SUITE [-p PROVIDER] [-k CASE_GLOB] [-t TAG] [-j N]
                  [-o results.json] [--markdown report.md] [--junit junit.xml]
                  [--threshold 0.9] [--baseline previous.json] [--no-cache]
evalkit validate SUITE...      # check config without calling any model
evalkit compare BASE CURRENT   # diff two results files
evalkit init [PATH]            # write an example suite
```

Exit codes: `0` passed · `1` below threshold or regressed · `2` invalid configuration.

## GitHub Action

```yaml
# .github/workflows/evals.yml
on: pull_request
jobs:
  evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: omergazi34/evalkit@main
        with:
          suite: evals/support-bot.yaml
          threshold: "0.9"
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

The Markdown report is written to the job summary, and the job fails on a threshold miss or a
regression. This repository's own CI runs the action against
[`examples/quickstart.yaml`](examples/quickstart.yaml) (the `dogfood` job).

## Design

[DESIGN.md](DESIGN.md) explains the architecture and the trade-offs: why the judge reasons before
scoring, why cached responses keep their original latency, why refusals are data rather than errors,
and why server-side model fallbacks are deliberately left off.

```text
YAML ─► config.py (pydantic) ─► runner.py ─┬─► providers/ (claude · openai_compat · mock)
                                           ├─► cache.py (SQLite, content-addressed)
                                           └─► assertions.py ─► judge.py
                                                  │
                    results.py ◄──────────────────┘
                        │
                        ├─► report.py (terminal · Markdown · JUnit)
                        └─► compare.py (baseline regressions)
```

## Development

```bash
uv sync
uv run pytest --cov      # 110 tests, ~95% coverage, no network or API key needed
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

## License

MIT
