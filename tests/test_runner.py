from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

from evalkit.cache import ResponseCache, cache_key
from evalkit.providers.base import Completion
from evalkit.runner import Runner
from tests.conftest import ScriptedProvider, make_suite

CASES = [
    {"id": "good", "input": "say ankara", "assert": [{"type": "contains", "value": "ankara"}]},
    {"id": "bad", "input": "say izmir", "assert": [{"type": "contains", "value": "ankara"}]},
]


def test_runs_every_provider_case_and_repeat_in_order() -> None:
    suite = make_suite(cases=CASES, repeat=2)
    a, b = ScriptedProvider("a"), ScriptedProvider("b")
    run = asyncio.run(Runner(suite, [a, b]).run())
    assert [(r.provider, r.case_id, r.repeat) for r in run.results] == [
        ("a", "good", 0),
        ("a", "good", 1),
        ("a", "bad", 0),
        ("a", "bad", 1),
        ("b", "good", 0),
        ("b", "good", 1),
        ("b", "bad", 0),
        ("b", "bad", 1),
    ]
    summary = {s.provider: s for s in run.summaries()}
    assert summary["a"].pass_rate == 0.5
    assert not run.passed


def test_concurrency_limit_is_respected() -> None:
    suite = make_suite(cases=[{"id": f"c{i}", "input": "x"} for i in range(20)])
    provider = ScriptedProvider(delay=0.01)
    asyncio.run(Runner(suite, [provider], concurrency=3).run())
    assert provider.max_in_flight == 3


def test_provider_errors_become_failed_trials_not_crashes() -> None:
    suite = make_suite(cases=CASES)
    run = asyncio.run(Runner(suite, [ScriptedProvider(fail=True)]).run())
    assert all(r.error and "boom" in r.error for r in run.results)
    assert run.summaries()[0].errors == 2


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[ResponseCache]:
    cache = ResponseCache(tmp_path)
    yield cache
    cache.close()


def test_cache_skips_repeat_calls_and_keeps_latency(cache: ResponseCache) -> None:
    suite = make_suite(cases=CASES)
    provider = ScriptedProvider(delay=0.02)
    first = asyncio.run(Runner(suite, [provider], cache=cache).run())
    second = asyncio.run(Runner(suite, [provider], cache=cache).run())
    assert len(provider.calls) == 2
    assert [r.cached for r in second.results] == [True, True]
    assert second.results[0].latency_ms == first.results[0].latency_ms
    assert (cache.hits, cache.misses) == (2, 2)


def test_cache_key_changes_with_anything_that_affects_output() -> None:
    base = cache_key({"model": "a"}, "sys", [{"role": "user", "content": "x"}], 0)
    assert base == cache_key({"model": "a"}, "sys", [{"role": "user", "content": "x"}], 0)
    assert base != cache_key({"model": "b"}, "sys", [{"role": "user", "content": "x"}], 0)
    assert base != cache_key({"model": "a"}, None, [{"role": "user", "content": "x"}], 0)
    assert base != cache_key({"model": "a"}, "sys", [{"role": "user", "content": "y"}], 0)
    assert base != cache_key({"model": "a"}, "sys", [{"role": "user", "content": "x"}], 1)


def test_cache_roundtrips_completion(cache: ResponseCache) -> None:
    completion = Completion(text="ünïcode", input_tokens=1, output_tokens=2, metadata={"k": "v"})
    cache.put("key", completion, 12.5)
    assert cache.get("key") == (completion, 12.5)
    assert cache.get("other") is None


def test_llm_judge_grades_through_judge_provider() -> None:
    suite = make_suite(
        judge={"type": "mock"},
        cases=[
            {
                "id": "polite",
                "input": "greet me",
                "assert": [{"type": "llm_judge", "rubric": "Is polite"}],
            }
        ],
    )
    judge = ScriptedProvider(
        "judge", outputs=lambda req: '{"reasoning": "warm greeting", "score": 0.9}'
    )
    run = asyncio.run(Runner(suite, [ScriptedProvider()], judge=judge).run())
    (result,) = run.results
    assert result.passed
    assert "warm greeting" in result.assertions[0].reason
    assert "<output>\ngreet me\n</output>" in judge.calls[0].messages[0]["content"]


def test_broken_judge_fails_the_assertion_not_the_run() -> None:
    suite = make_suite(
        judge={"type": "mock"},
        cases=[{"id": "x", "input": "hi", "assert": [{"type": "llm_judge", "rubric": "r"}]}],
    )
    judge = ScriptedProvider("judge", outputs=lambda req: "I think it's fine")
    (result,) = asyncio.run(Runner(suite, [ScriptedProvider()], judge=judge).run()).results
    assert not result.passed
    assert "grading failed" in result.assertions[0].reason


def test_flaky_cases_are_detected() -> None:
    outputs = ["ankara", "izmir", "ankara"]
    suite = make_suite(cases=CASES[:1], repeat=3)
    run = asyncio.run(Runner(suite, [ScriptedProvider(outputs=outputs)], concurrency=1).run())
    (summary,) = run.summaries()
    assert summary.flaky_cases == ("good",)
    assert summary.passed == 2
