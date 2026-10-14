from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from evalkit.assertions import AssertionResult
from evalkit.compare import compare
from evalkit.report import to_junit, to_markdown
from evalkit.results import CaseResult, RunResult, percentile


def trial(
    case_id: str, passed: bool, provider: str = "p", repeat: int = 0, latency: float = 100.0
) -> CaseResult:
    assertion = AssertionResult(
        type="contains", passed=passed, score=float(passed), reason="r | pipe"
    )
    return CaseResult(
        case_id=case_id,
        provider=provider,
        repeat=repeat,
        output="out",
        latency_ms=latency,
        input_tokens=10,
        output_tokens=2,
        assertions=[assertion],
    )


def run_of(*results: CaseResult, threshold: float = 1.0) -> RunResult:
    providers = list(dict.fromkeys(r.provider for r in results))
    return RunResult("suite", "2026-10-14T00:00:00+00:00", 1.0, threshold, providers, list(results))


@pytest.mark.parametrize(("pct", "expected"), [(50, 2), (95, 4), (100, 4), (1, 1)])
def test_percentile_nearest_rank(pct: float, expected: float) -> None:
    assert percentile([4, 1, 3, 2], pct) == expected


def test_percentile_of_nothing_is_zero() -> None:
    assert percentile([], 50) == 0.0


def test_weighted_score_and_error_handling() -> None:
    r = CaseResult(
        "c",
        "p",
        0,
        "o",
        1.0,
        assertions=[
            AssertionResult("a", True, 1.0, "", weight=3),
            AssertionResult("b", False, 0.0, "", weight=1),
        ],
    )
    assert r.score == 0.75
    assert not r.passed
    assert CaseResult("c", "p", 0, "", 0.0, error="x").score == 0.0
    assert CaseResult("c", "p", 0, "", 0.0).score == 1.0


def test_threshold_decides_overall_pass() -> None:
    results = [trial("a", True), trial("b", True), trial("c", False)]
    assert not run_of(*results).passed
    assert run_of(*results, threshold=0.6).passed


def test_results_roundtrip_through_json(tmp_path: Path) -> None:
    original = run_of(trial("a", True), trial("b", False, provider="q"))
    path = tmp_path / "results.json"
    original.save(path)
    loaded = RunResult.load(path)
    assert loaded.to_dict() == original.to_dict()


def test_results_reject_unknown_format() -> None:
    with pytest.raises(ValueError, match="unsupported results format"):
        RunResult.from_dict({"format": 99})


def test_compare_finds_regressions_fixes_and_membership_changes() -> None:
    baseline = run_of(
        trial("stable", True), trial("breaks", True), trial("heals", False), trial("gone", True)
    )
    current = run_of(
        trial("stable", True), trial("breaks", False), trial("heals", True), trial("new", True)
    )
    cmp = compare(baseline, current)
    assert [c.case_id for c in cmp.regressions] == ["breaks"]
    assert [c.case_id for c in cmp.fixes] == ["heals"]
    assert cmp.added == (("p", "new"),)
    assert cmp.removed == (("p", "gone"),)
    assert cmp.has_regressions


def test_markdown_report_has_summary_and_escaped_failures() -> None:
    run = run_of(trial("a", True), trial("b", False))
    md = to_markdown(run, compare(run_of(trial("b", True)), run))
    assert md.startswith("## ❌ evalkit: suite")
    assert "| `p` | 50.0% (1/2)" in md
    assert "🔻 | `b`" in md
    assert "r \\| pipe" in md


def test_junit_counts_failures_and_errors() -> None:
    errored = CaseResult("e", "p", 0, "", 0.0, error="timeout")
    xml = to_junit(run_of(trial("a", True), trial("b", False), trial("b", True, repeat=1), errored))
    suite = ET.fromstring(xml.split("\n", 1)[1]).find("testsuite")
    assert suite is not None
    assert (suite.get("tests"), suite.get("failures"), suite.get("errors")) == ("4", "1", "1")
    names = [tc.get("name") for tc in suite.iter("testcase")]
    assert names == ["a", "b", "b#1", "e"]
