from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from evalkit.assertions import Context, Verdict, evaluate, parse_json_output
from evalkit.config import AssertionSpec

_adapter: TypeAdapter[AssertionSpec] = TypeAdapter(AssertionSpec)


def check(spec: dict[str, Any], output: str, **ctx: Any) -> tuple[bool, float, str]:
    parsed = _adapter.validate_python(spec)
    context = Context(output=output, latency_ms=ctx.pop("latency_ms", 100.0), case_id="c", **ctx)
    result = asyncio.run(evaluate(parsed, context))
    return result.passed, result.score, result.reason


@pytest.mark.parametrize(
    ("spec", "output", "passed"),
    [
        ({"type": "equals", "value": "Ankara"}, "  Ankara\n", True),
        ({"type": "equals", "value": "ankara", "ignore_case": True}, "Ankara", True),
        ({"type": "equals", "value": "Ankara", "strip": False}, " Ankara", False),
        ({"type": "contains", "value": "30 days"}, "Refunds within 30 days.", True),
        ({"type": "contains", "value": "REFUND", "ignore_case": True}, "refund policy", True),
        ({"type": "contains", "value": "x"}, "abc", False),
        ({"type": "contains", "value": "x", "negate": True}, "abc", True),
        ({"type": "contains_any", "values": ["yes", "sure"]}, "Sure thing", False),
        (
            {"type": "contains_any", "values": ["yes", "sure"], "ignore_case": True},
            "Sure thing",
            True,
        ),
        ({"type": "starts_with", "value": "Dear"}, "\n Dear user", True),
        ({"type": "regex", "pattern": r"\b\d{4}-\d{2}-\d{2}\b"}, "Due 2026-10-14.", True),
        ({"type": "regex", "pattern": "^yes$", "flags": "im"}, "Well\nYES", True),
        ({"type": "regex", "pattern": "^yes$"}, "Well\nYES", False),
        ({"type": "json_valid"}, '{"a": 1}', True),
        ({"type": "json_valid"}, '```json\n{"a": 1}\n```', True),
        ({"type": "json_valid"}, "{a: 1}", False),
        ({"type": "length", "min": 2, "max": 4}, "abc", True),
        ({"type": "length", "max": 2}, "abc", False),
        ({"type": "latency", "max_ms": 50}, "x", False),
    ],
)
def test_deterministic_assertions(spec: dict[str, Any], output: str, passed: bool) -> None:
    assert check(spec, output)[0] is passed


def test_contains_all_reports_partial_score() -> None:
    passed, score, reason = check({"type": "contains_all", "values": ["a", "b", "c", "d"]}, "a b")
    assert not passed
    assert score == 0.5
    assert "'c'" in reason


def test_negation_inverts_score_and_name() -> None:
    parsed = _adapter.validate_python({"type": "contains", "value": "x", "negate": True})
    result = asyncio.run(evaluate(parsed, Context(output="x", latency_ms=1, case_id="c")))
    assert (result.type, result.passed, result.score) == ("not_contains", False, 0.0)


def test_assertion_values_are_templated_with_case_vars() -> None:
    assert check(
        {"type": "contains", "value": "{{city}}"}, "I live in Izmir", vars={"city": "Izmir"}
    )[0]


SCHEMA = {
    "type": "object",
    "required": ["name", "age"],
    "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
}


@pytest.mark.parametrize(
    ("output", "passed", "reason"),
    [
        ('{"name": "Ada", "age": 36}', True, "matches schema"),
        ('{"name": "Ada", "age": -1}', False, "age"),
        ('{"name": "Ada"}', False, "'age' is a required property"),
        ("not json", False, "invalid JSON"),
    ],
)
def test_json_schema(output: str, passed: bool, reason: str) -> None:
    ok, _, why = check({"type": "json_schema", "schema": SCHEMA}, output)
    assert ok is passed
    assert reason in why


def test_parse_json_output_only_strips_a_single_fence() -> None:
    assert parse_json_output("```\n[1, 2]\n```") == [1, 2]
    with pytest.raises(ValueError, match="Expecting value"):
        parse_json_output("Here you go: ```json\n[1]\n```")


def test_llm_judge_uses_threshold() -> None:
    async def judge(rubric: str, output: str) -> Verdict:
        assert rubric == "Is polite to Ada"
        return Verdict(score=0.6, reasoning="mostly polite")

    spec = {"type": "llm_judge", "rubric": "Is polite to {{name}}", "threshold": 0.5}
    passed, score, reason = check(spec, "hello", judge=judge, vars={"name": "Ada"})
    assert passed
    assert score == 0.6
    assert "mostly polite" in reason
    assert not check({**spec, "threshold": 0.7}, "hello", judge=judge, vars={"name": "Ada"})[0]


@pytest.fixture
def checks_module(tmp_path: Path) -> str:
    (tmp_path / "my_checks.py").write_text(
        "def is_short(output, ctx):\n    return len(output) < 5\n"
        "def with_reason(output, ctx):\n    return (False, f'case {ctx[\"case_id\"]} failed')\n"
        "def as_dict(output, ctx):\n    return {'pass': True, 'score': 0.8, 'reason': 'ok'}\n"
        "def broken(output, ctx):\n    raise RuntimeError('bad')\n"
        "def weird(output, ctx):\n    return 42\n"
    )
    sys.path.insert(0, str(tmp_path))
    yield "my_checks"
    sys.path.remove(str(tmp_path))
    sys.modules.pop("my_checks", None)


@pytest.mark.parametrize(
    ("function", "passed", "reason"),
    [
        ("is_short", True, "-> True"),
        ("with_reason", False, "case c failed"),
        ("as_dict", True, "ok"),
        ("broken", False, "raised RuntimeError: bad"),
        ("weird", False, "unsupported value"),
        ("missing", False, "cannot load"),
    ],
)
def test_python_assertions(checks_module: str, function: str, passed: bool, reason: str) -> None:
    ok, _, why = check({"type": "python", "function": f"{checks_module}:{function}"}, "abc")
    assert ok is passed
    assert reason in why
