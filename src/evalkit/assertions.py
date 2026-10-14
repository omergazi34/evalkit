"""Assertion implementations. Each one maps (spec, output) -> AssertionResult with a 0..1 score."""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from evalkit import config as c
from evalkit.config import AssertionSpec, render

_FENCE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


@dataclass(frozen=True, slots=True)
class AssertionResult:
    type: str
    passed: bool
    score: float
    reason: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "passed": self.passed,
            "score": round(self.score, 4),
            "reason": self.reason,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class Outcome:
    passed: bool
    reason: str
    score: float | None = None

    @property
    def resolved_score(self) -> float:
        if self.score is not None:
            return self.score
        return 1.0 if self.passed else 0.0


JudgeFn = Callable[[str, str], Awaitable["Verdict"]]


@dataclass(frozen=True, slots=True)
class Verdict:
    score: float
    reasoning: str


@dataclass(slots=True)
class Context:
    output: str
    latency_ms: float
    case_id: str
    vars: dict[str, Any] = field(default_factory=dict)
    input_text: str = ""
    judge: JudgeFn | None = None


def parse_json_output(text: str) -> Any:
    """Parse model output as JSON, tolerating a single surrounding Markdown code fence."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fenced = _FENCE.match(text)
        if fenced:
            return json.loads(fenced.group(1))
        raise


def _fold(text: str, ignore_case: bool) -> str:
    return text.casefold() if ignore_case else text


def _short(text: str, limit: int = 60) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def _check(spec: AssertionSpec, ctx: Context) -> Outcome:
    out = ctx.output
    v = ctx.vars

    if isinstance(spec, c.Equals):
        expected = render(spec.value, v)
        actual = out.strip() if spec.strip else out
        ok = _fold(actual, spec.ignore_case) == _fold(expected, spec.ignore_case)
        return Outcome(ok, f"expected {expected!r}, got {_short(actual)!r}")

    if isinstance(spec, c.Contains):
        needle = render(spec.value, v)
        ok = _fold(needle, spec.ignore_case) in _fold(out, spec.ignore_case)
        return Outcome(ok, f"{'found' if ok else 'missing'} {needle!r}")

    if isinstance(spec, c.ContainsAll | c.ContainsAny):
        needles = [render(x, v) for x in spec.values]
        hay = _fold(out, spec.ignore_case)
        hits = [n for n in needles if _fold(n, spec.ignore_case) in hay]
        if isinstance(spec, c.ContainsAll):
            missing = [n for n in needles if n not in hits]
            score = len(hits) / len(needles)
            return Outcome(not missing, f"missing {missing}" if missing else "all found", score)
        return Outcome(bool(hits), f"found {hits}" if hits else f"none of {needles}")

    if isinstance(spec, c.StartsWith):
        prefix = render(spec.value, v)
        ok = _fold(out.lstrip(), spec.ignore_case).startswith(_fold(prefix, spec.ignore_case))
        return Outcome(ok, f"{'starts' if ok else 'does not start'} with {prefix!r}")

    if isinstance(spec, c.Regex):
        flags = 0
        for ch in spec.flags:
            flags |= {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}[ch]
        m = re.search(spec.pattern, out, flags)
        return Outcome(m is not None, f"matched {_short(m.group(0))!r}" if m else "no match")

    if isinstance(spec, c.JsonValid):
        try:
            parse_json_output(out)
        except json.JSONDecodeError as exc:
            return Outcome(False, f"invalid JSON: {exc.msg} at line {exc.lineno} col {exc.colno}")
        return Outcome(True, "valid JSON")

    if isinstance(spec, c.JsonSchema):
        try:
            data = parse_json_output(out)
        except json.JSONDecodeError as exc:
            return Outcome(False, f"invalid JSON: {exc.msg}")
        validator = jsonschema.Draft202012Validator(spec.schema_)
        errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
        if errors:
            first = errors[0]
            where = "/".join(str(p) for p in first.absolute_path) or "<root>"
            return Outcome(False, f"{len(errors)} schema error(s); {where}: {first.message}")
        return Outcome(True, "matches schema")

    if isinstance(spec, c.Length):
        n = len(out)
        ok = (spec.min is None or n >= spec.min) and (spec.max is None or n <= spec.max)
        return Outcome(ok, f"length {n} (min={spec.min}, max={spec.max})")

    if isinstance(spec, c.Latency):
        ok = ctx.latency_ms <= spec.max_ms
        return Outcome(ok, f"{ctx.latency_ms:.0f}ms (max {spec.max_ms:.0f}ms)")

    if isinstance(spec, c.LlmJudge):
        if ctx.judge is None:
            return Outcome(False, "no judge configured")
        verdict = await ctx.judge(render(spec.rubric, v), out)
        ok = verdict.score >= spec.threshold
        return Outcome(
            ok, f"judge {verdict.score:.2f} (≥{spec.threshold}): {verdict.reasoning}", verdict.score
        )

    if isinstance(spec, c.Python):
        return _run_python(spec, ctx)

    raise AssertionError(f"unhandled assertion type {spec!r}")  # pragma: no cover


def _run_python(spec: c.Python, ctx: Context) -> Outcome:
    module_name, func_name = spec.function.split(":")
    try:
        func = getattr(importlib.import_module(module_name), func_name)
    except (ImportError, AttributeError) as exc:
        return Outcome(False, f"cannot load {spec.function}: {exc}")
    context = {"case_id": ctx.case_id, "vars": ctx.vars, "latency_ms": ctx.latency_ms}
    try:
        result = func(ctx.output, context)
    except Exception as exc:  # user code: report, don't crash the run
        return Outcome(False, f"{spec.function} raised {type(exc).__name__}: {exc}")
    if isinstance(result, bool):
        return Outcome(result, f"{spec.function} -> {result}")
    if isinstance(result, tuple) and len(result) == 2:
        return Outcome(bool(result[0]), str(result[1]))
    if isinstance(result, dict) and "pass" in result:
        score = result.get("score")
        return Outcome(
            bool(result["pass"]),
            str(result.get("reason", "")),
            float(score) if score is not None else None,
        )
    return Outcome(False, f"{spec.function} returned unsupported value {result!r:.80}")


async def evaluate(spec: AssertionSpec, ctx: Context) -> AssertionResult:
    outcome = await _check(spec, ctx)
    passed, score, reason = outcome.passed, outcome.resolved_score, outcome.reason
    name: str = spec.type
    if spec.negate:
        passed, score, reason, name = not passed, 1.0 - score, f"negated: {reason}", f"not_{name}"
    return AssertionResult(type=name, passed=passed, score=score, reason=reason, weight=spec.weight)
