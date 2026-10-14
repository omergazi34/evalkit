"""Run results: per-trial records, per-provider summaries, and JSON (de)serialisation."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalkit.assertions import AssertionResult

RESULTS_FORMAT = 1


@dataclass(slots=True)
class CaseResult:
    """One trial: one case, run once against one provider."""

    case_id: str
    provider: str
    repeat: int
    output: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached: bool = False
    refused: bool = False
    error: str | None = None
    assertions: list[AssertionResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.error is None and all(a.passed for a in self.assertions)

    @property
    def score(self) -> float:
        if self.error is not None:
            return 0.0
        if not self.assertions:
            return 1.0
        total = sum(a.weight for a in self.assertions)
        return sum(a.score * a.weight for a in self.assertions) / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "provider": self.provider,
            "repeat": self.repeat,
            "passed": self.passed,
            "score": round(self.score, 4),
            "output": self.output,
            "latency_ms": round(self.latency_ms, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached": self.cached,
            "refused": self.refused,
            "error": self.error,
            "assertions": [a.to_dict() for a in self.assertions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseResult:
        return cls(
            case_id=data["case_id"],
            provider=data["provider"],
            repeat=data.get("repeat", 0),
            output=data.get("output", ""),
            latency_ms=data.get("latency_ms", 0.0),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            cached=data.get("cached", False),
            refused=data.get("refused", False),
            error=data.get("error"),
            assertions=[
                AssertionResult(
                    type=a["type"],
                    passed=a["passed"],
                    score=a["score"],
                    reason=a.get("reason", ""),
                    weight=a.get("weight", 1.0),
                )
                for a in data.get("assertions", [])
            ],
        )


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; 0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True, slots=True)
class ProviderSummary:
    provider: str
    trials: int
    passed: int
    errors: int
    refusals: int
    mean_score: float
    latency_p50_ms: float
    latency_p95_ms: float
    input_tokens: int
    output_tokens: int
    flaky_cases: tuple[str, ...]

    @property
    def pass_rate(self) -> float:
        return self.passed / self.trials if self.trials else 0.0


@dataclass(slots=True)
class RunResult:
    suite: str
    started_at: str
    duration_s: float
    threshold: float
    providers: list[str]
    results: list[CaseResult]

    def for_provider(self, provider: str) -> list[CaseResult]:
        return [r for r in self.results if r.provider == provider]

    def summaries(self) -> list[ProviderSummary]:
        out = []
        for provider in self.providers:
            rows = self.for_provider(provider)
            by_case: dict[str, set[bool]] = defaultdict(set)
            for r in rows:
                by_case[r.case_id].add(r.passed)
            live = [r.latency_ms for r in rows if r.error is None]
            out.append(
                ProviderSummary(
                    provider=provider,
                    trials=len(rows),
                    passed=sum(r.passed for r in rows),
                    errors=sum(r.error is not None for r in rows),
                    refusals=sum(r.refused for r in rows),
                    mean_score=sum(r.score for r in rows) / len(rows) if rows else 0.0,
                    latency_p50_ms=percentile(live, 50),
                    latency_p95_ms=percentile(live, 95),
                    input_tokens=sum(r.input_tokens or 0 for r in rows),
                    output_tokens=sum(r.output_tokens or 0 for r in rows),
                    flaky_cases=tuple(
                        sorted(c for c, outcomes in by_case.items() if len(outcomes) > 1)
                    ),
                )
            )
        return out

    @property
    def passed(self) -> bool:
        return all(s.pass_rate >= self.threshold for s in self.summaries())

    def case_pass_fraction(self) -> dict[tuple[str, str], float]:
        """(provider, case_id) -> fraction of repeats that passed."""
        groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
        for r in self.results:
            groups[(r.provider, r.case_id)].append(r.passed)
        return {key: sum(v) / len(v) for key, v in groups.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": RESULTS_FORMAT,
            "suite": self.suite,
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 3),
            "threshold": self.threshold,
            "providers": self.providers,
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        if data.get("format") != RESULTS_FORMAT:
            raise ValueError(f"unsupported results format {data.get('format')!r}")
        return cls(
            suite=data["suite"],
            started_at=data["started_at"],
            duration_s=data["duration_s"],
            threshold=data["threshold"],
            providers=list(data["providers"]),
            results=[CaseResult.from_dict(r) for r in data["results"]],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
