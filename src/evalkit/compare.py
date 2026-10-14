"""Diff two runs to find regressions: cases that pass less often than they did in the baseline."""

from __future__ import annotations

from dataclasses import dataclass

from evalkit.results import RunResult


@dataclass(frozen=True, slots=True)
class Change:
    provider: str
    case_id: str
    before: float
    after: float


@dataclass(frozen=True, slots=True)
class Comparison:
    regressions: tuple[Change, ...]
    fixes: tuple[Change, ...]
    added: tuple[tuple[str, str], ...]
    removed: tuple[tuple[str, str], ...]

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions)


def compare(baseline: RunResult, current: RunResult) -> Comparison:
    before = baseline.case_pass_fraction()
    after = current.case_pass_fraction()
    regressions, fixes = [], []
    for key in sorted(after.keys() & before.keys()):
        b, a = before[key], after[key]
        if a < b:
            regressions.append(Change(key[0], key[1], b, a))
        elif a > b:
            fixes.append(Change(key[0], key[1], b, a))
    return Comparison(
        regressions=tuple(regressions),
        fixes=tuple(fixes),
        added=tuple(sorted(after.keys() - before.keys())),
        removed=tuple(sorted(before.keys() - after.keys())),
    )
