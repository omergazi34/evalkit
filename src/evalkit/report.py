"""Renderers for a RunResult: terminal (rich), Markdown (PR comments / job summaries), JUnit XML."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from rich.console import Console
from rich.table import Table
from rich.text import Text

from evalkit.compare import Comparison
from evalkit.results import CaseResult, RunResult


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _clip(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _failures(run: RunResult) -> list[CaseResult]:
    return [r for r in run.results if not r.passed]


# --------------------------------------------------------------------------- terminal


def print_report(run: RunResult, console: Console, comparison: Comparison | None = None) -> None:
    table = Table(title=f"evalkit · {run.suite}", title_justify="left", expand=False)
    for col, justify in [
        ("provider", "left"),
        ("pass rate", "right"),
        ("score", "right"),
        ("p50", "right"),
        ("p95", "right"),
        ("tokens in/out", "right"),
        ("errors", "right"),
        ("flaky", "right"),
    ]:
        table.add_column(col, justify=justify)  # type: ignore[arg-type]
    for s in run.summaries():
        ok = s.pass_rate >= run.threshold
        table.add_row(
            s.provider,
            Text(f"{_pct(s.pass_rate)} ({s.passed}/{s.trials})", style="green" if ok else "red"),
            f"{s.mean_score:.2f}",
            f"{s.latency_p50_ms:.0f}ms",
            f"{s.latency_p95_ms:.0f}ms",
            f"{s.input_tokens:,}/{s.output_tokens:,}",
            str(s.errors),
            str(len(s.flaky_cases)),
        )
    console.print(table)

    failures = _failures(run)
    if failures:
        console.print(f"\n[bold red]{len(failures)} failing trial(s)[/]")
        for r in failures:
            console.print(
                f"[red]✗[/] [bold]{r.case_id}[/] [dim]({r.provider}, repeat {r.repeat})[/]"
            )
            if r.error:
                console.print(f"    error: {r.error}", markup=False)
            if r.refused:
                console.print("    model refused")
            for a in r.assertions:
                if not a.passed:
                    console.print(f"    {a.type}: {a.reason}", markup=False, highlight=False)
            if not r.error:
                console.print(
                    f"    output: {_clip(r.output)!r}", markup=False, style="dim", highlight=False
                )

    if comparison is not None:
        _print_comparison(comparison, console)

    verdict = "[bold green]PASS[/]" if run.passed else "[bold red]FAIL[/]"
    console.print(f"\n{verdict} threshold {_pct(run.threshold)} · {run.duration_s:.1f}s")


def _print_comparison(cmp: Comparison, console: Console) -> None:
    if cmp.regressions:
        console.print(f"\n[bold red]{len(cmp.regressions)} regression(s) vs baseline[/]")
        for ch in cmp.regressions:
            console.print(f"  ↓ {ch.case_id} ({ch.provider}): {_pct(ch.before)} → {_pct(ch.after)}")
    if cmp.fixes:
        console.print(f"\n[bold green]{len(cmp.fixes)} fix(es) vs baseline[/]")
        for ch in cmp.fixes:
            console.print(f"  ↑ {ch.case_id} ({ch.provider}): {_pct(ch.before)} → {_pct(ch.after)}")
    if not cmp.regressions and not cmp.fixes:
        console.print("\nNo changes vs baseline.")


# --------------------------------------------------------------------------- markdown


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def to_markdown(run: RunResult, comparison: Comparison | None = None) -> str:
    icon = "✅" if run.passed else "❌"
    lines = [
        f"## {icon} evalkit: {run.suite}",
        "",
        "| Provider | Pass rate | Score | p50 | p95 | Tokens in/out | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in run.summaries():
        lines.append(
            f"| `{s.provider}` | {_pct(s.pass_rate)} ({s.passed}/{s.trials}) | {s.mean_score:.2f} "
            f"| {s.latency_p50_ms:.0f}ms | {s.latency_p95_ms:.0f}ms "
            f"| {s.input_tokens:,}/{s.output_tokens:,} | {s.errors} |"
        )
    lines += ["", f"Threshold: **{_pct(run.threshold)}** · Duration: {run.duration_s:.1f}s"]

    if comparison is not None and (comparison.regressions or comparison.fixes):
        lines += [
            "",
            "### Changes vs baseline",
            "",
            "| | Case | Provider | Before | After |",
            "|---|---|---|---:|---:|",
        ]
        for ch in comparison.regressions:
            lines.append(
                f"| 🔻 | `{ch.case_id}` | `{ch.provider}` | {_pct(ch.before)} | {_pct(ch.after)} |"
            )
        for ch in comparison.fixes:
            lines.append(
                f"| 🔺 | `{ch.case_id}` | `{ch.provider}` | {_pct(ch.before)} | {_pct(ch.after)} |"
            )

    failures = _failures(run)
    if failures:
        lines += [
            "",
            "<details><summary>Failing trials</summary>",
            "",
            "| Case | Provider | Why |",
            "|---|---|---|",
        ]
        for r in failures:
            reasons = (
                [r.error]
                if r.error
                else [f"{a.type}: {a.reason}" for a in r.assertions if not a.passed]
            )
            lines.append(f"| `{r.case_id}` | `{r.provider}` | {_md_escape('; '.join(reasons))} |")
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- junit


def to_junit(run: RunResult) -> str:
    root = ET.Element("testsuites", name=run.suite)
    for provider in run.providers:
        rows = run.for_provider(provider)
        suite = ET.SubElement(
            root,
            "testsuite",
            name=f"{run.suite} [{provider}]",
            tests=str(len(rows)),
            failures=str(sum(not r.passed and r.error is None for r in rows)),
            errors=str(sum(r.error is not None for r in rows)),
            time=f"{sum(r.latency_ms for r in rows) / 1000:.3f}",
        )
        for r in rows:
            name = r.case_id if r.repeat == 0 else f"{r.case_id}#{r.repeat}"
            case = ET.SubElement(
                suite,
                "testcase",
                name=name,
                classname=f"{run.suite}.{provider}",
                time=f"{r.latency_ms / 1000:.3f}",
            )
            if r.error is not None:
                ET.SubElement(case, "error", message=r.error)
            elif not r.passed:
                reasons = "; ".join(f"{a.type}: {a.reason}" for a in r.assertions if not a.passed)
                failure = ET.SubElement(case, "failure", message=reasons)
                failure.text = r.output
    ET.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"
