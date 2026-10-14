"""Command-line interface.

Exit codes: 0 pass, 1 eval failure or regression, 2 configuration error.
"""

from __future__ import annotations

import asyncio
import fnmatch
import io
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from evalkit import __version__
from evalkit.cache import ResponseCache
from evalkit.compare import compare
from evalkit.config import Case, ConfigError, Suite, load_suite
from evalkit.providers import Provider, build_provider
from evalkit.providers.base import ProviderError
from evalkit.report import print_report, to_junit, to_markdown
from evalkit.results import CaseResult, RunResult
from evalkit.runner import Runner

EXIT_FAIL = 1
EXIT_CONFIG = 2

EXAMPLE_SUITE = """\
# evalkit suite: run with `evalkit run evalkit.yaml`
name: quickstart
description: Offline example using the deterministic mock provider. Swap in a real provider below.

provider:
  type: mock
  responses:
    capital: "The capital of Turkey is Ankara."
    json-extract: '{"name": "Ada", "age": 36}'
# provider:
#   type: anthropic            # needs ANTHROPIC_API_KEY
#   model: claude-opus-5-5
#   effort: low
# provider:
#   type: openai               # any OpenAI-compatible server, e.g. Ollama
#   base_url: http://localhost:11434/v1
#   model: llama3

system: You are a concise assistant.

cases:
  - id: capital
    input: What is the capital of Turkey?
    assert:
      - type: contains
        value: Ankara
      - type: not_contains
        value: Istanbul
      - type: length
        max: 200

  - id: json-extract
    input: 'Return JSON with keys name and age for: "Ada is 36."'
    assert:
      - type: json_schema
        schema:
          type: object
          required: [name, age]
          properties:
            name: {type: string}
            age: {type: integer}
"""


def _fail(message: str, code: int = EXIT_CONFIG) -> None:
    click.echo(message, err=True)
    sys.exit(code)


def _select_cases(suite: Suite, patterns: tuple[str, ...], tags: tuple[str, ...]) -> list[Case]:
    cases = list(suite.cases)
    if patterns:
        cases = [c for c in cases if any(fnmatch.fnmatchcase(c.id, p) for p in patterns)]
    if tags:
        cases = [c for c in cases if set(tags) & set(c.tags)]
    return cases


async def _execute(
    suite: Suite,
    labels: tuple[str, ...],
    cases: list[Case],
    concurrency: int,
    cache: ResponseCache | None,
    console: Console,
    show_progress: bool,
) -> RunResult:
    configs = [p for p in suite.providers if not labels or p.label in labels]
    providers: list[Provider] = []
    judge: Provider | None = None
    try:
        providers = [build_provider(p) for p in configs]
        judge = build_provider(suite.judge) if suite.judge else None
        total = len(providers) * len(cases) * suite.repeat
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
            disable=not show_progress,
        ) as progress:
            task = progress.add_task(f"running {suite.name}", total=total)

            def on_result(_: CaseResult) -> None:
                progress.advance(task)

            runner = Runner(
                suite,
                providers,
                judge=judge,
                cache=cache,
                concurrency=concurrency,
                on_result=on_result,
            )
            return await runner.run(cases)
    finally:
        for p in [*providers, *([judge] if judge else [])]:
            await p.aclose()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="evalkit")
def main() -> None:
    """Regression tests for LLM applications."""
    _tolerate_narrow_encodings()


def _tolerate_narrow_encodings() -> None:
    """Model output is arbitrary Unicode, but a Windows console or a redirected stream may use a
    legacy code page (e.g. cp1254). Replace unencodable characters instead of crashing mid-report."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and stream.encoding.lower() not in {"utf-8", "utf8"}:
            stream.reconfigure(errors="replace")


@main.command()
@click.argument("suite_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-p",
    "--provider",
    "labels",
    multiple=True,
    help="Only run the provider with this label (repeatable).",
)
@click.option(
    "-k",
    "--case",
    "patterns",
    multiple=True,
    help="Only run cases whose id matches this glob (repeatable).",
)
@click.option(
    "-t", "--tag", "tags", multiple=True, help="Only run cases with this tag (repeatable)."
)
@click.option("-j", "--concurrency", default=8, show_default=True, type=click.IntRange(1, 256))
@click.option(
    "--cache/--no-cache",
    default=True,
    show_default=True,
    help="Reuse responses from previous runs.",
)
@click.option(
    "--cache-dir", default=".evalkit_cache", show_default=True, type=click.Path(path_type=Path)
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write results JSON here.",
)
@click.option(
    "--markdown",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write a Markdown report here.",
)
@click.option(
    "--junit", type=click.Path(dir_okay=False, path_type=Path), help="Write JUnit XML here."
)
@click.option(
    "--threshold", type=click.FloatRange(0, 1), help="Override the suite's minimum pass rate."
)
@click.option(
    "--baseline",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Results JSON to diff against.",
)
@click.option("--fail-on-regression/--no-fail-on-regression", default=True, show_default=True)
@click.option("-q", "--quiet", is_flag=True, help="Only print the summary table.")
def run(
    suite_path: Path,
    labels: tuple[str, ...],
    patterns: tuple[str, ...],
    tags: tuple[str, ...],
    concurrency: int,
    cache: bool,
    cache_dir: Path,
    output: Path | None,
    markdown: Path | None,
    junit: Path | None,
    threshold: float | None,
    baseline: Path | None,
    fail_on_regression: bool,
    quiet: bool,
) -> None:
    """Run SUITE_PATH and report results."""
    console = Console()
    try:
        suite = load_suite(suite_path)
    except ConfigError as exc:
        _fail(str(exc))
        return
    if threshold is not None:
        suite = suite.model_copy(update={"threshold": threshold})

    known = [p.label for p in suite.providers]
    unknown = [label for label in labels if label not in known]
    if unknown:
        _fail(f"unknown provider label(s) {unknown}; suite defines {known}")
    cases = _select_cases(suite, patterns, tags)
    if not cases:
        _fail("no cases match the given --case/--tag filters")

    # Custom `python` assertions are imported relative to the suite file.
    sys.path.insert(0, str(suite_path.resolve().parent))

    response_cache = ResponseCache(cache_dir) if cache else None
    try:
        result = asyncio.run(
            _execute(
                suite, labels, cases, concurrency, response_cache, console, show_progress=not quiet
            )
        )
    except ProviderError as exc:
        _fail(f"provider setup failed: {exc}")
        return
    finally:
        if response_cache is not None:
            response_cache.close()

    comparison = compare(RunResult.load(baseline), result) if baseline else None
    print_report(result, console, comparison)
    if response_cache is not None and not quiet:
        console.print(
            f"[dim]cache: {response_cache.hits} hit(s), {response_cache.misses} miss(es)[/]"
        )

    if output:
        result.save(output)
    if markdown:
        markdown.write_text(to_markdown(result, comparison), encoding="utf-8")
    if junit:
        junit.write_text(to_junit(result), encoding="utf-8")

    regressed = comparison is not None and comparison.has_regressions and fail_on_regression
    if not result.passed or regressed:
        sys.exit(EXIT_FAIL)


@main.command()
@click.argument("suite_paths", nargs=-1, required=True, type=click.Path(path_type=Path))
def validate(suite_paths: tuple[Path, ...]) -> None:
    """Check suite files without calling any model."""
    failed = False
    for path in suite_paths:
        try:
            suite = load_suite(path)
        except ConfigError as exc:
            click.echo(f"✗ {exc}", err=True)
            failed = True
            continue
        click.echo(f"✓ {path}: {len(suite.cases)} case(s), {len(suite.providers)} provider(s)")
    if failed:
        sys.exit(EXIT_CONFIG)


@main.command()
@click.argument("path", default="evalkit.yaml", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--force", is_flag=True, help="Overwrite an existing file.")
def init(path: Path, force: bool) -> None:
    """Write an example suite to PATH (default: evalkit.yaml)."""
    if path.exists() and not force:
        _fail(f"{path} already exists; pass --force to overwrite")
    path.write_text(EXAMPLE_SUITE, encoding="utf-8")
    click.echo(f"wrote {path}; run it with: evalkit run {path}")


@main.command(name="compare")
@click.argument("baseline", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("current", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def compare_cmd(baseline: Path, current: Path) -> None:
    """Diff two results files; exit 1 if CURRENT regressed."""
    cmp = compare(RunResult.load(baseline), RunResult.load(current))
    for ch in cmp.regressions:
        click.echo(f"↓ {ch.case_id} ({ch.provider}): {ch.before:.0%} → {ch.after:.0%}")
    for ch in cmp.fixes:
        click.echo(f"↑ {ch.case_id} ({ch.provider}): {ch.before:.0%} → {ch.after:.0%}")
    click.echo(f"{len(cmp.regressions)} regression(s), {len(cmp.fixes)} fix(es)")
    if cmp.has_regressions:
        sys.exit(EXIT_FAIL)
