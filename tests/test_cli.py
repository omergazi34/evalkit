from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from evalkit.cli import main

PASSING = """\
name: cli
provider: {type: mock, responses: {a: "Ankara"}}
cases:
  - id: a
    input: capital?
    tags: [geo]
    assert: [{type: contains, value: Ankara}]
  - id: b
    input: echo me
"""

FAILING = PASSING.replace('responses: {a: "Ankara"}', 'responses: {a: "Istanbul"}')


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    monkeypatch.chdir(tmp_path)
    return CliRunner()


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_init_then_run_quickstart_passes(cli: CliRunner, tmp_path: Path) -> None:
    assert cli.invoke(main, ["init"]).exit_code == 0
    result = cli.invoke(main, ["run", "evalkit.yaml", "-q"])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_init_refuses_to_overwrite(cli: CliRunner) -> None:
    cli.invoke(main, ["init"])
    assert cli.invoke(main, ["init"]).exit_code == 2
    assert cli.invoke(main, ["init", "--force"]).exit_code == 0


def test_failing_suite_exits_1_and_writes_reports(cli: CliRunner, tmp_path: Path) -> None:
    suite = write(tmp_path / "s.yaml", FAILING)
    result = cli.invoke(
        main,
        ["run", str(suite), "-o", "r.json", "--markdown", "r.md", "--junit", "r.xml", "--no-cache"],
    )
    assert result.exit_code == 1, result.output
    assert "contains: missing 'Ankara'" in result.output
    assert json.loads((tmp_path / "r.json").read_text())["passed"] is False
    assert "❌" in (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "<failure" in (tmp_path / "r.xml").read_text()


def test_threshold_override(cli: CliRunner, tmp_path: Path) -> None:
    suite = write(tmp_path / "s.yaml", FAILING)
    assert cli.invoke(main, ["run", str(suite), "--threshold", "0.5", "-q"]).exit_code == 0


def test_invalid_suite_exits_2(cli: CliRunner, tmp_path: Path) -> None:
    suite = write(tmp_path / "s.yaml", "name: x\nprovider: {type: mock}\ncases: []\n")
    result = cli.invoke(main, ["run", str(suite)])
    assert result.exit_code == 2
    assert "cases" in result.output


def test_filters(cli: CliRunner, tmp_path: Path) -> None:
    suite = write(tmp_path / "s.yaml", FAILING)
    assert cli.invoke(main, ["run", str(suite), "-k", "b", "-q"]).exit_code == 0
    assert cli.invoke(main, ["run", str(suite), "-t", "geo", "-q"]).exit_code == 1
    assert cli.invoke(main, ["run", str(suite), "-k", "zzz"]).exit_code == 2
    assert cli.invoke(main, ["run", str(suite), "-p", "nope"]).exit_code == 2


def test_baseline_regression_fails_even_above_threshold(cli: CliRunner, tmp_path: Path) -> None:
    good = write(tmp_path / "good.yaml", PASSING)
    bad = write(tmp_path / "bad.yaml", FAILING)
    assert cli.invoke(main, ["run", str(good), "-o", "base.json", "-q"]).exit_code == 0
    args = ["run", str(bad), "--threshold", "0.5", "--baseline", "base.json"]
    result = cli.invoke(main, args)
    assert result.exit_code == 1
    assert "1 regression(s)" in result.output
    assert cli.invoke(main, [*args, "--no-fail-on-regression"]).exit_code == 0


def test_validate_and_compare_commands(cli: CliRunner, tmp_path: Path) -> None:
    good = write(tmp_path / "good.yaml", PASSING)
    broken = write(tmp_path / "broken.yaml", "name: x\n")
    assert cli.invoke(main, ["validate", str(good)]).exit_code == 0
    assert cli.invoke(main, ["validate", str(good), str(broken)]).exit_code == 2

    bad = write(tmp_path / "bad.yaml", FAILING)
    cli.invoke(main, ["run", str(good), "-o", "a.json", "-q"])
    cli.invoke(main, ["run", str(bad), "-o", "b.json", "-q"])
    result = cli.invoke(main, ["compare", "a.json", "b.json"])
    assert result.exit_code == 1
    assert "↓ a (mock)" in result.output
    assert cli.invoke(main, ["compare", "b.json", "a.json"]).exit_code == 0


def test_python_assertions_resolve_relative_to_suite(cli: CliRunner, tmp_path: Path) -> None:
    folder = tmp_path / "suites"
    folder.mkdir()
    write(folder / "suite_checks.py", "def shouts(output, ctx):\n    return output.isupper()\n")
    suite = write(
        folder / "s.yaml",
        "name: py\nprovider: {type: mock, default_response: LOUD}\n"
        "cases: [{id: a, input: x, assert: [{type: python, function: 'suite_checks:shouts'}]}]\n",
    )
    assert cli.invoke(main, ["run", str(suite), "-q", "--no-cache"]).exit_code == 0
