from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from evalkit.config import ConfigError, Contains, Suite, load_suite, render
from tests.conftest import make_suite


def test_single_provider_shorthand_becomes_list() -> None:
    suite = make_suite()
    assert [p.label for p in suite.providers] == ["mock"]


def test_provider_labels_default_to_type_and_model() -> None:
    suite = make_suite(
        provider=None,
        providers=[{"type": "openai", "model": "llama3"}, {"type": "mock", "id": "fake"}],
    )
    assert [p.label for p in suite.providers] == ["openai:llama3", "fake"]


def test_not_prefix_expands_to_negated_assertion() -> None:
    suite = make_suite(
        cases=[{"id": "a", "input": "x", "assert": [{"type": "not_contains", "value": "y"}]}]
    )
    (spec,) = suite.cases[0].assertions
    assert isinstance(spec, Contains)
    assert spec.negate is True


def test_defaults_are_prepended_to_case_assertions() -> None:
    suite = make_suite(
        defaults={"assert": [{"type": "length", "max": 10}]},
        cases=[{"id": "a", "input": "x", "assert": [{"type": "contains", "value": "x"}]}],
    )
    assert [a.type for a in suite.assertions_for(suite.cases[0])] == ["length", "contains"]


def test_prompt_template_renders_case_vars() -> None:
    suite = make_suite(
        prompt="Translate to {{ lang }}: {{text}}",
        cases=[{"id": "a", "vars": {"lang": "TR", "text": "hi"}}],
    )
    assert suite.messages_for(suite.cases[0]) == [
        {"role": "user", "content": "Translate to TR: hi"}
    ]


def test_multi_turn_messages_are_supported() -> None:
    suite = make_suite(
        cases=[
            {
                "id": "a",
                "messages": [
                    {"role": "user", "content": "My name is Ada."},
                    {"role": "assistant", "content": "Nice to meet you."},
                    {"role": "user", "content": "What's my name?"},
                ],
            }
        ]
    )
    assert len(suite.messages_for(suite.cases[0])) == 3


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cases": [{"id": "a", "input": "x"}, {"id": "a", "input": "y"}]}, "duplicate case ids"),
        ({"cases": [{"id": "a"}]}, "has no input"),
        ({"prompt": "{{missing}}", "cases": [{"id": "a"}]}, "'missing' is not defined"),
        (
            {"cases": [{"id": "a", "input": "x", "assert": [{"type": "nope"}]}]},
            "does not match any of the expected tags",
        ),
        (
            {
                "cases": [
                    {"id": "a", "input": "x", "assert": [{"type": "llm_judge", "rubric": "r"}]}
                ]
            },
            "no 'judge' provider",
        ),
        (
            {"cases": [{"id": "a", "input": "x", "assert": [{"type": "regex", "pattern": "("}]}]},
            "invalid regex",
        ),
        (
            {"cases": [{"id": "a", "input": "x", "assert": [{"type": "length"}]}]},
            "needs 'min' and/or 'max'",
        ),
        (
            {
                "cases": [
                    {"id": "a", "input": "x", "assert": [{"type": "contains", "value": "{{v}}"}]}
                ]
            },
            "undefined variable",
        ),
        (
            {"cases": [{"id": "a", "messages": [{"role": "assistant", "content": "x"}]}]},
            "end with a user message",
        ),
        ({"provider": {"type": "anthropic"}}, "requires a 'model'"),
        ({"cases": [{"id": "a", "input": "x", "typo": 1}]}, "Extra inputs are not permitted"),
    ],
)
def test_invalid_suites_fail_at_load_time(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make_suite(**overrides)


def test_render_rejects_unknown_variables() -> None:
    assert render("{{ a }}-{{b}}", {"a": 1, "b": "x"}) == "1-x"
    with pytest.raises(ConfigError):
        render("{{c}}", {})


def test_load_suite_reports_file_and_field_paths(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    path.write_text(
        "name: s\nprovider: {type: mock}\n"
        "cases:\n  - id: a\n    input: x\n    assert:\n      - type: length\n"
    )
    with pytest.raises(ConfigError) as info:
        load_suite(path)
    assert str(path) in str(info.value)
    assert "cases.0.assert.0" in str(info.value)


@pytest.mark.parametrize(
    ("content", "message"), [("[1, 2]", "must be a YAML mapping"), ("a: [", "invalid YAML")]
)
def test_load_suite_rejects_bad_yaml(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "suite.yaml"
    path.write_text(content)
    with pytest.raises(ConfigError, match=message):
        load_suite(path)


def test_load_suite_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="file not found"):
        load_suite(tmp_path / "nope.yaml")


def test_example_suites_are_valid() -> None:
    examples = Path(__file__).parent.parent / "examples"
    for path in sorted(examples.glob("*.yaml")):
        assert isinstance(load_suite(path), Suite)
