"""Suite configuration: the YAML schema, validated with pydantic.

A suite file is parsed once, up front, into immutable models. Every error a user can make in YAML
(typos in field names, unknown assertion types, missing template variables, duplicate ids) is
reported here with a path into the file, before a single model call is made.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

_VAR = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class ConfigError(Exception):
    """A suite file is missing, malformed, or fails validation."""


def template_vars(template: str) -> set[str]:
    return set(_VAR.findall(template))


def render(template: str, variables: dict[str, Any]) -> str:
    """Substitute ``{{ name }}`` placeholders. Unknown names are a hard error, never silent."""

    def sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            raise ConfigError(f"template variable '{name}' is not defined")
        return str(variables[name])

    return _VAR.sub(sub, template)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- providers

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ProviderConfig(_Strict):
    type: Literal["anthropic", "openai", "mock"]
    id: str | None = None
    model: str = ""
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float | None = None
    effort: Effort | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = Field(default=120.0, gt=0)
    params: dict[str, Any] = Field(default_factory=dict)
    # mock provider only
    responses: dict[str, str] = Field(default_factory=dict)
    default_response: str | None = None
    latency_ms: float = 0.0

    @model_validator(mode="after")
    def _requires_model(self) -> ProviderConfig:
        if self.type != "mock" and not self.model:
            raise ValueError(f"provider type '{self.type}' requires a 'model'")
        if self.type != "mock" and (self.responses or self.default_response):
            raise ValueError(
                "'responses' / 'default_response' are only valid for the mock provider"
            )
        return self

    @property
    def label(self) -> str:
        if self.id:
            return self.id
        return f"{self.type}:{self.model}" if self.model else self.type


# --------------------------------------------------------------------------- assertions


class _Assertion(_Strict):
    negate: bool = False
    weight: float = Field(default=1.0, gt=0)


class Equals(_Assertion):
    type: Literal["equals"]
    value: str
    ignore_case: bool = False
    strip: bool = True


class Contains(_Assertion):
    type: Literal["contains"]
    value: str
    ignore_case: bool = False


class ContainsAll(_Assertion):
    type: Literal["contains_all"]
    values: list[str] = Field(min_length=1)
    ignore_case: bool = False


class ContainsAny(_Assertion):
    type: Literal["contains_any"]
    values: list[str] = Field(min_length=1)
    ignore_case: bool = False


class StartsWith(_Assertion):
    type: Literal["starts_with"]
    value: str
    ignore_case: bool = False


class Regex(_Assertion):
    type: Literal["regex"]
    pattern: str
    flags: str = ""

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc
        return value

    @field_validator("flags")
    @classmethod
    def _known_flags(cls, value: str) -> str:
        unknown = set(value) - set("ims")
        if unknown:
            raise ValueError(f"unknown regex flags {sorted(unknown)}; use any of 'i', 'm', 's'")
        return value


class JsonValid(_Assertion):
    type: Literal["json_valid"]


class JsonSchema(_Assertion):
    type: Literal["json_schema"]
    schema_: dict[str, Any] = Field(alias="schema")

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class Length(_Assertion):
    type: Literal["length"]
    min: int | None = Field(default=None, ge=0)
    max: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _bounds(self) -> Length:
        if self.min is None and self.max is None:
            raise ValueError("length needs 'min' and/or 'max'")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("length 'min' is greater than 'max'")
        return self


class Latency(_Assertion):
    type: Literal["latency"]
    max_ms: float = Field(gt=0)


class LlmJudge(_Assertion):
    type: Literal["llm_judge"]
    rubric: str
    threshold: float = Field(default=0.7, ge=0, le=1)


class Python(_Assertion):
    type: Literal["python"]
    function: str = Field(pattern=r"^[\w.]+:[\w]+$", description="module.path:function_name")


AssertionSpec = Annotated[
    Equals
    | Contains
    | ContainsAll
    | ContainsAny
    | StartsWith
    | Regex
    | JsonValid
    | JsonSchema
    | Length
    | Latency
    | LlmJudge
    | Python,
    Field(discriminator="type"),
]


def _assertion_templates(spec: AssertionSpec) -> list[str]:
    """The text fields of an assertion that may contain ``{{ var }}`` placeholders."""
    if isinstance(spec, Equals | Contains | StartsWith):
        return [spec.value]
    if isinstance(spec, ContainsAll | ContainsAny):
        return list(spec.values)
    if isinstance(spec, LlmJudge):
        return [spec.rubric]
    return []


def _expand_negations(raw: Any) -> Any:
    """``not_contains`` -> ``contains`` with ``negate: true``. Works for every assertion type."""
    if isinstance(raw, list):
        return [_expand_negations(item) for item in raw]
    if (
        isinstance(raw, dict)
        and isinstance(raw.get("type"), str)
        and raw["type"].startswith("not_")
    ):
        return {
            **raw,
            "type": raw["type"].removeprefix("not_"),
            "negate": not raw.get("negate", False),
        }
    return raw


# --------------------------------------------------------------------------- cases & suite


class Message(_Strict):
    role: Literal["user", "assistant"]
    content: str


class Case(_Strict):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    id: str = Field(pattern=r"^[\w.\-]+$")
    description: str | None = None
    input: str | None = None
    messages: list[Message] | None = None
    vars: dict[str, Any] = Field(default_factory=dict)
    assertions: list[AssertionSpec] = Field(default_factory=list, alias="assert")
    tags: list[str] = Field(default_factory=list)

    @field_validator("assertions", mode="before")
    @classmethod
    def _negations(cls, value: Any) -> Any:
        return _expand_negations(value)

    @model_validator(mode="after")
    def _one_input(self) -> Case:
        if self.input is not None and self.messages is not None:
            raise ValueError("use either 'input' or 'messages', not both")
        if self.messages is not None and (not self.messages or self.messages[-1].role != "user"):
            raise ValueError("'messages' must be non-empty and end with a user message")
        return self


class Defaults(_Strict):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    assertions: list[AssertionSpec] = Field(default_factory=list, alias="assert")

    @field_validator("assertions", mode="before")
    @classmethod
    def _negations(cls, value: Any) -> Any:
        return _expand_negations(value)


class Suite(_Strict):
    name: str
    description: str | None = None
    system: str | None = None
    prompt: str | None = None
    providers: list[ProviderConfig] = Field(min_length=1)
    judge: ProviderConfig | None = None
    defaults: Defaults = Field(default_factory=Defaults)
    repeat: int = Field(default=1, ge=1, le=50)
    threshold: float = Field(default=1.0, ge=0, le=1)
    cases: list[Case] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _single_provider(cls, data: Any) -> Any:
        if isinstance(data, dict) and "provider" in data:
            if "providers" in data:
                raise ValueError("use either 'provider' or 'providers', not both")
            data = {**data, "providers": [data["provider"]]}
            del data["provider"]
        return data

    @model_validator(mode="after")
    def _consistency(self) -> Suite:
        ids = [case.id for case in self.cases]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate case ids: {dupes}")

        labels = [p.label for p in self.providers]
        dupes = sorted({x for x in labels if labels.count(x) > 1})
        if dupes:
            raise ValueError(f"duplicate provider labels {dupes}; give each provider a unique 'id'")

        for case in self.cases:
            if case.input is None and case.messages is None and self.prompt is None:
                raise ValueError(
                    f"case '{case.id}' has no input, no messages and the suite has no prompt"
                )
            # Render eagerly so a missing variable fails at load time, not mid-run.
            try:
                self.messages_for(case)
                self.system_for(case)
            except ConfigError as exc:
                raise ValueError(f"case '{case.id}': {exc}") from exc
            for spec in self.assertions_for(case):
                missing = (
                    set().union(*map(template_vars, _assertion_templates(spec))) - case.vars.keys()
                )
                if missing:
                    raise ValueError(
                        f"case '{case.id}': {spec.type} uses undefined "
                        f"variable(s) {sorted(missing)}"
                    )
            uses_judge = any(isinstance(a, LlmJudge) for a in self.assertions_for(case))
            if uses_judge and self.judge is None:
                raise ValueError(
                    f"case '{case.id}' uses llm_judge but the suite has no 'judge' provider"
                )
        return self

    def assertions_for(self, case: Case) -> list[AssertionSpec]:
        return [*self.defaults.assertions, *case.assertions]

    def system_for(self, case: Case) -> str | None:
        return render(self.system, case.vars) if self.system else None

    def messages_for(self, case: Case) -> list[dict[str, str]]:
        if case.messages is not None:
            return [
                {"role": m.role, "content": render(m.content, case.vars)} for m in case.messages
            ]
        template = case.input if case.input is not None else self.prompt
        assert template is not None  # guaranteed by _consistency
        return [{"role": "user", "content": render(template, case.vars)}]


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [f"{path}: invalid suite"]
    for err in exc.errors():
        location = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"  - {location}: {err['msg']}")
    return "\n".join(lines)


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"{path}: file not found") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: a suite must be a YAML mapping")
    try:
        return Suite.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(path, exc)) from exc
