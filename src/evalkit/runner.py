"""Executes a suite: fans trials out concurrently, caches responses, grades outputs."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from evalkit.assertions import AssertionResult, Context, JudgeFn, Verdict, evaluate
from evalkit.cache import ResponseCache, cache_key
from evalkit.config import Case, Suite
from evalkit.judge import Judge, JudgeError
from evalkit.providers.base import Completion, Provider, ProviderError, Request
from evalkit.results import CaseResult, RunResult


class Runner:
    def __init__(
        self,
        suite: Suite,
        providers: Sequence[Provider],
        *,
        judge: Provider | None = None,
        cache: ResponseCache | None = None,
        concurrency: int = 8,
        on_result: Callable[[CaseResult], None] | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.suite = suite
        self.providers = list(providers)
        self.cache = cache
        self.on_result = on_result
        self._sem = asyncio.Semaphore(concurrency)
        self._judge_provider = judge
        self._judge = Judge(self._judge_complete) if judge is not None else None

    async def _complete(
        self, provider: Provider, request: Request, repeat: int
    ) -> tuple[Completion, float, bool]:
        """Returns (completion, latency_ms, cached). Cached entries keep their original latency so
        latency assertions stay meaningful on re-runs."""
        key = None
        if self.cache is not None:
            key = cache_key(provider.cache_identity(), request.system, request.messages, repeat)
            hit = self.cache.get(key)
            if hit is not None:
                return hit[0], hit[1], True
        async with self._sem:
            start = time.perf_counter()
            completion = await provider.complete(request)
            latency_ms = (time.perf_counter() - start) * 1000
        if self.cache is not None and key is not None:
            self.cache.put(key, completion, latency_ms)
        return completion, latency_ms, False

    async def _judge_complete(self, request: Request) -> Completion:
        assert self._judge_provider is not None
        completion, _, _ = await self._complete(self._judge_provider, request, repeat=0)
        return completion

    def _judge_for(self, input_text: str) -> JudgeFn | None:
        if self._judge is None:
            return None
        judge = self._judge

        async def grade(rubric: str, output: str) -> Verdict:
            return await judge.grade(rubric, input_text, output)

        return grade

    async def _run_trial(self, provider: Provider, case: Case, repeat: int) -> CaseResult:
        request = Request(
            messages=self.suite.messages_for(case),
            system=self.suite.system_for(case),
            case_id=case.id,
        )
        try:
            completion, latency_ms, cached = await self._complete(provider, request, repeat)
        except ProviderError as exc:
            result = CaseResult(
                case_id=case.id,
                provider=provider.label,
                repeat=repeat,
                output="",
                latency_ms=0.0,
                error=str(exc),
            )
            self._emit(result)
            return result

        input_text = request.messages[-1]["content"]
        ctx = Context(
            output=completion.text,
            latency_ms=latency_ms,
            case_id=case.id,
            vars=case.vars,
            input_text=input_text,
            judge=self._judge_for(input_text),
        )
        graded: list[AssertionResult] = []
        for spec in self.suite.assertions_for(case):
            try:
                graded.append(await evaluate(spec, ctx))
            except (JudgeError, ProviderError) as exc:
                graded.append(
                    AssertionResult(
                        type=spec.type,
                        passed=False,
                        score=0.0,
                        reason=f"grading failed: {exc}",
                        weight=spec.weight,
                    )
                )

        result = CaseResult(
            case_id=case.id,
            provider=provider.label,
            repeat=repeat,
            output=completion.text,
            latency_ms=latency_ms,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cached=cached,
            refused=completion.refused,
            assertions=graded,
        )
        self._emit(result)
        return result

    def _emit(self, result: CaseResult) -> None:
        if self.on_result is not None:
            self.on_result(result)

    async def run(self, cases: Sequence[Case] | None = None) -> RunResult:
        selected = list(cases) if cases is not None else list(self.suite.cases)
        started = datetime.now(UTC)
        t0 = time.perf_counter()
        trials = [
            self._run_trial(provider, case, repeat)
            for provider in self.providers
            for case in selected
            for repeat in range(self.suite.repeat)
        ]
        # gather preserves input order, so results come back grouped provider -> case -> repeat.
        results = await asyncio.gather(*trials)
        return RunResult(
            suite=self.suite.name,
            started_at=started.isoformat(timespec="seconds"),
            duration_s=time.perf_counter() - t0,
            threshold=self.suite.threshold,
            providers=[p.label for p in self.providers],
            results=list(results),
        )

    @property
    def total_trials(self) -> int:
        return len(self.providers) * len(self.suite.cases) * self.suite.repeat
