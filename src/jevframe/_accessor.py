"""The common public API; integrations supply only row and result conversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Generic, TypeVar

from typesafe_sdk import AsyncTypeSafeClient, JSONContent, Noul, Score

from ._cache import MemoryCache
from ._context import RowSource
from ._engine import Options, evaluate_rows
from ._schema import QuestionPlan, choice_question, make_plan
from ._types import Errors, Nulls, Output, ProgressCallback, Question, State

FrameT = TypeVar("FrameT")
SeriesT = TypeVar("SeriesT")


def _validate_output(output: Output) -> None:
    if output not in ("columns", "struct"):
        raise ValueError("output must be 'columns' or 'struct'")


class BaseAccessor(ABC, Generic[FrameT, SeriesT]):
    @abstractmethod
    def _source(self, state: State) -> RowSource: ...

    @abstractmethod
    def _frame(
        self, values: list[tuple[Any, ...]], plan: QuestionPlan, source: RowSource, output: Output
    ) -> FrameT: ...

    @abstractmethod
    def _series(self, values: list[tuple[Any, ...]], source: RowSource) -> SeriesT: ...

    async def noul(
        self,
        question: str,
        *,
        state: State,
        client: AsyncTypeSafeClient | None = None,
        model: str | None = None,
        max_concurrency: int = 16,
        cache: MemoryCache | None = None,
        progress: bool | ProgressCallback = False,
        errors: Errors = "raise",
        nulls: Nulls = "include",
    ) -> SeriesT:
        """Return a yes-probability Series, without thresholding or changing rows."""
        options = Options(client, model, max_concurrency, cache, progress, errors, nulls)
        plan = make_plan({"result": Noul(instructions=question)}, prefix=False)
        source = self._source(state)
        values = await evaluate_rows(source, state, plan, options)
        return self._series(values, source)

    async def choice(
        self,
        question: str,
        *,
        choices: Sequence[str] | Mapping[str, JSONContent | None],
        state: State,
        output: Output = "columns",
        client: AsyncTypeSafeClient | None = None,
        model: str | None = None,
        max_concurrency: int = 16,
        cache: MemoryCache | None = None,
        progress: bool | ProgressCallback = False,
        errors: Errors = "raise",
        nulls: Nulls = "include",
    ) -> FrameT:
        """Return labels, confidence, and probabilities as columns or one result struct."""
        _validate_output(output)
        options = Options(client, model, max_concurrency, cache, progress, errors, nulls)
        plan = make_plan({"result": choice_question(question, choices)}, prefix=False)
        source = self._source(state)
        values = await evaluate_rows(source, state, plan, options)
        return self._frame(values, plan, source, output)

    async def score(
        self,
        question: str,
        *,
        levels: Sequence[JSONContent],
        state: State,
        output: Output = "columns",
        client: AsyncTypeSafeClient | None = None,
        model: str | None = None,
        max_concurrency: int = 16,
        cache: MemoryCache | None = None,
        progress: bool | ProgressCallback = False,
        errors: Errors = "raise",
        nulls: Nulls = "include",
    ) -> FrameT:
        """Return level/label, expected score, and probabilities as columns or a struct."""
        _validate_output(output)
        options = Options(client, model, max_concurrency, cache, progress, errors, nulls)
        if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence):
            raise TypeError("levels must be an ordered sequence of descriptions")
        plan = make_plan({"result": Score(instructions=question, criteria=levels)}, prefix=False)
        source = self._source(state)
        values = await evaluate_rows(source, state, plan, options)
        return self._frame(values, plan, source, output)

    async def evaluate(
        self,
        *,
        state: State,
        questions: Mapping[str, Question],
        output: Output = "columns",
        client: AsyncTypeSafeClient | None = None,
        model: str | None = None,
        max_concurrency: int = 16,
        cache: MemoryCache | None = None,
        progress: bool | ProgressCallback = False,
        errors: Errors = "raise",
        nulls: Nulls = "include",
    ) -> FrameT:
        """Evaluate questions per row, with prefixed fields in columns or a result struct."""
        _validate_output(output)
        options = Options(client, model, max_concurrency, cache, progress, errors, nulls)
        plan = make_plan(questions, prefix=True)
        source = self._source(state)
        values = await evaluate_rows(source, state, plan, options)
        return self._frame(values, plan, source, output)
