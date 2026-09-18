"""Public controls and diagnostics shared by both dataframe integrations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from typesafe_sdk import Choice, JSONContent, Noul, Score

Question: TypeAlias = Noul | Choice | Score
State: TypeAlias = str | Sequence[str] | Callable[[Mapping[str, Any]], JSONContent]
Errors: TypeAlias = Literal["raise", "coerce"]
Nulls: TypeAlias = Literal["include", "skip", "raise"]
Output: TypeAlias = Literal["columns", "struct"]


@dataclass(frozen=True)
class Progress:
    """A completion snapshot. Counts refer to rows, never request attempts.

    ``cache_hits`` includes rows sharing another row's in-flight request.
    Successful rows equal ``completed - failed - skipped``.
    """

    completed: int
    total: int
    failed: int = 0
    skipped: int = 0
    cache_hits: int = 0


ProgressCallback: TypeAlias = Callable[[Progress], None]


@dataclass(frozen=True)
class RowFailure:
    """A failed inference at a zero-based position in the evaluated dataframe."""

    position: int
    cause: Exception


class EvaluationError(RuntimeError):
    """A row failed inference, validation, or context construction.

    ``position`` is authoritative even with duplicate indexes. The underlying
    exception is available as ``cause`` and through exception chaining.
    """

    def __init__(self, position: int, cause: Exception, *, phase: str = "inference"):
        self.position = position
        self.cause = cause
        self.phase = phase
        # Avoid copying SDK response bodies or user context into our diagnostics.
        super().__init__(f"Row {position}: {phase} failed ({type(cause).__name__})")


class EvaluationWarning(UserWarning):
    """Coerced row failures. Inspect ``failures`` for positions and causes."""

    def __init__(self, failures: Sequence[RowFailure]):
        self.failures = tuple(sorted(failures, key=lambda failure: failure.position))
        positions = ", ".join(str(f.position) for f in self.failures[:10])
        if len(self.failures) > 10:
            positions += ", …"
        super().__init__(
            f"Inference failed for {len(self.failures)} row(s) at positions {positions}; "
            "all results for these rows are missing. Inspect .failures for causes."
        )
