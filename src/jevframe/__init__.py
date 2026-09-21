"""Semantic dataframe evaluation using TypeSafe Jev.

Import ``jevframe.pandas`` or ``jevframe.polars`` to register the native accessor.

Jev is reached through TypeSafe's API by default, or through OpenRouter with
``provider="openrouter"`` (or automatically when only ``OPENROUTER_API_KEY`` is set).
"""

from ._cache import MemoryCache
from ._providers import openrouter_client
from ._types import EvaluationError, EvaluationWarning, Progress, RowFailure

__all__ = [
    "EvaluationError",
    "EvaluationWarning",
    "MemoryCache",
    "Progress",
    "RowFailure",
    "openrouter_client",
]
