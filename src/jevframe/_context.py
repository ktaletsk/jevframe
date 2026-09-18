"""Context construction without importing either dataframe library."""

from __future__ import annotations

import datetime as dt
import inspect
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ._types import State


def identity(value: Any) -> Any:
    return value


@dataclass
class RowSource:
    rows: Iterator[dict[str, Any]]
    count: int
    convert_scalar: Callable[[Any], Any] = identity
    index: Any = None


def selected_columns(columns: Sequence[Any], state: State) -> list[str]:
    """Validate selection before inference; callable context sees every column."""
    if callable(state):
        if inspect.iscoroutinefunction(state):
            raise TypeError("state must be a synchronous callable")
        selected = list(columns)
    elif isinstance(state, str):
        selected = [state]
    elif isinstance(state, Sequence) and not isinstance(state, bytes):
        selected = list(state)
    else:
        raise TypeError("state must be a column name, sequence of names, or synchronous callable")
    if not selected and not callable(state):
        raise ValueError("state must select at least one column")
    if any(not isinstance(name, str) for name in selected):
        raise TypeError("Context column names must be strings")
    if len(set(selected)) != len(selected):
        raise ValueError("Context column names must be unique")
    for name in selected:
        occurrences = list(columns).count(name)
        if occurrences == 0:
            raise ValueError(f"Unknown context column: {name!r}")
        if occurrences > 1:
            raise ValueError(f"Ambiguous context column: {name!r}")
    return selected


def normalize(value: Any, convert_scalar: Callable[[Any], Any] = identity) -> Any:
    """Return strict JSON values, keeping map/list order and temporal precision."""
    value = convert_scalar(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError("Infinite numbers are not valid context")
        return value
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        # Timedeltas are exact at Python's microsecond resolution, including negatives.
        microseconds = (value.days * 86400 + value.seconds) * 1_000_000 + value.microseconds
        sign = "-" if microseconds < 0 else ""
        seconds, micros = divmod(abs(microseconds), 1_000_000)
        return f"{sign}PT{seconds}.{micros:06d}S"
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Context object keys must be strings")
        return {key: normalize(item, convert_scalar) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item, convert_scalar) for item in value]
    raise TypeError(f"Unsupported context type: {type(value).__name__}")


def construct_context(row: Mapping[str, Any], state: State, source: RowSource) -> Any:
    # Deep copying is necessary: pandas object columns can contain lists/dicts.
    value = state(deepcopy(dict(row))) if callable(state) else row
    if inspect.isawaitable(value):
        if inspect.iscoroutine(value):
            value.close()
        raise TypeError("state must return context synchronously")
    result = normalize(value, source.convert_scalar)
    if not isinstance(result, (str, dict, list)):
        raise TypeError("Constructed state must be text, an object, or an array")
    return result


def contains_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(contains_null(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_null(item) for item in value)
    return False
