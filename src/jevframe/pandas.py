"""Import this module to register ``pandas.DataFrame.jev``."""

from __future__ import annotations

from typing import Any

try:
    import numpy as np
    import pandas as pd
except ImportError as error:
    raise ImportError("Install jevframe[pandas] to use the pandas integration") from error

from ._accessor import BaseAccessor
from ._context import RowSource, selected_columns
from ._schema import QuestionPlan, pack_rows
from ._types import Output, State


def _scalar(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.datetime64, np.timedelta64)):
        if np.isnat(value):
            return None
        if isinstance(value, np.datetime64):
            return str(value)
        return pd.Timedelta(value).isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return list(value) if value.ndim else _scalar(value[()])
    return value


@pd.api.extensions.register_dataframe_accessor("jev")
class JevAccessor(BaseAccessor[pd.DataFrame, pd.Series]):
    """Async semantic operations preserving pandas row positions and indexes."""

    def __init__(self, frame: pd.DataFrame):
        self._data = frame

    def _source(self, state: State) -> RowSource:
        columns = selected_columns(list(self._data.columns), state)
        data = self._data.loc[:, columns]
        rows = (
            dict(zip(columns, row, strict=True)) for row in data.itertuples(index=False, name=None)
        )
        if not columns:
            # itertuples(index=False) yields no rows when there are no columns.
            rows = ({} for _ in range(len(data)))
        return RowSource(rows, len(data), _scalar, self._data.index.copy(deep=True))

    def _frame(
        self, values: list[tuple[Any, ...]], plan: QuestionPlan, source: RowSource, output: Output
    ) -> pd.DataFrame:
        types = {"float": "float64", "int": "Int64", "str": "string"}
        if output == "struct":
            return pd.DataFrame(
                {"result": pd.array(pack_rows(values, plan), dtype="object")}, index=source.index
            )
        # Arrays, not index-keyed dicts/Series: duplicate index labels must not align.
        arrays = {
            column.name: pd.array([row[i] for row in values], dtype=types[column.kind])
            for i, column in enumerate(plan.columns)
        }
        return pd.DataFrame(arrays, index=source.index)

    def _series(self, values: list[tuple[Any, ...]], source: RowSource) -> pd.Series:
        return pd.Series(
            [row[0] for row in values], index=source.index, dtype="float64", name="probability"
        )
