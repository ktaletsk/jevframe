"""Import this module to register eager ``polars.DataFrame.jev``."""

from __future__ import annotations

from typing import Any

try:
    import polars as pl
except ImportError as error:
    raise ImportError("Install jevframe[polars] to use the Polars integration") from error

from ._accessor import BaseAccessor
from ._context import RowSource, selected_columns
from ._schema import QuestionPlan, pack_rows
from ._types import Output, State


def _duration(value: int, unit: str) -> str:
    nanos = value * {"ns": 1, "us": 1_000, "ms": 1_000_000}[unit]
    sign = "-" if nanos < 0 else ""
    seconds, fractional = divmod(abs(nanos), 1_000_000_000)
    return f"{sign}PT{seconds}.{fractional:09d}S"


def _lossless_temporal(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    # iter_rows converts temporal values to Python, which truncates nanoseconds.
    # Convert before extraction, including temporal values nested in containers.
    if isinstance(dtype, pl.Datetime):
        fmt = "%Y-%m-%dT%H:%M:%S%.f" + ("%:z" if dtype.time_zone else "")
        return expr.dt.to_string(fmt)
    if dtype == pl.Time:
        return expr.dt.to_string("%H:%M:%S%.f")
    if isinstance(dtype, pl.Duration):
        return expr.cast(pl.Int64).map_elements(
            lambda value: _duration(value, dtype.time_unit), return_dtype=pl.String
        )
    if isinstance(dtype, pl.List):
        return expr.list.eval(_lossless_temporal(pl.element(), dtype.inner))
    if isinstance(dtype, pl.Array):
        return expr.arr.to_list().list.eval(_lossless_temporal(pl.element(), dtype.inner))
    if isinstance(dtype, pl.Struct):
        if not dtype.fields:
            return expr
        fields = [
            _lossless_temporal(expr.struct.field(field.name), field.dtype).alias(field.name)
            for field in dtype.fields
        ]
        return pl.when(expr.is_null()).then(None).otherwise(pl.struct(fields))
    return expr


@pl.api.register_dataframe_namespace("jev")
class JevNamespace(BaseAccessor[pl.DataFrame, pl.Series]):
    """Async semantic operations for eager Polars; outputs stay native."""

    def __init__(self, frame: pl.DataFrame):
        self._data = frame

    def _source(self, state: State) -> RowSource:
        columns = selected_columns(self._data.columns, state)
        data = self._data.select(
            _lossless_temporal(pl.col(name), self._data.schema[name]).alias(name)
            for name in columns
        )
        rows = (dict(zip(columns, row, strict=True)) for row in data.iter_rows())
        return RowSource(rows, data.height)

    def _frame(
        self, values: list[tuple[Any, ...]], plan: QuestionPlan, source: RowSource, output: Output
    ) -> pl.DataFrame:
        types = {"float": pl.Float64, "int": pl.Int64, "str": pl.String}
        if output == "struct":
            dtype = pl.Struct({column.name: types[column.kind] for column in plan.columns})
            return pl.DataFrame(pl.Series("result", pack_rows(values, plan), dtype=dtype))
        return pl.DataFrame(
            {
                column.name: pl.Series(
                    column.name, [row[i] for row in values], dtype=types[column.kind]
                )
                for i, column in enumerate(plan.columns)
            }
        )

    def _series(self, values: list[tuple[Any, ...]], source: RowSource) -> pl.Series:
        return pl.Series("probability", [row[0] for row in values], dtype=pl.Float64)
