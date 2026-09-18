"""The two output layouts preserve the same complete decisions and row identity."""

import asyncio

import httpx2
import pandas as pd
import polars as pl
import pytest
from conftest import response_body
from pandas.testing import assert_index_equal
from polars.testing import assert_frame_equal
from typesafe_sdk import Choice, Noul, Score

import jevframe.pandas  # noqa: F401
import jevframe.polars  # noqa: F401
from jevframe import EvaluationWarning, MemoryCache


def frame_for(backend, values):
    if backend == "pandas":
        index = pd.MultiIndex.from_tuples(
            [("team", i % 2) for i in range(len(values))], names=["team", "id"]
        )
        return pd.DataFrame({"text": values}, index=index)
    return pl.DataFrame({"text": values}, schema={"text": pl.String})


async def evaluate(frame, method, **kwargs):
    if method == "choice":
        return await frame.jev.choice("Topic?", choices=["a__b", "other"], state="text", **kwargs)
    if method == "score":
        return await frame.jev.score("Grade?", levels=[{"grade": "F"}, "A"], state="text", **kwargs)
    return await frame.jev.evaluate(
        state="text",
        questions={
            "yes__no": Noul(instructions="Yes?"),
            "topic": Choice(criteria={"a__b": None, "other": None}),
            "grade": Score(criteria=[{"grade": "F"}, "A"]),
        },
        **kwargs,
    )


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["choice", "score", "evaluate"])
async def test_layout_equivalence_order_and_cache_reuse(backend, method, make_client):
    completed = []

    async def handler(payload, _):
        position = int(payload["state"]["text"])
        await asyncio.sleep((3 - position) * 0.01)
        completed.append(position)
        return response_body(payload, position / 4)

    client, calls, _ = make_client(handler)
    frame = frame_for(backend, ["0", "1", "0", "2"])
    cache = MemoryCache()
    columns = await evaluate(frame, method, client=client, cache=cache)
    packed = await evaluate(frame, method, client=client, cache=cache, output="struct")
    assert completed == [2, 1, 0]
    assert len(calls) == 3
    assert packed.shape == (4, 1)
    assert list(packed.columns) == ["result"]
    if backend == "pandas":
        assert_index_equal(packed.index, frame.index, exact=True)
        assert packed["result"].dtype == object
        assert packed["result"].tolist() == columns.to_dict("records")
        assert list(packed["result"].iloc[0]) == list(columns.columns)
        assert "result" not in frame.columns
        frame["decisions"] = packed["result"]
        assert frame["decisions"].tolist() == columns.to_dict("records")
        # Mutable cells are independent of other rows and of cached tuples.
        field = next(iter(packed["result"].iloc[0]))
        original = packed["result"].iloc[2][field]
        packed["result"].iloc[0][field] = "changed"
        assert packed["result"].iloc[2][field] == original
        again = await evaluate(frame, method, client=client, cache=cache, output="struct")
        assert again["result"].iloc[0][field] == original
        assert len(calls) == 3
    else:
        assert_frame_equal(packed.unnest("result"), columns)
        assigned = frame.with_columns(packed["result"].alias("decisions"))
        assert assigned["decisions"].to_list() == columns.to_dicts()


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["choice", "score", "evaluate"])
async def test_struct_schema_for_empty_skipped_and_failed_rows(backend, method, make_client):
    client, calls, _ = make_client(lambda *_: httpx2.Response(401))
    empty = await evaluate(frame_for(backend, []), method, output="struct")
    frame = frame_for(backend, [None, "bad"])
    with pytest.warns(EvaluationWarning) as warnings:
        missing = await evaluate(
            frame, method, output="struct", client=client, nulls="skip", errors="coerce"
        )
    assert len(calls) == 1
    assert [failure.position for failure in warnings[0].message.failures] == [1]
    assert empty.shape == (0, 1)
    assert missing.shape == (2, 1)
    if backend == "pandas":
        assert empty["result"].dtype == missing["result"].dtype == object
        assert missing["result"].tolist() == [None, None]
        assert missing["result"].isna().all()
        assert_index_equal(missing.index, frame.index, exact=True)
    else:
        assert empty.schema == missing.schema
        assert missing["result"].to_list() == [None, None]
        assert missing["result"].is_null().all()
        # Even an empty struct has the full field schema, with native scalar types.
        columns = await evaluate(frame_for(backend, []), method)
        assert_frame_equal(empty.unnest("result"), columns)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["choice", "score", "evaluate"])
@pytest.mark.parametrize("output", ["unknown", None])
async def test_invalid_layout_rejected_before_inference(backend, method, output, make_client):
    client, calls, _ = make_client()
    with pytest.raises(ValueError, match="output must be"):
        await evaluate(frame_for(backend, ["hello"]), method, client=client, output=output)
    assert calls == []
