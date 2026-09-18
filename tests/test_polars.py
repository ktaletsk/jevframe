import asyncio

import httpx2
import polars as pl
import pytest
from conftest import response_body
from polars.testing import assert_frame_equal
from typesafe_sdk import Choice, Noul, Score

import jevframe.polars  # noqa: F401
from jevframe import EvaluationWarning


async def test_native_order_and_unchanged_input(make_client):
    async def handler(payload, _):
        i = payload["state"]["id"]
        await asyncio.sleep((3 - i) * 0.005)
        return response_body(payload, i / 4)

    client, _, _ = make_client(handler)
    frame = pl.DataFrame({"id": [0, 1, 2, 3]})
    before = frame.clone()
    result = await frame.jev.noul("Yes?", state="id", client=client)
    assert isinstance(result, pl.Series)
    assert result.dtype == pl.Float64
    assert result.name == "probability"
    assert result.to_list() == [0, 0.25, 0.5, 0.75]
    assert_frame_equal(frame, before)
    assigned = frame.with_columns(result.alias("answer"))
    assert assigned["answer"].to_list() == result.to_list()


async def test_mixed_native_schema_and_filter(make_client):
    client, calls, _ = make_client()
    frame = pl.DataFrame({"x": ["a", "b", "c"]}).filter(pl.col("x") != "b")
    result = await frame.jev.evaluate(
        state="x",
        client=client,
        questions={
            "q": Noul(),
            "c": Choice(criteria={"a": None, "b": None}),
            "s": Score(criteria=["F", "A"]),
        },
    )
    assert isinstance(result, pl.DataFrame)
    assert result.height == 2
    assert [call["state"]["x"] for call in calls] == ["a", "c"]
    assert result["q__probability"].to_list() == [0.75, 0.75]
    assert result["s__score"].to_list() == [0.75, 0.75]
    assert result["s__level"].dtype == pl.Int64
    assert result["c__label"].dtype == pl.String
    assert result["c__p__b"].dtype == pl.Float64


async def test_empty_and_all_failed_native_schema(make_client, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    frame = pl.DataFrame(schema={"x": pl.String})
    empty = await frame.jev.score("Grade?", levels=["F", "A"], state="x")
    assert empty.shape == (0, 6)
    assert empty.schema == {
        "level": pl.Int64,
        "label": pl.String,
        "score": pl.Float64,
        "confidence": pl.Float64,
        "p__0": pl.Float64,
        "p__1": pl.Float64,
    }
    client, _, _ = make_client(lambda *_: httpx2.Response(401))
    with pytest.warns(EvaluationWarning):
        failed = await pl.DataFrame({"x": ["bad"]}).jev.score(
            "Grade?", levels=["F", "A"], state="x", client=client, errors="coerce"
        )
    assert failed.schema == empty.schema
    assert failed.row(0) == (None,) * 6


async def test_lossless_temporal_and_nested_context(make_client):
    client, calls, _ = make_client()
    ts = pl.Series("ts", [1_000_000_001, None], dtype=pl.Datetime("ns", time_zone="UTC"))
    frame = pl.DataFrame(
        {
            "ts": ts,
            "duration": pl.Series([1, -1], dtype=pl.Duration("ns")),
            "time": pl.Series([1, None], dtype=pl.Time),
            "id": [2**60 + 1, 2],
            "list": pl.Series([[1_000_000_001], [None]], dtype=pl.List(pl.Datetime("ns"))),
        }
    ).with_columns(pl.struct("ts").alias("struct"))
    await frame.jev.noul("Yes?", state=frame.columns, client=client)
    context = calls[0]["state"]
    assert context["ts"] == "1970-01-01T00:00:01.000000001+00:00"
    assert context["duration"] == "PT0.000000001S"
    assert context["time"] == "00:00:00.000000001"
    assert context["list"] == ["1970-01-01T00:00:01.000000001"]
    assert context["struct"] == {"ts": context["ts"]}
    assert context["id"] == 2**60 + 1
    assert calls[1]["state"]["ts"] is None
    assert calls[1]["state"]["duration"] == "-PT0.000000001S"


async def test_callable_and_null_policy_after_transformation(make_client):
    client, calls, _ = make_client()
    frame = pl.DataFrame({"x": [None, "real"]})
    result = await frame.jev.noul(
        "Yes?",
        state=lambda row: {"text": row["x"] or "fallback"},
        nulls="raise",
        client=client,
    )
    assert result.to_list() == [0.75, 0.75]
    assert calls[0]["state"] == {"text": "fallback"}
    skipped = await frame.jev.noul("Yes?", state="x", nulls="skip", client=client)
    assert skipped.to_list() == [None, 0.75]


def test_no_lazy_namespace():
    assert not hasattr(pl.LazyFrame({"x": [1]}), "jev")


async def test_empty_struct_context(make_client):
    client, calls, _ = make_client()
    frame = pl.DataFrame({"x": [{}, {}]})
    result = await frame.jev.noul("Yes?", state="x", client=client)
    assert result.to_list() == [0.75, 0.75]
    assert calls[0]["state"] == {"x": {}}
