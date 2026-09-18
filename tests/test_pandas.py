import asyncio
import datetime as dt
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from conftest import response_body
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
from typesafe_sdk import Choice, Noul, Score

import jevframe.pandas  # noqa: F401
from jevframe import EvaluationError, EvaluationWarning


async def test_out_of_order_duplicate_index_and_assignment(make_client):
    completed = []

    async def handler(payload, _):
        i = payload["state"]["id"]
        await asyncio.sleep((3 - i) * 0.01)
        completed.append(i)
        return response_body(payload, i / 4)

    client, calls, _ = make_client(handler)
    frame = pd.DataFrame({"id": [0, 1, 2, 3]}, index=pd.Index([8, 2, 8, 1], name="ticket"))
    before = frame.copy(deep=True)
    result = await frame.jev.noul("Is it useful?", state="id", client=client)
    assert completed == [3, 2, 1, 0]
    assert len(calls) == 4
    assert_index_equal(result.index, frame.index, exact=True)
    assert_series_equal(result, pd.Series([0, 0.25, 0.5, 0.75], frame.index, name="probability"))
    assert_frame_equal(frame, before)
    frame["answer"] = result
    assert frame["answer"].tolist() == [0, 0.25, 0.5, 0.75]


async def test_multiindex_and_filtered_rows(make_client):
    client, calls, _ = make_client()
    index = pd.MultiIndex.from_tuples([("a", 2), ("a", 2), ("b", 1)], names=["team", "id"])
    frame = pd.DataFrame({"text": ["first", "second", "third"]}, index=index).iloc[[1, 0]]
    result = await frame.jev.choice(
        "Topic?", choices=["bug", "billing"], state="text", client=client
    )
    assert_index_equal(result.index, frame.index, exact=True)
    assert [call["state"] for call in calls] == [{"text": "second"}, {"text": "first"}]
    assert result.columns.tolist() == ["label", "confidence", "p__bug", "p__billing"]
    assert result["p__bug"].tolist() == [1.0, 1.0]


async def test_mixed_questions_share_one_request_per_row(make_client):
    client, calls, _ = make_client()
    frame = pd.DataFrame({"review": ["broken", "great"], "private": [1, 2]})
    questions = {
        "urgent": Noul(instructions="Urgent?"),
        "topic": Choice(instructions="Topic?", criteria={"bug": "Broken", "other": None}),
        "quality": Score(instructions="Quality?", criteria=["poor", "good", "great"]),
    }
    result = await frame.jev.evaluate(state="review", questions=questions, client=client)
    assert len(calls) == 2
    assert all(set(call["questions"]) == set(questions) for call in calls)
    assert all("private" not in call["state"] for call in calls)
    assert result.columns.tolist() == [
        "urgent__probability",
        "topic__label",
        "topic__confidence",
        "topic__p__bug",
        "topic__p__other",
        "quality__level",
        "quality__label",
        "quality__score",
        "quality__confidence",
        "quality__p__0",
        "quality__p__1",
        "quality__p__2",
    ]
    assert result["quality__label"].tolist() == ["good", "good"]
    assert result["quality__score"].tolist() == [0.75, 0.75]


async def test_score_structured_levels_and_ties(make_client):
    def handler(payload, _):
        body = response_body(payload)
        body["answers"]["result"].update(score=0.5, probabilities={"0": 0.5, "1": 0.5})
        return body

    client, _, _ = make_client(handler)
    result = await pd.DataFrame({"x": ["a"]}).jev.score(
        "Grade?", levels=[{"grade": "F"}, ["A", "excellent"]], state="x", client=client
    )
    assert result.iloc[0].to_dict() == {
        "level": 0,
        "label": '{"grade":"F"}',
        "score": 0.5,
        "confidence": 0.5,
        "p__0": 0.5,
        "p__1": 0.5,
    }
    assert str(result["level"].dtype) == "Int64"


async def test_empty_frames_have_schema_without_credentials(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    frame = pd.DataFrame({"x": pd.Series(dtype="string")})
    result = await frame.jev.score("Grade?", levels=["F", "A"], state="x")
    assert result.empty
    assert result.dtypes.astype(str).to_dict() == {
        "level": "Int64",
        "label": "string",
        "score": "float64",
        "confidence": "float64",
        "p__0": "float64",
        "p__1": "float64",
    }
    noul = await frame.jev.noul("Yes?", state="x")
    assert str(noul.dtype) == "float64"
    assert noul.name == "probability"


async def test_nullable_nested_numeric_and_temporal_context(make_client):
    client, calls, _ = make_client()
    frame = pd.DataFrame(
        {
            "id": pd.Series([2**60 + 1], dtype="Int64"),
            "missing": pd.Series([pd.NA], dtype="string"),
            "nested": [[None, np.nan, np.int64(7), np.bool_(True), np.array([1, 2])]],
            "timestamp": [pd.Timestamp("2026-09-18T10:00:00.123456789Z")],
            "date": [dt.date(2026, 9, 18)],
            "duration": [pd.Timedelta(1, unit="ns")],
        }
    )
    await frame.jev.noul("Yes?", state=frame.columns.tolist(), client=client)
    context = calls[0]["state"]
    assert context == {
        "id": 2**60 + 1,
        "missing": None,
        "nested": [None, None, 7, True, [1, 2]],
        "timestamp": "2026-09-18T10:00:00.123456789+00:00",
        "date": "2026-09-18",
        "duration": "P0DT0H0M0.000000001S",
    }


async def test_callable_has_detached_nested_values_and_excludes_index(make_client):
    client, calls, _ = make_client()
    original = {"notes": ["original"]}
    frame = pd.DataFrame({"data": [deepcopy(original)]}, index=["secret-index"])

    def context(row):
        row["data"]["notes"].append("added")
        return row["data"]

    await frame.jev.noul("Yes?", state=context, client=client)
    assert frame.iloc[0, 0] == original
    assert calls[0]["state"] == {"notes": ["original", "added"]}
    assert "secret-index" not in str(calls)


@pytest.mark.parametrize("bad", [object(), float("inf"), {3: "value"}])
async def test_context_errors_never_coerced(make_client, bad):
    client, calls, _ = make_client()
    with pytest.raises(EvaluationError) as caught:
        await pd.DataFrame({"x": [bad]}).jev.noul("Yes?", state="x", client=client, errors="coerce")
    assert caught.value.position == 0
    assert caught.value.phase == "context construction"
    assert calls == []


async def test_null_policies_and_progress(make_client):
    client, calls, _ = make_client()
    frame = pd.DataFrame({"x": ["hello", pd.NA, {"nested": None}, pd.NaT]})
    updates = []
    result = await frame.jev.noul(
        "Yes?", state="x", nulls="skip", client=client, progress=updates.append
    )
    assert len(calls) == 1
    assert result.isna().tolist() == [False, True, True, True]
    assert updates[-1].completed == 4
    assert updates[-1].skipped == 3
    with pytest.raises(EvaluationError) as caught:
        await frame.iloc[1:].jev.noul("Yes?", state="x", nulls="raise", client=client)
    assert caught.value.position == 0


async def test_all_failed_and_all_skipped_outputs(make_client, monkeypatch):
    import httpx2

    client, _, _ = make_client(lambda *_: httpx2.Response(401, json={"error": "denied"}))
    frame = pd.DataFrame({"x": ["a", "b"]})
    with pytest.warns(EvaluationWarning) as warnings:
        result = await frame.jev.choice(
            "Which?", choices=["a", "b"], state="x", client=client, errors="coerce"
        )
    assert result.isna().all().all()
    assert str(result["p__a"].dtype) == "float64"
    assert [f.position for f in warnings[0].message.failures] == [0, 1]
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    skipped = await pd.DataFrame({"x": [None]}).jev.noul("Yes?", state="x", nulls="skip")
    assert skipped.isna().all()


@pytest.mark.parametrize("state", [[], ["x", "x"], "missing", [1]])
async def test_invalid_state_before_requests(make_client, state):
    client, calls, _ = make_client()
    with pytest.raises((ValueError, TypeError)):
        await pd.DataFrame({"x": [1]}).jev.noul("Yes?", state=state, client=client)
    assert calls == []


async def test_ambiguous_columns_and_generated_column_collision(make_client):
    client, calls, _ = make_client()
    with pytest.raises(ValueError, match="Ambiguous"):
        await pd.DataFrame([[1, 2]], columns=["x", "x"]).jev.noul("Yes?", state="x", client=client)
    with pytest.raises(ValueError, match="collide"):
        await pd.DataFrame({"x": [1]}).jev.evaluate(
            state="x",
            client=client,
            questions={
                "a": Choice(criteria={"b__probability": None}),
                "a__p__b": Noul(instructions="Yes?"),
            },
        )
    assert calls == []


async def test_choice_mapping_and_reject_duplicate_labels(make_client):
    client, calls, _ = make_client()
    frame = pd.DataFrame({"x": ["hi"]})
    with pytest.raises(ValueError, match="unique"):
        await frame.jev.choice("Topic?", choices=["a", "a"], state="x", client=client)
    await frame.jev.choice(
        "Topic?", choices={"a": {"description": "A"}, "b": None}, state="x", client=client
    )
    assert calls[0]["questions"]["result"]["criteria"] == {"a": {"description": "A"}, "b": None}


async def test_callable_on_zero_column_frame_preserves_row_count(make_client):
    client, calls, _ = make_client()
    frame = pd.DataFrame(index=pd.Index(["duplicate", "duplicate"], name="id"))
    result = await frame.jev.noul("Yes?", state=lambda row: "constant context", client=client)
    assert result.tolist() == [0.75, 0.75]
    assert len(calls) == 2
    assert_index_equal(result.index, frame.index)
