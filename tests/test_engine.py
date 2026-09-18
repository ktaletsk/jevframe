import asyncio

import httpx2
import pandas as pd
import pytest
from conftest import response_body
from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeAuthenticationError

import jevframe.pandas  # noqa: F401
from jevframe import EvaluationError, EvaluationWarning, Progress, _engine


async def test_bounded_workers_and_request_concurrency(make_client):
    active = peak = 0
    worker_ids = set()

    async def handler(payload, _):
        nonlocal active, peak
        worker_ids.add(id(asyncio.current_task()))
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.001)
            return response_body(payload)
        finally:
            active -= 1

    client, calls, _ = make_client(handler)
    result = await pd.DataFrame({"x": range(100)}).jev.noul(
        "Yes?", state="x", client=client, max_concurrency=3
    )
    assert len(result) == len(calls) == 100
    assert peak == 3
    assert len(worker_ids) == 3
    assert active == 0


async def test_cancellation_drains_workers_and_keeps_supplied_client_open(make_client):
    entered = asyncio.Event()
    active = 0

    async def handler(payload, _):
        nonlocal active
        active += 1
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1

    client, _, http = make_client(handler)
    task = asyncio.create_task(
        pd.DataFrame({"x": range(20)}).jev.noul(
            "Yes?", state="x", client=client, max_concurrency=3, errors="coerce"
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert active == 0
    assert not http.is_closed


async def test_failure_cancels_other_rows_and_preserves_input(make_client):
    release = asyncio.Event()
    active = 0

    async def handler(payload, _):
        nonlocal active
        active += 1
        if active == 3:
            release.set()
        try:
            await release.wait()
            if payload["state"]["x"] == 1:
                return httpx2.Response(401)
            await asyncio.Event().wait()
        finally:
            active -= 1

    client, calls, _ = make_client(handler)
    frame = pd.DataFrame({"x": range(10)}, index=["duplicate"] * 10)
    before = frame.copy(deep=True)
    with pytest.raises(EvaluationError) as caught:
        await frame.jev.noul("Yes?", state="x", client=client, max_concurrency=3)
    assert caught.value.position == 1
    assert isinstance(caught.value.cause, TypeSafeAuthenticationError)
    assert active == 0
    assert len(calls) == 3
    pd.testing.assert_frame_equal(frame, before)


async def test_sdk_retry_policy_and_progress_counts(make_client):
    def handler(payload, call):
        if call < 3:
            return httpx2.Response(429 if call == 1 else 503, headers={"retry-after-ms": "1"})
        return response_body(payload)

    client, calls, _ = make_client(
        handler, retry=RetryPolicy(max_retries=2, backoff_initial=0, backoff_max=0)
    )
    updates = []
    result = await pd.DataFrame({"x": [1]}).jev.noul(
        "Yes?", state="x", client=client, progress=updates.append
    )
    assert result.tolist() == [0.75]
    assert len(calls) == 3
    assert updates == [Progress(completed=1, total=1)]


async def test_sdk_connection_retry(make_client):
    def handler(payload, call):
        if call == 1:
            raise httpx2.ConnectError("temporary")
        return response_body(payload)

    client, calls, _ = make_client(handler, retry=RetryPolicy(max_retries=1, backoff_initial=0))
    await pd.DataFrame({"x": [1]}).jev.noul("Yes?", state="x", client=client)
    assert len(calls) == 2


async def test_auth_failure_never_retried(make_client):
    client, calls, _ = make_client(lambda *_: httpx2.Response(401), retry=RetryPolicy())
    with pytest.raises(EvaluationError):
        await pd.DataFrame({"x": [1]}).jev.noul("Yes?", state="x", client=client)
    assert len(calls) == 1


async def test_owned_client_configuration_and_cleanup(make_client, monkeypatch):
    client, calls, http = make_client()
    created = []

    def factory(**kwargs):
        created.append(kwargs)
        return client

    monkeypatch.setattr(_engine, "_create_client", factory)
    monkeypatch.setenv("TYPESAFE_API_KEY", " local-test-key ")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", " pinned-test-model ")
    monkeypatch.setenv("TYPESAFE_BASE_URL", " https://example.test ")
    await pd.DataFrame({"x": [1, 2]}).jev.noul("Yes?", state="x")
    assert created == [
        {
            "api_key": "local-test-key",
            "model": "pinned-test-model",
            "base_url": "https://example.test",
        }
    ]
    assert all(call["model"] == "pinned-test-model" for call in calls)
    assert http.is_closed


async def test_supplied_client_defaults_and_override(make_client):
    client, calls, http = make_client(model="client-model")
    frame = pd.DataFrame({"x": [1]})
    await frame.jev.noul("Yes?", state="x", client=client)
    await frame.jev.noul("Yes?", state="x", client=client, model="override-model")
    assert [call["model"] for call in calls] == ["client-model", "override-model"]
    assert not http.is_closed


async def test_question_snapshot_is_used_after_mutation(make_client):
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(payload, _):
        entered.set()
        await release.wait()
        return response_body(payload)

    client, calls, _ = make_client(handler)
    question = Choice(
        instructions="Original?", criteria={"a": {"description": "original"}, "b": None}
    )
    questions = {"topic": question}
    task = asyncio.create_task(
        pd.DataFrame({"x": [1, 2]}).jev.evaluate(
            state="x", questions=questions, client=client, max_concurrency=1
        )
    )
    await entered.wait()
    question.instructions = "Changed?"
    question.criteria["a"]["description"] = "changed"
    question.criteria["c"] = None
    questions["added"] = Noul()
    release.set()
    result = await task
    assert "topic__p__c" not in result
    assert all(call["questions"] == calls[0]["questions"] for call in calls)
    assert calls[0]["questions"]["topic"]["criteria"]["a"] == {"description": "original"}


@pytest.mark.parametrize(
    "bad_answer",
    [
        {"type": "noul", "noul": -0.1},
        {"type": "noul", "noul": 1.1},
        {"type": "noul", "noul": True},
        {"type": "future_answer", "value": 0.75},
        {"type": "choice", "choice": "a", "confidence": 1, "probabilities": {"a": 1}},
    ],
)
async def test_invalid_noul_never_becomes_a_probability(make_client, bad_answer):
    def handler(payload, _):
        body = response_body(payload)
        body["answers"]["result"] = bad_answer
        return body

    client, _, _ = make_client(handler)
    with pytest.raises(EvaluationError):
        await pd.DataFrame({"x": [1]}).jev.noul("Yes?", state="x", client=client)


@pytest.mark.parametrize(
    "patch",
    [
        {"probabilities": {"a": 1.0}},
        {"probabilities": {"a": 0.2, "b": 0.2}},
        {"probabilities": {"a": 1.1, "b": -0.1}},
        {"probabilities": {"a": 0.5, "b": 0.5, "extra": 0.0}},
        {"choice": "unknown"},
        {"confidence": -0.2},
    ],
)
async def test_invalid_choice_distribution(make_client, patch):
    def handler(payload, _):
        body = response_body(payload)
        body["answers"]["result"].update(patch)
        return body

    client, _, _ = make_client(handler)
    with pytest.raises(EvaluationError):
        await pd.DataFrame({"x": [1]}).jev.choice(
            "Topic?", choices=["a", "b"], state="x", client=client
        )


@pytest.mark.parametrize(
    "patch",
    [
        {"score": 0.25},
        {"score": 1.1},
        {"legend": {"0": "changed", "1": "A"}},
        {"probabilities": {"0": 1.0}},
        {"confidence": 2.0},
    ],
)
async def test_invalid_score(make_client, patch):
    def handler(payload, _):
        body = response_body(payload)
        body["answers"]["result"].update(patch)
        return body

    client, _, _ = make_client(handler)
    with pytest.raises(EvaluationError):
        await pd.DataFrame({"x": [1]}).jev.score(
            "Grade?", levels=["F", "A"], state="x", client=client
        )


async def test_partial_or_extra_answers_fail_the_whole_row(make_client):
    def handler(payload, call):
        body = response_body(payload)
        if call == 1:
            del body["answers"]["b"]
        else:
            body["answers"]["extra"] = {"type": "noul", "noul": 0.5}
        return body

    client, _, _ = make_client(handler)
    with pytest.warns(EvaluationWarning):
        result = await pd.DataFrame({"x": [1, 2]}).jev.evaluate(
            state="x", questions={"a": Noul(), "b": Noul()}, client=client, errors="coerce"
        )
    assert result.isna().all().all()


async def test_callback_errors_propagate_and_owned_client_closes(make_client, monkeypatch):
    client, _, http = make_client()
    monkeypatch.setattr(_engine, "_create_client", lambda **_: client)

    def progress(_):
        raise LookupError("callback failed")

    with pytest.raises(LookupError, match="callback failed"):
        await pd.DataFrame({"x": [1]}).jev.noul(
            "Yes?", state="x", errors="coerce", progress=progress
        )
    assert http.is_closed


async def test_tqdm_progress_closes(make_client, monkeypatch):
    import tqdm.auto

    bars = []

    class Bar:
        def __init__(self, **kwargs):
            self.total = kwargs["total"]
            self.completed = 0
            self.closed = False
            bars.append(self)

        def update(self, n):
            self.completed += n

        def close(self):
            self.closed = True

    monkeypatch.setattr(tqdm.auto, "tqdm", Bar)
    client, _, _ = make_client()
    await pd.DataFrame({"x": [1, 2]}).jev.noul("Yes?", state="x", client=client, progress=True)
    assert bars[0].completed == bars[0].total == 2
    assert bars[0].closed


@pytest.mark.parametrize(
    "options",
    [
        {"max_concurrency": 0},
        {"max_concurrency": True},
        {"max_concurrency": 1.5},
        {"errors": "ignore"},
        {"nulls": "ignore"},
        {"cache": True},
        {"progress": "yes"},
        {"model": ""},
        {"client": object()},
    ],
)
async def test_invalid_options_on_empty_input(options):
    with pytest.raises((ValueError, TypeError)):
        await pd.DataFrame({"x": []}).jev.noul("Yes?", state="x", **options)


async def test_invalid_question_objects_before_requests(make_client):
    client, calls, _ = make_client()
    for questions in ({}, {"x": {"type": "noul"}}, {"x": Score(criteria=["one"])}, {"": Noul()}):
        with pytest.raises((ValueError, TypeError)):
            await pd.DataFrame({"x": [1]}).jev.evaluate(
                state="x", questions=questions, client=client
            )
    assert calls == []


async def test_async_context_and_progress_rejected(make_client):
    async def callback(_):
        return "text"

    client, calls, _ = make_client()
    frame = pd.DataFrame({"x": [1]})
    with pytest.raises(TypeError, match="synchronous"):
        await frame.jev.noul("Yes?", state=callback, client=client)
    with pytest.raises(TypeError, match="synchronous"):
        await frame.jev.noul("Yes?", state="x", progress=callback, client=client)
    assert calls == []


async def test_unknown_extra_answer_is_not_silently_discarded(make_client):
    def handler(payload, _):
        body = response_body(payload)
        body["answers"]["unexpected"] = {"type": "future_answer", "value": 0.8}
        return body

    client, _, _ = make_client(handler)
    with pytest.raises(EvaluationError):
        await pd.DataFrame({"x": [1]}).jev.noul("Yes?", state="x", client=client)
