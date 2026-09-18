import asyncio

import httpx2
import pandas as pd
import pytest
from conftest import response_body
from typesafe_sdk import Choice

import jevframe.pandas  # noqa: F401
from jevframe import EvaluationWarning, MemoryCache, _engine


async def test_disabled_by_default(make_client):
    client, calls, _ = make_client()
    frame = pd.DataFrame({"x": [1, 1]})
    for _ in range(2):
        await frame.jev.noul("Yes?", state="x", client=client)
    assert len(calls) == 4


async def test_coalesce_inflight_and_reuse_across_calls(make_client):
    async def handler(payload, _):
        await asyncio.sleep(0.005)
        return response_body(payload)

    client, calls, _ = make_client(handler)
    cache = MemoryCache()
    frame = pd.DataFrame({"x": ["same"] * 20}, index=[1] * 20)
    updates = []
    first = await frame.jev.noul(
        "Yes?", state="x", client=client, cache=cache, progress=updates.append
    )
    assert len(calls) == len(cache) == 1
    assert updates[-1].cache_hits == 19
    assert updates[-1].completed == 20
    # Row labels do not form part of the cache key; output mutation cannot poison it.
    first.iloc[0] = 0
    second = await frame.reset_index(drop=True).jev.noul(
        "Yes?", state="x", client=client, cache=cache
    )
    assert second.tolist() == [0.75] * 20
    assert len(calls) == 1
    cache.clear()
    assert len(cache) == 0
    await frame.jev.noul("Yes?", state="x", client=client, cache=cache)
    assert len(calls) == 2


async def test_cache_keys_include_questions_model_and_client(make_client):
    first, calls, _ = make_client(model="first-model")
    second, other_calls, _ = make_client(model="second-model")
    frame = pd.DataFrame({"x": [1]})
    cache = MemoryCache()
    for question, model in [("One?", None), ("Two?", None), ("One?", "override")]:
        await frame.jev.noul(question, state="x", client=first, model=model, cache=cache)
    await frame.jev.noul("One?", state="x", client=second, cache=cache)
    assert len(calls) == 3
    assert len(other_calls) == 1
    assert len(cache) == 4


async def test_owned_cache_identity_and_env_invalidation(make_client, monkeypatch):
    created = []

    def factory(**kwargs):
        client, calls, _ = make_client()
        created.append((kwargs, calls))
        return client

    monkeypatch.setattr(_engine, "_create_client", factory)
    monkeypatch.setenv("TYPESAFE_API_KEY", "first-key")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "model-a")
    frame = pd.DataFrame({"x": [1]})
    cache = MemoryCache()
    await frame.jev.noul("Yes?", state="x", cache=cache)
    await frame.jev.noul("Yes?", state="x", cache=cache)
    assert sum(len(calls) for _, calls in created) == 1
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "model-b")
    await frame.jev.noul("Yes?", state="x", cache=cache)
    monkeypatch.setenv("TYPESAFE_API_KEY", "second-key")
    await frame.jev.noul("Yes?", state="x", cache=cache)
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://another.test")
    await frame.jev.noul("Yes?", state="x", cache=cache)
    assert sum(len(calls) for _, calls in created) == 4


async def test_criterion_order_and_mutation_invalidate_cache(make_client):
    client, calls, _ = make_client()
    frame = pd.DataFrame({"x": [1]})
    cache = MemoryCache()
    question = Choice(criteria={"a": None, "b": None})
    for criteria in (
        {"a": None, "b": None},
        {"b": None, "a": None},
        {"b": "description", "a": None},
    ):
        question.criteria = criteria
        await frame.jev.evaluate(
            state="x", questions={"topic": question}, client=client, cache=cache
        )
    assert len(calls) == 3


async def test_failures_are_not_cached_and_coerced_per_row(make_client):
    async def handler(payload, call):
        await asyncio.sleep(0.005)
        return httpx2.Response(401) if call == 1 else response_body(payload)

    client, calls, _ = make_client(handler)
    frame = pd.DataFrame({"x": [1, 1, 1]})
    cache = MemoryCache()
    with pytest.warns(EvaluationWarning) as caught:
        result = await frame.jev.noul(
            "Yes?", state="x", client=client, cache=cache, errors="coerce"
        )
    assert result.isna().all()
    assert [f.position for f in caught[0].message.failures] == [0, 1, 2]
    assert len(cache) == 0
    assert len(calls) == 1
    result = await frame.jev.noul("Yes?", state="x", client=client, cache=cache)
    assert result.tolist() == [0.75] * 3
    assert len(calls) == 2


async def test_lru_eviction(make_client):
    client, calls, _ = make_client()
    cache = MemoryCache(max_entries=2)
    for value in [1, 2, 1, 3, 2]:
        await pd.DataFrame({"x": [value]}).jev.noul("Yes?", state="x", client=client, cache=cache)
    assert [call["state"]["x"] for call in calls] == [1, 2, 3, 2]
    assert len(cache) == 2


async def test_clear_during_request_discards_pending_result(make_client):
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(payload, _):
        entered.set()
        await release.wait()
        return response_body(payload)

    client, _, _ = make_client(handler)
    cache = MemoryCache()
    task = asyncio.create_task(
        pd.DataFrame({"x": [1]}).jev.noul("Yes?", state="x", client=client, cache=cache)
    )
    await entered.wait()
    cache.clear()
    release.set()
    await task
    assert len(cache) == 0


async def test_cancellation_with_coalesced_waiters(make_client):
    entered = asyncio.Event()

    async def handler(payload, _):
        entered.set()
        await asyncio.Event().wait()

    client, calls, _ = make_client(handler)
    cache = MemoryCache()
    task = asyncio.create_task(
        pd.DataFrame({"x": [1] * 10}).jev.noul("Yes?", state="x", client=client, cache=cache)
    )
    await entered.wait()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(calls) == 1
    assert len(cache) == 0


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_cache_size(size):
    with pytest.raises(ValueError):
        MemoryCache(size)
