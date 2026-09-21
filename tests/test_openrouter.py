from __future__ import annotations

import json
import warnings

import httpx2
import pandas as pd
import pytest
from conftest import response_body
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeAuthenticationError

import jevframe.pandas  # noqa: F401
from jevframe import EvaluationError, EvaluationWarning, _engine, openrouter_client
from jevframe._providers import (
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_DECISIONS_PATH,
    OPENROUTER_DEFAULT_BASE_URL,
    OPENROUTER_DEFAULT_MODEL,
    DecisionsTransport,
    adapt_request,
    qualify_model,
    resolve_backend,
    select_provider,
)

TYPESAFE_KEY_ENV = "TYPESAFE_API_KEY"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (
        TYPESAFE_KEY_ENV,
        "TYPESAFE_BASE_URL",
        "TYPESAFE_DEFAULT_MODEL",
        OPENROUTER_API_KEY_ENV,
        "OPENROUTER_BASE_URL",
        "OPENROUTER_DEFAULT_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def mock_transport(handler=None):
    """A DecisionsTransport whose inner transport records the rewritten requests."""
    requests = []

    def handle(request):
        requests.append(request)
        payload = json.loads(request.content)
        result = handler(payload, len(requests)) if handler else response_body(payload)
        return result if isinstance(result, httpx2.Response) else httpx2.Response(200, json=result)

    return DecisionsTransport(httpx2.MockTransport(handle)), requests


def system_one_request(model: str = "jev-1.13") -> httpx2.Request:
    body = {"state": {"x": 1}, "model": model, "questions": {"result": {"type": "noul"}}}
    content = json.dumps(body).encode()
    return httpx2.Request(
        "POST",
        "https://openrouter.ai/v1/systemone",
        headers={"Authorization": "Bearer test-key", "Content-Type": "application/json"},
        content=content,
        extensions={"timeout": {"connect": 3.0}},
    )


def test_qualify_model_namespaces_bare_names():
    assert qualify_model("jev-1.13") == "typesafe/jev-1.13"
    assert qualify_model("typesafe/jev-1.13") == "typesafe/jev-1.13"
    assert qualify_model("other/model") == "other/model"


def test_adapt_request_targets_the_decisions_endpoint():
    adapted = adapt_request(system_one_request())
    assert str(adapted.url) == OPENROUTER_DEFAULT_BASE_URL + OPENROUTER_DECISIONS_PATH
    assert adapted.method == "POST"
    assert adapted.headers["Authorization"] == "Bearer test-key"
    assert adapted.extensions["timeout"] == {"connect": 3.0}
    body = json.loads(adapted.content)
    assert body["model"] == "typesafe/jev-1.13"
    assert body["state"] == {"x": 1}
    assert body["questions"] == {"result": {"type": "noul"}}
    assert int(adapted.headers["Content-Length"]) == len(adapted.content)


def test_adapt_request_leaves_other_endpoints_untouched():
    request = httpx2.Request("GET", "https://openrouter.ai/v1/models")
    assert adapt_request(request) is request


def test_select_provider_prefers_typesafe_when_configured(monkeypatch):
    monkeypatch.setenv(TYPESAFE_KEY_ENV, "typesafe-key")
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "openrouter-key")
    assert select_provider("auto") == "typesafe"
    assert select_provider("openrouter") == "openrouter"


def test_select_provider_falls_back_to_openrouter(monkeypatch):
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "openrouter-key")
    assert select_provider("auto") == "openrouter"


def test_select_provider_rejects_unknown_names():
    with pytest.raises(ValueError):
        select_provider("anthropic")


def test_openrouter_backend_resolution(monkeypatch):
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, " openrouter-key ")
    backend = resolve_backend("openrouter")
    assert backend.name == "openrouter"
    assert backend.api_key == "openrouter-key"
    assert backend.model == OPENROUTER_DEFAULT_MODEL
    assert backend.base_url == OPENROUTER_DEFAULT_BASE_URL
    assert isinstance(backend.transport, DecisionsTransport)
    assert set(backend.client_kwargs()) == {"api_key", "model", "base_url", "transport"}
    assert resolve_backend("openrouter", "jev-1.13").model == "typesafe/jev-1.13"
    monkeypatch.setenv("OPENROUTER_DEFAULT_MODEL", "jev-1.12")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://proxy.test")
    backend = resolve_backend("openrouter")
    assert (backend.model, backend.base_url) == ("typesafe/jev-1.12", "https://proxy.test")


def test_openrouter_never_borrows_the_typesafe_key(monkeypatch):
    monkeypatch.setenv(TYPESAFE_KEY_ENV, "typesafe-key")
    with pytest.raises(ValueError, match=OPENROUTER_API_KEY_ENV):
        resolve_backend("openrouter")


def test_default_backend_is_unchanged_by_an_openrouter_key(monkeypatch):
    monkeypatch.setenv(TYPESAFE_KEY_ENV, "typesafe-key")
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "openrouter-key")
    backend = resolve_backend("auto")
    assert backend.name == "typesafe"
    assert backend.client_kwargs() == {
        "api_key": "typesafe-key",
        "model": "jev-latest",
        "base_url": "https://api.typesafe.ai",
    }


async def test_owned_openrouter_client_configuration_and_request(monkeypatch):
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "openrouter-key")
    transport, requests = mock_transport()
    created = []

    def factory(**kwargs):
        created.append(kwargs)
        return AsyncTypeSafeClient(
            api_key=kwargs["api_key"],
            model=kwargs["model"],
            base_url=kwargs["base_url"],
            transport=transport,
            retry=RetryPolicy(max_retries=0),
        )

    monkeypatch.setattr(_engine, "_create_client", factory)
    result = await pd.DataFrame({"x": [1, 2]}).jev.noul(
        "Yes?", state="x", provider="openrouter", max_concurrency=1
    )
    assert result.tolist() == [0.75, 0.75]
    assert len(created) == 1
    assert created[0]["api_key"] == "openrouter-key"
    assert created[0]["model"] == OPENROUTER_DEFAULT_MODEL
    assert created[0]["base_url"] == OPENROUTER_DEFAULT_BASE_URL
    assert isinstance(created[0]["transport"], DecisionsTransport)
    assert len(requests) == 2
    for request in requests:
        assert str(request.url) == "https://openrouter.ai/api/alpha/decisions"
        assert request.headers["Authorization"] == "Bearer openrouter-key"
        assert request.headers["Content-Type"] == "application/json"
        body = json.loads(request.content)
        assert body["model"] == "typesafe/jev-1.13"
        assert body["questions"] == {"result": {"type": "noul", "instructions": "Yes?"}}
    assert [json.loads(r.content)["state"] for r in requests] == [{"x": 1}, {"x": 2}]


async def test_openrouter_client_factory_evaluates_choices(monkeypatch):
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "openrouter-key")
    transport, requests = mock_transport()
    async with openrouter_client(
        model="jev-1.13", transport=transport, retry=RetryPolicy(max_retries=0)
    ) as client:
        frame = await pd.DataFrame({"x": ["late invoice"]}).jev.choice(
            "What is it about?", choices=["billing", "bug"], state="x", client=client
        )
    assert frame["label"].tolist() == ["billing"]
    assert frame["p__billing"].tolist() == [1.0]
    assert json.loads(requests[0].content)["model"] == "typesafe/jev-1.13"
    assert str(requests[0].url).endswith(OPENROUTER_DECISIONS_PATH)


async def test_supplied_openrouter_client_qualifies_a_per_call_model(monkeypatch):
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "openrouter-key")
    transport, requests = mock_transport()
    async with openrouter_client(transport=transport, retry=RetryPolicy(max_retries=0)) as client:
        await pd.DataFrame({"x": [1]}).jev.noul("Yes?", state="x", client=client, model="jev-1.12")
    assert json.loads(requests[0].content)["model"] == "typesafe/jev-1.12"


async def test_openrouter_errors_map_to_sdk_exceptions(monkeypatch):
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "openrouter-key")
    transport, requests = mock_transport(lambda *_: httpx2.Response(401, json={"error": "nope"}))
    async with openrouter_client(transport=transport, retry=RetryPolicy(max_retries=0)) as client:
        with pytest.raises(EvaluationError) as caught:
            await pd.DataFrame({"x": [1]}).jev.noul("Yes?", state="x", client=client)
        assert isinstance(caught.value.cause, TypeSafeAuthenticationError)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            coerced = await pd.DataFrame({"x": [1]}).jev.noul(
                "Yes?", state="x", client=client, errors="coerce"
            )
    assert coerced.isna().tolist() == [True]
    assert isinstance(caught_warnings[0].message, EvaluationWarning)
    assert caught_warnings[0].message.failures[0].position == 0
    assert len(requests) == 2


async def test_provider_is_rejected_with_a_supplied_client(monkeypatch):
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "openrouter-key")
    transport, _ = mock_transport()
    async with openrouter_client(transport=transport) as client:
        with pytest.raises(ValueError, match="provider applies only"):
            await pd.DataFrame({"x": [1]}).jev.noul(
                "Yes?", state="x", client=client, provider="openrouter"
            )


async def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="provider must be"):
        await pd.DataFrame({"x": [1]}).jev.noul("Yes?", state="x", provider="typesafe-ai")


async def test_openrouter_provider_without_a_key_sends_nothing(monkeypatch):
    monkeypatch.setenv(TYPESAFE_KEY_ENV, "typesafe-key")
    monkeypatch.setattr(
        _engine, "_create_client", lambda **_: pytest.fail("no client may be created")
    )
    with pytest.raises(ValueError, match=OPENROUTER_API_KEY_ENV):
        await pd.DataFrame({"x": [1]}).jev.noul("Yes?", state="x", provider="openrouter")
