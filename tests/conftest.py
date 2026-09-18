from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest_asyncio
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy


def response_body(payload: dict[str, Any], probability: float = 0.75) -> dict[str, Any]:
    answers = {}
    for name, question in payload["questions"].items():
        if question["type"] == "noul":
            answers[name] = {"type": "noul", "noul": probability}
        elif question["type"] == "choice":
            labels = list(question["criteria"])
            probabilities = dict.fromkeys(labels, 0.0)
            probabilities[labels[0]] = 1.0
            answers[name] = {
                "type": "choice",
                "choice": labels[0],
                "confidence": 1.0,
                "probabilities": probabilities,
            }
        else:
            levels = question["criteria"]
            p = [0.0] * len(levels)
            p[0], p[1] = 0.25, 0.75
            answers[name] = {
                "type": "score",
                "score": 0.75,
                "confidence": 0.5,
                "legend": {str(i): label for i, label in enumerate(levels)},
                "probabilities": {str(i): value for i, value in enumerate(p)},
            }
    return {"model": payload["model"], "answers": answers, "usage": {"input_tokens": 12}}


@pytest_asyncio.fixture
async def make_client():
    clients = []

    def create(handler: Callable | None = None, **kwargs):
        calls = []

        async def handle(request):
            payload = json.loads(request.content)
            calls.append(payload)
            result = handler(payload, len(calls)) if handler else response_body(payload)
            if inspect.isawaitable(result):
                result = await result
            return (
                result if isinstance(result, httpx2.Response) else httpx2.Response(200, json=result)
            )

        http = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
        retry = kwargs.pop("retry", RetryPolicy(max_retries=0))
        client = AsyncTypeSafeClient(api_key="test-key", http_client=http, retry=retry, **kwargs)
        clients.append(client)
        return client, calls, http

    yield create
    for client in clients:
        await client.aclose()
