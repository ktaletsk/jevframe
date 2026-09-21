"""Backend selection: TypeSafe's own API, or the same Jev model served by OpenRouter.

Both backends speak the identical System One wire shape, so only the URL path, the
credential, and the model namespace differ. OpenRouter is reached by giving the
official SDK client an adapting transport, which keeps retries, timeouts, error
mapping, and response validation exactly as the TypeSafe backend has them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import httpx2
from typesafe_sdk import AsyncTypeSafeClient, constants

Provider: TypeAlias = Literal["auto", "typesafe", "openrouter"]
PROVIDERS = ("auto", "typesafe", "openrouter")

# The SDK builds `base_url + SYSTEM_ONE_PATH`; the constant is internal, so it is
# repeated here and only ever compared against, never imported from the SDK.
TYPESAFE_SYSTEM_ONE_PATH = "/v1/systemone"

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
"""Environment variable holding the OpenRouter API key."""

OPENROUTER_BASE_URL_ENV = "OPENROUTER_BASE_URL"
"""Environment variable overriding the OpenRouter API root."""

OPENROUTER_DEFAULT_MODEL_ENV = "OPENROUTER_DEFAULT_MODEL"
"""Environment variable overriding the OpenRouter model."""

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai"
"""Default OpenRouter API root."""

OPENROUTER_DECISIONS_PATH = "/api/alpha/decisions"
"""OpenRouter's decisions endpoint; Jev is not served by chat completions."""

OPENROUTER_DEFAULT_MODEL = "typesafe/jev-1.13"
"""Default OpenRouter model; OpenRouter has no moving `jev-latest` alias."""

OPENROUTER_MODEL_PREFIX = "typesafe/"

_MISSING_KEY = (
    f"No OpenRouter API key was found. Set the {OPENROUTER_API_KEY_ENV} environment "
    "variable, or pass a client configured for OpenRouter."
)


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def qualify_model(model: str) -> str:
    """Namespace a bare TypeSafe model name for OpenRouter's catalog."""
    return model if "/" in model else OPENROUTER_MODEL_PREFIX + model


def _qualify_body_model(content: bytes | None) -> bytes | None:
    """Qualify a bare model name in a prepared System One body, if there is one."""
    if not content:
        return content
    try:
        body = json.loads(content)
    except ValueError:
        return content
    if not isinstance(body, dict) or not isinstance(body.get("model"), str):
        return content
    qualified = qualify_model(body["model"])
    if qualified == body["model"]:
        return content
    body["model"] = qualified
    return json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def adapt_request(request: httpx2.Request) -> httpx2.Request:
    """Retarget a System One request at OpenRouter's decisions endpoint.

    Any other path is returned untouched, so a caller reusing the client for a
    different SDK resource still sees the SDK's own behavior and errors.
    """
    if request.url.path != TYPESAFE_SYSTEM_ONE_PATH:
        return request
    content = _qualify_body_model(request.content)
    # Content-Length is recomputed from the (possibly rewritten) body.
    headers = [
        (name, value) for name, value in request.headers.raw if name.lower() != b"content-length"
    ]
    return httpx2.Request(
        request.method,
        request.url.copy_with(path=OPENROUTER_DECISIONS_PATH),
        headers=headers,
        content=content,
        extensions=request.extensions,
    )


class DecisionsTransport(httpx2.AsyncBaseTransport):
    """An httpx transport that maps System One requests onto OpenRouter's endpoint."""

    def __init__(self, inner: httpx2.AsyncBaseTransport | None = None) -> None:
        self._inner = httpx2.AsyncHTTPTransport() if inner is None else inner

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        return await self._inner.handle_async_request(adapt_request(request))

    async def aclose(self) -> None:
        await self._inner.aclose()


@dataclass(frozen=True)
class Backend:
    """Resolved configuration for one owned client."""

    name: Literal["typesafe", "openrouter"]
    api_key: str | None
    model: str
    base_url: str
    transport: httpx2.AsyncBaseTransport | None = None

    def client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
        }
        if self.transport is not None:
            kwargs["transport"] = self.transport
        return kwargs


def select_provider(provider: Provider) -> Literal["typesafe", "openrouter"]:
    """Resolve ``"auto"``; OpenRouter is used only when TypeSafe is not configured."""
    if provider not in PROVIDERS:
        raise ValueError("provider must be 'auto', 'typesafe', or 'openrouter'")
    if provider != "auto":
        return provider
    if not _env(constants.API_KEY_ENV) and _env(OPENROUTER_API_KEY_ENV):
        return "openrouter"
    return "typesafe"


def resolve_backend(provider: Provider, model: str | None = None) -> Backend:
    """Resolve credentials, model, and base URL for the selected backend."""
    name = select_provider(provider)
    if name == "typesafe":
        resolved = model or _env(constants.DEFAULT_MODEL_ENV) or constants.DEFAULT_MODEL
        base_url = _env(constants.BASE_URL_ENV) or constants.DEFAULT_BASE_URL
        return Backend("typesafe", _env(constants.API_KEY_ENV) or None, resolved, base_url)
    api_key = _env(OPENROUTER_API_KEY_ENV)
    if not api_key:
        # Never fall through to the TypeSafe key: it is not valid at OpenRouter.
        raise ValueError(_MISSING_KEY)
    resolved = model or _env(OPENROUTER_DEFAULT_MODEL_ENV) or OPENROUTER_DEFAULT_MODEL
    base_url = _env(OPENROUTER_BASE_URL_ENV) or OPENROUTER_DEFAULT_BASE_URL
    return Backend("openrouter", api_key, qualify_model(resolved), base_url, DecisionsTransport())


def openrouter_client(
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    transport: httpx2.AsyncBaseTransport | None = None,
    **kwargs: Any,
) -> AsyncTypeSafeClient:
    """Build an official SDK client that talks to Jev through OpenRouter.

    Pass the result as ``client=`` to keep ownership of its lifetime, and to set
    SDK options such as ``timeout`` and ``retry``. Arguments default to the
    ``OPENROUTER_*`` environment variables; bare model names are namespaced
    (``jev-1.13`` becomes ``typesafe/jev-1.13``).

    Examples:
        ```python
        from jevframe import openrouter_client

        async with openrouter_client(timeout=30) as client:
            result = await df.jev.noul("Is this urgent?", state="review", client=client)
        ```
    """
    resolved_key = api_key or _env(OPENROUTER_API_KEY_ENV)
    if not resolved_key:
        raise ValueError(_MISSING_KEY)
    resolved_model = model or _env(OPENROUTER_DEFAULT_MODEL_ENV) or OPENROUTER_DEFAULT_MODEL
    return AsyncTypeSafeClient(
        api_key=resolved_key,
        model=qualify_model(resolved_model),
        base_url=base_url or _env(OPENROUTER_BASE_URL_ENV) or OPENROUTER_DEFAULT_BASE_URL,
        transport=DecisionsTransport(transport),
        **kwargs,
    )
