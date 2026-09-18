"""Bounded async row evaluation, independent of pandas and Polars."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import warnings
from dataclasses import dataclass
from typing import Any

from typesafe_sdk import AsyncTypeSafeClient, constants

from ._cache import MemoryCache, client_token
from ._context import RowSource, construct_context, contains_null
from ._schema import QuestionPlan, encode, flatten
from ._types import (
    Errors,
    EvaluationError,
    EvaluationWarning,
    Nulls,
    Progress,
    ProgressCallback,
    RowFailure,
    State,
)


@dataclass(frozen=True)
class Options:
    client: AsyncTypeSafeClient | None = None
    model: str | None = None
    max_concurrency: int = 16
    cache: MemoryCache | None = None
    progress: bool | ProgressCallback = False
    errors: Errors = "raise"
    nulls: Nulls = "include"

    def __post_init__(self) -> None:
        if self.client is not None and not isinstance(self.client, AsyncTypeSafeClient):
            raise TypeError("client must be an official AsyncTypeSafeClient")
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError("model must be a nonempty string")
        if (
            isinstance(self.max_concurrency, bool)
            or not isinstance(self.max_concurrency, int)
            or self.max_concurrency < 1
        ):
            raise ValueError("max_concurrency must be a positive integer")
        if self.cache is not None and not isinstance(self.cache, MemoryCache):
            raise TypeError("cache must be a MemoryCache instance or None")
        if not isinstance(self.progress, bool) and not callable(self.progress):
            raise TypeError("progress must be a bool or synchronous callback")
        if inspect.iscoroutinefunction(self.progress):
            raise TypeError("progress callback must be synchronous")
        if self.errors not in ("raise", "coerce"):
            raise ValueError("errors must be 'raise' or 'coerce'")
        if self.nulls not in ("include", "skip", "raise"):
            raise ValueError("nulls must be 'include', 'skip', or 'raise'")


def _create_client(**kwargs: Any) -> AsyncTypeSafeClient:
    return AsyncTypeSafeClient(**kwargs)


async def evaluate_rows(
    source: RowSource, state: State, plan: QuestionPlan, options: Options
) -> list[tuple[Any, ...]]:
    """One request per row, with a fixed number of workers and positional output."""
    missing = (None,) * len(plan.columns)
    results = [missing] * source.count
    rows = enumerate(source.rows)
    cache = options.cache
    failures: list[RowFailure] = []
    client = options.client
    owned = client is None

    # Resolve owned configuration once, using only documented SDK constants.
    # A supplied client captures its own defaults: isolate its cache by identity.
    model = options.model
    client_kwargs: dict[str, Any] = {}
    if owned:
        model = model or os.environ.get(constants.DEFAULT_MODEL_ENV, "").strip()
        model = model or constants.DEFAULT_MODEL
        base_url = os.environ.get(constants.BASE_URL_ENV, "").strip() or constants.DEFAULT_BASE_URL
        api_key = os.environ.get(constants.API_KEY_ENV, "").strip() or None
        client_kwargs = {"api_key": api_key, "model": model, "base_url": base_url}
        credential_fingerprint = hashlib.sha256((api_key or "").encode()).digest()
        scope: tuple[Any, ...] = ("owned", base_url.rstrip("/"), model, credential_fingerprint)
    else:
        scope = ("supplied", client_token(client), model) if cache is not None else ()

    # Futures are only created for active distinct requests, not for every row.
    # Store outcomes as values so failed owners cannot leave unobserved exceptions.
    pending: dict[
        tuple[Any, ...], asyncio.Future[tuple[tuple[Any, ...] | None, Exception | None]]
    ] = {}
    completed = failed = skipped = cache_hits = 0
    bar: Any = None

    def report(*, failure: bool = False, skip: bool = False, hit: bool = False) -> None:
        nonlocal completed, failed, skipped, cache_hits
        completed += 1
        failed += int(failure)
        skipped += int(skip)
        cache_hits += int(hit)
        if bar is not None:
            bar.update(1)
        if callable(options.progress):
            returned = options.progress(
                Progress(completed, source.count, failed, skipped, cache_hits)
            )
            if inspect.isawaitable(returned):
                if inspect.iscoroutine(returned):
                    returned.close()
                raise TypeError("progress callback must return synchronously")

    async def infer(context: Any) -> tuple[tuple[Any, ...], bool]:
        key = None
        generation = 0
        if cache is not None:
            generation = cache._generation
            digest = hashlib.sha256(encode([context, plan.serialized]).encode()).digest()
            key = (generation, scope, digest)
            cached = cache._get(key)
            if cached is not None:
                return cached, True
            if key in pending:
                value, error = await asyncio.shield(pending[key])
                if error is not None:
                    raise error
                assert value is not None
                return value, True
            pending[key] = asyncio.get_running_loop().create_future()
        try:
            assert client is not None
            response = await client.system_one(state=context, questions=plan.questions, model=model)
            value = flatten(response, plan)
            if cache is not None and key is not None:
                cache._put(key, value, generation)
                pending[key].set_result((value, None))
            return value, False
        except Exception as error:
            if key is not None:
                pending[key].set_result((None, error))
            raise
        finally:
            if key is not None:
                future = pending.pop(key)
                if not future.done():
                    future.cancel()

    async def worker() -> None:
        nonlocal client
        for position, row in rows:
            try:
                context = construct_context(row, state, source)
                has_null = contains_null(context)
                if has_null and options.nulls == "raise":
                    raise ValueError("Row context contains null values")
            except Exception as error:
                raise EvaluationError(position, error, phase="context construction") from error
            if has_null and options.nulls == "skip":
                report(skip=True)
            else:
                # Client configuration errors are never converted to missing probabilities.
                # No await occurs here, so exactly one owned client is created.
                if client is None:
                    client = _create_client(**client_kwargs)
                try:
                    results[position], hit = await infer(context)
                except Exception as error:
                    if options.errors == "raise":
                        raise EvaluationError(position, error) from error
                    failures.append(RowFailure(position, error))
                    report(failure=True)
                else:
                    report(hit=hit)
            # Cached and skipped rows otherwise never yield, starving cancellation/UI.
            await asyncio.sleep(0)

    tasks: list[asyncio.Task[None]] = []
    try:
        if options.progress is True:
            from tqdm.auto import tqdm

            bar = tqdm(total=source.count, desc="Evaluating", unit="row")
        tasks = [
            asyncio.create_task(worker()) for _ in range(min(source.count, options.max_concurrency))
        ]
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            if owned and client is not None:
                await client.aclose()
        finally:
            if bar is not None:
                bar.close()
    if failures:
        warnings.warn(EvaluationWarning(failures), stacklevel=3)
    return results
