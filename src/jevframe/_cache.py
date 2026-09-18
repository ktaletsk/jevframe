"""Explicit, bounded process-local caching; no credentials or row text as keys."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Any
from weakref import WeakKeyDictionary

_client_tokens: WeakKeyDictionary[Any, object] = WeakKeyDictionary()
_client_lock = RLock()


def client_token(client: Any) -> object:
    # Keep an opaque token in keys: a collected client's id must never be reused.
    with _client_lock:
        if client not in _client_tokens:
            _client_tokens[client] = object()
        return _client_tokens[client]


class MemoryCache:
    """LRU cache of successful complete row evaluations.

    Pass the same instance to subsequent calls to reuse results. Model aliases
    can move; call ``clear()`` when fresh inference is required. Cache contents
    are process-local and never written to disk.
    """

    def __init__(self, max_entries: int = 10_000):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._entries: OrderedDict[tuple[Any, ...], tuple[Any, ...]] = OrderedDict()
        self._lock = RLock()
        self._generation = 0

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        """Discard entries, including results of requests already in progress."""
        with self._lock:
            self._entries.clear()
            self._generation += 1

    def _get(self, key: tuple[Any, ...]) -> tuple[Any, ...] | None:
        with self._lock:
            result = self._entries.get(key)
            if result is not None:
                self._entries.move_to_end(key)
            return result

    def _put(self, key: tuple[Any, ...], value: tuple[Any, ...], generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
