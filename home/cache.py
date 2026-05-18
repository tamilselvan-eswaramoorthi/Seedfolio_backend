"""
Lightweight in-memory TTL cache for the home module.

Two logical caches share one dict:
  - "market"    : index data (Nifty / Sensex), shared across all users
  - "portfolio" : PortfolioPerformance object, one entry per user_id

Entries are keyed by (name, key, date) and auto-invalidate at midnight.
"""

import threading
from datetime import date
from typing import Any, Optional

# Protects _store writes only — NOT held during factory() calls
_write_lock = threading.Lock()

# { (cache_name, key, date_str) -> value }
_store: dict[tuple, Any] = {}


def _day_key(name: str, key: str) -> tuple:
    return (name, key, date.today().isoformat())


def get(name: str, key: str) -> Optional[Any]:
    return _store.get(_day_key(name, key))


def set(name: str, key: str, value: Any) -> None:  # noqa: A001
    today = date.today().isoformat()
    with _write_lock:
        # Evict stale entries for this (name, key) from previous days
        stale = [k for k in _store if k[0] == name and k[1] == key and k[2] != today]
        for k in stale:
            del _store[k]
        _store[(name, key, today)] = value


def get_or_create(name, key, factory):
    cached = get(name, key)
    if cached is not None:
        return cached
    value = factory()
    set(name, key, value)
    return value