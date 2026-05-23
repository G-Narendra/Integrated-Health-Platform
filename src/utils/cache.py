"""
Simple in-memory and disk-based cache for LLM responses and retrieval results.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional


class TTLCache:
    """Simple TTL-based cache with optional disk persistence."""

    def __init__(self, ttl_seconds: int = 300, persist_path: Optional[str] = None):
        self._cache: dict[str, dict] = {}
        self.ttl = ttl_seconds
        self._persist_path = Path(persist_path) if persist_path else None
        self._load_from_disk()

    def get(self, key: str) -> Optional[Any]:
        data = self._cache.get(key)
        if data is None:
            return None
        if time.time() - data["ts"] > self.ttl:
            del self._cache[key]
            return None
        return data["value"]

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = {"value": value, "ts": time.time()}

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()

    def _load_from_disk(self) -> None:
        if self._persist_path and self._persist_path.exists():
            try:
                data = json.loads(self._persist_path.read_text())
                self._cache = {k: v for k, v in data.items() if time.time() - v["ts"] < self.ttl}
            except Exception:
                pass

    def _save_to_disk(self) -> None:
        if self._persist_path:
            try:
                self._persist_path.parent.mkdir(parents=True, exist_ok=True)
                self._persist_path.write_text(json.dumps(self._cache))
            except Exception:
                pass


# Global caches
llm_cache = TTLCache(ttl_seconds=300)
retrieval_cache = TTLCache(ttl_seconds=600)
