"""Content-addressed response cache in SQLite.

Re-running a suite after editing one case should only pay for that case. The key is a hash of
everything that can change the output: provider identity (model + params), system prompt,
messages and the repeat index. Changing any of them is a cache miss by construction.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evalkit.providers.base import Completion

SCHEMA_VERSION = 1


def cache_key(
    identity: dict[str, Any], system: str | None, messages: list[dict[str, str]], repeat: int
) -> str:
    payload = json.dumps(
        {
            "v": SCHEMA_VERSION,
            "provider": identity,
            "system": system,
            "messages": messages,
            "repeat": repeat,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, directory: str | Path) -> None:
        self.path = Path(directory) / "responses.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY,"
            " completion TEXT NOT NULL,"
            " latency_ms REAL NOT NULL,"
            " created_at REAL NOT NULL)"
        )
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> tuple[Completion, float] | None:
        row = self._db.execute(
            "SELECT completion, latency_ms FROM responses WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return Completion(**json.loads(row[0])), float(row[1])

    def put(self, key: str, completion: Completion, latency_ms: float) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?)",
            (key, json.dumps(asdict(completion), ensure_ascii=False), latency_ms, time.time()),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()
