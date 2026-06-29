"""Structured, persisted audit log (JSON Lines).

Every attribution decision and every appeal is appended as one JSON object per
line to an append-only ``.jsonl`` file. This is the single source of truth for
the audit trail: it survives restarts, is trivially greppable, and each line is
independently parseable.

The file path is resolved per call from the ``PROVENANCEGUARD_AUDIT_LOG``
environment variable (falling back to ``audit_log.jsonl`` at the project root),
so tests can point it at a temp file without re-importing anything.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "audit_log.jsonl"


def _path() -> Path:
    return Path(os.environ.get("PROVENANCEGUARD_AUDIT_LOG", str(_DEFAULT_PATH)))


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a trailing ``Z``."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def append(entry: Dict[str, Any]) -> None:
    """Append one structured entry to the audit log."""
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def read(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return audit entries, oldest first. ``limit`` keeps the most recent N."""
    path = _path()
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        entries = [json.loads(line) for line in fh if line.strip()]
    return entries[-limit:] if limit is not None else entries


def find_submission(content_id: str) -> Optional[Dict[str, Any]]:
    """Return the most recent ``submission`` entry for ``content_id``."""
    for entry in reversed(read()):
        if entry.get("content_id") == content_id and entry.get("event") == "submission":
            return entry
    return None


def clear() -> None:
    """Delete the audit log file (used by tests)."""
    path = _path()
    if path.exists():
        path.unlink()
