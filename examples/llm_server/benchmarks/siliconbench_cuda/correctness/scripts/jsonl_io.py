"""Shared I/O helpers for correctness/ scripts."""

import json
from pathlib import Path
from typing import Iterator


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield one parsed JSON object per non-empty line."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
