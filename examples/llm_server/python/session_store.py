# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit, model-bound storage for named serving sessions."""

import asyncio
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .checkpoint import (
    checkpoint_identity,
    CPUCheckpoint,
    load_checkpoint,
    save_checkpoint,
)


@dataclass(frozen=True)
class SessionStore:
    """Configure once at worker startup with its artifacts and execution settings.

    ``create`` hashes the supplied artifacts off the event loop. Include external
    weights, tokenizer, adapters, backend and activation-affecting configuration.
    Saves replace this session's previous checkpoint atomically. Closing a live
    session does not delete its saved state. Conversation messages belong to the
    application and should be persisted independently.
    """

    directory: Path
    identity: str = field(repr=False)

    @classmethod
    async def create(
        cls,
        directory: os.PathLike,
        *,
        model_files: Sequence[os.PathLike],
        configuration: Mapping[str, Any],
    ) -> "SessionStore":
        files = tuple(Path(path).expanduser().resolve() for path in model_files)
        if not files:
            raise ValueError("session storage requires model artifacts")
        identity = await asyncio.to_thread(
            checkpoint_identity, files, dict(configuration)
        )
        path = Path(directory).expanduser().resolve()
        await asyncio.to_thread(path.mkdir, mode=0o700, parents=True, exist_ok=True)
        return cls(path, identity)

    def _path(self, session_id: str) -> Path:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        name = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self.directory / f"{name}.etcp"

    def save(self, session_id: str, checkpoint: CPUCheckpoint) -> None:
        """Blocking file operation; async clients call this off their event loop."""
        checkpoint.validate(self.identity)
        save_checkpoint(checkpoint, self._path(session_id))

    def load(self, session_id: str) -> CPUCheckpoint:
        """Blocking read with integrity and model/configuration validation."""
        return load_checkpoint(self._path(session_id), expected_identity=self.identity)
