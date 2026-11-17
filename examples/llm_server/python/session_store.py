# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit, model-bound storage for named serving sessions."""

import asyncio
import hashlib
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .checkpoint import (
    _DEFAULT_MAX_BYTES,
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
    max_bytes: int = _DEFAULT_MAX_BYTES

    def __post_init__(self):
        if type(self.max_bytes) is not int or self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        info = self.directory.stat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("session storage must be a directory")
        if os.name == "posix" and (info.st_uid != os.getuid() or info.st_mode & 0o022):
            raise ValueError(
                "session storage must be owned by the current user and not writable by others"
            )

    @classmethod
    async def create(
        cls,
        directory: os.PathLike,
        *,
        model_files: Sequence[os.PathLike],
        configuration: Mapping[str, Any],
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> "SessionStore":
        files = tuple(Path(path).expanduser().resolve() for path in model_files)
        if not files:
            raise ValueError("session storage requires model artifacts")
        identity = await asyncio.to_thread(
            checkpoint_identity, files, dict(configuration)
        )
        path = Path(directory).expanduser().resolve()
        await asyncio.to_thread(path.mkdir, mode=0o700, parents=True, exist_ok=True)
        return cls(path, identity, max_bytes)

    def _path(self, session_id: str) -> Path:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        name = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self.directory / f"{name}.etcp"

    def save(self, session_id: str, checkpoint: CPUCheckpoint) -> None:
        """Blocking file operation; async clients call this off their event loop."""
        checkpoint.validate(self.identity)
        if len(checkpoint.data) > self.max_bytes:
            raise ValueError("checkpoint payload exceeds session storage max_bytes")
        save_checkpoint(checkpoint, self._path(session_id))

    def load(self, session_id: str) -> CPUCheckpoint:
        """Blocking read with integrity and model/configuration validation."""
        return load_checkpoint(
            self._path(session_id),
            expected_identity=self.identity,
            max_bytes=self.max_bytes,
        )

    def delete(self, session_id: str) -> bool:
        """Remove saved state only; callers must first settle concurrent writes."""
        try:
            self._path(session_id).unlink()
        except FileNotFoundError:
            return False
        directory = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return True
