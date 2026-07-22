# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Common utilities for Gemma 4 export scripts."""

from __future__ import annotations

import logging
import os
import platform
import sys
import tempfile

from iopath.common.file_io import PathManager


USE_OSS: bool = platform.system() == "Darwin" or sys.platform == "darwin"

logger: logging.Logger = logging.getLogger(__name__)


def setup_path_manager(
    max_parallel: int = 4,
    num_retries: int = 2,
    ttl: int | None = None,
    has_user_data: bool = False,
) -> PathManager:
    """Initialize and configure PathManager with ManifoldPathHandler.

    Args:
        max_parallel: Maximum number of parallel operations.
        num_retries: Number of retry attempts.
        ttl: Time to live for cached objects.
        has_user_data: Whether the handler deals with user data.

    Returns:
        Configured PathManager instance. In OSS mode, returns a bare
        PathManager without the Manifold handler.
    """
    pathmgr = PathManager()
    if not USE_OSS:
        from iopath.fb.manifold import ManifoldPathHandler

        pathmgr.register_handler(
            ManifoldPathHandler(
                max_parallel=max_parallel,
                num_retries=num_retries,
                ttl=ttl,
                has_user_data=has_user_data,
            )
        )
    return pathmgr


def resolve_local_path(pathmgr: PathManager, path: str) -> str:
    """Resolve any iopath-supported path to a local filesystem path.

    iopath's `get_local_path()` only supports single files. For
    `manifold://` directory paths, list the directory and download every
    file via `pathmgr`, then symlink them into a single local directory
    that mirrors the original layout. Local paths pass through unchanged.

    Args:
        pathmgr: Configured PathManager (see `setup_path_manager`).
        path: Local path or `manifold://bucket/path` (file or directory).

    Returns:
        Local filesystem path containing the requested contents.
    """
    if not path.startswith("manifold://"):
        return path

    if not pathmgr.isdir(path):
        return pathmgr.get_local_path(path)

    files = pathmgr.ls(path)
    cache_dir = tempfile.mkdtemp(prefix="iopath_dir_")
    logger.info(f"Resolving {len(files)} files from {path} to {cache_dir}")
    for fname in files:
        local_file = pathmgr.get_local_path(f"{path}/{fname}")
        os.symlink(local_file, os.path.join(cache_dir, fname))
    return cache_dir
