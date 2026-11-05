# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Relay native diagnostics through logging while the terminal HUD is enabled."""

import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar

worker_stderr = ContextVar("llm_worker_stderr", default=None)
logger = logging.getLogger(__name__)
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


async def _relay(reader):
    while chunk := await reader.read(4096):
        for line in _ANSI.sub("", chunk.decode("utf-8", errors="replace")).splitlines():
            if line.strip():
                logger.warning("%s", line)


@asynccontextmanager
async def capture_worker_logs():
    read_fd, write_fd = os.pipe()
    pipe = os.fdopen(read_fd, "rb", buffering=0)
    reader = asyncio.StreamReader()
    try:
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), pipe
        )
    except BaseException:
        pipe.close()
        os.close(write_fd)
        raise
    relay = asyncio.create_task(_relay(reader), name="hud-worker-logs")
    token = worker_stderr.set(write_fd)
    try:
        yield
    finally:
        worker_stderr.reset(token)
        os.close(write_fd)
        try:
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(relay, timeout=1)
        finally:
            relay.cancel()
            transport.close()
            await asyncio.gather(relay, return_exceptions=True)
