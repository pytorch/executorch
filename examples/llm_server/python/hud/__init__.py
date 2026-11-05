# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Terminal HUD shared by model-specific serving applications."""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from .metrics import Metrics
from .terminal import terminal_available, TerminalHUD


def install(
    app, model_id, *, model_path=None, context=None, capacity=None, listen_address=None
):
    app.state.telemetry = None
    if not terminal_available(sys.stderr):
        return
    metrics = Metrics()
    app.state.telemetry = metrics
    model = f"{model_id} | {Path(model_path).name}" if model_path else model_id
    if listen_address:
        model = f"{listen_address} | {model}"
    original = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with original(app) as state:
            metrics.started = time.monotonic()
            async with TerminalHUD(metrics, model, context, capacity):
                yield state

    app.router.lifespan_context = lifespan
