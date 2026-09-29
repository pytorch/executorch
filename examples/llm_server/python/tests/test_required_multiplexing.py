# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Batching-required startup must never silently select a serial worker."""

import json
import subprocess
import sys

import pytest

from executorch.examples.llm_server.python.worker_client import (
    spawn_worker,
    WorkerError,
)


@pytest.fixture
def readiness_worker():
    children = []

    def launch(capabilities, **options):
        readiness = {"ready": True, "max_named_sessions": 8, **capabilities}
        program = (
            "import sys; "
            f"print({json.dumps(readiness)!r}, flush=True); "
            "sys.stdin.read()"
        )

        def record_process(*args, **kwargs):
            proc = subprocess.Popen(*args, **kwargs)
            children.append((proc, proc.stdin, proc.stdout))
            return proc

        return spawn_worker(
            [sys.executable, "-u", "-c", program], popen=record_process, **options
        )

    yield launch, children
    for proc, stdin, stdout in children:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
        stdin.close()
        stdout.close()


@pytest.mark.parametrize(
    "capabilities", [{}, {"multiplexed": False}, {"multiplexed": 1}]
)
def test_required_multiplexing_rejects_and_reaps_legacy_worker(
    readiness_worker, capabilities
):
    launch, children = readiness_worker
    with pytest.raises(WorkerError, match="multiplexing") as caught:
        launch(capabilities, require_multiplexing=True)
    assert caught.value.code == "unsupported_multiplexing"
    assert len(children) == 1
    proc, stdin, stdout = children[0]
    assert proc.poll() is not None
    assert stdin.closed and stdout.closed


def test_required_multiplexing_accepts_explicit_capability(readiness_worker):
    launch, _ = readiness_worker
    client = launch({"multiplexed": True}, require_multiplexing=True)
    try:
        assert client.supports_multiplexing
    finally:
        client.close()


def test_default_negotiation_preserves_legacy_fallback(readiness_worker):
    launch, _ = readiness_worker
    client = launch({})
    try:
        assert not client.supports_multiplexing
    finally:
        client.close()
