# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import io
import logging
from contextlib import asynccontextmanager
from functools import partial

import httpx
import pytest
from executorch.examples.llm_server.python import hud
from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.hud import terminal
from executorch.examples.llm_server.python.hud.metrics import Metrics
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import SessionRuntime
from executorch.examples.llm_server.python.tests.conftest import FakeRunner
from executorch.examples.llm_server.python.tests.test_openrouter_recovery import (
    CapacityWorker,
)


@pytest.mark.parametrize("stream,late", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("recover", [False, True])
def test_hud_records_request_outcome_after_admission_recovery(stream, late, recover):
    async def scenario():
        worker = CapacityWorker(False)
        runtime = SessionRuntime(worker)
        serving = ServingChat(
            runtime,
            ChatTemplate(allow_fallback=True),
            "test",
            evict_idle_sessions=recover,
            streaming_preflight_timeout=0.001,
        )
        app = build_app(serving, "test")
        metrics = app.state.telemetry = Metrics()

        async def release_after_headers(scope, receive, send):
            async def observed_send(message):
                if message["type"] == "http.response.start":
                    worker.admission.set()
                await send(message)

            await app(scope, receive, observed_send)

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=release_after_headers),
                base_url="http://test",  # @lint-ignore: in-process ASGI test URL
            ) as client:
                for session in ["a", "b", "c"]:
                    if session == "c" and late:
                        worker.admission.clear()
                    response = await client.post(
                        "/v1/chat/completions",
                        json={
                            "messages": [{"role": "user", "content": "hi"}],
                            "session_id": session,
                            "stream": stream,
                        },
                    )
                    if session != "c" or recover:
                        assert response.status_code == 200
                        if stream:
                            assert '"content":"ok"' in response.text
                        else:
                            assert (
                                response.json()["choices"][0]["message"]["content"]
                                == "ok"
                            )
                    else:
                        assert "capacity_exhausted" in response.text
            assert metrics.completed == (3 if recover else 2)
            assert metrics.errors == (0 if recover else 1)
            assert metrics.totals.reported == (3 if recover else 2)
            assert metrics.turns[3].state == ("done" if recover else "error")
            assert metrics.turns[3].error == (None if recover else "capacity_exhausted")
            assert worker.attempts == (
                ["a", "b", "c", "c"] if recover else ["a", "b", "c"]
            )
        finally:
            worker.admission.set()
            await runtime.aclose_worker()

    asyncio.run(scenario())


class _Terminal(io.StringIO):
    fail_entry = False

    def isatty(self):
        return True

    def write(self, text):
        if self.fail_entry and "\x1b[?1049h" in text:
            super().write(text)
            raise OSError("terminal unavailable")
        return super().write(text)


@pytest.fixture
def terminal_output(monkeypatch):
    stream = _Terminal()
    monkeypatch.setenv("TERM", "xterm")
    for name in ["", "uvicorn", "uvicorn.error", "uvicorn.access"]:
        logger = logging.getLogger(name)
        monkeypatch.setattr(logger, "handlers", [logging.StreamHandler(stream)])
        monkeypatch.setattr(logger, "level", logging.INFO)
    return stream


@pytest.mark.parametrize("failure", ["storage", "terminal"])
def test_hud_initialization_failure_keeps_serving(
    monkeypatch, terminal_output, failure
):
    archives = []

    def archive_factory(**kwargs):
        if failure == "storage":
            raise OSError("temporary storage unavailable")
        archive = io.BytesIO()
        archives.append(archive)
        return archive

    monkeypatch.setattr(terminal.tempfile, "TemporaryFile", archive_factory)
    monkeypatch.setattr(hud, "terminal_available", lambda stream: True)
    monkeypatch.setattr(
        hud, "TerminalHUD", partial(terminal.TerminalHUD, stream=terminal_output)
    )
    terminal_output.fail_entry = failure == "terminal"
    handlers = logging.getLogger().handlers[:]
    stages = []

    async def scenario():
        @asynccontextmanager
        async def factory():
            runtime = SessionRuntime(FakeRunner(["ok"]))
            stages.append("started")
            try:
                yield ServingChat(runtime, ChatTemplate(allow_fallback=True), "test")
            finally:
                await runtime.aclose_worker()
                stages.append("closed")

        app = build_app(None, "test", serving_factory=factory, hud=True)
        async with app.router.lifespan_context(app):
            assert stages == ["started"]
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",  # @lint-ignore: in-process ASGI test URL
            ) as client:
                assert (await client.get("/health")).status_code == 200
            assert logging.getLogger().handlers == handlers

    asyncio.run(scenario())
    assert stages == ["started", "closed"]
    assert all(archive.closed for archive in archives)
    assert "HUD unavailable; continuing with plain logs" in terminal_output.getvalue()
    if failure == "terminal":
        assert "\x1b[?25h\x1b[?1049l" in terminal_output.getvalue()


class _Archive(io.BytesIO):
    failure = None

    def write(self, data):
        if self.failure == "raise":
            raise OSError("storage full")
        if self.failure == "zero":
            return 0
        return super().write(data[:3])


@pytest.mark.parametrize("failure", ["raise", "zero"])
def test_archive_failure_replays_logs_and_restores_logging(
    monkeypatch, terminal_output, failure
):
    archive = _Archive()
    monkeypatch.setattr(terminal.tempfile, "TemporaryFile", lambda **kwargs: archive)
    logger = logging.getLogger()
    handlers = logger.handlers[:]

    async def scenario():
        async with terminal.TerminalHUD(
            Metrics(), "test", stream=terminal_output
        ) as dashboard:
            logger.warning("previous diagnostic: 模型")
            archive.failure = failure
            logger.error("diagnostic during failure")
            assert logger.handlers == handlers
            logger.error("diagnostic after failure")
            dashboard.handle(
                logging.makeLogRecord(
                    {"levelno": logging.ERROR, "msg": "queued diagnostic"}
                )
            )
            restored = terminal_output.getvalue()
            dashboard.draw()
            await asyncio.wait_for(dashboard._task, 1)
            assert terminal_output.getvalue() == restored
        assert logger.handlers == handlers

    asyncio.run(scenario())
    assert archive.closed
    output = terminal_output.getvalue()
    for message in [
        "previous diagnostic: 模型",
        "diagnostic during failure",
        "diagnostic after failure",
        "queued diagnostic",
    ]:
        assert output.count(message) == 1
    assert output.index("\x1b[?25h\x1b[?1049l") < output.index("previous diagnostic")
