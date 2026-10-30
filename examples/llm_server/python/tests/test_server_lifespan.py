# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Serving factories own native resources on the ASGI lifespan loop."""

import asyncio
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest

from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.multiplexed_worker_client import (
    spawn_multiplexed_worker,
)
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import SessionRuntime


_WORKER = r"""
import json, sys
print(json.dumps(dict(ready=True, multiplexed=True, max_inflight_requests=4)), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    request_id = request['request_id']
    if request['op'] == 'generate':
        print(json.dumps(dict(request_id=request_id, token='hello')), flush=True)
        print(json.dumps(dict(request_id=request_id, done=True, finish_reason='stop', prompt_tokens=3, completion_tokens=1)), flush=True)
    else:
        ack = {'reset': 'reset', 'close': 'closed', 'open': 'opened'}[request['op']]
        print(json.dumps(dict(request_id=request_id, **{ack: True})), flush=True)
"""


def test_factory_validation():
    serving = SimpleNamespace(healthy=True)
    with pytest.raises(ValueError, match="exactly one"):
        build_app(None, "test-model")
    with pytest.raises(ValueError, match="exactly one"):
        build_app(serving, "test-model", serving_factory=lambda: None)
    with pytest.raises(TypeError, match="async context manager factory"):
        build_app(None, "test-model", serving_factory=serving)


def test_factory_native_startup_requests_and_shutdown_share_loop():
    loops = []
    resources = {}

    class ObservedServing(ServingChat):
        async def create(self, req):
            loops.append(asyncio.get_running_loop())
            return await super().create(req)

        async def reset_session(self, session_id):
            loops.append(asyncio.get_running_loop())
            await super().reset_session(session_id)

        async def close_session(self, session_id):
            loops.append(asyncio.get_running_loop())
            await super().close_session(session_id)

    @asynccontextmanager
    async def factory():
        loops.append(asyncio.get_running_loop())
        worker = await spawn_multiplexed_worker([sys.executable, "-u", "-c", _WORKER])
        resources["worker"] = worker
        runtime = None
        try:
            runtime = SessionRuntime(worker)
            resources["runtime"] = runtime
            yield ObservedServing(
                runtime,
                ChatTemplate(hf_tokenizer_path=None, allow_fallback=True),
                "test-model",
            )
        finally:
            loops.append(asyncio.get_running_loop())
            if runtime is None:
                await worker.close()
            else:
                await runtime.aclose_worker()

    app = build_app(None, "test-model", serving_factory=factory)
    assert not resources

    async def scenario():
        initial_tasks = asyncio.all_tasks()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",  # @lint-ignore
        ) as client:
            with pytest.raises(RuntimeError, match="application lifespan"):
                await client.get("/health")
            async with app.router.lifespan_context(app):
                assert (await client.get("/health")).json() == {"status": "ok"}
                assert (await client.get("/v1/models")).json()["data"][0][
                    "id"
                ] == "test-model"
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "session_id": "s",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert response.status_code == 200
                assert response.json()["choices"][0]["message"]["content"] == "hello"
                assert response.json()["usage"]["completion_tokens"] == 1
                assert (await client.post("/v1/sessions/s/reset")).json() == {
                    "reset": True,
                    "session_id": "s",
                }
                assert (await client.delete("/v1/sessions/s")).json() == {
                    "closed": True,
                    "session_id": "s",
                }
            with pytest.raises(RuntimeError, match="after shutdown"):
                await client.get("/health")
        assert app.state.serving is None
        assert len(loops) == 5
        assert all(loop is asyncio.get_running_loop() for loop in loops)
        worker = resources["worker"]
        assert worker._loop is asyncio.get_running_loop()
        assert worker.closed and worker._proc.returncode is not None
        assert worker._reader.done() and worker._writer.done()
        assert not resources["runtime"]._settlements
        assert not (asyncio.all_tasks() - initial_tasks)

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["runtime", "adapter"])
def test_factory_construction_failure_after_spawn_reaps_worker(stage):
    resources = {}

    @asynccontextmanager
    async def factory():
        worker = await spawn_multiplexed_worker([sys.executable, "-u", "-c", _WORKER])
        resources["worker"] = worker
        runtime = None
        try:
            runtime = SessionRuntime(
                worker, cancel_grace_seconds=-1 if stage == "runtime" else 0.01
            )
            raise ValueError("adapter construction failed")
            yield  # pragma: no cover - startup fails before yielding an adapter
        finally:
            if runtime is None:
                await worker.close()
            else:
                await runtime.aclose_worker()

    app = build_app(None, "test-model", serving_factory=factory)

    async def scenario():
        initial_tasks = asyncio.all_tasks()
        message = (
            "cancellation timeouts"
            if stage == "runtime"
            else "adapter construction failed"
        )
        with pytest.raises(ValueError, match=message):
            async with app.router.lifespan_context(app):
                pytest.fail("failed factory must not start the application")
        assert app.state.serving is None
        worker = resources["worker"]
        assert worker.closed and worker._proc.returncode is not None
        assert worker._reader.done() and worker._writer.done()
        assert not (asyncio.all_tasks() - initial_tasks)

    asyncio.run(scenario())


def test_factory_clears_adapter_when_shutdown_fails():
    @asynccontextmanager
    async def factory():
        try:
            yield SimpleNamespace(healthy=True)
        finally:
            raise RuntimeError("shutdown failed")

    app = build_app(None, "test-model", serving_factory=factory)

    async def scenario():
        with pytest.raises(RuntimeError, match="shutdown failed"):
            async with app.router.lifespan_context(app):
                assert app.state.serving.healthy
        assert app.state.serving is None

    asyncio.run(scenario())


def test_factory_rejects_missing_adapter():
    exited = []

    @asynccontextmanager
    async def factory():
        try:
            yield None
        finally:
            exited.append(True)

    app = build_app(None, "test-model", serving_factory=factory)

    async def scenario():
        with pytest.raises(RuntimeError, match="yielded no serving adapter"):
            async with app.router.lifespan_context(app):
                pytest.fail("missing adapter must fail startup")
        assert app.state.serving is None
        assert exited == [True]

    asyncio.run(scenario())


def test_direct_instance_keeps_legacy_event_handlers_and_external_ownership():
    events = []

    class InjectedServing:
        healthy = True

        def close(self):
            pytest.fail("the application must not close an injected adapter")

        async def aclose(self):
            pytest.fail("the application must not close an injected adapter")

    serving = InjectedServing()
    app = build_app(serving, "test-model")

    @app.on_event("startup")
    def startup():
        events.append("startup")

    @app.on_event("shutdown")
    def shutdown():
        events.append("shutdown")

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",  # @lint-ignore
        ) as client:
            assert (await client.get("/health")).status_code == 200
            async with app.router.lifespan_context(app):
                assert events == ["startup"]
                assert (await client.get("/health")).status_code == 200
            assert events == ["startup", "shutdown"]
            assert app.state.serving is serving
            assert (await client.get("/health")).status_code == 200

    asyncio.run(scenario())
