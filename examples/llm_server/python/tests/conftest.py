# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixtures for serving contract tests.

Most tests use SessionRuntime over a fake worker to exercise the HTTP boundary
without a model or GPU. Tests depending on native_worker are marked for a
separate CI pass after the native executables have been built.
"""

import asyncio
import json
import os
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.multiplexed_worker_client import (
    MultiplexedWorkerClient,
)
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import SessionRuntime
from executorch.examples.llm_server.python.tool_parsers import HermesDetector
from executorch.examples.llm_server.python.worker_client import WorkerError
from fastapi.testclient import TestClient


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "native_worker: requires a native test worker executable"
    )


def pytest_collection_modifyitems(items):
    for item in items:
        if "native_worker" in item.fixturenames:
            item.add_marker(pytest.mark.native_worker)


@pytest.fixture(scope="module")
def native_worker():
    """Require an explicitly selected, freshly built native test executable."""
    executable = os.environ.get("EXECUTORCH_TEST_MULTIPLEXED_WORKER")
    if not executable:
        pytest.skip("set EXECUTORCH_TEST_MULTIPLEXED_WORKER to the native test target")
    assert Path(executable).is_file(), executable
    return executable


@pytest.fixture
def active_peer(native_worker):
    @asynccontextmanager
    async def start():
        loop = asyncio.get_running_loop()
        parent, child = socket.socketpair()
        parent.setblocking(False)
        proc = None
        client = None
        try:
            proc = await asyncio.create_subprocess_exec(
                native_worker,
                "--validation-worker",
                str(child.fileno()),
                pass_fds=(child.fileno(),),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                limit=1024 * 1024,
            )
            child.close()
            ready = json.loads(await proc.stdout.readline())
            assert ready["ready"] and ready["multiplexed"]
            client = MultiplexedWorkerClient(
                proc,
                max_named_sessions=ready["max_named_sessions"],
                max_inflight_requests=2,
            )
            peer = client.generate("active peer", SimpleNamespace(max_new_tokens=3))
            # Native execute() is blocked on this socket until explicitly released.
            assert await loop.sock_recv(parent, 1) == b"A"
            assert not peer._state.completion.done()
            yield client, peer, parent
        finally:
            parent.close()
            child.close()
            try:
                if client is not None:
                    await client.close()
            finally:
                if proc is not None:
                    if proc.returncode is None:
                        proc.kill()
                    await proc.wait()

    return start


class _FakeStats:
    num_prompt_tokens = 5
    num_generated_tokens = 0
    finish_reason = None


class FakeRunner:
    """Canned engine: emits fixed tokens, records the config it was given.

    With max_named_sessions > 0 it also models the worker's session admission:
    open_session() enforces capacity and reports structured WorkerError codes,
    matching the real worker's contract."""

    supports_ignore_eos = True

    def __init__(
        self,
        tokens,
        fail=False,
        finish_reason=None,
        max_named_sessions=0,
        gen_ids=None,
        reuse=0,
    ):
        self._tokens = list(tokens)
        self._fail = fail
        self._finish_reason = finish_reason  # worker-reported stop reason, if any
        self._gen_ids = list(gen_ids) if gen_ids is not None else None
        self._reuse = reuse  # reused_prompt_tokens reported in done stats
        self.captured_config = None
        self.stopped = False
        self.reset_count = 0
        self.max_named_sessions = max_named_sessions
        self.open_named = set()  # currently-open named sessions
        self.opened_log = []  # every successful open, in order
        self.reset_log = []  # every reset_session call, in order

    def reset(self):
        self.reset_count += 1

    def stop(self):
        self.stopped = True

    def open_session(self, session_id):
        if session_id in self.open_named:
            return  # idempotent
        if self.max_named_sessions == 0:
            raise WorkerError("no named sessions", code="unsupported_session")
        if len(self.open_named) >= self.max_named_sessions:
            raise WorkerError("session capacity exhausted", code="capacity_exhausted")
        self.open_named.add(session_id)
        self.opened_log.append(session_id)

    def close_session(self, session_id):
        self.open_named.discard(session_id)

    def reset_session(self, session_id):
        # Clears context but keeps the slot (stays in open_named).
        self.reset_log.append(session_id)

    def generate(self, prompt, config, token_callback=None, stats_callback=None):
        self.captured_config = config
        if self._fail:
            raise RuntimeError("Generation failed")
        for tok in self._tokens:
            if token_callback:
                token_callback(tok)
        if stats_callback:
            stats = _FakeStats()
            stats.num_generated_tokens = len(self._tokens)
            stats.finish_reason = self._finish_reason
            stats.generated_token_ids = (
                list(self._gen_ids) if self._gen_ids is not None else None
            )
            stats.reused_prompt_tokens = self._reuse
            stats_callback(stats)


class _FakeTokenizer:
    """Minimal stand-in for an HF tokenizer (counting + templating)."""

    all_special_tokens: list = []

    def __init__(self, prompt_tokens):
        self._n = prompt_tokens

    def encode(self, text, add_special_tokens=False):
        return [0] * self._n

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, **kwargs
    ):
        return "PROMPT"


@pytest.fixture
def make_client():
    def _make(
        tokens=("Hello", ", ", "world"),
        max_context=None,
        prompt_tokens=None,
        fail=False,
        finish_reason=None,
        max_named_sessions=0,
        gen_ids=None,
        reuse=0,
        reasoning_extractor=None,
        supports_ignore_eos=True,
        default_stop_strings=("<|im_end|>",),
        streaming_parser_factory=None,
        content_filter=None,
        special_tokens=None,
        api_dialect="openai",
        evict_idle_sessions=False,
    ):
        fake = FakeRunner(
            tokens,
            fail=fail,
            finish_reason=finish_reason,
            max_named_sessions=max_named_sessions,
            gen_ids=gen_ids,
            reuse=reuse,
        )
        fake.supports_ignore_eos = supports_ignore_eos
        fake.default_stop_strings = (
            None if default_stop_strings is None else list(default_stop_strings)
        )
        runtime = SessionRuntime(fake)  # one fake worker
        template = ChatTemplate(hf_tokenizer_path=None, allow_fallback=True)
        if prompt_tokens is not None or special_tokens is not None:
            template._hf = _FakeTokenizer(
                prompt_tokens if prompt_tokens is not None else 5
            )
            if special_tokens is not None:
                template._hf.all_special_tokens = special_tokens
        serving = ServingChat(
            runtime,
            template,
            "test-model",
            max_context=max_context,
            tool_detector_cls=HermesDetector,
            reasoning_extractor=reasoning_extractor,
            streaming_parser_factory=streaming_parser_factory,
            content_filter=content_filter,
            api_dialect=api_dialect,
            evict_idle_sessions=evict_idle_sessions,
        )
        return TestClient(build_app(serving, "test-model")), fake

    return _make
