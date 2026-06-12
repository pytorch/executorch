# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixtures for hermetic contract tests.

We build a SessionRuntime over a single FakeRunner worker, so the tests exercise
the real server, protocol, templating, and streaming code over the HTTP boundary
with NO model, tokenizer, GPU, or worker subprocess. This mirrors ExecuTorch's
fake_llm_executor approach: fake the worker, test the real surface.

The control plane imports no runtime pybind; only fastapi, pydantic, httpx, and
pytest are required.
"""

import pytest

from executorch.extension.llm.server.python.chat_template import ChatTemplate
from executorch.extension.llm.server.python.server import build_app
from executorch.extension.llm.server.python.serving_chat import ServingChat
from executorch.extension.llm.server.python.session_runtime import SessionRuntime
from executorch.extension.llm.server.python.tool_parsers import HermesDetector
from executorch.extension.llm.server.python.worker_client import WorkerError
from fastapi.testclient import TestClient


class _FakeStats:
    num_prompt_tokens = 5
    num_generated_tokens = 0
    finish_reason = None


class FakeRunner:
    """Canned engine: emits fixed tokens, records the config it was given.

    With max_named_sessions > 0 it also models the worker's session admission:
    open_session() enforces capacity and reports structured WorkerError codes,
    matching the real worker's contract."""

    def __init__(
        self, tokens, fail=False, finish_reason=None, max_named_sessions=0, gen_ids=None
    ):
        self._tokens = list(tokens)
        self._fail = fail
        self._finish_reason = finish_reason  # worker-reported stop reason, if any
        self._gen_ids = list(gen_ids or [])  # ids reported per turn (V2b.1.5)
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
            stats.generated_token_ids = list(self._gen_ids)
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
    ):
        fake = FakeRunner(
            tokens,
            fail=fail,
            finish_reason=finish_reason,
            max_named_sessions=max_named_sessions,
            gen_ids=gen_ids,
        )
        runtime = SessionRuntime(fake)  # one fake worker
        template = ChatTemplate(hf_tokenizer_path=None, allow_fallback=True)
        if prompt_tokens is not None:
            template._hf = _FakeTokenizer(prompt_tokens)
        serving = ServingChat(
            runtime,
            template,
            "test-model",
            max_context=max_context,
            tool_detector_cls=HermesDetector,
        )
        return TestClient(build_app(serving, "test-model")), fake

    return _make
