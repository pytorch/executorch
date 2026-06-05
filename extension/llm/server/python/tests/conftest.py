# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixtures for hermetic contract tests.

We build a RunnerPool over a single FakeRunner worker handle, so the tests
exercise the real server, protocol, templating, and streaming code over the
HTTP boundary with NO model, tokenizer, GPU, or worker subprocess. This mirrors
ExecuTorch's fake_llm_executor approach: fake the worker, test the real surface.

The control plane imports no runtime pybind; only fastapi, pydantic, httpx, and
pytest are required.
"""

import pytest

from executorch.extension.llm.server.python.chat_template import ChatTemplate
from executorch.extension.llm.server.python.runner_pool import RunnerPool
from executorch.extension.llm.server.python.server import build_app
from executorch.extension.llm.server.python.serving_chat import ServingChat
from executorch.extension.llm.server.python.tool_parsers import HermesDetector
from fastapi.testclient import TestClient


class _FakeStats:
    num_prompt_tokens = 5
    num_generated_tokens = 0
    finish_reason = None


class FakeRunner:
    """Canned engine: emits fixed tokens, records the config it was given."""

    def __init__(self, tokens, fail=False, finish_reason=None):
        self._tokens = list(tokens)
        self._fail = fail
        self._finish_reason = finish_reason  # worker-reported stop reason, if any
        self.captured_config = None
        self.stopped = False
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def stop(self):
        self.stopped = True

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
    ):
        fake = FakeRunner(tokens, fail=fail, finish_reason=finish_reason)
        pool = RunnerPool([fake])  # one fake worker handle
        template = ChatTemplate(hf_tokenizer_path=None, allow_fallback=True)
        if prompt_tokens is not None:
            template._hf = _FakeTokenizer(prompt_tokens)
        serving = ServingChat(
            pool,
            template,
            "test-model",
            max_context=max_context,
            tool_detector_cls=HermesDetector,
        )
        return TestClient(build_app(serving, "test-model")), fake

    return _make
