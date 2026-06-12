# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plain-chat streaming special-token cleanup (WI1).

Non-streaming scrubs broad content specials (the full all_special_tokens set) from
visible content via _strip_specials. Plain-chat streaming must be consistent: a
broad special that is NOT a turn terminator (e.g. <|fim_pad|>) must not reach the
client, and -- since trimming it makes the turn's visible text != generated text
-- the worker halts (omitting ids), so the turn is recorded non-resumable. Tool
turns keep the narrow terminator set so a <tool_call> is never cut before parsing.
"""

import json

from executorch.extension.llm.server.python.chat_template import ChatTemplate
from executorch.extension.llm.server.python.server import build_app
from executorch.extension.llm.server.python.serving_chat import ServingChat
from executorch.extension.llm.server.python.session_runtime import SessionRuntime
from executorch.extension.llm.server.python.tool_parsers import HermesDetector
from executorch.extension.llm.server.python.worker_client import WorkerError
from fastapi.testclient import TestClient

FIM = "<|fim_pad|>"  # a broad content special that is NOT a turn terminator
WEATHER_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
]


class _SpecialTok:
    """Fake HF tokenizer whose special set is broader than the turn terminators:
    eos=<|im_end|> (a terminator) plus <|fim_pad|> (broad-only)."""

    eos_token = "<|im_end|>"
    all_special_tokens = ["<|im_end|>", FIM]

    def encode(self, text, add_special_tokens=False):
        return [0] * 5

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, **kw
    ):
        return "PROMPT"


class _Runner:
    """Fake worker. With honor_stops it models the real worker's stop-trim: a
    stop string halts generation, the stop and everything after is dropped, and
    the turn is non-resumable (generated_token_ids omitted)."""

    def __init__(self, tokens, gen_ids=None, honor_stops=False, max_named=4):
        self._tokens = list(tokens)
        self._gen_ids = list(gen_ids or [])
        self._honor = honor_stops
        self.max_named_sessions = max_named
        self.open_named = set()
        self.captured_config = None

    def reset(self):
        pass

    def stop(self):
        pass

    def open_session(self, sid):
        if sid in self.open_named:
            return
        if self.max_named_sessions == 0:
            raise WorkerError("no named sessions", code="unsupported_session")
        if len(self.open_named) >= self.max_named_sessions:
            raise WorkerError("capacity", code="capacity_exhausted")
        self.open_named.add(sid)

    def close_session(self, sid):
        self.open_named.discard(sid)

    def reset_session(self, sid):
        pass

    def generate(self, prompt, config, token_callback=None, stats_callback=None):
        self.captured_config = config
        stops = list(getattr(config, "stop", []) or []) if self._honor else []
        emitted, trimmed = 0, False
        for tok in self._tokens:
            if any(s and s in tok for s in stops):
                trimmed = True
                break
            if token_callback:
                token_callback(tok)
            emitted += 1
        if stats_callback:
            stats = type("S", (), {})()
            stats.num_prompt_tokens = 5
            stats.num_generated_tokens = emitted
            stats.finish_reason = "stop" if trimmed else None
            stats.generated_token_ids = [] if trimmed else list(self._gen_ids)
            stats_callback(stats)


def _serving(tokens, honor_stops=False, gen_ids=None):
    runner = _Runner(tokens, gen_ids=gen_ids, honor_stops=honor_stops)
    template = ChatTemplate(hf_tokenizer_path=None, allow_fallback=True)
    template._hf = _SpecialTok()
    serving = ServingChat(
        SessionRuntime(runner), template, "test-model", tool_detector_cls=HermesDetector
    )
    return serving, runner


def _client(serving):
    return TestClient(build_app(serving, "test-model"))


def _sse_content(text):
    content, finish = "", None
    for line in text.splitlines():
        if line.startswith("data:") and "[DONE]" not in line:
            d = json.loads(line[5:])
            for ch in d.get("choices", []):
                content += (ch.get("delta", {}) or {}).get("content") or ""
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return content, finish


def test_plain_chat_worker_stops_include_broad_specials():
    serving, runner = _serving(["hi"])
    _client(serving).post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert FIM in (runner.captured_config.stop or [])


def test_tool_path_worker_stops_exclude_broad_specials():
    serving, runner = _serving(["hi"])
    _client(serving).post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": WEATHER_TOOLS,
        },
    )
    # Tool turns keep the narrow set so a structural/tool delimiter isn't cut.
    assert FIM not in (runner.captured_config.stop or [])


def test_plain_chat_streaming_does_not_leak_broad_special():
    serving, _ = _serving(["Hi", FIM, "leak"])
    r = _client(serving).post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    content, finish = _sse_content(r.text)
    assert content == "Hi"  # the special and everything after is cut
    assert FIM not in content
    assert finish == "stop"


def test_plain_chat_nonstreaming_matches_streaming_visible():
    serving, _ = _serving(["Hi", FIM, "leak"])
    r = _client(serving).post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    choice = r.json()["choices"][0]
    assert choice["message"]["content"] == "Hi"
    assert choice["finish_reason"] == "stop"


def test_tool_streaming_not_broken_by_broad_special():
    tc = '<tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>'
    serving, _ = _serving([tc])
    r = _client(serving).post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": WEATHER_TOOLS,
            "stream": True,
        },
    )
    chunks = [
        json.loads(line[5:])
        for line in r.text.splitlines()
        if line.startswith("data:") and "[DONE]" not in line
    ]
    finishes = [
        c["choices"][0]["finish_reason"]
        for c in chunks
        if c["choices"][0].get("finish_reason")
    ]
    has_tool = any(
        (c["choices"][0].get("delta") or {}).get("tool_calls") for c in chunks
    )
    assert "tool_calls" in finishes and has_tool


def test_plain_chat_broad_stop_marks_turn_nonresumable():
    # honor_stops: the worker trims at the broad special and omits ids; the
    # transcript must record the turn as non-resumable (ids=None), not splice ids
    # for text the client never saw.
    serving, _ = _serving(["Hi", FIM, "leak"], honor_stops=True, gen_ids=[1, 2, 3])
    r = _client(serving).post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "session_id": "s",
        },
    )
    assert r.json()["choices"][0]["message"]["content"] == "Hi"
    assert serving._transcript._turns["s"][0]["ids"] is None


def test_user_request_stop_trims_and_nonresumable():
    serving, _ = _serving(["keep", "STOPHERE", "drop"], honor_stops=True, gen_ids=[9])
    r = _client(serving).post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "session_id": "s",
            "stop": "STOPHERE",
        },
    )
    assert r.json()["choices"][0]["message"]["content"] == "keep"
    assert serving._transcript._turns["s"][0]["ids"] is None
