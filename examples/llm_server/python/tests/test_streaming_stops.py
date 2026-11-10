# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plain-chat streaming special-token cleanup.

Non-streaming scrubs broad content specials (the full all_special_tokens set) from
visible content via _strip_specials. Plain-chat streaming must be consistent: a
broad special that is NOT a turn terminator (e.g. <|fim_pad|>) must not reach the
client, and -- since trimming it makes the turn's visible text != generated text
-- the worker halts (omitting ids), so the turn is recorded non-resumable. Tool
turns keep the narrow terminator set so a <tool_call> is never cut before parsing.
"""

import asyncio
import json

import pytest

from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.protocol import ChatCompletionRequest
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import (
    GenStats,
    SessionRuntime,
)
from executorch.examples.llm_server.python.tool_parsers import HermesDetector
from executorch.examples.llm_server.python.worker_client import WorkerError
from fastapi.testclient import TestClient

FIM = "<|fim_pad|>"  # a broad content special that is NOT a turn terminator
CHANNEL_OPEN = "<|channel>"
CHANNEL_CLOSE = "<channel|>"
THINK = "<|think|>"
CHANNEL_SPECIALS = {CHANNEL_OPEN, CHANNEL_CLOSE, THINK}
WEATHER_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
]


class _SpecialTok:
    """Fake HF tokenizer whose special set is broader than the turn terminators:
    eos=<|im_end|> (a terminator) plus <|fim_pad|> (broad-only)."""

    eos_token = "<|im_end|>"
    all_special_tokens = ["<|im_end|>", FIM, CHANNEL_OPEN, CHANNEL_CLOSE, THINK]

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
        self._gen_ids = list(gen_ids) if gen_ids is not None else None
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
            stats.generated_token_ids = (
                None if trimmed or self._gen_ids is None else list(self._gen_ids)
            )
            stats_callback(stats)


def _serving(
    tokens,
    honor_stops=False,
    gen_ids=None,
    content_filter=None,
    content_filter_specials=None,
):
    runner = _Runner(tokens, gen_ids=gen_ids, honor_stops=honor_stops)
    template = ChatTemplate(hf_tokenizer_path=None, allow_fallback=True)
    template._hf = _SpecialTok()
    serving = ServingChat(
        SessionRuntime(runner),
        template,
        "test-model",
        tool_detector_cls=HermesDetector,
        content_filter=content_filter,
        content_filter_specials=content_filter_specials,
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


def test_plain_chat_streaming_applies_content_filter():
    def strip_channel(text):
        return text.replace(f"{CHANNEL_OPEN}secret{CHANNEL_CLOSE}", "").strip()

    serving, runner = _serving(
        ["Visible ", CHANNEL_OPEN, "secret", CHANNEL_CLOSE],
        content_filter=strip_channel,
        content_filter_specials=CHANNEL_SPECIALS,
    )
    r = _client(serving).post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    content, _ = _sse_content(r.text)
    assert content == "Visible"
    assert CHANNEL_OPEN not in content and CHANNEL_CLOSE not in content
    assert CHANNEL_OPEN not in (runner.captured_config.stop or [])


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


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prefix", [[], ["keep"]])
def test_user_request_stop_trims_and_nonresumable(stream, prefix):
    serving, _ = _serving(prefix + ["STOPHERE", "drop"], honor_stops=True, gen_ids=[9])
    r = _client(serving).post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "session_id": "s",
            "stop": "STOPHERE",
            "stream": stream,
        },
    )
    assert r.status_code == 200
    content = (
        _sse_content(r.text)[0]
        if stream
        else r.json()["choices"][0]["message"].get("content")
    )
    assert (content or "") == "".join(prefix)
    assert serving._transcript._turns["s"][0]["ids"] is None


def _replay_ignored_eos(chunks, stop=None, finish_reason="length"):
    serving, _ = _serving([])
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        ignore_eos=True,
        stop=stop,
    )

    class Source:
        def __init__(self):
            self.chunks = iter(chunks)
            self.position = 0
            self.cancellations = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                chunk = next(self.chunks)
            except StopIteration:
                raise StopAsyncIteration from None
            self.position += 1
            return chunk

        def cancel(self):
            self.cancellations += 1

    async def collect():
        source = Source()
        stop_hit = [False]
        emitted = []
        async for text in serving._clean_ignored_eos(
            request, source, stop_hit, GenStats(finish_reason=finish_reason)
        ):
            emitted.append((source.position, text))
        return emitted, stop_hit[0], source.cancellations

    return asyncio.run(collect())


def test_ignored_eos_streams_safe_text_before_reading_next_chunk():
    emitted, stopped, cancellations = _replay_ignored_eos(["Hello", " world"])
    assert emitted == [(1, "Hello"), (2, " world")]
    assert not stopped
    assert cancellations == 0


def test_ignored_eos_retains_only_possible_stop_fragment():
    emitted, stopped, cancellations = _replay_ignored_eos(
        ["Hi <|im_", "end", "|>hidden"]
    )
    assert emitted == [(1, "Hi ")]
    assert not stopped
    assert cancellations == 0


def test_ignored_eos_releases_fragment_when_it_is_not_a_stop():
    emitted, _, _ = _replay_ignored_eos(["猫 <|im_", "possible"])
    assert emitted == [(1, "猫 "), (2, "<|im_possible")]


@pytest.mark.parametrize(
    "text,stop,finish_reason,expected,stopped",
    [
        ("猫 says hi<|im_end|>hidden", None, "length", "猫 says hi", False),
        (f"Hi{FIM}hidden", None, "length", "Hi", False),
        ("HiSTOPignored", "STOP", "length", "Hi", True),
        ("Hi<|im_end|>hidden STOPignored", "STOP", "length", "Hi", True),
        ("Hi<|im_end|>STOPignored", "<|im_end|>STOP", "length", "Hi", True),
        ("Hi <|im_", None, "length", "Hi <|im_", False),
        ("Hi <|im_", "end|>", "stop", "Hi ", False),
    ],
)
def test_ignored_eos_cleanup_across_chunk_boundaries(
    text, stop, finish_reason, expected, stopped
):
    chunkings = [[text[:split], text[split:]] for split in range(len(text) + 1)]
    chunkings.append(list(text))
    for chunks in chunkings:
        emitted, stop_hit, cancellations = _replay_ignored_eos(
            chunks, stop=stop, finish_reason=finish_reason
        )
        assert "".join(part for _, part in emitted) == expected, chunks
        assert stop_hit == stopped, chunks
        assert cancellations == int(stopped), chunks
