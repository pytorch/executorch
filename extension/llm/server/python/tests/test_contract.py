# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hermetic OpenAI-contract tests (fake engine, no model/GPU).

These assert on the public HTTP/wire contract only — response object shapes,
the streaming chunk protocol, status codes — never on internal classes or
methods. Implementation can change freely as long as these pass.
"""

import json

import pytest


def _sse_chunks(text):
    chunks, done = [], False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            done = True
            continue
        chunks.append(json.loads(payload))
    return chunks, done


def test_health(make_client):
    client, _ = make_client()
    assert client.get("/health").json() == {"status": "ok"}


def test_models_listing_shape(make_client):
    client, _ = make_client()
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "test-model"
    assert body["data"][0]["object"] == "model"


def test_chat_nonstreaming_shape(make_client):
    client, _ = make_client(tokens=["Hello", ", ", "world"])
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hello, world"
    assert choice["finish_reason"] == "stop"
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert k in body["usage"]
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )


def test_chat_streaming_protocol(make_client):
    client, _ = make_client(tokens=["a", "b", "c"])
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    chunks, done = _sse_chunks(resp.text)
    assert done, "stream must terminate with data: [DONE]"
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert content == "abc"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_request_params_forwarded_to_generation(make_client):
    # Contract behavior: the server must honor max_tokens/temperature.
    client, fake = make_client()
    client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 7,
            "temperature": 0.1,
        },
    )
    assert fake.captured_config.max_new_tokens == 7
    assert abs(fake.captured_config.temperature - 0.1) < 1e-6


def test_special_tokens_forwarded_to_worker_as_stops(make_client):
    # The worker must be told to stop at the model's end-of-turn special tokens
    # (e.g. <|im_end|>), not just request `stop` sequences. Otherwise a worker
    # whose EOS-by-token-id check misses the turn end runs to max_new (or errors
    # forwarding its own end token) past it — the text_llm_worker "decode failed"
    # seen on a real model. (Forwarding only request stops would leave this [].)
    client, fake = make_client()
    client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert "<|im_end|>" in (fake.captured_config.stop or [])


def test_request_stop_forwarded_to_worker(make_client):
    # A request `stop` sequence must reach the worker (so it can terminate early),
    # not only be applied by the Python backstop. The stop tests elsewhere pass
    # via that backstop even if forwarding regresses; this asserts forwarding.
    client, fake = make_client()
    client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["STOP"],
        },
    )
    # stop == special tokens + request stops, so check membership (not equality).
    assert "STOP" in (fake.captured_config.stop or [])


def test_tools_field_accepted(make_client):
    # tools is part of the contract even before parsing is enforced (M2).
    client, _ = make_client()
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "function", "function": {"name": "f", "parameters": {}}}
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["role"] == "assistant"


def test_invalid_request_returns_422(make_client):
    client, _ = make_client()
    resp = client.post(
        "/v1/chat/completions", json={"model": "test-model"}
    )  # no messages
    assert resp.status_code == 422


def test_special_tokens_stripped_nonstreaming(make_client):
    # The runner may decode EOS/special tokens to text; they must not leak.
    client, _ = make_client(tokens=["Hello", " world", "<|im_end|>", "LEAK"])
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.json()["choices"][0]["message"]["content"] == "Hello world"


def test_usage_populated_when_special_token_cuts_early(make_client):
    # Regression: cutting at a special token must not skip usage stats.
    client, _ = make_client(tokens=["Hello", "<|im_end|>", "LEAK"])
    body = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    ).json()
    assert body["choices"][0]["message"]["content"] == "Hello"
    assert body["usage"]["completion_tokens"] > 0


def test_special_tokens_stripped_streaming(make_client):
    client, _ = make_client(tokens=["Hello", " world", "<|im_end|>", "LEAK"])
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    chunks, _ = _sse_chunks(resp.text)
    content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert content == "Hello world"
    assert "LEAK" not in content and "<|im_end|>" not in content


# (1) Context-size-exceeded -> structured 400, both modes (not a dropped socket).
def test_context_length_exceeded_returns_400(make_client):
    client, _ = make_client(max_context=2048, prompt_tokens=2940)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "x" * 100}],
        },
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "context_length_exceeded"
    assert err["type"] == "invalid_request_error"


def test_context_length_exceeded_streaming_returns_400(make_client):
    client, _ = make_client(max_context=2048, prompt_tokens=2940)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
        },
    )
    # Pre-flight check rejects before the stream starts -> clean 400, no SSE.
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "context_length_exceeded"


def test_prompt_plus_max_tokens_exceeding_context_returns_400(make_client):
    # Prompt fits (100 < 2048) but prompt + max_tokens (100 + 2000) > 2048: must
    # reject up front, not run until the worker hits the limit mid-decode.
    client, _ = make_client(max_context=2048, prompt_tokens=100)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2000,
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "context_length_exceeded"


def test_prompt_plus_max_tokens_within_context_ok(make_client):
    # Prompt + max_tokens (100 + 100) <= 2048: must NOT be rejected.
    client, _ = make_client(max_context=2048, prompt_tokens=100)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        },
    )
    assert resp.status_code == 200


# (1) Mid-generation failure -> structured error, never a dropped connection.
def test_generation_failure_returns_structured_error(make_client):
    client, _ = make_client(fail=True)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 500
    assert resp.json()["error"]["type"] == "server_error"


def test_generation_failure_streaming_emits_error_event(make_client):
    client, _ = make_client(fail=True)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200  # headers already sent; error arrives as an event
    chunks, done = _sse_chunks(resp.text)
    assert done
    assert any("error" in c for c in chunks)


# (3) finish_reason == "length" when max_tokens is hit.
def test_finish_reason_length_when_max_tokens_hit(make_client):
    client, _ = make_client(tokens=["a", "b", "c"])  # 3 generated tokens
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3,
        },
    )
    assert resp.json()["choices"][0]["finish_reason"] == "length"


def test_finish_reason_stop_when_under_max_tokens(make_client):
    client, _ = make_client(tokens=["a", "b", "c"])
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        },
    )
    assert resp.json()["choices"][0]["finish_reason"] == "stop"


# Worker-reported finish_reason: the worker may silently clamp max_new to the
# context window, so the token-count heuristic can't tell a real stop from a
# truncation. Trust the worker's reason.
def test_worker_reported_length_overrides_token_count(make_client):
    # 3 tokens generated (< requested 100) but the worker says it ran to the cap
    # (a context clamp): finish_reason must be "length", not "stop".
    client, _ = make_client(tokens=["a", "b", "c"], finish_reason="length")
    body = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        },
    ).json()
    assert body["choices"][0]["finish_reason"] == "length"


def test_worker_reported_stop_overrides_token_count(make_client):
    # 3 tokens with max_tokens=3 (heuristic would say "length"), but the worker
    # reports EOS: finish_reason must be "stop".
    client, _ = make_client(tokens=["a", "b", "c"], finish_reason="stop")
    body = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3,
        },
    ).json()
    assert body["choices"][0]["finish_reason"] == "stop"


def test_server_stop_sequence_wins_over_worker_length(make_client):
    # A server-side stop sequence is truncation in the control plane; it must
    # win even when the worker would report "length".
    client, _ = make_client(
        tokens=["Hello ", "world ", "STOP", " x"], finish_reason="length"
    )
    body = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["STOP"],
            "max_tokens": 100,
        },
    ).json()
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "STOP" not in (body["choices"][0]["message"]["content"] or "")


# (4) Error-variant matrix: malformed requests -> consistent 422.
@pytest.mark.parametrize(
    "bad_body",
    [
        {"model": "m"},  # missing messages
        {"model": "m", "messages": "not-a-list"},
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": "hot",
        },
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": "maybe",
        },
        {"model": "m", "messages": [{"content": "no role"}]},
    ],
)
def test_invalid_requests_return_422(make_client, bad_body):
    client, _ = make_client()
    assert client.post("/v1/chat/completions", json=bad_body).status_code == 422


# (6) Streaming usage when stream_options.include_usage is set.
def test_streaming_usage_included(make_client):
    client, _ = make_client(tokens=["a", "b"])
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    chunks, _ = _sse_chunks(resp.text)
    usage_chunks = [c for c in chunks if c.get("usage")]
    assert usage_chunks, "expected a chunk carrying usage"
    u = usage_chunks[-1]["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_streaming_usage_absent_by_default(make_client):
    client, _ = make_client(tokens=["a", "b"])
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    chunks, _ = _sse_chunks(resp.text)
    assert not any(c.get("usage") for c in chunks)


# (2) Unicode/multibyte content survives streaming intact.
def test_unicode_streaming_integrity(make_client):
    pieces = ["café ", "日本語 ", "😀", "🎉"]
    client, _ = make_client(tokens=pieces)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    chunks, _ = _sse_chunks(
        resp.text
    )  # _sse_chunks uses json.loads -> validates UTF-8/JSON
    content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert content == "".join(pieces)


def test_unicode_nonstreaming_integrity(make_client):
    pieces = ["café ", "日本語 ", "😀"]
    client, _ = make_client(tokens=pieces)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.json()["choices"][0]["message"]["content"] == "".join(pieces)
