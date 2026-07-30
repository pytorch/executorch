# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Language-neutral OpenAI-contract conformance tests.

Runs against any base URL (ExecuTorch, llama.cpp, mlx-lm, ...) so every server
implementation is validated against one shared spec. Point it at a running
server:

    OPENAI_BASE_URL=http://127.0.0.1:8000/v1 pytest test_openai_contract.py

Skips automatically if no server is reachable.
"""

import json
import os
import urllib.error
import urllib.request

import pytest

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
MODEL = os.environ.get("OPENAI_MODEL", "executorch")


def _post(path: str, body: dict, stream: bool = False):
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=120)


def _server_up() -> bool:
    try:
        urllib.request.urlopen(f"{BASE_URL}/models", timeout=5)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_up(), reason="no OpenAI server at OPENAI_BASE_URL"
)


def test_models_listing():
    with urllib.request.urlopen(f"{BASE_URL}/models", timeout=10) as r:
        data = json.loads(r.read())
    assert data["object"] == "list"
    assert any("id" in m for m in data["data"])


def test_chat_completion_nonstreaming():
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    with _post("/chat/completions", body) as r:
        data = json.loads(r.read())
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(data["choices"][0]["message"]["content"], str)
    assert data["choices"][0]["finish_reason"] is not None


def test_chat_completion_streaming():
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Count to three."}],
        "max_tokens": 32,
        "stream": True,
    }
    saw_role = saw_content = saw_done = False
    with _post("/chat/completions", body, stream=True) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(payload)
            assert chunk["object"] == "chat.completion.chunk"
            delta = chunk["choices"][0]["delta"]
            saw_role = saw_role or delta.get("role") == "assistant"
            saw_content = saw_content or bool(delta.get("content"))
    assert saw_role and saw_content and saw_done


def test_multibyte_streaming_integrity():
    # Byte-level BPE can split a multi-byte character across tokens; the stream
    # must reassemble it, not abort with a UTF-8 decode error.
    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Reply with exactly: 你好世界 🌍 café"}
        ],
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": True,
    }
    content, saw_done, saw_error = "", False, False
    with _post("/chat/completions", body, stream=True) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(payload)
            if "error" in chunk:
                saw_error = True
            content += (
                chunk["choices"][0]["delta"].get("content", "")
                if chunk.get("choices")
                else ""
            )
    assert saw_done and not saw_error
    assert isinstance(content, str) and content  # reassembled, valid UTF-8


def test_usage_chunk_in_stream():
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 16,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    usage = None
    with _post("/chat/completions", body, stream=True) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
    assert usage is not None, "no usage chunk emitted with include_usage"
    assert usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def test_tool_call_response_shape():
    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is the weather in Paris? Use the tool."}
        ],
        "tools": [WEATHER_TOOL],
        "max_tokens": 128,
        "temperature": 0.0,
    }
    with _post("/chat/completions", body) as r:
        data = json.loads(r.read())
    calls = data["choices"][0]["message"].get("tool_calls")
    assert calls, "expected tool_calls in response"
    tc = calls[0]
    assert tc["type"] == "function"
    assert tc["id"]
    assert tc["function"]["name"] == "get_weather"
    json.loads(tc["function"]["arguments"])  # arguments is a JSON string
    assert data["choices"][0]["finish_reason"] == "tool_calls"


def test_error_body_shape():
    # Over-long prompt -> structured 400 (OpenAI error envelope), not a 500/drop.
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "word " * 40000}],
        "max_tokens": 8,
    }
    try:
        _post("/chat/completions", body)
        raise AssertionError("expected an HTTP error for over-long prompt")
    except urllib.error.HTTPError as e:
        assert 400 <= e.code < 500
        err = json.loads(e.read())["error"]
        assert err["message"] and err["type"]
