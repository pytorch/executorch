# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tool-calling tests (HTTP contract via the server).

Hermes/Qwen format only. The server buffers the model's full output and parses
it once into complete OpenAI tool_calls; parse failures degrade to visible text.
"""

import json

WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
]


def _call_text(name, args):
    return f'<tool_call>\n{{"name": "{name}", "arguments": {json.dumps(args)}}}\n</tool_call>'


# --- HTTP contract: non-streaming ---------------------------------------


def test_tool_call_nonstreaming(make_client):
    client, _ = make_client(tokens=[_call_text("get_weather", {"city": "Paris"})])
    body = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": WEATHER_TOOLS,
        },
    ).json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tc = choice["message"]["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}


def test_tool_call_streaming(make_client):
    client, _ = make_client(tokens=[_call_text("get_weather", {"city": "Paris"})])
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": WEATHER_TOOLS,
            "stream": True,
        },
    )
    # Reuse the contract helper's SSE parsing.
    chunks = []
    for line in resp.text.splitlines():
        line = line.strip()
        if line.startswith("data:") and "[DONE]" not in line:
            chunks.append(json.loads(line[len("data:") :].strip()))
    tool_deltas = [
        c["choices"][0]["delta"]["tool_calls"][0]
        for c in chunks
        if c["choices"] and c["choices"][0]["delta"].get("tool_calls")
    ]
    assert tool_deltas and tool_deltas[0]["function"]["name"] == "get_weather"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_undefined_tool_is_not_called(make_client):
    # Model calls a tool not in the request's tools -> no tool_calls; the raw
    # call stays visible as content (degrade to text, never silent drop).
    client, _ = make_client(tokens=[_call_text("rm_rf", {"path": "/"})])
    body = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": WEATHER_TOOLS,
        },
    ).json()
    msg = body["choices"][0]["message"]
    assert msg.get("tool_calls") is None
    assert "rm_rf" in (msg.get("content") or "")  # not dropped — visible as text


def test_mixed_valid_and_undefined_tool_degrades_to_text(make_client):
    # A response with one valid + one undefined call must NOT emit the valid one
    # while silently dropping the undefined one — the whole response degrades to
    # visible text so the model's full intent is preserved.
    client, _ = make_client(
        tokens=[
            _call_text("get_weather", {"city": "Paris"})
            + _call_text("rm_rf", {"path": "/"})
        ]
    )
    msg = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": WEATHER_TOOLS,
        },
    ).json()["choices"][0]["message"]
    assert msg.get("tool_calls") is None  # no partial set
    content = msg.get("content") or ""
    assert "rm_rf" in content and "get_weather" in content  # full intent visible


def test_tool_choice_none_omits_tools_from_prompt():
    from executorch.examples.llm_server.python.chat_template import ChatTemplate
    from executorch.examples.llm_server.python.server import build_app
    from executorch.examples.llm_server.python.serving_chat import ServingChat
    from executorch.examples.llm_server.python.session_runtime import SessionRuntime
    from executorch.examples.llm_server.python.tool_parsers import HermesDetector

    # tool_choice="none" must NOT inject tool schemas into the chat template; if it
    # did, the model could still emit a <tool_call> that we'd surface as plain text
    # (parsing is disabled for "none"). Assert via a recording tokenizer.
    from fastapi.testclient import TestClient

    class _RecordingTok:
        all_special_tokens: list = []

        def __init__(self):
            self.tools_seen = "UNSET"

        def encode(self, text):
            return [0]

        def apply_chat_template(
            self, messages, tools, add_generation_prompt, tokenize, **kwargs
        ):
            self.tools_seen = tools
            return "PROMPT"

    class _Runner:
        def reset(self):
            pass

        def stop(self):
            pass

        def generate(self, prompt, config, token_callback=None, stats_callback=None):
            if token_callback:
                token_callback("ok")

    rec = _RecordingTok()
    template = ChatTemplate(hf_tokenizer_path=None, allow_fallback=True)
    template._hf = rec
    runtime = SessionRuntime(_Runner())
    serving = ServingChat(
        runtime, template, "test-model", tool_detector_cls=HermesDetector
    )
    client = TestClient(build_app(serving, "test-model"))
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": WEATHER_TOOLS,
        "max_tokens": 8,
    }

    assert (
        client.post(
            "/v1/chat/completions", json={**body, "tool_choice": "none"}
        ).status_code
        == 200
    )
    assert rec.tools_seen is None  # tools omitted from the rendered prompt

    # Control: default ("auto") still passes the tools through to the template.
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    assert rec.tools_seen == WEATHER_TOOLS


def test_malformed_tool_call_falls_back_to_text(make_client):
    # Broken JSON inside the markers must not crash; degrade to visible text.
    client, _ = make_client(tokens=["<tool_call>\n{not json}\n</tool_call>"])
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": WEATHER_TOOLS,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"].get("tool_calls") is None


def test_no_tools_field_means_text_even_if_markers_present(make_client):
    # Without a tools array, tool markers are just content (not parsed).
    client, _ = make_client(tokens=[_call_text("get_weather", {"city": "X"})])
    body = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    ).json()
    assert body["choices"][0]["message"].get("tool_calls") is None


def test_parallel_calls_in_one_message(make_client):
    # Two complete <tool_call> blocks in one output -> two structured calls.
    tokens = [
        _call_text("get_weather", {"city": "A"})
        + _call_text("get_weather", {"city": "B"})
    ]
    client, _ = make_client(tokens=tokens)
    body = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather in A and B?"}],
            "tools": WEATHER_TOOLS,
        },
    ).json()
    calls = body["choices"][0]["message"]["tool_calls"]
    assert [json.loads(c["function"]["arguments"])["city"] for c in calls] == ["A", "B"]
