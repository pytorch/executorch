# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contract tests for sampling/control params: stop sequences, n, tool_choice.

These exercise the real server over the HTTP boundary with a FakeRunner.
"""

import json

CALL = (
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
)
WEATHER_TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {}},
}


def _body(**kw):
    b = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
    b.update(kw)
    return b


def test_n_greater_than_one_is_rejected(make_client):
    client, _ = make_client()
    r = client.post("/v1/chat/completions", json=_body(n=2))
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert r.json()["error"]["code"] == "unsupported_parameter"


def test_unsupported_params_rejected(make_client):
    client, _ = make_client()
    for param in (
        {"top_p": 0.5},
        {"top_p": 2.0},  # > 1.0 is not a no-op either — only exactly 1.0/unset is
        {"seed": 42},
        {"reasoning_effort": "high"},
        {"frequency_penalty": 1.0},
        {"presence_penalty": -0.5},
        {"top_k": 40},
        {"logit_bias": {"123": 5.0}},
        {"response_format": {"type": "json_object"}},
        {"logprobs": True},
        {"top_logprobs": 5},
        {"parallel_tool_calls": False},
    ):
        r = client.post("/v1/chat/completions", json=_body(**param))
        assert r.status_code == 400, param
        assert r.json()["error"]["code"] == "unsupported_parameter", param


def test_nonpositive_max_tokens_rejected(make_client):
    # max_tokens=0/-2 must not be silently treated as "unbounded" (the `or` bug);
    # 0 and negatives are invalid for both field names.
    client, _ = make_client()
    for param in (
        {"max_tokens": 0},
        {"max_tokens": -2},
        {"max_completion_tokens": 0},
        {"max_completion_tokens": -2},
    ):
        r = client.post("/v1/chat/completions", json=_body(**param))
        assert r.status_code == 400, param
        assert r.json()["error"]["type"] == "invalid_request_error", param


def test_temperature_range_rejected(make_client):
    client, _ = make_client()
    for param in (
        {"temperature": -1},
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"temperature": 3},
    ):
        r = client.post("/v1/chat/completions", json=_body(**param))
        assert r.status_code == 400, param
        assert r.json()["error"]["code"] == "invalid_value", param


def test_noop_output_contract_fields_accepted(make_client):
    # The default/no-op forms must NOT be rejected (don't break OpenAI clients
    # that send them explicitly).
    client, _ = make_client()
    r = client.post(
        "/v1/chat/completions",
        json=_body(
            response_format={"type": "text"},
            parallel_tool_calls=True,
            max_tokens=8,
        ),
    )
    assert r.status_code == 200


def test_zero_penalties_and_unknown_fields_accepted(make_client):
    # frequency/presence_penalty=0.0 are no-ops; unknown non-generation fields
    # (user/store/metadata) are ignored, not rejected (don't break OpenAI clients).
    client, _ = make_client()
    r = client.post(
        "/v1/chat/completions",
        json=_body(
            frequency_penalty=0.0,
            presence_penalty=0.0,
            user="abc",
            store=False,
            metadata={"k": "v"},
            max_tokens=8,
        ),
    )
    assert r.status_code == 200


def test_unsupported_tool_choice_rejected(make_client):
    # "required" / a specific-function choice would need constrained decoding to
    # force/restrict the call; the server rejects rather than silently treating as "auto".
    client, _ = make_client()
    for choice in (
        "required",
        {"type": "function", "function": {"name": "get_weather"}},
    ):
        r = client.post(
            "/v1/chat/completions",
            json=_body(tools=[WEATHER_TOOL], tool_choice=choice, max_tokens=8),
        )
        assert r.status_code == 400, choice
        assert r.json()["error"]["code"] == "unsupported_parameter", choice


def test_supported_params_accepted(make_client):
    # top_p=1.0 (no-op) and temperature/max_tokens must NOT be rejected; neither
    # should tool_choice "auto" / "none".
    client, _ = make_client()
    for temperature in (0.0, 1.0, 2.0):
        r = client.post(
            "/v1/chat/completions",
            json=_body(top_p=1.0, temperature=temperature, max_tokens=8),
        )
        assert r.status_code == 200, temperature
    for choice in ("auto", "none"):
        r = client.post(
            "/v1/chat/completions",
            json=_body(tools=[WEATHER_TOOL], tool_choice=choice, max_tokens=8),
        )
        assert r.status_code == 200, choice


def test_stop_sequence_truncates_nonstreaming(make_client):
    client, _ = make_client(tokens=["Hello ", "world ", "STOP", " ignored"])
    r = client.post("/v1/chat/completions", json=_body(stop=["STOP"], max_tokens=32))
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert content == "Hello world "
    assert "STOP" not in content


def test_stop_sequence_truncates_streaming(make_client):
    client, _ = make_client(tokens=["Hello ", "world ", "STOP", " ignored"])
    content = ""
    with client.stream(
        "POST", "/v1/chat/completions", json=_body(stop=["STOP"], stream=True)
    ) as r:
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            delta = json.loads(payload)["choices"][0]["delta"]
            content += delta.get("content") or ""
    assert content == "Hello world "
    assert "STOP" not in content


def test_stop_forces_finish_reason_over_length_nonstreaming(make_client):
    # 4 tokens emitted (completion reaches max_tokens=4) AND a stop is hit:
    # finish_reason must be "stop" (boundary), not "length".
    client, _ = make_client(tokens=["a ", "b ", "STOP", " c"])
    body = client.post(
        "/v1/chat/completions", json=_body(stop=["STOP"], max_tokens=4)
    ).json()
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "STOP" not in (body["choices"][0]["message"]["content"] or "")


def test_stop_forces_finish_reason_over_length_streaming(make_client):
    client, _ = make_client(tokens=["a ", "b ", "STOP", " c"])
    finish = None
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json=_body(stop=["STOP"], max_tokens=4, stream=True),
    ) as r:
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            p = line[len("data:") :].strip()
            if p == "[DONE]":
                break
            fr = json.loads(p)["choices"][0].get("finish_reason")
            if fr:
                finish = fr
    assert finish == "stop"


_STOP_THEN_CALL = (
    'Answer STOP <tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>'
)


def test_stop_before_tool_call_nonstreaming(make_client):
    # Stop boundary precedes a tool call (in one chunk, so truncation — not just
    # early-stop — must catch it): the call must NOT be parsed/emitted.
    client, _ = make_client(tokens=[_STOP_THEN_CALL])
    r = client.post(
        "/v1/chat/completions",
        json=_body(tools=[WEATHER_TOOL], stop=["STOP"], max_tokens=64),
    )
    msg = r.json()["choices"][0]["message"]
    assert msg.get("tool_calls") is None
    assert "STOP" not in (msg.get("content") or "")


def test_stop_before_tool_call_streaming(make_client):
    client, _ = make_client(tokens=[_STOP_THEN_CALL])
    saw_tool, content = False, ""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json=_body(tools=[WEATHER_TOOL], stop=["STOP"], stream=True),
    ) as r:
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            p = line[len("data:") :].strip()
            if p == "[DONE]":
                break
            delta = json.loads(p)["choices"][0]["delta"]
            if delta.get("tool_calls"):
                saw_tool = True
            content += delta.get("content") or ""
    assert not saw_tool
    assert "STOP" not in content


def test_tool_choice_none_disables_tools(make_client):
    client, _ = make_client(tokens=[CALL])
    r = client.post(
        "/v1/chat/completions",
        json=_body(tools=[WEATHER_TOOL], tool_choice="none", max_tokens=64),
    )
    msg = r.json()["choices"][0]["message"]
    assert msg.get("tool_calls") is None  # tools disabled -> returned as text
    assert r.json()["choices"][0]["finish_reason"] != "tool_calls"


def test_tool_choice_default_still_parses(make_client):
    # Sanity: without tool_choice="none", the same call IS parsed as a tool call.
    client, _ = make_client(tokens=[CALL])
    r = client.post(
        "/v1/chat/completions", json=_body(tools=[WEATHER_TOOL], max_tokens=64)
    )
    calls = r.json()["choices"][0]["message"].get("tool_calls")
    assert calls and calls[0]["function"]["name"] == "get_weather"
