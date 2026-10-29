# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contract tests for sampling/control params: stop sequences, n, tool_choice.

These exercise the real server over the HTTP boundary with a FakeRunner.
"""

import json

import pytest

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
        {"reasoning_effort": "high"},
        {"frequency_penalty": 1.0},
        {"presence_penalty": -0.5},
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


def test_sampling_parameter_ranges_rejected(make_client):
    client, _ = make_client()
    for param in (
        {"top_p": 0.0},
        {"top_p": -0.1},
        {"top_p": 1.1},
        {"top_k": -1},
        {"top_k": 2**31},
        {"seed": 0},
        {"seed": -1},
        {"seed": 2**63},
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
    # Sampling controls and temperature/max_tokens must not be rejected; neither
    # should tool_choice "auto" / "none".
    client, _ = make_client()
    for temperature in (0.0, 1.0, 2.0):
        r = client.post(
            "/v1/chat/completions",
            json=_body(
                top_p=0.9,
                top_k=40,
                seed=42,
                temperature=temperature,
                max_tokens=8,
            ),
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


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("ignore_eos", [False, True])
def test_ignore_eos_hides_text_after_template_terminators(
    make_client, stream, ignore_eos
):
    client, runner = make_client(
        tokens=["before", "<|im_end|>", "after"], finish_reason="length"
    )
    response = client.post(
        "/v1/chat/completions",
        json=_body(ignore_eos=ignore_eos, stream=stream, max_tokens=3),
    )
    assert response.status_code == 200
    if stream:
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        finish = chunks[-1]["choices"][0]["finish_reason"]
    else:
        content = response.json()["choices"][0]["message"]["content"]
        finish = response.json()["choices"][0]["finish_reason"]
    assert content == "before"
    assert finish == ("length" if ignore_eos else "stop")
    assert runner.captured_config.ignore_eos is ignore_eos


@pytest.mark.parametrize("stream", [False, True])
def test_ignore_eos_still_honors_explicit_stop(make_client, stream):
    client, runner = make_client(tokens=["before<|im_end|>after", "ST", "OP", "hidden"])
    response = client.post(
        "/v1/chat/completions",
        json=_body(ignore_eos=True, stop="STOP", stream=stream, max_tokens=32),
    )
    assert response.status_code == 200
    if stream:
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        finish = chunks[-1]["choices"][0]["finish_reason"]
    else:
        content = response.json()["choices"][0]["message"]["content"]
        finish = response.json()["choices"][0]["finish_reason"]
    assert content == "before"
    assert finish == "stop"
    assert runner.captured_config.stop == ["STOP"]


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_ignore_eos_requires_a_boolean(make_client, value):
    client, runner = make_client()
    response = client.post("/v1/chat/completions", json=_body(ignore_eos=value))
    assert response.status_code == 422
    assert runner.captured_config is None


@pytest.mark.parametrize("stream", [False, True])
def test_ignore_eos_rejects_unsupported_worker_before_streaming(make_client, stream):
    client, runner = make_client(supports_ignore_eos=False)
    response = client.post(
        "/v1/chat/completions", json=_body(ignore_eos=True, stream=stream)
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert runner.captured_config is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_before_eos", [False, True])
@pytest.mark.parametrize("eos", ["<|im_end|>", "<|native_eos|>"])
def test_ignore_eos_parses_only_first_turn_and_counts_full_generation(
    make_client, stream, tool_before_eos, eos
):
    first = CALL if tool_before_eos else "answer"
    client, runner = make_client(
        tokens=[first, eos[:4], eos[4:], CALL],
        finish_reason="length",
        default_stop_strings=[eos],
    )
    response = client.post(
        "/v1/chat/completions",
        json=_body(
            ignore_eos=True,
            tools=[WEATHER_TOOL],
            stream=stream,
            stream_options={"include_usage": True},
            max_tokens=4,
        ),
    )
    assert response.status_code == 200
    if stream:
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        choices = [c["choices"][0] for c in chunks if c["choices"]]
        calls = [call for c in choices for call in c["delta"].get("tool_calls", [])]
        content = "".join(c["delta"].get("content", "") for c in choices)
        finish = choices[-1]["finish_reason"]
        usage = chunks[-1]["usage"]
    else:
        choice = response.json()["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        content = choice["message"].get("content") or ""
        finish = choice["finish_reason"]
        usage = response.json()["usage"]
    assert len(calls) == int(tool_before_eos)
    assert content == ("" if tool_before_eos else "answer")
    assert finish == ("tool_calls" if tool_before_eos else "length")
    assert usage["completion_tokens"] == 4
    assert not runner.stopped


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("eos", ["<|im_end|>", "<|native_eos|>"])
def test_ignore_eos_reasoning_excludes_later_turns(make_client, stream, eos):
    def extract(text):
        reasoning, _, content = text.partition("</think>")
        return reasoning.removeprefix("<think>"), content

    client, runner = make_client(
        tokens=["<think>reason</think>answer", eos, "hidden"],
        default_stop_strings=[eos],
        reasoning_extractor=extract,
        finish_reason="length",
    )
    response = client.post(
        "/v1/chat/completions", json=_body(ignore_eos=True, stream=stream)
    )
    assert response.status_code == 200
    if stream:
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        deltas = [c["choices"][0]["delta"] for c in chunks]
        content = "".join(d.get("content", "") for d in deltas)
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    else:
        message = response.json()["choices"][0]["message"]
        content, reasoning = message["content"], message["reasoning_content"]
    assert content == "answer"
    assert reasoning == "reason"
    assert not runner.stopped


@pytest.mark.parametrize("eos", ["<|im_end|>", "<|native_eos|>"])
def test_ignore_eos_streaming_parser_never_receives_later_turn(make_client, eos):
    from executorch.examples.llm_server.python.protocol import DeltaMessage

    seen = []

    class Parser:
        def feed(self, text):
            seen.append(text)
            return [DeltaMessage(content=text)]

        def finish(self):
            return []

    client, runner = make_client(
        tokens=["answer", eos[:4], eos[4:], "hidden"],
        default_stop_strings=[eos],
        streaming_parser_factory=Parser,
        finish_reason="length",
    )
    response = client.post(
        "/v1/chat/completions", json=_body(ignore_eos=True, stream=True)
    )
    assert response.status_code == 200
    assert "".join(seen) == "answer"
    assert not runner.stopped


def test_ignore_eos_rejects_old_worker_before_context_validation(make_client):
    client, _ = make_client(supports_ignore_eos=False, max_context=1, prompt_tokens=10)
    response = client.post("/v1/chat/completions", json=_body(ignore_eos=True))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"


def test_ignore_eos_content_filter_preserves_explicit_stop_reason(make_client):
    client, runner = make_client(
        tokens=["answer", "<|im_end|>", "STOP", "hidden"],
        content_filter=lambda text: text.upper(),
        finish_reason="length",
    )
    response = client.post(
        "/v1/chat/completions",
        json=_body(
            ignore_eos=True,
            stream=True,
            stop="STOP",
            max_tokens=4,
        ),
    )
    assert response.status_code == 200
    chunks = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert (
        "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "ANSWER"
    )
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert runner.stopped


@pytest.mark.parametrize("stream", [False, True])
def test_ignore_eos_requires_reported_native_terminators(make_client, stream):
    client, runner = make_client(default_stop_strings=None)
    response = client.post(
        "/v1/chat/completions", json=_body(ignore_eos=True, stream=stream)
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert "native EOS markers" in response.json()["error"]["message"]
