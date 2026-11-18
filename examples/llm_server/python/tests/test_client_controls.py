# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import pytest


@pytest.mark.parametrize("dialect", ["openai", "openrouter"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [("thinking_token_budget", value) for value in (-2, 0.6, 10.5, 100, "100")]
    + [
        ("grammar", 'root ::= "ok"'),
        ("grammar", ""),
        ("guided_json", "x"),
        ("guided_regex", "x"),
        ("guided_choice", "x"),
        ("guided_grammar", "x"),
        ("structured_outputs", "x"),
        ("regex", "x"),
        ("ebnf", "x"),
        ("stop_regex", ".+"),
        ("add_generation_prompt", False),
        ("continue_final_message", True),
        ("cache_salt", "tenant-a"),
        ("cache_salt", ""),
    ],
)
def test_unsupported_controls_fail_before_generation(
    make_client, dialect, stream, field, value
):
    client, worker = make_client(api_dialect=dialect)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            field: value,
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unsupported_parameter"
    assert field in error["message"]
    assert worker.captured_config is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("include", [False, True])
def test_shared_reasoning_visibility_alias(make_client, stream, include):
    client, _ = make_client(
        tokens=["thought|answer"],
        reasoning_extractor=lambda text: tuple(text.split("|")),
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            "include_reasoning": include,
        },
    )
    assert response.status_code == 200
    if stream:
        messages = [
            json.loads(line[6:])["choices"][0]["delta"]
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
    else:
        messages = [response.json()["choices"][0]["message"]]
    assert "".join(m.get("content") or "" for m in messages) == "answer"
    assert "".join(m.get("reasoning_content") or "" for m in messages) == (
        "thought" if include else ""
    )


@pytest.mark.parametrize("value", [0, 1, "false", [], {}])
def test_reasoning_visibility_requires_boolean(make_client, value):
    client, worker = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "include_reasoning": value,
        },
    )
    assert response.status_code == 422
    assert worker.captured_config is None


def test_harmless_metadata_and_null_extensions_remain_accepted(make_client):
    client, _ = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "user": "caller",
            "store": False,
            "metadata": {"trace": "123"},
            "thinking_token_budget": None,
            "cache_salt": None,
            "include_reasoning": None,
        },
    )
    assert response.status_code == 200


@pytest.mark.parametrize("dialect", ["openai", "openrouter"])
@pytest.mark.parametrize("stream", [False, True])
def test_explicit_default_generation_controls_are_accepted(
    make_client, dialect, stream
):
    client, worker = make_client(api_dialect=dialect)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            "add_generation_prompt": True,
            "continue_final_message": False,
        },
    )
    assert response.status_code == 200
    assert worker.captured_config is not None


@pytest.mark.parametrize(
    "options",
    [
        {"include_reasoning": False, "reasoning": {"exclude": False}},
        {
            "chat_template_kwargs": {"return_reasoning": False},
            "reasoning": {"exclude": False},
        },
    ],
)
def test_shared_visibility_conflicts_are_rejected(make_client, options):
    client, worker = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], **options},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_value"
    assert worker.captured_config is None


@pytest.mark.parametrize("include", [None, False, True])
def test_unknown_model_has_consistent_error_precedence(make_client, include):
    client, worker = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "model": "missing",
            "temperature": -1,
            "include_reasoning": include,
        },
    )
    assert response.status_code == 404
    assert worker.captured_config is None


@pytest.mark.parametrize("dialect", ["openai", "openrouter"])
@pytest.mark.parametrize("stream", [False, True])
def test_shared_exclude_alias_hides_reasoning(make_client, dialect, stream):
    client, _ = make_client(
        api_dialect=dialect,
        tokens=["thought|answer"],
        reasoning_extractor=lambda text: tuple(text.split("|")),
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            "reasoning": {"exclude": True},
        },
    )
    assert response.status_code == 200
    assert "thought" not in response.text
    assert "answer" in response.text


@pytest.mark.parametrize("include", [None, False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_shared_null_exclude_is_unset(make_client, include, stream):
    client, _ = make_client(
        tokens=["thought|answer"],
        reasoning_extractor=lambda text: tuple(text.split("|")),
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            "reasoning": {"exclude": None},
            "include_reasoning": include,
        },
    )
    assert response.status_code == 200
    assert "answer" in response.text
    assert ("thought" in response.text) == (include is not False)
