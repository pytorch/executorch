# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import re

import pytest
from executorch.examples.llm_server.python.errors import ContextLengthExceeded


def muse_overflow_counts(body):
    error = body["error"]
    assert error["metadata"]["error_type"] == "context_length_exceeded"
    # Muse's OpenRouter contract: metadata classifies it, message gives the
    # input size and usable limit for its one-shot changed-input retry.
    match = re.match(
        r"This model's maximum context length is (\d+) tokens\. "
        r"However, your messages resulted in (\d+) tokens[. (]",
        error["message"],
    )
    return tuple(map(int, match.groups())) if match else None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "prompt,completion,budget",
    [(110, None, 99), (100, None, 99), (80, 30, 70), (110, 30, 70)],
)
def test_openrouter_http_overflow_supplies_actionable_prompt_counts(
    make_client, stream, prompt, completion, budget
):
    client, worker = make_client(
        max_context=100, prompt_tokens=prompt, openrouter_errors=True
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
            **({"max_tokens": completion} if completion else {}),
        },
    )
    assert response.status_code == 400
    assert muse_overflow_counts(response.json()) == (budget, prompt)
    assert response.json()["error"]["metadata"]["context_length"] == 100
    assert worker.captured_config is None


@pytest.mark.parametrize("prompt", [20, 110])
def test_impossible_output_reservation_does_not_invent_recoverable_counts(
    make_client, prompt
):
    client, _ = make_client(
        max_context=100, prompt_tokens=prompt, openrouter_errors=True
    )
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
    )
    assert response.status_code == 400
    assert muse_overflow_counts(response.json()) is None


def test_default_openai_error_contract_is_unchanged(make_client):
    client, _ = make_client(max_context=100, prompt_tokens=110)
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.json() == ContextLengthExceeded(110, 100).body()
    assert "metadata" not in response.json()["error"]


def test_late_context_error_has_same_openrouter_contract(make_client):
    client, worker = make_client(openrouter_errors=True)

    def fail(*args, **kwargs):
        raise ContextLengthExceeded(110, 100)

    worker.generate = fail
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert response.status_code == 200
    errors = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith('data: {"error"')
    ]
    assert len(errors) == 1
    assert muse_overflow_counts(errors[0]) == (99, 110)
    assert '"finish_reason"' not in response.text
