# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preserve unknown versus known-empty generated IDs across the serving stack."""

import json
import sys
from contextlib import asynccontextmanager

import pytest

from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.multiplexed_worker_client import (
    spawn_multiplexed_worker,
)
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import (
    GenStats,
    SessionRuntime,
)
from executorch.examples.llm_server.python.worker_client import spawn_worker
from fastapi.testclient import TestClient


_WORKER = r"""
import json, os, sys
multiplexed = sys.argv[1] == 'True'
known_empty = sys.argv[2] == 'eos'
fd = os.environ.get('EXECUTORCH_LLM_WORKER_CONTROL_FD')
if fd is not None:
    os.close(int(fd))
print(json.dumps(dict(ready=True, multiplexed=multiplexed, max_named_sessions=2, max_inflight_requests=2)), flush=True)
def send(request, **fields):
    if multiplexed:
        fields['request_id'] = request['request_id']
    print(json.dumps(fields), flush=True)
turn = 0
for line in sys.stdin:
    request = json.loads(line)
    if request.get('op') == 'open':
        send(request, opened=True)
        continue
    turn += 1
    if turn == 1:
        assert 'prompt_segments' not in request
        assert 'STOP' in request['stop']
        # Both terminal-only EOS and an entirely trimmed string stop have no
        # visible text; only explicit ID metadata authorizes warm splicing.
        metadata = dict(generated_token_ids=[]) if known_empty else {}
        send(request, done=True, finish_reason='stop', completion_tokens=0, **metadata)
    else:
        assert turn == 2
        if known_empty:
            assert dict(ids=[]) in request['prompt_segments']
        else:
            assert 'prompt_segments' not in request
            assert request['prompt']
        send(request, token='next')
        send(request, done=True, finish_reason='stop', completion_tokens=1, generated_token_ids=[7])
"""


@pytest.mark.parametrize("multiplexed", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("completion", ["string_stop", "eos"])
def test_empty_visible_completion_preserves_id_presence_through_http_runtime_transcript(
    monkeypatch, multiplexed, stream, completion
):
    command = [sys.executable, "-u", "-c", _WORKER, str(multiplexed), completion]
    runtime = None
    serving = None
    generations = []

    def configure(worker):
        nonlocal runtime, serving
        runtime = SessionRuntime(worker)
        serving = ServingChat(
            runtime,
            ChatTemplate(hf_tokenizer_path=None, allow_fallback=True),
            "test-model",
        )
        generate_stream = runtime.generate_stream

        def observe_generation(*args, **kwargs):
            generation = generate_stream(*args, **kwargs)
            generations.append(generation)
            return generation

        monkeypatch.setattr(runtime, "generate_stream", observe_generation)
        return serving

    @asynccontextmanager
    async def native_serving():
        worker = await spawn_multiplexed_worker(command)
        try:
            yield configure(worker)
        finally:
            if runtime is not None:
                await runtime.aclose_worker()
            else:
                await worker.close()

    if multiplexed:
        app = build_app(None, "test-model", serving_factory=native_serving)
    else:
        app = build_app(configure(spawn_worker(command)), "test-model")
    try:
        assert GenStats().generated_token_ids is None
        with TestClient(app) as client:
            messages = [{"role": "user", "content": "hi"}]
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "session_id": "s",
                    "messages": messages,
                    "stop": "STOP",
                    "stream": stream,
                },
            )
            assert response.status_code == 200
            assert "generated_token_ids" not in response.text
            if stream:
                chunks = [
                    json.loads(line[5:])
                    for line in response.text.splitlines()
                    if line.startswith("data:") and "[DONE]" not in line
                ]
                choices = [
                    choice for chunk in chunks for choice in chunk.get("choices", [])
                ]
                visible = "".join(
                    choice.get("delta", {}).get("content") or "" for choice in choices
                )
                assert any(choice.get("finish_reason") == "stop" for choice in choices)
                assert response.text.rstrip().endswith("data: [DONE]")
            else:
                choice = response.json()["choices"][0]
                assert "content" not in choice["message"]
                visible = choice["message"].get("content") or ""
                assert choice["finish_reason"] == "stop"
            assert visible == ""
            expected_ids = [] if completion == "eos" else None
            assert generations[0].stats.generated_token_ids == expected_ids
            assert serving._transcript._turns["s"][0]["ids"] == expected_ids

            messages += [
                {"role": "assistant", "content": visible},
                {"role": "user", "content": "continue"},
            ]
            successor = client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "session_id": "s", "messages": messages},
            )
            assert successor.status_code == 200
            assert successor.json()["choices"][0]["message"]["content"] == "next"
            assert runtime.healthy
    finally:
        if not multiplexed:
            runtime.close_worker()
