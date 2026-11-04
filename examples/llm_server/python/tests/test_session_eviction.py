# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import pytest
from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.openai_transcript import (
    OpenAITranscriptState,
)
from executorch.examples.llm_server.python.protocol import ChatMessage


def test_idle_session_eviction_releases_capacity_and_transcript(make_client):
    client, worker = make_client(
        max_named_sessions=2, evict_idle_sessions=True, gen_ids=[1]
    )

    def chat(sid):
        return client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "session_id": sid},
        )

    assert chat("a").status_code == chat("b").status_code == 200
    assert chat("a").status_code == 200  # a is newer than b
    assert chat("c").status_code == 200
    assert worker.open_named == {"a", "c"}
    serving = client.app.state.serving
    assert "b" not in serving._transcript._turns
    for i in range(20):
        assert chat(f"next-{i}").status_code == 200
    assert len(worker.open_named) == len(serving._idle_sessions) == 2


def test_history_identity_prevents_same_answer_collision():
    template = ChatTemplate(allow_fallback=True)
    state = OpenAITranscriptState(template)
    u1 = ChatMessage(role="user", content="removed question")
    u2 = ChatMessage(role="user", content="retained question")
    answer = ChatMessage(role="assistant", content="OK")
    for i, history in enumerate(([u1], [u1, answer, u2])):
        state.record_assistant_turn(
            session_id="s",
            request_messages=history,
            content="OK",
            tool_calls=None,
            generated_token_ids=[100 + i],
            prior_turns=i,
        )
    messages = [u2, answer, ChatMessage(role="user", content="continue")]
    rendered = template.render(messages)
    prompt = state.build_prompt_input(
        session_id="s",
        messages=messages,
        rendered_prompt=rendered,
        tools=None,
        template_kwargs=None,
    )
    assert prompt.text == rendered
    assert prompt.segments is None
    assert not state._turns["s"]


@pytest.mark.parametrize("changed_role", ["system", "developer", "user", "tool"])
def test_changed_preceding_context_invalidates_stored_reply(changed_role):
    template = ChatTemplate(allow_fallback=True)
    state = OpenAITranscriptState(template)
    history = [ChatMessage(role=changed_role, content="before")]
    state.record_assistant_turn(
        session_id="s",
        request_messages=history,
        content="OK",
        tool_calls=None,
        generated_token_ids=[123],
        prior_turns=0,
    )
    messages = [
        ChatMessage(role=changed_role, content="after"),
        ChatMessage(role="assistant", content="OK"),
    ]
    prompt = state.build_prompt_input(
        session_id="s",
        messages=messages,
        rendered_prompt=template.render(messages),
        tools=None,
        template_kwargs=None,
    )
    assert prompt.segments is None


def test_idle_eviction_protects_active_transactions(make_client):
    from executorch.examples.llm_server.python.protocol import ChatCompletionRequest

    client, worker = make_client(max_named_sessions=2, evict_idle_sessions=True)
    serving = client.app.state.serving

    async def scenario():
        def request(sid):
            return ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hi")], session_id=sid
            )

        await serving.create(request("a"))
        await serving.create(request("b"))
        async with serving._transactions.hold("a"):
            await serving.create(request("c"))
            assert worker.open_named == {"a", "c"}
            async with serving._transactions.hold("c"):
                assert not await serving._evict_idle_session()

    asyncio.run(scenario())
