# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Session-routing contract tests (fake worker, no model/GPU).

V2a: one worker hosts multiple isolated sessions, routed by session_id, admitted
up front so capacity refusals are HTTP statuses rather than mid-stream errors.
These assert the HTTP/wire contract only.
"""

import asyncio

import pytest

from executorch.extension.llm.server.python.chat_template import ChatTemplate
from executorch.extension.llm.server.python.errors import GenerationError
from executorch.extension.llm.server.python.serving_chat import ServingChat
from executorch.extension.llm.server.python.worker_client import WorkerError


def _chat(client, *, session_id=None, headers=None):
    body = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
    if session_id is not None:
        body["session_id"] = session_id
    return client.post("/v1/chat/completions", json=body, headers=headers or {})


def test_session_id_routed_and_opened(make_client):
    client, fake = make_client(max_named_sessions=2)
    resp = _chat(client, session_id="abc")
    assert resp.status_code == 200
    # Admitted before generation, and forwarded to the worker's generate().
    assert fake.opened_log == ["abc"]
    assert fake.captured_config.session_id == "abc"


def test_reusing_session_id_is_idempotent(make_client):
    client, fake = make_client(max_named_sessions=1)
    assert _chat(client, session_id="s").status_code == 200
    assert _chat(client, session_id="s").status_code == 200
    # Same id reused, not re-admitted into a second slot.
    assert fake.opened_log == ["s"]


def test_anonymous_request_does_not_open_named(make_client):
    client, fake = make_client(max_named_sessions=2)
    assert _chat(client).status_code == 200
    assert fake.opened_log == []
    assert fake.captured_config.session_id is None


def test_explicit_session_unsupported_when_single_session(make_client):
    # max_named_sessions=0: backend hosts only the scratch session.
    client, _ = make_client(max_named_sessions=0)
    resp = _chat(client, session_id="abc")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_session"


def test_capacity_exhausted_returns_429(make_client):
    client, _ = make_client(max_named_sessions=1)
    assert _chat(client, session_id="a").status_code == 200
    resp = _chat(client, session_id="b")  # second distinct id, no free slot
    assert resp.status_code == 429
    assert resp.json()["error"]["code"] == "capacity_exhausted"


def test_close_frees_a_slot(make_client):
    client, _ = make_client(max_named_sessions=1)
    assert _chat(client, session_id="a").status_code == 200
    deleted = client.delete("/v1/sessions/a")
    assert deleted.status_code == 200
    assert deleted.json() == {"closed": True, "session_id": "a"}
    # The freed slot now admits a different session.
    assert _chat(client, session_id="b").status_code == 200


def test_invalid_session_id_rejected_before_worker(make_client):
    client, fake = make_client(max_named_sessions=2)
    resp = _chat(client, session_id="has space")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_session_id"
    assert fake.opened_log == []  # never reached the worker


def test_session_id_too_long_rejected(make_client):
    client, _ = make_client(max_named_sessions=2)
    resp = _chat(client, session_id="x" * 129)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_session_id"


def test_session_id_header_alias(make_client):
    client, fake = make_client(max_named_sessions=2)
    resp = _chat(client, headers={"X-ExecuTorch-Session-ID": "from-header"})
    assert resp.status_code == 200
    assert fake.opened_log == ["from-header"]


def test_body_session_id_wins_over_header(make_client):
    client, fake = make_client(max_named_sessions=2)
    resp = _chat(
        client, session_id="from-body", headers={"X-ExecuTorch-Session-ID": "from-hdr"}
    )
    assert resp.status_code == 200
    assert fake.opened_log == ["from-body"]


def test_session_id_underscore_header_alias(make_client):
    # pi's sendSessionAffinityHeaders emits a verbatim `session_id` header.
    client, fake = make_client(max_named_sessions=2)
    resp = _chat(client, headers={"session_id": "from-session-id"})
    assert resp.status_code == 200
    assert fake.opened_log == ["from-session-id"]


def test_x_session_affinity_header_alias(make_client):
    client, fake = make_client(max_named_sessions=2)
    resp = _chat(client, headers={"x-session-affinity": "from-affinity"})
    assert resp.status_code == 200
    assert fake.opened_log == ["from-affinity"]


def test_session_header_precedence(make_client):
    # X-ExecuTorch-Session-ID > session_id > x-session-affinity (no body field).
    client, fake = make_client(max_named_sessions=3)
    resp = _chat(
        client,
        headers={
            "X-ExecuTorch-Session-ID": "xet",
            "session_id": "sid",
            "x-session-affinity": "aff",
        },
    )
    assert resp.status_code == 200
    assert fake.opened_log == ["xet"]


def _chat_msgs(client, messages, session_id):
    return client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "session_id": session_id, "messages": messages},
    )


# The fake worker streams tokens ("Hello", ", ", "world"), so the assistant
# content we return (and the client must echo back to match the fingerprint) is:
_FAKE_REPLY = "Hello, world"


def test_token_id_segments_splice_prior_assistant_turn(make_client):
    # V2b.1.5: the server stores turn-1's generated ids and, on turn 2, sends
    # prompt_segments that splice them back as an exact {ids} run (not text) --
    # but only because the client echoes back the assistant turn we generated.
    client, fake = make_client(max_named_sessions=2, gen_ids=[7, 8, 9])
    assert (
        _chat_msgs(client, [{"role": "user", "content": "hi"}], "s").status_code == 200
    )
    # First turn has no prior assistant turn -> plain text prompt.
    assert fake.captured_config.prompt_segments is None

    turn2 = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": _FAKE_REPLY},  # matches what we returned
        {"role": "user", "content": "more"},
    ]
    assert _chat_msgs(client, turn2, "s").status_code == 200
    segs = fake.captured_config.prompt_segments
    assert segs is not None, "expected token-ID segments on the second turn"
    # The stored generated ids are spliced in as an exact id run...
    assert any(s.get("ids") == [7, 8, 9] for s in segs)
    # ...bracketed by text segments (template glue + the new user turn).
    assert any("text" in s for s in segs)


def test_edited_assistant_turn_not_spliced(make_client):
    # P1 guard: if the client edits a prior assistant turn (or reuses the session
    # for a different conversation), the stale ids must NOT be spliced -- the
    # fingerprint mismatches and we fall back to text.
    client, fake = make_client(max_named_sessions=2, gen_ids=[7, 8, 9])
    _chat_msgs(client, [{"role": "user", "content": "hi"}], "s")
    turn2 = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "EDITED - not what the model generated"},
        {"role": "user", "content": "more"},
    ]
    assert _chat_msgs(client, turn2, "s").status_code == 200
    assert fake.captured_config.prompt_segments is None


def test_stop_trimmed_turn_not_spliced(make_client):
    # P1/P2 guard: a stop-trimmed turn (worker omits generated_token_ids ->
    # recorded ids=None) is never spliced, even when the turn fingerprint matches,
    # so unseen post-stop tokens can't be injected into a later prompt.
    client, fake = make_client(max_named_sessions=2, gen_ids=[])  # [] => ids None
    _chat_msgs(client, [{"role": "user", "content": "hi"}], "s")
    turn2 = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": _FAKE_REPLY},  # fingerprint matches
        {"role": "user", "content": "more"},
    ]
    assert _chat_msgs(client, turn2, "s").status_code == 200
    assert fake.captured_config.prompt_segments is None


def test_no_segments_for_anonymous_requests(make_client):
    client, fake = make_client(max_named_sessions=2, gen_ids=[1, 2])
    client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert fake.captured_config.prompt_segments is None


def test_reset_clears_stored_ids(make_client):
    # After reset, the next turn has no stored ids to splice -> plain text again.
    client, fake = make_client(max_named_sessions=2, gen_ids=[5, 6])
    _chat_msgs(client, [{"role": "user", "content": "hi"}], "s")
    client.post("/v1/sessions/s/reset")
    turn2 = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "more"},
    ]
    _chat_msgs(client, turn2, "s")
    assert fake.captured_config.prompt_segments is None


def test_reset_endpoint_clears_context_but_keeps_slot(make_client):
    # max_named=1: open "a", reset it, then a *different* id must still 429 —
    # proving reset cleared context without freeing the slot (unlike DELETE).
    client, fake = make_client(max_named_sessions=1)
    assert _chat(client, session_id="a").status_code == 200
    r = client.post("/v1/sessions/a/reset")
    assert r.status_code == 200
    assert r.json() == {"reset": True, "session_id": "a"}
    assert fake.reset_log == ["a"]
    assert _chat(client, session_id="b").status_code == 429  # slot still held


def test_reset_invalid_session_id_rejected(make_client):
    client, _ = make_client(max_named_sessions=2)
    r = client.post("/v1/sessions/has%20space/reset")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_session_id"


class _RaisingRuntime:
    """Runtime whose worker ops fail, to exercise the lockstep invariant."""

    async def open(self, sid):
        pass

    async def reset(self, sid):
        raise WorkerError("worker down")

    async def close(self, sid):
        raise WorkerError("worker down")


@pytest.mark.parametrize("op", ["reset_session", "close_session"])
def test_worker_op_failure_keeps_transcript(op):
    # Lockstep invariant: if the worker reset/close fails, the adapter transcript
    # must NOT be cleared -- both retain old state so they never drift.
    template = ChatTemplate(hf_tokenizer_path=None, allow_fallback=True)
    serving = ServingChat(_RaisingRuntime(), template, "test-model")
    serving._transcript.record_assistant_turn(
        session_id="s",
        content="hi",
        tool_calls=None,
        generated_token_ids=[1, 2],
        prior_turns=0,
    )

    async def go():
        with pytest.raises(GenerationError):
            await getattr(serving, op)("s")

    asyncio.run(go())
    assert serving._transcript._turns.get(
        "s"
    ), "transcript cleared despite worker failure"


def test_record_assistant_turn_replaces_stale_at_position():
    # A regenerated/branched turn under the same session_id must REPLACE the
    # record at its position (prior_turns), not append, so a later turn can still
    # splice the regenerated ids instead of breaking on a stale fingerprint.
    from executorch.extension.llm.server.python.openai_transcript import (
        OpenAITranscriptState,
    )

    t = OpenAITranscriptState(ChatTemplate(hf_tokenizer_path=None, allow_fallback=True))
    t.record_assistant_turn(
        session_id="s",
        content="a0",
        tool_calls=None,
        generated_token_ids=[1],
        prior_turns=0,
    )
    t.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=[2],
        prior_turns=1,
    )
    assert [r["ids"] for r in t._turns["s"]] == [[1], [2]]
    # regenerate turn 2 (same prior_turns) -> replaces stale [2], no stale tail
    t.record_assistant_turn(
        session_id="s",
        content="a1b",
        tool_calls=None,
        generated_token_ids=[3],
        prior_turns=1,
    )
    assert [r["ids"] for r in t._turns["s"]] == [[1], [3]]


def test_divergence_truncates_stale_tail():
    # Editing an EARLIER assistant turn (divergence at k) prunes the stale tail
    # from k so it can't keep shadowing future requests; nothing is spliced and
    # the matched prefix is kept. (Restoring hits for the edited turn isn't
    # possible -- we never generated its ids -- but staleness is bounded.)
    from executorch.extension.llm.server.python.openai_transcript import (
        OpenAITranscriptState,
    )
    from executorch.extension.llm.server.python.protocol import ChatMessage

    t = OpenAITranscriptState(ChatTemplate(hf_tokenizer_path=None, allow_fallback=True))
    t.record_assistant_turn(
        session_id="s",
        content="a0",
        tool_calls=None,
        generated_token_ids=[1],
        prior_turns=0,
    )
    t.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=[2],
        prior_turns=1,
    )
    msgs = [
        ChatMessage(role="user", content="u0"),
        ChatMessage(role="assistant", content="a0-EDITED"),
        ChatMessage(role="user", content="u1"),
    ]
    out = t.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt="X",
        tools=None,
        template_kwargs=None,
    )
    assert out.text == "X"  # diverged -> plain text fallback
    assert t._turns["s"] == []  # stale tail pruned from the first mismatch


class _HFToolSpecials:
    # Tokenizer that marks a turn terminator AND tool/structural delimiters special.
    all_special_tokens = ["<|im_end|>", "<tool_call>", "</tool_call>", "<|box_start|>"]
    eos_token = "<|im_end|>"


def test_stop_set_narrow_but_strip_set_broad():
    # Two-set split (work item 1): the generation/pre-parse-truncation set is
    # NARROW (turn terminators only) so a <tool_call> is never halted or cut
    # before the parser sees it; the final content-strip set stays BROAD so stray
    # specials can't leak into visible content.
    from types import SimpleNamespace

    template = ChatTemplate(hf_tokenizer_path=None, allow_fallback=True)
    template._hf = _HFToolSpecials()
    serving = ServingChat(_RaisingRuntime(), template, "test-model")

    assert "<|im_end|>" in serving._stops  # terminator kept
    assert "<tool_call>" not in serving._stops  # delimiter excluded
    assert "</tool_call>" not in serving._stops
    assert "<|box_start|>" not in serving._stops

    assert "<tool_call>" in serving._content_specials  # broad strip keeps it
    assert "<|box_start|>" in serving._content_specials

    # The insidious site: _truncate_raw must NOT cut at <tool_call> (it uses the
    # narrow set), so the full tool-call markup survives to the parser.
    raw = (
        "sure<tool_call>\n<function=f>\n<parameter=x>\n1\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    assert serving._truncate_raw(raw, SimpleNamespace(stop=None)) == raw


def test_injected_assistant_exemplar_falls_back_to_text():
    # 5d: a client-injected assistant turn we never generated (few-shot exemplar /
    # pre-seeded turn) shifts the ordinal alignment -> fingerprint mismatch ->
    # safe text fallback (no stale ids spliced).
    from executorch.extension.llm.server.python.openai_transcript import (
        OpenAITranscriptState,
    )
    from executorch.extension.llm.server.python.protocol import ChatMessage

    t = OpenAITranscriptState(ChatTemplate(hf_tokenizer_path=None, allow_fallback=True))
    t.record_assistant_turn(
        session_id="s",
        content="a0",
        tool_calls=None,
        generated_token_ids=[1, 2],
        prior_turns=0,
    )
    msgs = [
        ChatMessage(role="user", content="u0"),
        ChatMessage(role="assistant", content="INJECTED EXEMPLAR"),  # not ours
        ChatMessage(role="user", content="u1"),
    ]
    out = t.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt="X",
        tools=None,
        template_kwargs=None,
    )
    assert out.text == "X" and out.segments is None  # safe text fallback
