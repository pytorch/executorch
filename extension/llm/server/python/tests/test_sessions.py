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
