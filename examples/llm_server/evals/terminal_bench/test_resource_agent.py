# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import copy
import json
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from harbor.models.agent.context import AgentContext

from .delegation import DelegationEstimate
from .resource_agent import ResourceAgent, SYSTEM

SNAPSHOT = {
    "max_inflight_requests": 4,
    "submitted_generations": 0,
    "waiting_for_runtime_lease": 0,
}
ESTIMATE = DelegationEstimate(
    uncached_input_tokens=200,
    output_tokens=200,
    prefill_tokens_s=100,
    decode_tokens_s=20,
    child_tool_s=0,
    parent_work_s=30,
    inline_s=50,
    integration_s=1,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def call(name, **arguments):
    return {
        "id": name,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class Environment:
    async def exec(self, command, timeout_sec):
        return SimpleNamespace(return_code=0, stdout="evidence", stderr="")


def agent(tmp_path, **kwargs):
    obj = ResourceAgent(
        logs_dir=tmp_path, model_name="openai/test", base_url="http://test/v1", **kwargs
    )
    obj.context = AgentContext(n_input_tokens=0, n_cache_tokens=0, n_output_tokens=0)
    return obj


@pytest.mark.anyio
async def test_background_child_overlaps_parent_and_releases_session(tmp_path):
    obj = agent(tmp_path, enabled=True)
    child_started = asyncio.Event()
    parent_work = asyncio.Event()
    closed = []
    messages = [
        {"role": "system", "content": SYSTEM + " parent-only context" * 10000},
        {
            "role": "assistant",
            "tool_calls": [
                call("spawn", question="inspect a", parent_command="inspect b")
            ],
        },
    ]
    original = copy.deepcopy(messages)

    async def report(turn):
        return SNAPSHOT

    obj.report = report
    obj.estimate_delegation = lambda messages, snapshot: replace(
        ESTIMATE, output_tokens=obj.child_output_tokens, decode_tokens_s=100
    )

    async def completion(role, history, cap):
        assert role == "child" and cap == obj.child_output_tokens
        child_started.set()
        await parent_work.wait()
        return {"message": {"content": "a is valid"}, "finish_reason": "stop"}

    async def close(role):
        closed.append(role)

    class ParentEnvironment:
        async def exec(self, command, timeout_sec):
            assert command == "inspect b"
            await child_started.wait()
            assert not obj.child.done()
            parent_work.set()
            return SimpleNamespace(return_code=0, stdout="b is valid", stderr="")

    obj.completion, obj.close_session = completion, close
    output = await asyncio.wait_for(
        obj.parent_tool(messages[-1]["tool_calls"][0], messages, ParentEnvironment()), 1
    )
    assert "b is valid" in output
    await asyncio.wait_for(obj.child, 1)
    assert obj.notify(messages)
    assert not obj.notify(messages)
    assert closed == ["child"]
    assert len(obj.child_messages) == 2
    assert "parent-only context" not in json.dumps(obj.child_messages)
    assert "inspect a" in obj.child_messages[-1]["content"]
    assert messages[: len(original)] == original
    assert obj.child.result()["findings"] == "a is valid"


@pytest.mark.parametrize("question", ["x" * 4096, "é" * 2048])
def test_child_assignment_byte_limit(question):
    messages = ResourceAgent.child_prompt(question, 100)
    assert len(messages) == 2
    assert messages[-1]["content"].startswith(question)
    assert "100 tokens" in messages[-1]["content"]
    with pytest.raises(ValueError, match="4096 bytes"):
        ResourceAgent.child_prompt(question + "x", 100)


@pytest.mark.anyio
async def test_child_omits_output_that_exceeds_context_budget(tmp_path):
    obj = agent(tmp_path, enabled=True)
    obj.child_messages = obj.child_prompt("inspect a", 100)
    requests = []

    async def completion(role, messages, cap):
        assert len(json.dumps(messages).encode()) <= 16384
        requests.append(copy.deepcopy(messages))
        message = (
            {"role": "assistant", "tool_calls": [call("shell", command="cat a")]}
            if len(requests) == 1
            else {"role": "assistant", "content": "Need a narrower inspection"}
        )
        messages.append(message)
        return {"message": message, "finish_reason": "stop"}

    class LargeOutputEnvironment:
        async def exec(self, command, timeout_sec):
            return SimpleNamespace(return_code=0, stdout="evidence" * 4000, stderr="")

    obj.completion = completion
    result = await obj.investigate(LargeOutputEnvironment())
    assert result["status"] == "complete"
    assert "Output omitted" in requests[-1][-1]["content"]
    assert "narrow the command" in requests[-1][-1]["content"]


@pytest.mark.anyio
async def test_child_stops_before_requesting_an_oversized_context(tmp_path):
    obj = agent(tmp_path, enabled=True)
    obj.child_messages = [{"role": "user", "content": "x" * 16384}]
    result = await obj.investigate(Environment())
    assert result["status"] == "context_budget_exhausted"
    assert not obj.sessions


@pytest.mark.anyio
async def test_run_keeps_telemetry_out_of_prompt_history(tmp_path):
    obj = agent(tmp_path)
    turns, requests = [], []

    async def report(turn):
        turns.append(turn)
        return {"submitted_generations": 900 + len(turns)}

    async def completion(role, messages, cap):
        requests.append(copy.deepcopy(messages))
        message = (
            {"role": "assistant", "tool_calls": [call("shell", command="ls")]}
            if len(requests) < 3
            else {"role": "assistant", "content": "done"}
        )
        messages.append(message)
        return {"message": message, "finish_reason": "stop"}

    obj.report, obj.completion = report, completion
    await obj.run("inspect a", Environment(), obj.context)
    assert turns == [0, 1, 2, "final"]
    for messages in requests:
        assert [m for m in messages if m["role"] == "user"] == [
            {"role": "user", "content": "inspect a"}
        ]
        assert "Delegation limit: 0 child" in messages[0]["content"]
        assert "submitted_generations" not in json.dumps(messages)


@pytest.mark.anyio
async def test_child_total_budget_limits_later_requests(tmp_path):
    obj = agent(tmp_path, enabled=True, child_output_tokens=100)
    caps = []

    async def completion(role, messages, cap):
        caps.append(cap)
        obj.output_tokens[role] += min(60, cap)
        return {
            "finish_reason": "stop",
            "message": {"tool_calls": [call("shell", command="ls")]},
        }

    async def close(role):
        pass

    obj.completion, obj.close_session = completion, close
    result = await obj.investigate(Environment())
    assert caps == [100, 40]
    assert result["status"] == "budget_exhausted"


@pytest.mark.anyio
async def test_cancel_waits_for_child_cleanup(tmp_path):
    obj = agent(tmp_path, enabled=True)
    started, closed = asyncio.Event(), asyncio.Event()

    async def completion(*args):
        started.set()
        await asyncio.Future()

    async def close(role):
        await asyncio.sleep(0)
        closed.set()

    obj.completion, obj.close_session = completion, close
    obj.child = asyncio.create_task(obj.investigate(Environment()))
    await started.wait()
    result = await obj.parent_tool(call("cancel"), [], Environment())
    assert json.loads(result)["status"] == "cancelled"
    assert closed.is_set() and obj.child.done()


@pytest.mark.anyio
async def test_inline_rejects_spawn_and_child_cannot_recurse(tmp_path):
    obj = agent(tmp_path)
    with pytest.raises(ValueError, match="No child budget"):
        await obj.parent_tool(
            call("spawn", question="a", parent_command="cat b"), [], Environment()
        )
    assert not obj.child_used


@pytest.mark.anyio
async def test_enabled_agent_rejects_unknown_cost_without_consuming_child(tmp_path):
    obj = agent(tmp_path, enabled=True)

    async def report(turn):
        return SNAPSHOT

    obj.report = report
    with pytest.raises(ValueError, match="unknown_cost"):
        await obj.parent_tool(
            call("spawn", question="inspect a", parent_command="inspect b"),
            [],
            Environment(),
        )
    assert obj.child is None and not obj.child_used
    assert not obj.sessions


@pytest.mark.anyio
async def test_run_hides_rates_and_declines_a_forced_spawn(tmp_path):
    obj = agent(tmp_path, enabled=True)
    requests = []

    async def report(turn):
        return SNAPSHOT | {"last_observation": {"prefill_tokens_per_second": 100}}

    async def completion(role, messages, cap):
        requests.append(role)
        assert "Resource report:" not in json.dumps(messages)
        assert "prefill_tokens_per_second" not in json.dumps(messages)
        message = (
            {
                "role": "assistant",
                "tool_calls": [call("spawn", question="a", parent_command="b")],
            }
            if len(requests) == 1
            else {"role": "assistant", "content": "finished inline"}
        )
        messages.append(message)
        return {"message": message, "finish_reason": "stop"}

    obj.report, obj.completion = report, completion
    await obj.run("task", Environment(), obj.context)
    assert requests == ["parent", "parent"]
    assert obj.child is None and not obj.child_used
    history = json.loads((tmp_path / "parent.json").read_text())
    assert any(
        m["role"] == "tool" and "Delegation declined: unknown_cost" in m["content"]
        for m in history
    )


@pytest.mark.anyio
async def test_launch_rechecks_capacity_and_enforces_estimated_output_cap(tmp_path):
    obj = agent(tmp_path, enabled=True)
    obj.estimate_delegation = lambda messages, snapshot: ESTIMATE
    assert obj.admission([], SNAPSHOT, "report")["allowed"]
    snapshot = SNAPSHOT | {"submitted_generations": 3}
    caps = []

    async def report(turn):
        return snapshot

    async def completion(role, messages, cap):
        caps.append(cap)
        return {"message": {"content": "finding"}, "finish_reason": "stop"}

    async def close(role):
        pass

    obj.report, obj.completion, obj.close_session = report, completion, close
    spawn = call("spawn", question="inspect a", parent_command="inspect b")
    messages = [{"role": "assistant", "tool_calls": [spawn]}]
    with pytest.raises(ValueError, match="request_capacity"):
        await obj.parent_tool(spawn, messages, Environment())
    assert not obj.child_used
    snapshot = SNAPSHOT
    await obj.parent_tool(spawn, messages, Environment())
    await obj.child
    assert caps == [ESTIMATE.output_tokens]
    assert obj.child_budget == ESTIMATE.output_tokens


@pytest.mark.anyio
@pytest.mark.parametrize("role", ["parent", "child"])
async def test_stream_accounting_and_role_tools(tmp_path, role):
    obj = agent(tmp_path)

    def respond(request):
        body = json.loads(request.content)
        names = [tool["function"]["name"] for tool in body["tools"]]
        assert names == (
            ["shell"] if role == "child" else ["shell", "spawn", "wait", "cancel"]
        )
        assert body["session_id"].endswith("-" + role)
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "finding"},
                        "finish_reason": "stop",
                    }
                ]
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 80},
                },
            },
        ]
        return httpx.Response(
            200,
            text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
            + "data: [DONE]\n\n",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as obj.client:
        messages = [{"role": "user", "content": "question"}]
        result = await obj.completion(role, messages, 10)
    assert result["message"]["content"] == "finding"
    assert obj.context.n_input_tokens - obj.context.n_cache_tokens == 20
    assert obj.output_tokens[role] == 5


@asynccontextmanager
async def buffered_server():
    ready, release = asyncio.Event(), asyncio.Event()
    handlers, deleted = set(), []
    snapshot = {
        "max_context": 4096,
        "max_inflight_requests": 4,
        "submitted_generations": 0,
        "waiting_for_runtime_lease": 0,
        "recent_generations": [],
    }
    role = 'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
    result = (
        'data: {"choices":[{"index":0,"delta":{"content":"finding"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":10,"completion_tokens":1}}\n\ndata: [DONE]\n\n'
    )

    async def handle(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            headers = (await reader.readuntil(b"\r\n\r\n")).decode()
            method, path, _ = headers.splitlines()[0].split()
            length = next(
                (
                    int(line.split(":", 1)[1])
                    for line in headers.splitlines()
                    if line.lower().startswith("content-length:")
                ),
                0,
            )
            await reader.readexactly(length)
            if path.endswith("/chat/completions"):
                body = (role + result).encode()
                writer.write(
                    f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                    + role.encode()
                )
                await writer.drain()
                ready.set()
                await release.wait()
                writer.write(result.encode())
            else:
                if path == "/stall":
                    await release.wait()
                if method == "DELETE":
                    deleted.append(path)
                body = json.dumps(
                    snapshot if path.endswith("/runtime/stats") else {}
                ).encode()
                writer.write(
                    f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                    + body
                )
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            handlers.remove(task)

    async with await asyncio.start_server(handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        try:
            yield SimpleNamespace(
                url=f"http://127.0.0.1:{port}/v1",
                ready=ready,
                release=release,
                deleted=deleted,
            )
        finally:
            release.set()
            await asyncio.gather(*handlers, return_exceptions=True)


@pytest.mark.anyio
async def test_buffered_generation_outlives_idle_timeout(tmp_path):
    obj = agent(tmp_path)
    async with buffered_server() as server:
        obj.base_url = server.url
        async with httpx.AsyncClient(timeout=0.1, trust_env=False) as obj.client:
            pending = asyncio.create_task(
                obj.completion("parent", [{"role": "user", "content": "task"}], 10)
            )
            try:
                await asyncio.wait_for(server.ready.wait(), 1)
                with pytest.raises(httpx.ReadTimeout):
                    await obj.client.get(server.url.removesuffix("/v1") + "/stall")
                await asyncio.sleep(0.05)
                assert not pending.done()
                server.release.set()
                result = await asyncio.wait_for(pending, 1)
                assert result["message"]["content"] == "finding"
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.anyio
async def test_task_deadline_cancels_buffered_generation_and_closes_session(tmp_path):
    obj = agent(tmp_path)
    async with buffered_server() as server:
        obj.base_url = server.url
        pending = asyncio.create_task(obj.run("task", Environment(), obj.context))
        await asyncio.wait_for(server.ready.wait(), 1)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(pending, 0.05)
        assert server.ready.is_set()
        assert server.deleted == ["/v1/sessions/" + obj.prefix + "-parent"]
        assert not obj.sessions
        assert obj.context.metadata["status"] == "error"


@pytest.mark.anyio
async def test_cancel_before_child_starts_is_not_a_trial_failure(tmp_path):
    obj = agent(tmp_path, enabled=True)

    async def pending():
        await asyncio.Future()

    obj.child = asyncio.create_task(pending())
    result = await obj.parent_tool(call("cancel"), [], Environment())
    assert json.loads(result)["status"] == "cancelled"


@pytest.mark.anyio
async def test_cleanup_error_preserves_primary_error_and_trace(tmp_path):
    obj = agent(tmp_path)

    async def report(turn):
        return {}

    async def completion(*args):
        raise RuntimeError("primary request failure")

    async def close(role):
        raise RuntimeError("cleanup failure")

    obj.report, obj.completion, obj.close_session = report, completion, close
    with pytest.raises(RuntimeError, match="primary request failure"):
        await obj.run("task", Environment(), obj.context)
    assert (
        json.loads((tmp_path / "summary.json").read_text())["metadata"]["status"]
        == "error"
    )
    assert any(e["type"] == "cleanup_error" for e in obj.records)
