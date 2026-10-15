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
from tokenizers import models, pre_tokenizers, Tokenizer

from .delegation import DelegationEstimate
from .resource_agent import ResourceAgent, SYSTEM
from .runner import resource_command

SNAPSHOT = {
    "max_inflight_requests": 4,
    "submitted_generations": 0,
    "waiting_for_runtime_lease": 0,
}
ESTIMATE = DelegationEstimate(30, 50, 1, 30, 12, 2)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def call(name, **arguments):
    if name == "spawn":
        arguments.setdefault("parent_output_tokens", 200)
        arguments.setdefault("inline_output_tokens", 1000)
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
                call(
                    "spawn",
                    question="inspect a",
                    parent_command="inspect b",
                    expected_output_tokens=200,
                )
            ],
        },
    ]
    original = copy.deepcopy(messages)

    async def report(turn):
        return SNAPSHOT

    obj.report = report
    obj.estimate_delegation = lambda messages, snapshot, proposal: replace(
        ESTIMATE, child_s=4
    )
    obj.estimate_thinking_tokens = lambda messages, snapshot, proposal: 300

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
    obj.child_expected_output_tokens = 20
    obj.child_expected_thinking_tokens = 30
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
    outcome = next(e for e in obj.records if e["type"] == "child_output")
    assert outcome["expected_output_tokens"] == 20
    assert outcome["expected_thinking_tokens"] == 30
    assert outcome["expected_total_tokens"] == 50
    assert outcome["actual_output_tokens"] == outcome["output_limit"] == 100
    assert outcome["status"] == "budget_exhausted"


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
    outcome = next(e for e in obj.records if e["type"] == "child_output")
    assert outcome["status"] == "cancelled"


@pytest.mark.anyio
async def test_inline_rejects_spawn_and_child_cannot_recurse(tmp_path):
    obj = agent(tmp_path)
    with pytest.raises(ValueError, match="No child budget"):
        await obj.parent_tool(
            call(
                "spawn",
                question="a",
                parent_command="cat b",
                expected_output_tokens=200,
            ),
            [],
            Environment(),
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
            call(
                "spawn",
                question="inspect a",
                parent_command="inspect b",
                expected_output_tokens=200,
            ),
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
                "tool_calls": [
                    call(
                        "spawn",
                        question="a",
                        parent_command="b",
                        expected_output_tokens=200,
                    )
                ],
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
async def test_launch_checks_proposal_and_preserves_configured_output_cap(tmp_path):
    obj = agent(tmp_path, enabled=True)
    proposals = []

    def estimate(messages, snapshot, proposal):
        proposals.append(proposal.copy())
        return ESTIMATE

    obj.estimate_delegation = estimate
    obj.estimate_thinking_tokens = lambda messages, snapshot, proposal: 50
    snapshot = SNAPSHOT | {"submitted_generations": 3}
    caps = []

    async def report(turn):
        return snapshot

    async def completion(role, messages, cap):
        caps.append(cap)
        obj.output_tokens[role] += 150
        return {
            "message": (
                {"tool_calls": [call("shell", command="cat a")]}
                if len(caps) == 1
                else {"content": "finding"}
            ),
            "finish_reason": "stop",
        }

    async def close(role):
        pass

    obj.report, obj.completion, obj.close_session = report, completion, close
    spawn = call(
        "spawn",
        question="inspect a",
        parent_command="inspect b",
        expected_output_tokens=200,
    )
    messages = [{"role": "assistant", "tool_calls": [spawn]}]
    with pytest.raises(ValueError, match="request_capacity"):
        await obj.parent_tool(spawn, messages, Environment())
    assert not obj.child_used
    snapshot = SNAPSHOT
    await obj.parent_tool(spawn, messages, Environment())
    await obj.child
    assert caps == [obj.child_output_tokens, obj.child_output_tokens - 150]
    assert obj.child.result()["status"] == "complete"
    assert proposals[-1] == {
        "question": "inspect a",
        "parent_command": "inspect b",
        "expected_output_tokens": 200,
        "parent_output_tokens": 200,
        "inline_output_tokens": 1000,
    }
    assert f"{obj.child_output_tokens} tokens" in obj.child_messages[1]["content"]
    outcomes = [e for e in obj.records if e["type"] == "child_output"]
    assert len(outcomes) == 1
    assert outcomes[0]["expected_output_tokens"] == 200
    assert outcomes[0]["expected_thinking_tokens"] == 50
    assert outcomes[0]["expected_total_tokens"] == 250
    assert outcomes[0]["actual_output_tokens"] == 300
    assert outcomes[0]["output_limit"] == obj.child_output_tokens
    assert outcomes[0]["status"] == "complete"


def test_thinking_is_unknown_without_observations_and_policy_can_be_overridden(
    tmp_path,
):
    obj = agent(tmp_path, enabled=True)
    obj.estimate_delegation = lambda messages, snapshot, proposal: ESTIMATE
    proposal = {
        "question": "inspect a",
        "parent_command": "inspect b",
        "expected_output_tokens": 200,
    }
    decision = obj.admission([], SNAPSHOT, proposal)
    assert not decision["allowed"]
    assert decision["reason"] == "unknown_cost"
    assert decision["expected_thinking_tokens"] is None
    assert decision["expected_total_tokens"] is None

    def thinking(messages, snapshot, proposed):
        assert snapshot is SNAPSHOT and proposed is proposal
        return 300

    obj.estimate_thinking_tokens = thinking
    decision = obj.admission([], SNAPSHOT, proposal)
    assert decision["allowed"]
    assert decision["expected_output_tokens"] == 200
    assert decision["expected_thinking_tokens"] == 300
    assert decision["expected_total_tokens"] == 500
    assert decision["expected_child_s"] == ESTIMATE.child_s


def test_thinking_average_pools_parent_and_child_tokens(tmp_path):
    obj = agent(tmp_path)
    proposal = {"expected_output_tokens": 11}
    assert obj.estimate_thinking_tokens([], SNAPSHOT, proposal) is None
    for role, thinking, total in [("parent", 10, 40), ("child", 40, 50)]:
        obj.observe_thinking_tokens(
            {
                "choices": [{"finish_reason": "stop", "message": {"content": "done"}}],
                "usage": {
                    "completion_tokens": total,
                    "completion_tokens_details": {"reasoning_tokens": thinking},
                },
            },
            role,
        )
    assert obj.observed_thinking_tokens == 50
    assert obj.observed_non_thinking_tokens == 40
    assert obj.estimate_thinking_tokens([], SNAPSHOT, proposal) == 14
    obj.observe_thinking_tokens(
        {
            "choices": [{"finish_reason": "stop", "message": {"content": "done"}}],
            "usage": {
                "completion_tokens": 10,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        },
        "parent",
    )
    assert obj.estimate_thinking_tokens([], SNAPSHOT, proposal) == 11


@pytest.mark.parametrize("thinking,expected", [(0, 0), (10, None)])
def test_thinking_average_handles_zero_counts(tmp_path, thinking, expected):
    obj = agent(tmp_path)
    obj.observe_thinking_tokens(
        {
            "choices": [{"finish_reason": "stop", "message": {"content": "done"}}],
            "usage": {
                "completion_tokens": 10,
                "completion_tokens_details": {"reasoning_tokens": thinking},
            },
        },
        "parent",
    )
    assert (
        obj.estimate_thinking_tokens([], SNAPSHOT, {"expected_output_tokens": 100})
        == expected
    )


@pytest.mark.parametrize(
    "thinking,finish",
    [(None, "stop"), (True, "stop"), (-1, "stop"), (11, "stop"), (8, "length")],
)
def test_unknown_invalid_or_capped_thinking_samples_are_ignored(
    tmp_path, thinking, finish
):
    obj = agent(tmp_path)
    obj.observe_thinking_tokens(
        {
            "choices": [{"finish_reason": finish, "message": {"content": "partial"}}],
            "usage": {
                "completion_tokens": 10,
                "completion_tokens_details": {"reasoning_tokens": thinking},
            },
        },
        "parent",
    )
    assert obj.observed_thinking_tokens == obj.observed_non_thinking_tokens == 0
    assert (
        obj.estimate_thinking_tokens([], SNAPSHOT, {"expected_output_tokens": 100})
        is None
    )


@pytest.mark.anyio
@pytest.mark.parametrize("reported_thinking", [None, 1])
async def test_stream_measures_thinking_with_model_tokenizer_or_usage(
    tmp_path, reported_thinking
):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "think": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    obj = agent(tmp_path, hf_tokenizer=str(tmp_path))

    def respond(request):
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "think think think",
                            "content": "finding",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "completion_tokens_details": {
                        "reasoning_tokens": reported_thinking
                    },
                },
            },
        ]
        return httpx.Response(
            200,
            text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
            + "data: [DONE]\n\n",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as obj.client:
        await obj.completion("parent", [{"role": "user", "content": "question"}], 10)
    thinking = 3 if reported_thinking is None else reported_thinking
    assert obj.observed_thinking_tokens == thinking
    assert obj.observed_non_thinking_tokens == 8 - thinking
    assert obj.estimate_thinking_tokens(
        [], SNAPSHOT, {"expected_output_tokens": 10}
    ) == (6 if reported_thinking is None else 2)
    observation = next(e for e in obj.records if e["type"] == "thinking_observation")
    assert observation["source"] == (
        "tokenizer" if reported_thinking is None else "usage"
    )


def test_resource_launcher_passes_configured_tokenizer(tmp_path):
    command = resource_command(
        {
            "host": "127.0.0.1",
            "port": 8000,
            "model_id": "test",
            "hf_tokenizer": str(tmp_path),
        },
        {
            "agent": "resource",
            "concurrency": 1,
            "attempts": 1,
            "max_output_tokens": 4096,
            "temperature": 0.8,
            "step_limit": 100,
            "child_output_tokens": 1536,
            "model_cost_path": str(tmp_path / "model-cost.json"),
        },
        tmp_path,
    )
    assert f"hf_tokenizer={tmp_path}" in command
    assert f"calibration_path={tmp_path / 'calibration.json'}" in command


def test_resource_launcher_requires_calibration_inputs(tmp_path):
    with pytest.raises(ValueError, match="model_cost_path"):
        resource_command({}, {"agent": "resource"}, tmp_path)


def test_agent_rejects_calibration_for_another_model(tmp_path):
    profile = tmp_path / "calibration.json"
    profile.write_text(json.dumps({"model_id": "different-model"}))
    with pytest.raises(ValueError, match="Calibration model"):
        agent(tmp_path, calibration_path=profile)


@pytest.mark.anyio
@pytest.mark.parametrize("forecast", [None, True, 0, 1537])
async def test_spawn_rejects_invalid_forecast_without_starting_work(tmp_path, forecast):
    obj = agent(tmp_path, enabled=True)

    async def report(turn):
        return SNAPSHOT

    def estimate(*args):
        pytest.fail("Invalid proposals must not reach the estimator")

    obj.report = report
    obj.estimate_delegation = estimate
    obj.estimate_thinking_tokens = estimate
    proposal = {"question": "inspect a", "parent_command": "inspect b"}
    if forecast is not None:
        proposal["expected_output_tokens"] = forecast
    with pytest.raises(
        ValueError, match="invalid_output_forecast|forecast_exceeds_budget"
    ):
        await obj.parent_tool(call("spawn", **proposal), [], Environment())
    assert obj.child is None and not obj.child_used
    assert not any(e["type"] == "tool" for e in obj.records)


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
        if role == "parent":
            parameters = body["tools"][1]["function"]["parameters"]
            assert "expected_output_tokens" in parameters["required"]
            assert (
                parameters["properties"]["expected_output_tokens"]["type"] == "integer"
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
async def test_child_prompt_count_uses_server_rendering(tmp_path):
    from .test_inference_cost import COST

    obj = agent(tmp_path)
    obj.cost_model = COST

    async def respond(request):
        assert request.url.path == "/v1/chat/tokenize"
        body = json.loads(request.content)
        assert "session_id" not in body
        assert body["messages"] == obj.child_prompt(
            "inspect a", obj.child_output_tokens
        )
        assert [tool["function"]["name"] for tool in body["tools"]] == ["shell"]
        return httpx.Response(200, json={"prompt_tokens": 123})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as obj.client:
        await obj.count_child_prompt("inspect a")
    assert obj.child_prompt_tokens == 123


@pytest.mark.anyio
async def test_reasoning_cap_error_keeps_consumed_token_accounting(tmp_path):
    obj = agent(tmp_path)
    chunk = {
        "choices": [
            {
                "index": 0,
                "delta": {"reasoning_content": "still thinking"},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 64, "total_tokens": 74},
    }

    async def respond(request):
        return httpx.Response(
            200, text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n"
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as obj.client:
        with pytest.raises(
            RuntimeError, match="Reasoning exhausted the 64-token response cap"
        ):
            await obj.completion("parent", [], 64)
    assert obj.context.n_output_tokens == 64
    assert obj.output_tokens["parent"] == 64


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


@pytest.mark.parametrize(
    "inline,shared,allowed",
    [(1000, False, True), (400, False, False), (400, True, True)],
)
def test_default_estimator_produces_admission_without_an_override(
    tmp_path, inline, shared, allowed
):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    obj = agent(tmp_path, enabled=True, hf_tokenizer=str(tmp_path))
    obj.observe_thinking_tokens(
        {
            "choices": [{"finish_reason": "tool_calls", "message": {}}],
            "usage": {
                "completion_tokens": 200,
                "completion_tokens_details": {"reasoning_tokens": 100},
            },
        },
        "parent",
    )
    from .inference_cost import AttentionGroup, InferenceCost, ModelCost

    obj.cost_model = InferenceCost(
        ModelCost(
            16_762_600_196,
            25_163_726_848,
            1_344_831_488,
            (
                AttentionGroup(13, 32, 2, 128, 2),
                AttentionGroup(39, 32, 2, 128, 2, 2048),
            ),
        ),
        438e9,
        11.64e12,
        1.0 if shared else 2.0,
        1.0,
        256,
        0.0,
    )
    obj.child_prompt_tokens = 200
    snapshot = SNAPSHOT | {
        "engine": {
            "max_decode_sequences": 2,
            "sessions": [
                {"session_id": obj.prefix + "-parent", "committed_position": 2048}
            ],
        }
    }
    proposal = {
        "question": "Inspect a",
        "parent_command": "cat b",
        "expected_output_tokens": 200,
        "parent_output_tokens": 200,
        "inline_output_tokens": inline,
    }
    decision = obj.admission([], snapshot, proposal)
    assert decision["allowed"] is allowed
    assert decision["reason"] != "unknown_cost"
    assert decision["estimate"]["prefill_s"] > 0
    assert decision["parent_delay_s"] > 0
