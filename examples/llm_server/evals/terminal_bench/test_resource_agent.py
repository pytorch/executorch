# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from harbor.models.agent.context import AgentContext

from .delegation import DelegationEstimate
from .resource_agent import ResourceAgent, SYSTEM, TOOLS

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
        {"role": "system", "content": SYSTEM},
        {
            "role": "assistant",
            "tool_calls": [
                call("spawn", question="inspect a", parent_work="inspect b")
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

    obj.completion, obj.close_session = completion, close
    await obj.parent_tool(messages[-1]["tool_calls"][0], messages, Environment())
    await asyncio.wait_for(child_started.wait(), 1)
    assert not obj.child.done()
    await obj.parent_tool(call("shell", command="inspect b"), messages, Environment())
    parent_work.set()
    await asyncio.wait_for(obj.child, 1)
    assert obj.notify(messages)
    assert not obj.notify(messages)
    assert closed == ["child"]
    assert obj.child_messages[: len(original)] == original
    assert messages[: len(original)] == original
    assert obj.child.result()["findings"] == "a is valid"


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
            call("spawn", question="a", parent_work="b"), [], Environment()
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
            call("spawn", question="inspect a", parent_work="inspect b"),
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
        report = json.loads(messages[-1]["content"].removeprefix("Resource report: "))
        assert report == {
            "available_children": 0,
            "admission_reason": "unknown_cost",
            "child_total_output_cap": 0,
        }
        message = (
            {
                "role": "assistant",
                "tool_calls": [call("spawn", question="a", parent_work="b")],
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
    spawn = call("spawn", question="inspect a", parent_work="inspect b")
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
async def test_stream_accounting_uses_reported_cache_and_identical_tools(tmp_path):
    obj = agent(tmp_path)

    def respond(request):
        body = json.loads(request.content)
        assert body["tools"] == TOOLS
        assert body["session_id"].endswith("-child")
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
        result = await obj.completion("child", messages, 10)
    assert result["message"]["content"] == "finding"
    assert obj.context.n_input_tokens - obj.context.n_cache_tokens == 20
    assert obj.output_tokens["child"] == 5


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
