# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json

import litellm

from executorch.examples.llm_server.evals.terminal_bench.cliff_agent import (
    compact,
    estimate_tokens,
    SUMMARY_HEADER,
    THRESHOLD_TOKENS,
)
from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.models.agent.context import AgentContext
from minisweagent.agents import get_agent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

HEAD = [{"role": "system", "content": "Use bash."}, {"role": "user", "content": "Task"}]


def turn(index, output="short result"):
    return [
        {
            "role": "assistant",
            "content": f"thought-{index} " + "a" * 600,
            "reasoning_content": "r" * 600,
            "tool_calls": [
                {
                    "id": f"{index}-{tool}",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "c" * 200}),
                    },
                }
                for tool in range(2)
            ],
        },
        *[
            {"role": "tool", "tool_call_id": f"{index}-{tool}", "content": output}
            for tool in range(2)
        ],
    ]


def test_compaction_keeps_head_and_complete_recent_turns():
    messages = HEAD + turn(0, "x" * 4097) + turn(1, "y" * 4096)
    messages += [message for i in range(2, 8) for message in turn(i, "z" * 5000)]
    original = copy.deepcopy(messages)
    result = compact(messages)
    assert messages == original
    assert result[:2] == HEAD
    assert all(new is old for new, old in zip(result[-18:], messages[-18:]))
    summary = result[2]["content"]
    assert "x" * 4097 not in summary
    assert "y" * 4096 in summary
    assert "r" * 600 in summary
    assert "a" * 600 in summary and "c" * 200 in summary
    assert summary.count("[bash]") == 4


def test_recompaction_discards_previous_summary():
    first = compact(HEAD + [message for i in range(8) for message in turn(i)])
    second = compact(first + turn(8))
    assert "thought-0" in first[2]["content"]
    assert "thought-0" not in second[2]["content"]
    assert "thought-2" in second[2]["content"]
    assert second[2]["content"].count(SUMMARY_HEADER) == 1
    assert second[-18:] == first[-15:] + turn(8)
    assert compact(HEAD + [message for i in range(6) for message in turn(i)]) is None


def test_native_agent_preserves_prefix_and_harbor_usage(monkeypatch, tmp_path):
    requests = []

    def completion(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return litellm.ModelResponse(
            model="test",
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call-{len(requests)}",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"printf ok"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 80},
            },
        )

    monkeypatch.setattr(litellm, "completion", completion)
    model = LitellmModel(
        model_name="openai/test",
        cost_tracking="ignore_errors",
        model_kwargs={"extra_headers": {"x-session-affinity": "cache-test"}},
    )
    agent = get_agent(
        model,
        LocalEnvironment(cwd=str(tmp_path)),
        {
            "agent_class": "executorch.examples.llm_server.evals.terminal_bench.cliff_agent.CliffAgent",
            "system_template": "system",
            "instance_template": "task",
            "mode": "yolo",
        },
    )
    agent.add_messages(*HEAD)
    agent.step()
    first_context = copy.deepcopy(agent.messages)
    agent.step()
    assert requests[1]["messages"][:2] == requests[0]["messages"]
    assert not agent.compactions
    agent.add_messages(
        *turn(0, "large output " * 3000),
        *[message for i in range(1, 7) for message in turn(i)],
    )
    assert estimate_tokens(agent.messages) >= THRESHOLD_TOKENS
    agent.step()
    assert len(agent.compactions) == 1
    assert SUMMARY_HEADER in requests[-1]["messages"][2]["content"]
    sent = requests[-1]["messages"]
    agent.step()
    assert requests[-1]["messages"][: len(sent)] == sent
    assert all(
        request["extra_headers"]["x-session-affinity"] == "cache-test"
        for request in requests
    )
    assert all("extra" not in message for message in sent)
    data = agent.save(tmp_path / "mini-swe-agent.trajectory.json")
    assert data["info"]["compaction"]["keep_recent"] == 6
    assert data["info"]["compaction"]["result_max_chars"] == 4096
    assert data["messages"][: len(first_context)] == first_context
    assert len(data["messages"]) > len(agent.messages)
    context = AgentContext()
    MiniSweAgent(logs_dir=tmp_path, model_name="openai/test").populate_context_post_run(
        context
    )
    assert (
        context.n_input_tokens,
        context.n_output_tokens,
        context.n_cache_tokens,
    ) == (400, 40, 320)
