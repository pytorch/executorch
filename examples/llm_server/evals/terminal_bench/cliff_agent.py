# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CliffCompaction for mini-swe-agent's text/tool-call conversation.

Adapted from nguyenvuthientrang/cliffcompaction (b48d660), under the MIT
license in LICENSE.cliffcompaction, with conservative retention settings.
"""

import json

from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

THRESHOLD_TOKENS = 8192
KEEP_RECENT = 6
RESULT_MAX_CHARS = 4096
SUMMARY_HEADER = (
    "The following is a summary of your previous actions (long observations omitted):"
)


def estimate_tokens(messages):
    request = {
        "messages": [
            {key: value for key, value in message.items() if key != "extra"}
            for message in messages
        ],
        "tools": [BASH_TOOL],
    }
    return len(json.dumps(request, ensure_ascii=False)) // 4


def summarize_message(message):
    if message.get("extra", {}).get("cliff_summary"):
        return ""
    role = message["role"]
    text = message.get("content") or ""
    if not isinstance(text, str):
        raise ValueError("Cliff compaction supports text messages only")
    text = text.strip()
    if role == "assistant":
        lines = []
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning:
            lines.append(f"thinking: {reasoning.strip()}")
        if text:
            lines.append(f"assistant: {text}")
        for call in message.get("tool_calls") or []:
            function = call["function"]
            lines.append(f"[{function['name']}] {function['arguments']}")
        return "\n".join(lines)
    if role == "tool":
        return f"result: {text}" if text and len(text) <= RESULT_MAX_CHARS else ""
    return f"{role}: {text}" if text else ""


def compact(messages):
    starts = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
    if len(starts) <= KEEP_RECENT:
        return None
    head, cut = starts[0], starts[-KEEP_RECENT]
    if head and messages[head - 1].get("extra", {}).get("cliff_summary"):
        head -= 1
    parts = [
        text for message in messages[head:cut] if (text := summarize_message(message))
    ]
    summary = {
        "role": "user",
        "content": SUMMARY_HEADER + "\n\n" + "\n\n---\n\n".join(parts),
        "extra": {"cliff_summary": True},
    }
    result = [*messages[:head], summary, *messages[cut:]]
    if len(result) >= len(messages) or estimate_tokens(result) >= estimate_tokens(
        messages
    ):
        return None
    return result


class CliffAgent(InteractiveAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory_messages = []
        self.compactions = []

    def run(self, task="", **kwargs):
        self.trajectory_messages = []
        self.compactions = []
        return super().run(task, **kwargs)

    def add_messages(self, *messages):
        self.trajectory_messages.extend(messages)
        return super().add_messages(*messages)

    def query(self):
        before = estimate_tokens(self.messages)
        if before >= THRESHOLD_TOKENS:
            compacted = compact(self.messages)
            if compacted is not None:
                after = estimate_tokens(compacted)
                self.compactions.append(
                    {
                        "before_model_call": self.n_calls + 1,
                        "trajectory_messages": len(self.trajectory_messages),
                        "estimated_tokens_before": before,
                        "estimated_tokens_after": after,
                        "messages": compacted.copy(),
                    }
                )
                self.messages = compacted
                self.logger.info(
                    "Cliff compaction: %s -> %s estimated tokens", before, after
                )
            if estimate_tokens(self.messages) >= THRESHOLD_TOKENS:
                self.logger.warning(
                    "Context remains above the 8192-token estimate after preserving "
                    "the task and %s recent turns",
                    KEEP_RECENT,
                )
        return super().query()

    def serialize(self, *extra_dicts):
        data = super().serialize(*extra_dicts)
        # Harbor sums usage from this full trace, not the compacted model context.
        data["messages"] = self.trajectory_messages
        data["info"]["compaction"] = {
            "algorithm": "cliff",
            "threshold_tokens": THRESHOLD_TOKENS,
            "keep_recent": KEEP_RECENT,
            "thought_max_chars": 0,
            "thinking_max_chars": 0,
            "cmd_max_chars": 0,
            "result_max_chars": RESULT_MAX_CHARS,
            "token_estimator": "json_chars_div_4",
            "events": self.compactions,
        }
        return data
