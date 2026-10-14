# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Assemble a single OpenAI chat completion from streamed deltas."""


class ChatCompletionStream:
    def __init__(self):
        self.message = {"role": "assistant"}
        self.calls = {}
        self.usage = {}
        self.finish_reason = None

    def add(self, chunk):
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            if choice["index"] != 0:
                raise ValueError("Expected a single streamed completion")
            if choice.get("finish_reason") is not None:
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta", {})
            for key in ("content", "reasoning_content", "refusal"):
                if delta.get(key):
                    self.message[key] = self.message.get(key, "") + delta[key]
            for update in delta.get("tool_calls") or []:
                call = self.calls.setdefault(
                    update["index"],
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                call["id"] += update.get("id") or ""
                for key in ("name", "arguments"):
                    call["function"][key] += update.get("function", {}).get(key) or ""

    def result(self):
        message = dict(self.message)
        if self.calls:
            message["tool_calls"] = [self.calls[i] for i in sorted(self.calls)]
        return {
            "choices": [{"message": message, "finish_reason": self.finish_reason}],
            "usage": self.usage,
        }
