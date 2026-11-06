# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse's role-complete template and canonical OpenAI tool-name translation."""

from copy import deepcopy
from pathlib import Path

from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.errors import APIError


def wire_tool_name(name: str) -> str:
    return name if "." in name else f"{name}.{name}"


def resolve_tool_result_names(messages):
    names = {}
    result = []
    for message in messages:
        for call in message.tool_calls or []:
            names[call.id] = call.function.name
        if (
            message.role == "tool"
            and not message.name
            and message.tool_call_id in names
        ):
            message = message.model_copy(update={"name": names[message.tool_call_id]})
        result.append(message)
    return result


class MuseGlimmerChatTemplate(ChatTemplate):
    def __init__(self, *args, **kwargs):
        kwargs["chat_template"] = Path(__file__).with_suffix(".jinja").read_text()
        super().__init__(*args, **kwargs)

    def render(self, messages, tools=None, template_kwargs=None):
        tools = deepcopy(tools)
        names = {}
        for tool in tools or []:
            fn = tool["function"]
            original = fn["name"]
            wire = wire_tool_name(original)
            if wire in names:
                raise APIError(
                    400,
                    "Ambiguous Muse tool names",
                    "invalid_request_error",
                    "invalid_value",
                )
            names[wire] = original
            fn["name"] = wire
        messages = [m.model_copy(deep=True) for m in messages]
        for message in messages:
            if message.role == "tool" and message.name:
                message.name = wire_tool_name(message.name)
            for call in message.tool_calls or []:
                if call.function.name:
                    call.function.name = wire_tool_name(call.function.name)
        return super().render(messages, tools=tools, template_kwargs=template_kwargs)
