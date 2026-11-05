# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse's role-complete template and canonical OpenAI tool-name translation."""

import re
from copy import deepcopy
from functools import cached_property
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

    @cached_property
    def _reasoning_markers(self):
        if self._hf is None:
            return None
        markers = [
            self._hf.encode(m, add_special_tokens=False)
            for m in ("<|message|>", "<|eom|>", "<|eot|>")
        ]
        if all(len(ids) == 1 for ids in markers):
            result = {ids[0] for ids in markers}
            if len(result) == 3:
                return result
        return None

    def count_reasoning_tokens(self, token_ids):
        """Count generated body IDs, excluding headers and end markers.

        Decode runs ending at atomic framing tokens to map character boundaries
        back to original IDs. Re-encoding rendered reasoning can merge tokens and
        cannot account correctly for byte-fallback IDs or hidden reasoning.
        """
        from .framing import recipient_headers
        from .tool_parsers.atem import parameter_value_spans

        if self._reasoning_markers is None:
            return None
        parts, ends = [], {}
        cursor = offset = 0
        for i, token in enumerate(token_ids):
            if token in self._reasoning_markers:
                part = self._hf.decode(
                    token_ids[cursor : i + 1],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                parts.append(part)
                offset += len(part)
                ends[offset] = i + 1
                cursor = i + 1
        parts.append(
            self._hf.decode(
                token_ids[cursor:],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        text = "".join(parts)
        count = 0
        for header in recipient_headers(text, parameter_value_spans(text)):
            if header.group(1) != "self":
                continue
            start = ends.get(header.end())
            if start is None:
                return None
            end = re.search(r"<\|eo[mt]\|>", text[header.end() :])
            if end is None:
                count += len(token_ids) - start
            else:
                stop = ends.get(header.end() + end.end())
                if stop is None:
                    return None
                count += stop - 1 - start
        return count

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
