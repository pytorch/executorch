# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse's role-complete template and canonical OpenAI tool-name translation."""

from copy import deepcopy
from functools import cached_property
from pathlib import Path

from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.errors import APIError

from .framing import body_spans, CONTROLS, recipient_bodies, wire_tool_name
from .tool_parsers.atem import parameter_value_spans


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
        markers = {}
        for marker in CONTROLS:
            ids = self._hf.encode(marker, add_special_tokens=False)
            if len(ids) == 1 and ids[0] in self._hf.all_special_ids:
                markers[ids[0]] = marker
        if all(
            marker in markers.values()
            for marker in ("<|message|>", "<|eom|>", "<|eot|>")
        ):
            return markers
        return None

    def count_reasoning_tokens(self, token_ids):
        """Count original generated IDs in the same body spans as extraction."""
        if self._reasoning_markers is None:
            return None
        parts, boundaries = [], {0: 0}
        cursor = offset = 0
        for i, token in enumerate(token_ids):
            if token in self._reasoning_markers:
                part = self._hf.decode(
                    token_ids[cursor : i + 1],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                marker = self._reasoning_markers[token]
                if not part.endswith(marker):
                    return None
                parts.append(part)
                offset += len(part)
                boundaries[offset - len(marker)] = i
                boundaries[offset] = i + 1
                cursor = i + 1
        parts.append(
            self._hf.decode(
                token_ids[cursor:],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        text = "".join(parts)
        boundaries[len(text)] = len(token_ids)
        count = 0
        protected = parameter_value_spans(text)
        for header, start, end in recipient_bodies(text, protected):
            spans = list(body_spans(text, start, end, protected))
            if header.group(1) != "self" or not any(
                text[a:b].strip() for a, b in spans
            ):
                continue
            for begin, stop in spans:
                if begin not in boundaries or stop not in boundaries:
                    return None
                count += boundaries[stop] - boundaries[begin]
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
