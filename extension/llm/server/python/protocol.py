# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenAI-compatible request/response schemas for the ExecuTorch LLM server.

This is the Python view of the contract defined in ``extension/llm/server/spec``.
Any language server must serialize to the same shapes; the conformance suite in
``extension/llm/server/conformance`` validates them.
"""

import time
import uuid
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


class FunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ToolCall(BaseModel):
    index: Optional[int] = None
    id: Optional[str] = None
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, list[dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    n: int = 1
    seed: Optional[int] = None
    # Sampling knobs that change generation output. We don't plumb these, so they
    # are modeled (not dropped) in order to be rejected with a clear error rather
    # than silently ignored — see serving_chat's unsupported-parameter check.
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    top_k: Optional[int] = None
    logit_bias: Optional[dict[str, float]] = None
    # Output-contract fields: modeled (not dropped) so we reject the ones we
    # can't honor rather than returning an output that violates what was asked.
    response_format: Optional[dict[str, Any]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    # Per-request chat-template controls, e.g. {"enable_thinking": false} for Qwen3.
    chat_template_kwargs: Optional[dict[str, Any]] = None
    # Vendor extension: route this request to a persistent, isolated session (its
    # own KV/recurrent context) on a multi-session worker; requests sharing a
    # session_id continue the same context. Anonymous (a transient scratch
    # session) when unset. Also accepted via the X-ExecuTorch-Session-ID header.
    session_id: Optional[str] = None
    # Accepted now so the contract is stable; parsing/enforcement land in M2/M5.
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    reasoning_effort: Optional[str] = None

    def resolved_max_tokens(self) -> int:
        # `is not None` (not `or`): an explicit 0 must not be treated as unset.
        # Callers validate positivity; -1 means "unset / auto".
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        if self.max_tokens is not None:
            return self.max_tokens
        return -1


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("chatcmpl"))
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChunkChoice]
    usage: Optional[Usage] = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "executorch"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
