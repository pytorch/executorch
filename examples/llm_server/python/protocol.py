# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenAI-compatible request/response schemas for the ExecuTorch LLM server.

This is the Python view of the contract defined in ``examples/llm_server/spec``.
Any language server must serialize to the same shapes; the conformance suite in
``examples/llm_server/conformance`` validates them.
"""

import json
import time
import uuid
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


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
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: Optional[Union[str, list[dict[str, Any]]]] = None
    name: Optional[str] = None
    # Optional prior-turn reasoning (chain-of-thought). Mirrors
    # ResponseMessage.reasoning_content so a multi-turn client can echo an assistant
    # turn's reasoning back into the request; a chat template that renders prior
    # reasoning needs it here, and without the field it is dropped at parse.
    # Omission/null allows stored-token replay; a string edit invalidates that turn.
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = Field(default=None, strict=True)
    top_p: Optional[float] = Field(default=None, strict=True)
    max_tokens: Optional[int] = Field(default=None, strict=True)
    max_completion_tokens: Optional[int] = Field(default=None, strict=True)
    stop: Optional[Union[str, list[str]]] = None
    n: int = Field(default=1, strict=True)
    seed: Optional[int] = Field(default=None, strict=True)
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    top_k: Optional[int] = Field(default=None, strict=True)
    # Disable model/template termination; explicit request stops still apply.
    ignore_eos: bool = Field(default=False, strict=True)
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
    reasoning: Optional[dict[str, Any]] = None
    include_reasoning: Optional[bool] = Field(default=None, strict=True)
    # Retain output/cache controls so capability checks can reject them before
    # generation, including malformed values and empty strings.
    thinking_token_budget: Any = None
    grammar: Any = None
    guided_json: Any = None
    guided_regex: Any = None
    guided_choice: Any = None
    guided_grammar: Any = None
    structured_outputs: Any = None
    regex: Any = None
    ebnf: Any = None
    stop_regex: Any = None
    add_generation_prompt: Any = None
    continue_final_message: Any = None
    cache_salt: Any = None

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, tools):
        names = set()
        for tool in tools or []:
            fn = tool.get("function")
            if tool.get("type") != "function" or not isinstance(fn, dict):
                raise ValueError("tools must contain function declarations")
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError("tool names must be unique nonempty strings")
            names.add(name)
            if fn.get("parameters") is not None and not isinstance(
                fn["parameters"], dict
            ):
                raise ValueError("tool parameters must be a JSON schema object")
        return tools

    @field_validator("messages")
    @classmethod
    def validate_history(cls, messages):
        for message in messages:
            calls = set()
            for call in message.tool_calls or []:
                if message.role != "assistant" or not call.id or not call.function.name:
                    raise ValueError(
                        "historical tool calls require assistant role, id and name"
                    )
                if call.id in calls:
                    raise ValueError("duplicate historical tool call id")
                calls.add(call.id)
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except (ValueError, RecursionError) as error:
                    raise ValueError(
                        "historical tool arguments must be valid JSON"
                    ) from error
                if not isinstance(arguments, dict):
                    raise ValueError("historical tool arguments must be an object")
            # Compaction can remove calls/results, and providers can reuse IDs
            # across turns. Validate each message without requiring a full log.
            if message.role == "tool" and not message.tool_call_id:
                raise ValueError("tool result requires a tool_call_id")
        return messages

    def resolved_max_tokens(self) -> int:
        # `is not None` (not `or`): an explicit 0 must not be treated as unset.
        # Callers validate positivity; -1 means "unset / auto".
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        if self.max_tokens is not None:
            return self.max_tokens
        return -1


class PromptTokensDetails(BaseModel):
    # Warm-resume accounting, SGLang-compatible shape: tokens reused from the
    # session's resident state or a shared prefix snapshot instead of prefilled.
    cached_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: Optional[PromptTokensDetails] = None
    completion_tokens_details: Optional[CompletionTokensDetails] = None


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
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
    reasoning_content: Optional[str] = None
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
    context_length: Optional[int] = None


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
