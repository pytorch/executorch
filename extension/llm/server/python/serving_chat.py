# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""/v1/chat/completions OpenAI adapter: validates requests, renders the chat
template, parses tool calls, and formats OpenAI responses. It owns no model or
session state -- generation goes through SessionRuntime, and the token-ID warm-
resume transcript lives in OpenAITranscriptState."""

import json
import logging
import math
from typing import AsyncIterator, Optional

from .chat_template import ChatTemplate
from .errors import APIError, ContextLengthExceeded, GenerationError
from .protocol import (
    _new_id,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChunkChoice,
    DeltaMessage,
    FunctionCall,
    ResponseMessage,
    ToolCall,
    Usage,
)
from .session_runtime import GenerationOptions, GenStats, PromptInput, SessionRuntime
from .tool_parsers import HermesDetector, ToolCallItem

logger = logging.getLogger(__name__)


def _earliest_stop(text: str, stops: list[str]) -> Optional[int]:
    """Index of the earliest special-token occurrence in `text`, or None."""
    best = None
    for s in stops:
        i = text.find(s)
        if i != -1 and (best is None or i < best):
            best = i
    return best


class ServingChat:
    def __init__(
        self,
        runtime: SessionRuntime,
        template: ChatTemplate,
        model_id: str,
        max_context: Optional[int] = None,
        tool_detector_cls: Optional[type[HermesDetector]] = None,
    ):
        self._runtime = runtime
        self._template = template
        self._model_id = model_id
        self._max_context = max_context
        # Detector CLASS; a fresh instance is created per request so streaming
        # state is never shared across concurrent requests.
        self._tool_detector_cls = tool_detector_cls
        # Special tokens (e.g. <|im_end|>) the runner decodes to text; we cut the
        # visible content at the first one so they don't leak into responses.
        self._stops = template.special_tokens()

    @staticmethod
    def _tool_schemas(req: ChatCompletionRequest) -> dict[str, dict]:
        """Map each defined tool name to its JSON-schema ``parameters`` object.

        The detector uses the key set to validate names and the schema to coerce
        values to their declared types (the Qwen XML format is stringly-typed)."""
        schemas = {}
        for t in req.tools or []:
            fn = t.get("function", {}) if isinstance(t, dict) else {}
            name = fn.get("name")
            if name:
                schemas[name] = fn.get("parameters") or {}
        return schemas

    def _strip_specials(self, text: str) -> str:
        cut = _earliest_stop(text, self._stops)
        return text[:cut] if cut is not None else text

    @staticmethod
    def _to_openai_tool_call(item: ToolCallItem) -> ToolCall:
        return ToolCall(
            index=item.tool_index,
            id=_new_id("call"),
            type="function",
            function=FunctionCall(name=item.name, arguments=item.arguments),
        )

    def _tools_active(self, req: ChatCompletionRequest) -> bool:
        # tool_choice="none" disables tools even when the client sends them.
        return bool(self._tool_detector_cls and req.tools and req.tool_choice != "none")

    @staticmethod
    def _request_stops(req: ChatCompletionRequest) -> list[str]:
        s = req.stop
        if not s:
            return []
        return [s] if isinstance(s, str) else [x for x in s if x]

    @staticmethod
    def _apply_stop(text: str, stops: list[str]) -> str:
        """Truncate at the earliest stop string (the stop itself is excluded)."""
        cut = _earliest_stop(text, stops)
        return text[:cut] if cut is not None else text

    def _truncate_raw(self, text: str, req: ChatCompletionRequest) -> str:
        """Cut raw model output at the earliest special token or request stop
        sequence BEFORE tool parsing, so a tool call (or any text) past the stop
        boundary is neither parsed nor emitted."""
        return self._apply_stop(text, self._stops + self._request_stops(req))

    async def _collect_until_stop(self, stream: AsyncIterator[str], stops: list[str]):
        """Accumulate a buffered (non-streamed) generation into one string,
        halting the runtime early once a stop string (special token or request
        stop) appears, then draining so stats finalize. Returns (text, stopped):
        `stopped` lets the caller force finish_reason="stop" even when tokens
        queued before the runtime observed stop() pushed the count to max_tokens."""
        text = ""
        stopped = False
        async for tok in stream:
            text += tok
            if stops and _earliest_stop(text, stops) is not None:
                stopped = True
                self._runtime.stop()
                async for _ in stream:  # drain so stats_cb fires
                    pass
                break
        return text, stopped

    def _extract_tools(self, req: ChatCompletionRequest, text: str):
        """Returns (tool_calls | None, content_text). Falls back to plain text."""
        if self._tools_active(req):
            parsed = self._tool_detector_cls().detect_and_parse(
                text, self._tool_schemas(req)
            )
            if parsed.calls:
                content = self._strip_specials(parsed.normal_text) or None
                return [self._to_openai_tool_call(c) for c in parsed.calls], content
            text = parsed.normal_text
        return None, self._strip_specials(text)

    async def _clean(
        self, stream: AsyncIterator[str], stops: list[str], on_stop=None
    ) -> AsyncIterator[str]:
        # Yield text up to the earliest stop string (special token or request
        # `stop`), buffering across tokens so a stop spanning chunks is caught.
        # On a hit: optionally stop the runner early, then drain the source so it
        # finalizes (usage stats recorded, worker thread joined).
        hold = (
            max((len(s) for s in stops), default=1) - 1
        )  # keep a possible partial-stop tail
        buf = ""
        async for token in stream:
            buf += token
            cut = _earliest_stop(buf, stops)
            if cut is not None:
                if cut > 0:
                    yield buf[:cut]
                if on_stop is not None:
                    on_stop()
                async for _ in stream:  # drain so stats_cb fires
                    pass
                return
            if hold == 0:
                yield buf
                buf = ""
            elif len(buf) > hold:
                yield buf[:-hold]
                buf = buf[-hold:]
        if buf:
            yield buf

    def _options(self, req: ChatCompletionRequest) -> GenerationOptions:
        return GenerationOptions(
            max_new_tokens=req.resolved_max_tokens(),
            temperature=req.temperature if req.temperature is not None else 0.0,
            # Let the worker terminate at the same boundary the control plane
            # would cut: the model's special tokens (e.g. <|im_end|>) AND request
            # stop sequences. This stops generation at end-of-turn even when the
            # worker's EOS-by-token-id check misses it, instead of running to
            # max_new (or erroring) past the turn. The server's
            # _clean/_collect_until_stop still re-apply these as a backstop.
            stop=self._stops + self._request_stops(req),
        )

    def _finish_reason(
        self,
        req: ChatCompletionRequest,
        completion_tokens: int,
        tool_calls=None,
        stopped: bool = False,
        worker_finish: Optional[str] = None,
    ) -> str:
        # Precedence: tool call > stop boundary > worker reason > length heuristic.
        # `stopped` (a server-side stop sequence / special token) wins even over
        # the worker, since that truncation happened in the control plane.
        if tool_calls:
            return "tool_calls"
        if stopped:
            return "stop"
        # The worker knows whether it hit EOS ("stop") or ran to max_new ("length",
        # possibly a clamp to the context window) — trust it over the token-count
        # heuristic, which can't see a silent clamp.
        if worker_finish in ("stop", "length"):
            return worker_finish
        mt = req.resolved_max_tokens()
        return "length" if mt and mt > 0 and completion_tokens >= mt else "stop"

    @staticmethod
    def _reject_invalid_values(req: ChatCompletionRequest) -> None:
        """Reject out-of-range values (invalid_value); these take precedence over
        the unsupported-parameter error."""
        if req.temperature is not None and (
            not math.isfinite(req.temperature)
            or req.temperature < 0.0
            or req.temperature > 2.0
        ):
            raise APIError(
                400,
                f"temperature must be between 0 and 2 (got {req.temperature}).",
                "invalid_request_error",
                "invalid_value",
            )
        # max_tokens / max_completion_tokens, if given, must be positive integers
        # (OpenAI rejects 0 and negatives; our -1 sentinel means "unset/auto").
        for field_name in ("max_tokens", "max_completion_tokens"):
            v = getattr(req, field_name)
            if v is not None and v <= 0:
                raise APIError(
                    400,
                    f"{field_name} must be a positive integer (got {v}).",
                    "invalid_request_error",
                    "invalid_value",
                )

    @staticmethod
    def _reject_unsupported_params(req: ChatCompletionRequest) -> None:
        """Reject params we don't honor rather than silently ignoring them (a
        client relying on e.g. top_p/seed/logprobs would otherwise get wrong
        behavior). Only the no-op/default value of each passes: top_p exactly
        1.0; penalties 0; response_format type "text"; tool_choice none/auto/
        unset; parallel_tool_calls true (false can't be guaranteed without
        constraining); logprobs are not returned at all."""
        rf = req.response_format
        flags = [
            (req.n != 1, "n>1"),
            (req.top_p is not None and req.top_p != 1.0, "top_p"),
            (req.seed is not None, "seed"),
            (req.reasoning_effort is not None, "reasoning_effort"),
            (bool(req.frequency_penalty), "frequency_penalty"),
            (bool(req.presence_penalty), "presence_penalty"),
            (req.top_k is not None, "top_k"),
            (bool(req.logit_bias), "logit_bias"),
            (
                bool(rf) and rf.get("type", "text") != "text",
                "response_format (only 'text')",
            ),
            (bool(req.logprobs), "logprobs"),
            (req.top_logprobs is not None, "top_logprobs"),
            (req.parallel_tool_calls is False, "parallel_tool_calls=false"),
            (
                req.tool_choice not in (None, "none", "auto"),
                "tool_choice (only 'none' or 'auto')",
            ),
        ]
        unsupported = [label for cond, label in flags if cond]
        if unsupported:
            raise APIError(
                400,
                f"Unsupported parameter(s): {', '.join(unsupported)}. This server honors "
                "temperature, max_tokens/max_completion_tokens, stop, and tools for the "
                "configured tool-call format.",
                "invalid_request_error",
                "unsupported_parameter",
            )

    async def create(self, req: ChatCompletionRequest):
        self._reject_invalid_values(req)
        self._reject_unsupported_params(req)
        # tool_choice="none" must hide tools from the model: if we still render
        # the tool schemas, the model can emit a <tool_call> that we'd surface as
        # plain text (parsing is disabled), instead of a normal answer.
        template_tools = None if req.tool_choice == "none" else req.tools
        prompt = self._template.render(
            req.messages, tools=template_tools, template_kwargs=req.chat_template_kwargs
        )
        # Pre-flight context check: reject cleanly instead of failing mid-generation
        # (only possible when a tokenizer is available to count, e.g. --hf-tokenizer).
        if self._max_context:
            count = self._template.count_tokens(prompt)
            if count is not None:
                if count >= self._max_context:
                    raise ContextLengthExceeded(count, self._max_context)
                # An explicit max_tokens that wouldn't fit alongside the prompt
                # must be rejected here, not run until the worker hits the context
                # limit mid-decode (a 500 / streaming error after partial output).
                requested = req.resolved_max_tokens()
                if requested > 0 and count + requested > self._max_context:
                    raise ContextLengthExceeded(count, self._max_context, requested)
        options = self._options(req)
        prompt_input = PromptInput(text=prompt)
        if req.stream:
            return self._stream(req, prompt_input, options)
        return await self._complete(req, prompt_input, options)

    async def _complete(
        self,
        req: ChatCompletionRequest,
        prompt: PromptInput,
        options: GenerationOptions,
    ) -> ChatCompletionResponse:
        stats = GenStats()
        try:
            # Collect raw text (markers intact for tool parsing), halting early
            # at a stop boundary (special token or request stop).
            text, stopped = await self._collect_until_stop(
                self._runtime.generate_stream(None, prompt, options, stats),
                self._stops + self._request_stops(req),
            )
        except Exception as e:  # noqa: BLE001 - surface as a structured API error
            raise GenerationError(str(e))
        # Bound the raw output at the first stop/special token BEFORE tool
        # parsing, so a call after the stop boundary is not parsed/emitted.
        tool_calls, content = self._extract_tools(req, self._truncate_raw(text, req))
        finish = self._finish_reason(
            req, stats.completion_tokens, tool_calls, stopped, stats.finish_reason
        )
        return ChatCompletionResponse(
            model=self._model_id,
            choices=[
                Choice(
                    message=ResponseMessage(content=content, tool_calls=tool_calls),
                    finish_reason=finish,
                )
            ],
            usage=Usage(
                prompt_tokens=stats.prompt_tokens,
                completion_tokens=stats.completion_tokens,
                total_tokens=stats.prompt_tokens + stats.completion_tokens,
            ),
        )

    async def _stream(
        self,
        req: ChatCompletionRequest,
        prompt: PromptInput,
        options: GenerationOptions,
    ) -> AsyncIterator[str]:
        cid = _new_id("chatcmpl")

        def chunk(delta: DeltaMessage, finish=None) -> str:
            c = ChatCompletionChunk(
                id=cid,
                model=self._model_id,
                choices=[ChunkChoice(delta=delta, finish_reason=finish)],
            )
            return f"data: {c.model_dump_json(exclude_none=True)}\n\n"

        yield chunk(DeltaMessage(role="assistant"))
        error: Optional[Exception] = None
        use_tools = self._tools_active(req)
        tool_calls = None
        content = None

        stats = GenStats()
        stop_hit = [False]  # set when a stop boundary is reached (forces finish="stop")
        stops = self._stops + self._request_stops(req)
        try:
            if use_tools:
                # v1: buffer the (usually short) tool response, parse once.
                # Halt early at a stop boundary, and bound the raw output
                # BEFORE parsing so post-stop tool calls / text don't leak.
                raw, stop_hit[0] = await self._collect_until_stop(
                    self._runtime.generate_stream(None, prompt, options, stats),
                    stops,
                )
                tool_calls, content = self._extract_tools(
                    req, self._truncate_raw(raw, req)
                )
            else:
                # Plain chat: stream tokens live (best UX), cutting at special
                # tokens or request stop sequences and halting early on a hit.
                def on_stop():
                    stop_hit[0] = True
                    self._runtime.stop()

                async for token in self._clean(
                    self._runtime.generate_stream(None, prompt, options, stats),
                    stops,
                    on_stop=on_stop,
                ):
                    yield chunk(DeltaMessage(content=token))
        except (
            Exception
        ) as e:  # noqa: BLE001 - emit a structured error event, never drop the socket
            error = e

        if error is not None:
            err = {
                "message": f"Generation failed: {error}",
                "type": "server_error",
                "code": None,
            }
            yield f"data: {json.dumps({'error': err})}\n\n"
            yield "data: [DONE]\n\n"
            return

        if use_tools:
            if content:
                yield chunk(DeltaMessage(content=content))
            for tc in tool_calls or []:
                yield chunk(DeltaMessage(tool_calls=[tc]))
            finish = self._finish_reason(
                req,
                stats.completion_tokens,
                tool_calls,
                stop_hit[0],
                stats.finish_reason,
            )
        else:
            finish = self._finish_reason(
                req,
                stats.completion_tokens,
                stopped=stop_hit[0],
                worker_finish=stats.finish_reason,
            )
        yield chunk(DeltaMessage(), finish=finish)
        if req.stream_options and req.stream_options.include_usage:
            usage_chunk = ChatCompletionChunk(
                id=cid,
                model=self._model_id,
                choices=[],
                usage=Usage(
                    prompt_tokens=stats.prompt_tokens,
                    completion_tokens=stats.completion_tokens,
                    total_tokens=stats.prompt_tokens + stats.completion_tokens,
                ),
            )
            yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"
