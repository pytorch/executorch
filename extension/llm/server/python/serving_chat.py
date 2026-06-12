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
from typing import AsyncIterator, Callable, Optional

from .chat_template import ChatTemplate
from .errors import (
    APIError,
    ContextLengthExceeded,
    GenerationError,
    InvalidSessionId,
    SessionCapacity,
)
from .openai_transcript import OpenAITranscriptState
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
from .worker_client import WorkerError

logger = logging.getLogger(__name__)

_SESSION_ID_MAX_LEN = 128


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
        prompt_token_offset: int = 0,
        content_filter: Optional[Callable[[str], str]] = None,
        content_filter_specials: Optional[set[str]] = None,
    ):
        self._runtime = runtime
        self._template = template
        self._model_id = model_id
        self._max_context = max_context
        self._prompt_token_offset = prompt_token_offset
        self._content_filter = content_filter
        # Detector CLASS; a fresh instance is created per request so streaming
        # state is never shared across concurrent requests.
        self._tool_detector_cls = tool_detector_cls
        # Two distinct sets (see chat_template); create() combines them per path:
        #  * _stops: NARROW turn terminators (e.g. <|im_end|>). The ONLY stop set
        #    for tool turns -- excludes structural/tool delimiters so a <tool_call>
        #    is never halted or cut before _extract_tools sees it. Also used by
        #    _truncate_raw (pre-parse truncation on the tool path).
        #  * _content_specials: BROAD all-special-tokens set. For PLAIN chat it is
        #    added to the worker/clean stop set (create() -> gen_stops) so a leaked
        #    special halts the worker (-> dirty, ids omitted, non-resumable) and
        #    never reaches the client, AND it backs _strip_specials for final
        #    cleanup of already-parsed visible content.
        self._stops = template.turn_stop_sequences()
        handled = content_filter_specials or set()
        self._content_specials = [
            t for t in template.special_tokens() if t not in handled
        ]
        # OpenAI/chat-template token-ID warm-resume state. Adapter-side,
        # not runtime; kept in lockstep with the worker's session state by
        # clearing both on reset/close.
        self._transcript = OpenAITranscriptState(template)

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
        # Broad set: scrub ANY special token that leaked into already-parsed
        # visible content (not the narrow generation-stop set).
        cut = _earliest_stop(text, self._content_specials)
        return text[:cut] if cut is not None else text

    def _visible_content(self, text: str) -> str:
        if self._content_filter is not None:
            text = self._content_filter(text)
        return self._strip_specials(text)

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
                content = self._visible_content(parsed.normal_text) or None
                return [self._to_openai_tool_call(c) for c in parsed.calls], content
            text = parsed.normal_text
        return None, self._visible_content(text)

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

    def _options(
        self, req: ChatCompletionRequest, stops: list[str]
    ) -> GenerationOptions:
        return GenerationOptions(
            max_new_tokens=req.resolved_max_tokens(),
            temperature=req.temperature if req.temperature is not None else 0.0,
            # Worker stop set, chosen per path in create() (see __init__ for the
            # two sets); the server re-applies it in _clean/_collect_until_stop.
            stop=stops,
        )

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        # Keep it boring: non-empty printable ASCII (no spaces/control), <=128.
        if not session_id or len(session_id) > _SESSION_ID_MAX_LEN:
            raise InvalidSessionId(f"must be 1-{_SESSION_ID_MAX_LEN} characters")
        if not all(0x21 <= ord(c) <= 0x7E for c in session_id):
            raise InvalidSessionId("must be printable ASCII with no spaces")

    async def _preflight_session(self, session_id: str) -> None:
        """Reserve the session before any response bytes are emitted so a
        capacity refusal becomes an HTTP status, not an SSE error event."""
        try:
            await self._runtime.open(session_id)
        except WorkerError as e:
            if e.code in ("capacity_exhausted", "unsupported_session"):
                raise SessionCapacity(e.code)
            raise GenerationError(str(e))

    async def close_session(self, session_id: str) -> None:
        # Lockstep: do the fallible worker op FIRST, then clear the (best-effort,
        # can't-fail) transcript. If the worker op fails both retain old state,
        # so they never drift.
        self._validate_session_id(session_id)
        try:
            await self._runtime.close(session_id)
        except WorkerError as e:
            raise GenerationError(str(e))
        self._transcript.close(session_id)

    async def reset_session(self, session_id: str) -> None:
        # Lockstep: worker op first (fallible), then clear the transcript.
        self._validate_session_id(session_id)
        try:
            await self._runtime.reset(session_id)
        except WorkerError as e:
            raise GenerationError(str(e))
        self._transcript.reset(session_id)

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

    def _count_prompt_tokens(self, prompt: PromptInput) -> Optional[int]:
        """Token count of what the worker will actually assemble: the rendered
        text, or for token-ID segments sum(len(ids)) for {ids} runs + the
        tokenized length of {text} chunks. None when no tokenizer is available to
        count text (the worker still enforces the real context limit)."""
        if prompt.text is not None:
            count = self._template.count_tokens(prompt.text)
            return None if count is None else count + self._prompt_token_offset
        total = self._prompt_token_offset
        for seg in prompt.segments:
            if "ids" in seg:
                total += len(seg["ids"])
            else:
                c = self._template.count_tokens(seg["text"])
                if c is None:
                    return None
                total += c
        return total

    async def create(self, req: ChatCompletionRequest):
        self._reject_invalid_values(req)
        self._reject_unsupported_params(req)
        if req.session_id is not None:
            self._validate_session_id(req.session_id)
        # tool_choice="none" must hide tools from the model: if we still render
        # the tool schemas, the model can emit a <tool_call> that we'd surface as
        # plain text (parsing is disabled), instead of a normal answer.
        template_tools = None if req.tool_choice == "none" else req.tools
        prompt = self._template.render(
            req.messages, tools=template_tools, template_kwargs=req.chat_template_kwargs
        )
        # Token-ID segments splice prior assistant turns' exact ids so warm resume
        # survives the template's lossy tool-call re-render; plain text when
        # there's nothing to splice or on ambiguity (the worker verifies the
        # exact-token prefix regardless).
        prompt_input = self._transcript.build_prompt_input(
            session_id=req.session_id,
            messages=req.messages,
            rendered_prompt=prompt,
            tools=template_tools,
            template_kwargs=req.chat_template_kwargs,
        )
        # Pre-flight context check against the tokens the worker will actually
        # assemble: for segments that is sum(len(ids)) + tokenized text, not the
        # rendered string, so a near-limit prompt agrees with the worker rather
        # than false-400ing or failing mid-decode. Only when a tokenizer exists.
        if self._max_context:
            count = self._count_prompt_tokens(prompt_input)
            if count is not None:
                if count >= self._max_context:
                    raise ContextLengthExceeded(count, self._max_context)
                # An explicit max_tokens that wouldn't fit alongside the prompt
                # must be rejected here, not run until the worker hits the context
                # limit mid-decode (a 500 / streaming error after partial output).
                requested = req.resolved_max_tokens()
                if requested > 0 and count + requested > self._max_context:
                    raise ContextLengthExceeded(count, self._max_context, requested)
        # Per-path worker stop set (see __init__ for the two sets and why): tool
        # turns use the narrow set; plain chat adds the broad content specials.
        if self._tools_active(req):
            gen_stops = self._stops + self._request_stops(req)
        else:
            gen_stops = self._stops + self._content_specials + self._request_stops(req)
        options = self._options(req, gen_stops)
        # The generation scaffold the worker will prefill ahead of this turn's
        # tokens (e.g. Qwen3 <think> block), resolved with the same per-request
        # mode AND tools as the render; recorded per turn so warm-resume splicing
        # reproduces the exact resident scaffold even if the mode changes between
        # requests.
        preamble = self._template.generation_preamble(
            req.chat_template_kwargs, tools=template_tools
        )
        # Admit the session up front (before the stream's first chunk) so a
        # capacity refusal is an HTTP status, not a mid-stream error event.
        if req.session_id is not None:
            await self._preflight_session(req.session_id)
        if req.stream:
            return self._stream(req, prompt_input, options, preamble, gen_stops)
        return await self._complete(req, prompt_input, options, preamble, gen_stops)

    async def _complete(
        self,
        req: ChatCompletionRequest,
        prompt: PromptInput,
        options: GenerationOptions,
        preamble: str = "",
        gen_stops: Optional[list[str]] = None,
    ) -> ChatCompletionResponse:
        # Same stop set the worker was given (per-path: narrow for tools, broad
        # content specials added for plain chat); falls back to narrow if a caller
        # didn't supply it.
        stops = (
            gen_stops
            if gen_stops is not None
            else self._stops + self._request_stops(req)
        )
        stats = GenStats()
        try:
            # Collect raw text (markers intact for tool parsing), halting early
            # at a stop boundary (special token or request stop).
            text, stopped = await self._collect_until_stop(
                self._runtime.generate_stream(req.session_id, prompt, options, stats),
                stops,
            )
        except Exception as e:  # noqa: BLE001 - surface as a structured API error
            raise GenerationError(str(e))
        # Bound the raw output at the first stop/special token BEFORE tool
        # parsing, so a call after the stop boundary is not parsed/emitted.
        tool_calls, content = self._extract_tools(req, self._truncate_raw(text, req))
        # Record after the response is finalized: the fingerprint is of exactly
        # what we return (content + tool_calls), so the next turn can confirm the
        # client echoed this turn before splicing its ids.
        self._transcript.record_assistant_turn(
            session_id=req.session_id,
            content=content,
            tool_calls=tool_calls,
            generated_token_ids=stats.generated_token_ids,
            prior_turns=sum(1 for m in req.messages if m.role == "assistant"),
            preamble=preamble,
        )
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

    async def _stream_plain_content(
        self,
        req: ChatCompletionRequest,
        prompt: PromptInput,
        options: GenerationOptions,
        stats: GenStats,
        stops: list[str],
        stop_hit: list[bool],
    ) -> AsyncIterator[str]:
        if self._content_filter is not None:
            raw, stop_hit[0] = await self._collect_until_stop(
                self._runtime.generate_stream(req.session_id, prompt, options, stats),
                stops,
            )
            content = self._visible_content(self._apply_stop(raw, stops))
            if content:
                yield content
            return

        def on_stop():
            stop_hit[0] = True
            self._runtime.stop()

        async for token in self._clean(
            self._runtime.generate_stream(req.session_id, prompt, options, stats),
            stops,
            on_stop=on_stop,
        ):
            yield token

    async def _stream(
        self,
        req: ChatCompletionRequest,
        prompt: PromptInput,
        options: GenerationOptions,
        preamble: str = "",
        gen_stops: Optional[list[str]] = None,
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
        # Per-path stop set from create(): for plain chat this includes the broad
        # content specials, so _clean cuts a leaked special out of the stream (and
        # the worker, given the same set, halts + omits ids -> non-resumable turn).
        stops = (
            gen_stops
            if gen_stops is not None
            else self._stops + self._request_stops(req)
        )
        try:
            if use_tools:
                # Buffer the (usually short) tool response, parse once.
                # Halt early at a stop boundary, and bound the raw output
                # BEFORE parsing so post-stop tool calls / text don't leak.
                raw, stop_hit[0] = await self._collect_until_stop(
                    self._runtime.generate_stream(
                        req.session_id, prompt, options, stats
                    ),
                    stops,
                )
                tool_calls, content = self._extract_tools(
                    req, self._truncate_raw(raw, req)
                )
            else:
                streamed: list[str] = []
                async for token in self._stream_plain_content(
                    req, prompt, options, stats, stops, stop_hit
                ):
                    streamed.append(token)
                    yield chunk(DeltaMessage(content=token))
                content = "".join(streamed)  # for the session fingerprint
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
        self._transcript.record_assistant_turn(
            session_id=req.session_id,
            content=content,
            tool_calls=tool_calls,
            generated_token_ids=stats.generated_token_ids,
            prior_turns=sum(1 for m in req.messages if m.role == "assistant"),
            preamble=preamble,
        )

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
