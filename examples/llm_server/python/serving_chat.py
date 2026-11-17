# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""/v1/chat/completions OpenAI adapter: validates requests, renders the chat
template, parses tool calls, and formats OpenAI responses. It owns no model or
session state -- generation goes through SessionRuntime, and the token-ID warm-
resume transcript lives in OpenAITranscriptState."""

import asyncio
import json
import logging
import math
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Iterable, Optional, Protocol

from .chat_template import ChatTemplate
from .errors import (
    APIError,
    ContextLengthExceeded,
    GenerationError,
    InvalidSessionId,
    ModelNotFound,
    SessionCapacity,
)
from .hud.metrics import record_error, Turn
from .openai_transcript import OpenAITranscriptState
from .protocol import (
    _new_id,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChunkChoice,
    CompletionTokensDetails,
    DeltaMessage,
    FunctionCall,
    PromptTokensDetails,
    ResponseMessage,
    ToolCall,
    Usage,
)
from .session_runtime import (
    _SessionLocks,
    GenerationOptions,
    GenStats,
    PromptInput,
    SessionRuntime,
)
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


class _ChatStream:
    """Own the transaction even when an HTTP stream is closed before first read."""

    def __init__(self, stream, lease, generation):
        self._stream = stream
        self._lease = lease
        self._generation = generation
        self._reader = None
        self._read_done = None
        self._close_task = None
        self._closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        if self._reader is not None:
            raise RuntimeError("chat stream already has an active reader")
        self._reader = asyncio.current_task()
        self._read_done = asyncio.get_running_loop().create_future()
        try:
            try:
                return await self._stream.__anext__()
            finally:
                self._reader = None
                self._read_done.set_result(None)
        except BaseException:
            await self.aclose()
            raise

    async def _close(self):
        if self._reader is not None:
            self._reader.cancel()
            await SessionRuntime._finish_cleanup(self._read_done)
        try:
            await self._stream.aclose()
        finally:
            try:
                await self._generation.aclose()
            finally:
                await self._lease.__aexit__(None, None, None)

    async def aclose(self):
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await SessionRuntime._finish_cleanup(self._close_task)


class StreamingResponseParser(Protocol):
    """Request-local, append-only content/reasoning parser; no SSE or tool state.

    For parsed streams, replaces buffered reasoning extraction and content cleanup
    (content_filter and _strip_specials). Raw stop-string handling still runs
    before feed(). finish() runs only at successful logical EOF, including length
    exhaustion and a drained stop, not on error, cancellation, or early close.
    """

    def feed(self, text: str) -> Iterable[DeltaMessage]: ...

    def finish(self) -> Iterable[DeltaMessage]: ...


def _session_capacity_exhausted(error):
    return (
        isinstance(error, WorkerError)
        and error.code == "session_capacity_exhausted"
        or isinstance(error, SessionCapacity)
        and error.session_slots_exhausted
    )


@dataclass
class _AdmissionState:
    pending: bool = False
    admitted: bool = False
    retried: bool = False


class _AdmissionRetryStream:
    """Retry one session-slot rejection before any raw model output."""

    def __init__(self, serving, session_id, generation, factory, admission):
        self._serving = serving
        self._session_id = session_id
        self._generation = generation
        self._factory = factory
        self._admission = admission
        self._started = False
        self._closed = False
        self._cancel_requested = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        while not self._closed:
            try:
                token = await anext(self._generation)
                self._started = True
                if self._session_id is not None:
                    self._admission.admitted = True
                return token
            except StopAsyncIteration:
                if self._session_id is not None:
                    self._admission.admitted = True
                raise
            except (WorkerError, APIError) as error:
                if not await self._serving._retry_admission(
                    error,
                    self._admission,
                    generation=self._generation,
                    started=self._started,
                ):
                    raise
                if self._closed or self._cancel_requested:
                    raise StopAsyncIteration
                if self._session_id is not None:
                    self._admission.pending = True
                self._generation = self._factory()
        raise StopAsyncIteration

    def cancel(self):
        self._cancel_requested = True
        return self._generation.cancel()

    async def aclose(self):
        self._closed = True
        await self._generation.aclose()


@dataclass
class ToolContext:
    schemas: dict = field(default_factory=dict)
    incomplete: bool = False


@dataclass
class _ToolResponse:
    tool_calls: Optional[list[ToolCall]] = None
    reasoning: Optional[str] = None
    content: Optional[str] = None
    remaining_reasoning: Optional[str] = None


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
        reasoning_extractor: Optional[
            Callable[[str], tuple[Optional[str], str]]
        ] = None,
        streaming_parser_factory: Optional[
            Callable[[], StreamingResponseParser]
        ] = None,
        evict_idle_sessions: bool = False,
        streaming_preflight_timeout: Optional[float] = None,
        api_dialect: str = "openai",
        tool_reasoning_parser_factory: Optional[
            Callable[[], StreamingResponseParser]
        ] = None,
        reasoning_token_counter: Optional[Callable[[list[int]], Optional[int]]] = None,
    ):
        self._runtime = runtime
        self._template = template
        self._model_id = model_id
        self._max_context = max_context
        self._prompt_token_offset = prompt_token_offset
        self._content_filter = content_filter
        self._reasoning_extractor = reasoning_extractor
        self._reasoning_token_counter = reasoning_token_counter
        self._streaming_parser_factory = streaming_parser_factory
        self._tool_reasoning_parser_factory = tool_reasoning_parser_factory
        self._evict_idle_sessions = evict_idle_sessions
        self._idle_sessions: dict[str, None] = {}
        self._cleanup_sessions: dict[str, None] = {}
        self._eviction_lock = asyncio.Lock()
        self._streaming_preflight_timeout = streaming_preflight_timeout
        if api_dialect not in ("openai", "openrouter"):
            raise ValueError("api_dialect must be openai or openrouter")
        self.api_dialect = api_dialect
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
        self._native_stops = getattr(runtime, "default_stop_strings", None)
        handled = content_filter_specials or set()
        self._content_specials = [
            t for t in template.special_tokens() if t not in handled
        ]
        # OpenAI/chat-template token-ID warm-resume state. Adapter-side,
        # not runtime; kept in lockstep with the worker's session state by
        # clearing both on reset/close.
        self._transcript = OpenAITranscriptState(template)
        self._multiplexed = getattr(runtime, "supports_multiplexing", False) is True
        self._transactions = _SessionLocks(
            runtime.max_concurrent_requests if self._multiplexed else None
        )

    @property
    def max_context(self) -> Optional[int]:
        return self._max_context

    @property
    def healthy(self) -> bool:
        """Whether the underlying model worker can accept new requests."""
        return self._runtime.healthy

    def error_body(self, error: APIError) -> dict:
        return error.body(openrouter=self.api_dialect == "openrouter")

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

    def _split_reasoning(self, text: str) -> tuple[Optional[str], str]:
        if self._reasoning_extractor is None:
            return None, text
        return self._reasoning_extractor(text)

    @staticmethod
    def _return_reasoning(req: ChatCompletionRequest) -> bool:
        # Response visibility only; the model still computes reasoning on opt-out.
        kwargs = req.chat_template_kwargs or {}
        return kwargs.get("return_reasoning", True)

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

    def _visible_stops(self, req: ChatCompletionRequest) -> list[str]:
        return (
            self._stops
            + ((self._native_stops or []) if req.ignore_eos else [])
            + ([] if self._tools_active(req) else self._content_specials)
        )

    def _generation_stops(self, req: ChatCompletionRequest) -> list[str]:
        stops = self._request_stops(req)
        if req.ignore_eos:
            return stops
        return stops + self._visible_stops(req)

    @staticmethod
    def _apply_stop(text: str, stops: list[str]) -> str:
        """Truncate at the earliest stop string (the stop itself is excluded)."""
        cut = _earliest_stop(text, stops)
        return text[:cut] if cut is not None else text

    def _trim_stop_fragment(self, text: str, req: ChatCompletionRequest) -> str:
        # A native string stop inside a special token can remove its remainder.
        # Hide only suffixes that would be completed by a requested stop.
        cut = len(text)
        for marker in self._visible_stops(req):
            for stop in self._request_stops(req):
                at = marker.find(stop)
                if at > 0 and text.endswith(marker[:at]):
                    cut = min(cut, len(text) - at)
        return text[:cut]

    def _truncate_raw(
        self, text: str, req: ChatCompletionRequest, stopped: bool = False
    ) -> str:
        """Cut raw model output at the earliest special token or request stop
        sequence BEFORE tool parsing, so a tool call (or any text) past the stop
        boundary is neither parsed nor emitted."""
        visible_cut = _earliest_stop(text, self._visible_stops(req))
        request_cut = _earliest_stop(text, self._request_stops(req))
        cuts = [cut for cut in (visible_cut, request_cut) if cut is not None]
        text = text[: min(cuts)] if cuts else text
        return (
            self._trim_stop_fragment(text, req)
            if req.ignore_eos
            and stopped
            and (visible_cut is None or request_cut == min(cuts))
            else text
        )

    def _cancel_generation(self, generation: AsyncIterator[str]) -> None:
        cancel = getattr(generation, "cancel", None)
        if callable(cancel):
            cancel()
        elif not self._multiplexed:
            self._runtime.stop()
        else:
            raise WorkerError(
                "multiplexed runtime requires request-scoped cancellation"
            )

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
                self._cancel_generation(stream)
                async for _ in stream:  # drain so stats_cb fires
                    pass
                break
        return text, stopped

    def _parse_tools(
        self, req, text, finish_reason, raw_text, context, request_stopped
    ):
        return self._tool_detector_cls().detect_and_parse(text, self._tool_schemas(req))

    def _extract_response(
        self,
        req: ChatCompletionRequest,
        text: str,
        *,
        finish_reason=None,
        context=None,
        stopped=False,
        matched_stop=None,
    ):
        """Return tool calls, optional reasoning, and user-visible content."""
        # Generic detectors see visible text; recipient-aware adapters also need
        # raw framing to distinguish tool messages from quoted examples.
        context = context if context is not None else ToolContext()
        request_cut = _earliest_stop(text, self._request_stops(req))
        visible_cut = _earliest_stop(text, self._visible_stops(req))
        request_stops = self._request_stops(req)
        request_stopped = (
            any(
                matched_stop and matched_stop.startswith(stop) for stop in request_stops
            )
            or (
                matched_stop is None and finish_reason == "stop" and bool(request_stops)
            )
            or (
                request_cut is not None
                and (visible_cut is None or request_cut <= visible_cut)
            )
        )
        raw_text = self._truncate_raw(text, req, stopped)
        reasoning, text = self._split_reasoning(raw_text)
        tool_calls = None
        if self._tools_active(req):
            parsed = self._parse_tools(
                req, text, finish_reason, raw_text, context, request_stopped
            )
            context.incomplete = parsed.incomplete
            if parsed.calls:
                tool_calls = [self._to_openai_tool_call(c) for c in parsed.calls]
            text = parsed.normal_text
        if not self._return_reasoning(req):
            reasoning = None
        return (
            tool_calls,
            reasoning,
            self._visible_content(text) or None,
        )

    def _usage(self, stats: GenStats, *, ignore_eos=False) -> Usage:
        reasoning_tokens = None
        if (
            not ignore_eos
            and self._reasoning_token_counter
            and stats.generated_token_ids is not None
        ):
            reasoning_tokens = self._reasoning_token_counter(stats.generated_token_ids)
        return Usage(
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            total_tokens=stats.prompt_tokens + stats.completion_tokens,
            prompt_tokens_details=PromptTokensDetails(
                cached_tokens=stats.reused_prompt_tokens
            ),
            completion_tokens_details=(
                CompletionTokensDetails(reasoning_tokens=reasoning_tokens)
                if reasoning_tokens is not None
                else None
            ),
        )

    @staticmethod
    def _log_generation_stats(
        session_id: Optional[str], stats: GenStats, finish: str
    ) -> None:
        message = (
            "llm_turn_stats session_id=%s reason=%s prompt_tokens=%d "
            "reused_prompt_tokens=%d prefilled_prompt_tokens=%d "
            "completion_tokens=%d prefill_ms=%.1f prefill_tok_s=%.1f "
            "decode_ms=%.1f decode_tok_s=%.1f total_ms=%.1f finish=%s"
        )
        args = (
            session_id or "<scratch>",
            stats.session_reset_reason,
            stats.prompt_tokens,
            stats.reused_prompt_tokens,
            stats.prefilled_prompt_tokens,
            stats.completion_tokens,
            stats.prefill_ms,
            stats.prefill_tok_s,
            stats.decode_ms,
            stats.decode_tok_s,
            stats.total_ms,
            finish,
        )
        if stats.vision_encoder_ms is not None:
            message += " vision_encoder_ms=%.1f"
            args += (stats.vision_encoder_ms,)
        logger.info(message, *args)

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
            top_p=req.top_p if req.top_p is not None else 1.0,
            top_k=req.top_k if req.top_k is not None else 0,
            seed=req.seed if req.seed is not None else 0,
            # Worker stop set, chosen per path in create() (see __init__ for the
            # two sets); the server re-applies it in _clean/_collect_until_stop.
            stop=stops,
            ignore_eos=req.ignore_eos,
        )

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        # Keep it boring: non-empty printable ASCII (no spaces/control), <=128.
        if not session_id or len(session_id) > _SESSION_ID_MAX_LEN:
            raise InvalidSessionId(f"must be 1-{_SESSION_ID_MAX_LEN} characters")
        if not all(0x21 <= ord(c) <= 0x7E for c in session_id):
            raise InvalidSessionId("must be printable ASCII with no spaces")

    @staticmethod
    def _generation_error(error: Exception) -> APIError:
        if isinstance(error, APIError):
            return error
        if isinstance(error, WorkerError):
            if (
                error.code == "context_length_exceeded"
                and error.context_overflow is not None
            ):
                counts = error.context_overflow
                return ContextLengthExceeded(
                    counts["prompt_tokens"],
                    counts["context_length"],
                    counts["completion_tokens"],
                )
            if error.code in (
                "capacity_exhausted",
                "session_capacity_exhausted",
                "unsupported_session",
            ):
                return SessionCapacity(error.code)
            if error.code is not None:
                status = {
                    "session_not_found": 404,
                    "session_busy": 409,
                    "not_ready": 503,
                    "invalid_argument": 400,
                    "context_length_exceeded": 400,
                    "unsupported_parameter": 400,
                }.get(error.code, 500)
                return APIError(
                    status,
                    f"Generation failed: {error}",
                    "invalid_request_error" if status < 500 else "server_error",
                    error.code,
                )
        return GenerationError(str(error))

    async def _preflight_session(self, session_id: str, admission) -> None:
        """Reserve the session before any response bytes are emitted so a
        capacity refusal becomes an HTTP status, not an SSE error event."""
        try:
            await self._runtime.open(session_id)
        except WorkerError as e:
            if not self._multiplexed and e.code == "capacity_exhausted":
                raise SessionCapacity("session_capacity_exhausted") from e
            raise self._generation_error(e)
        if self._evict_idle_sessions:
            admission.admitted = True

    async def close_session(self, session_id: str) -> None:
        self._validate_session_id(session_id)
        try:
            async with self._transactions.hold(session_id):
                await self._close_session(session_id)
        except WorkerError as e:
            raise self._generation_error(e)

    async def _close_session(self, session_id):
        try:
            existed = await self._runtime.close(session_id)
        except WorkerError as error:
            if error.code == "session_not_found":
                self._transcript.close(session_id)
                self._idle_sessions.pop(session_id, None)
                self._cleanup_sessions.pop(session_id, None)
            raise
        self._transcript.close(session_id)
        self._idle_sessions.pop(session_id, None)
        self._cleanup_sessions.pop(session_id, None)
        return existed

    async def reset_session(self, session_id: str) -> None:
        self._validate_session_id(session_id)
        try:
            async with self._transactions.hold(session_id):
                try:
                    await self._runtime.reset(session_id)
                except BaseException as error:
                    missing = (
                        isinstance(error, WorkerError)
                        and error.code == "session_not_found"
                    )
                    # Missing native state is already reset at the HTTP boundary.
                    # A failed cold replacement does not restore the old state.
                    if self._multiplexed or missing:
                        self._transcript.reset(session_id)
                    if not missing:
                        raise
                else:
                    self._transcript.reset(session_id)
        except WorkerError as e:
            raise self._generation_error(e)

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
    def _reject_invalid_visibility(req: ChatCompletionRequest) -> None:
        """Require boolean, mutually consistent reasoning visibility controls."""
        template_kwargs = req.chat_template_kwargs or {}
        if not isinstance(template_kwargs.get("return_reasoning", True), bool):
            raise APIError(
                400,
                "chat_template_kwargs.return_reasoning must be a boolean.",
                "invalid_request_error",
                "invalid_value",
            )
        visibility = []
        if "return_reasoning" in template_kwargs:
            visibility.append(template_kwargs["return_reasoning"])
        if req.include_reasoning is not None:
            visibility.append(req.include_reasoning)
        if req.reasoning is not None and req.reasoning.get("exclude") is not None:
            if type(req.reasoning["exclude"]) is not bool:
                raise APIError(
                    400,
                    "reasoning.exclude must be a boolean",
                    "invalid_request_error",
                    "invalid_value",
                )
            visibility.append(not req.reasoning["exclude"])
        if len(set(visibility)) > 1:
            raise APIError(
                400,
                "Conflicting reasoning visibility controls",
                "invalid_request_error",
                "invalid_value",
            )

    @staticmethod
    def _reject_invalid_values(req: ChatCompletionRequest) -> None:
        """Reject invalid types/ranges (invalid_value); these take precedence over
        the unsupported-parameter error."""
        ServingChat._reject_invalid_visibility(req)
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
        if req.top_p is not None and (
            not math.isfinite(req.top_p) or req.top_p <= 0.0 or req.top_p > 1.0
        ):
            raise APIError(
                400,
                f"top_p must be between 0 (exclusive) and 1 (got {req.top_p}).",
                "invalid_request_error",
                "invalid_value",
            )
        if req.top_k is not None and not (0 <= req.top_k <= 2**31 - 1):
            raise APIError(
                400,
                f"top_k must be between 0 and {2**31 - 1} (got {req.top_k}).",
                "invalid_request_error",
                "invalid_value",
            )
        if req.seed is not None and not (0 < req.seed <= 2**63 - 1):
            raise APIError(
                400,
                f"seed must be between 1 and {2**63 - 1} (got {req.seed}).",
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

    def _reject_unsupported_params(self, req: ChatCompletionRequest) -> None:
        """Reject params we don't honor rather than silently ignoring them (a
        client relying on e.g. penalties/logprobs would otherwise get wrong
        behavior). Only the no-op/default value of each passes: penalties 0;
        response_format type "text"; tool_choice none/auto/unset;
        parallel_tool_calls true (false can't be guaranteed without
        constraining); logprobs are not returned at all."""
        rf = req.response_format
        flags = [
            (req.thinking_token_budget is not None, "thinking_token_budget"),
            (req.grammar is not None, "grammar"),
            (req.guided_json is not None, "guided_json"),
            (req.guided_regex is not None, "guided_regex"),
            (req.guided_choice is not None, "guided_choice"),
            (req.guided_grammar is not None, "guided_grammar"),
            (req.structured_outputs is not None, "structured_outputs"),
            (req.regex is not None, "regex"),
            (req.ebnf is not None, "ebnf"),
            (req.stop_regex is not None, "stop_regex"),
            (
                req.add_generation_prompt is not None
                and req.add_generation_prompt is not True,
                "add_generation_prompt",
            ),
            (
                req.continue_final_message is not None
                and req.continue_final_message is not False,
                "continue_final_message",
            ),
            (req.cache_salt is not None, "cache_salt"),
            (
                req.ignore_eos
                and getattr(self._runtime, "supports_ignore_eos", False) is not True,
                "ignore_eos (requires an updated multiplexed worker)",
            ),
            (
                req.ignore_eos and self._native_stops is None,
                "ignore_eos (requires decoded native EOS markers from the worker)",
            ),
            (req.n != 1, "n>1"),
            (req.reasoning_effort is not None, "reasoning_effort"),
            (
                req.reasoning
                not in (
                    None,
                    {},
                    {"exclude": None},
                    {"exclude": False},
                    {"exclude": True},
                ),
                "reasoning",
            ),
            (bool(req.frequency_penalty), "frequency_penalty"),
            (bool(req.presence_penalty), "presence_penalty"),
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
                "temperature, top_p, top_k, seed, max_tokens/max_completion_tokens, "
                "stop, and tools for the configured tool-call format.",
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

    @asynccontextmanager
    async def _transaction(self, session_id):
        admission = _AdmissionState()
        async with self._transactions.hold(session_id):
            known = session_id in self._idle_sessions
            self._idle_sessions.pop(session_id, None)
            try:
                yield admission
            finally:
                if self._evict_idle_sessions and session_id is not None:
                    if known or admission.admitted:
                        self._cleanup_sessions.pop(session_id, None)
                        self._idle_sessions[session_id] = None
                    elif admission.pending:
                        self._cleanup_sessions[session_id] = None
                        if not self._runtime.session_in_use(session_id):
                            cleanup = asyncio.create_task(
                                self._close_session(session_id)
                            )
                            try:
                                await SessionRuntime._finish_cleanup(cleanup)
                            except WorkerError as error:
                                if error.code != "session_not_found":
                                    logger.warning(
                                        "Could not close unconfirmed session %s",
                                        session_id,
                                    )

    async def _evict_idle_session(self) -> bool:
        if not self._evict_idle_sessions:
            return False
        async with self._eviction_lock:
            for session_id in [*self._cleanup_sessions, *self._idle_sessions]:
                if self._transactions.in_use(
                    session_id
                ) or self._runtime.session_in_use(session_id):
                    continue  # Includes queued callers, not just the owner.
                # The admitting chat already owns a request slot.
                async with self._transactions.hold(session_id, count_request=False):
                    try:
                        existed = await self._close_session(session_id)
                        if existed is False:
                            continue
                    except WorkerError as error:
                        if error.code == "session_not_found":
                            continue
                        if error.admitted is False:
                            return False  # The close never reached the worker.
                        candidates = (
                            self._cleanup_sessions
                            if session_id in self._cleanup_sessions
                            else self._idle_sessions
                        )
                        candidates.pop(session_id, None)
                        candidates[session_id] = None
                        return False
                return True
        return False

    async def create(
        self, req: ChatCompletionRequest, *, observer: Optional[Turn] = None
    ):
        """Create a response; activate an optional HUD turn under the session lock."""
        if req.model is not None and req.model != self._model_id:
            raise ModelNotFound(req.model, self._model_id)
        self._reject_invalid_values(req)
        visible = req.include_reasoning
        if (
            visible is None
            and req.reasoning is not None
            and req.reasoning.get("exclude") is not None
        ):
            visible = not req.reasoning["exclude"]
        if visible is not None:
            kwargs = dict(req.chat_template_kwargs or {})
            kwargs["return_reasoning"] = visible
            req = req.model_copy(update={"chat_template_kwargs": kwargs})
        if req.session_id is not None:
            self._validate_session_id(req.session_id)
        lease = self._transaction(req.session_id)
        entered = False
        try:
            admission = await lease.__aenter__()
            entered = True
            if observer is not None:
                observer.activate()
            result = await self._create(req, lease, admission, observer)
            if req.stream:
                entered = False  # The returned iterator owns the transaction.
            return result
        except WorkerError as error:
            raise self._generation_error(error)
        finally:
            if entered:
                await lease.__aexit__(None, None, None)

    async def _create(
        self,
        req: ChatCompletionRequest,
        lease,
        admission,
        observer: Optional[Turn] = None,
    ):
        self._reject_unsupported_params(req)
        if req.session_id is not None:
            self._validate_session_id(req.session_id)
        # tool_choice="none" must hide tools from the model: if we still render
        # the tool schemas, the model can emit a <tool_call> that we'd surface as
        # plain text (parsing is disabled), instead of a normal answer.
        template_tools = None if req.tool_choice == "none" else req.tools
        template_kwargs = dict(req.chat_template_kwargs or {})
        template_kwargs.pop("return_reasoning", None)
        prompt = self._template.render(
            req.messages, tools=template_tools, template_kwargs=template_kwargs
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
            template_kwargs=template_kwargs,
        )
        # Pre-flight context check against the tokens the worker will actually
        # assemble: for segments that is sum(len(ids)) + tokenized text, not the
        # rendered string, so a near-limit prompt agrees with the worker rather
        # than false-400ing or failing mid-decode. Only when a tokenizer exists.
        if self._max_context:
            count = self._count_prompt_tokens(prompt_input)
            if count is not None:
                requested = req.resolved_max_tokens()
                if count >= self._max_context:
                    raise ContextLengthExceeded(count, self._max_context, requested)
                # An explicit max_tokens that wouldn't fit alongside the prompt
                # must be rejected here, not run until the worker hits the context
                # limit mid-decode (a 500 / streaming error after partial output).
                if requested > 0 and count + requested > self._max_context:
                    raise ContextLengthExceeded(count, self._max_context, requested)
        # Per-path worker stop set (see __init__ for the two sets and why): tool
        # turns use the narrow set; plain chat adds the broad content specials.
        gen_stops = self._generation_stops(req)
        options = self._options(req, gen_stops)
        # The generation scaffold the worker will prefill ahead of this turn's
        # tokens (e.g. Qwen3 <think> block), resolved with the same per-request
        # mode AND tools as the render; recorded per turn so warm-resume splicing
        # reproduces the exact resident scaffold even if the mode changes between
        # requests.
        preamble = self._template.generation_preamble(
            template_kwargs, tools=template_tools
        )
        return await self._create_response(
            req,
            prompt_input,
            options,
            preamble,
            gen_stops,
            lease,
            admission,
            observer=observer,
        )

    async def _retry_admission(
        self, error, admission, *, generation=None, started=False
    ):
        slots_exhausted = _session_capacity_exhausted(error)
        if not started and (
            slots_exhausted
            or isinstance(error, WorkerError)
            and error.admitted is False
        ):
            admission.pending = False
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        if started or admission.retried or not slots_exhausted:
            return False
        admission.retried = True
        if generation is not None:
            await generation.aclose()
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        evicted = await self._evict_idle_session()
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        return evicted

    async def _create_response(
        self,
        req,
        prompt_input,
        options,
        preamble,
        gen_stops,
        lease,
        admission,
        tool_context=None,
        observer: Optional[Turn] = None,
    ):
        tool_context = tool_context if tool_context is not None else ToolContext()
        while True:
            try:
                return await self._start_response(
                    req,
                    prompt_input,
                    options,
                    preamble,
                    gen_stops,
                    lease,
                    admission,
                    tool_context,
                    observer,
                )
            except (WorkerError, APIError) as error:
                if not await self._retry_admission(error, admission):
                    raise

    async def _start_response(
        self,
        req,
        prompt_input,
        options,
        preamble,
        gen_stops,
        lease,
        admission,
        tool_context,
        observer: Optional[Turn] = None,
    ):
        # Legacy workers need explicit admission. Multiplexed generation opens
        # implicitly from its full prompt, preserving new-session prefix reuse.
        if req.session_id is not None and not self._multiplexed:
            await self._preflight_session(req.session_id, admission)
        if (
            self._evict_idle_sessions
            and self._multiplexed
            and req.session_id is not None
        ):
            admission.pending = True
        stats = GenStats()
        generation = self._runtime.generate_stream(
            req.session_id,
            prompt_input,
            options,
            stats,
            observer=observer,
        )
        if req.stream and self._multiplexed:
            await generation.wait_ready(timeout=self._streaming_preflight_timeout)
        if self._multiplexed and self._evict_idle_sessions:
            generation = _AdmissionRetryStream(
                self,
                req.session_id,
                generation,
                lambda: self._runtime.generate_stream(
                    req.session_id,
                    prompt_input,
                    options,
                    stats,
                    observer=observer,
                ),
                admission,
            )
        if req.stream:
            return _ChatStream(
                self._stream(
                    req, preamble, gen_stops, generation, stats, tool_context, observer
                ),
                lease,
                generation,
            )
        return await self._complete(
            req, preamble, gen_stops, generation, stats, tool_context
        )

    async def _complete(
        self,
        req: ChatCompletionRequest,
        preamble: str,
        gen_stops: list[str],
        generation,
        stats: GenStats,
        tool_context,
    ) -> ChatCompletionResponse:
        # Same stop set the worker was given (narrow for tools, broad for chat).
        stops = gen_stops
        try:
            # Collect raw text (markers intact for tool parsing), halting early
            # at a stop boundary (special token or request stop).
            async with aclosing(generation):
                text, stopped = await self._collect_until_stop(generation, stops)
        except (WorkerError, APIError):
            raise
        except Exception as e:  # noqa: BLE001 - surface as a structured API error
            raise self._generation_error(e)
        # Bound the raw output at the first stop/special token BEFORE tool
        # parsing, so a call after the stop boundary is not parsed/emitted.
        tool_calls, reasoning, content = self._extract_response(
            req,
            text,
            context=tool_context,
            matched_stop=stats.matched_stop,
            stopped=stopped or stats.finish_reason == "stop",
            finish_reason=self._finish_reason(
                req,
                stats.completion_tokens,
                stopped=stopped,
                worker_finish=stats.finish_reason,
            ),
        )
        # Compare future echoes against the finalized client-visible response.
        self._transcript.record_assistant_turn(
            session_id=req.session_id,
            request_messages=req.messages,
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning,
            # Hidden post-EOS tokens must not be spliced into a later echo.
            generated_token_ids=(
                None
                if req.ignore_eos or tool_context.incomplete
                else stats.generated_token_ids
            ),
            prior_turns=sum(1 for m in req.messages if m.role == "assistant"),
            preamble=preamble,
        )
        finish = self._finish_reason(
            req, stats.completion_tokens, tool_calls, stopped, stats.finish_reason
        )
        self._log_generation_stats(req.session_id, stats, finish)
        return ChatCompletionResponse(
            model=self._model_id,
            choices=[
                Choice(
                    message=ResponseMessage(
                        content=content,
                        reasoning_content=reasoning,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish,
                )
            ],
            usage=self._usage(stats, ignore_eos=req.ignore_eos),
        )

    async def _stream_plain_content(
        self,
        generation: AsyncIterator[str],
        stops: list[str],
        stop_hit: list[bool],
    ) -> AsyncIterator[str]:
        if self._content_filter is not None:
            raw, stopped = await self._collect_until_stop(generation, stops)
            stop_hit[0] = stop_hit[0] or stopped
            content = self._visible_content(self._apply_stop(raw, stops))
            if content:
                yield content
            return

        def on_stop():
            stop_hit[0] = True
            self._cancel_generation(generation)

        async for token in self._clean(
            generation,
            stops,
            on_stop=on_stop,
        ):
            yield token

    async def _clean_ignored_eos(
        self,
        req: ChatCompletionRequest,
        generation: AsyncIterator[str],
        stop_hit: list[bool],
        stats: GenStats,
    ) -> AsyncIterator[str]:
        request_stops = self._request_stops(req)
        visible_stops = self._visible_stops(req) + request_stops
        hold = max((len(s) for s in visible_stops), default=1) - 1
        buf = ""
        visible = True
        async for token in generation:
            buf += token
            request_cut = _earliest_stop(buf, request_stops)
            if not stop_hit[0] and request_cut is not None:
                stop_hit[0] = True
                self._cancel_generation(generation)
            if visible:
                cut = _earliest_stop(buf, visible_stops)
                if cut is not None:
                    text = buf[:cut]
                    if cut == request_cut:
                        text = self._trim_stop_fragment(text, req)
                    if text:
                        yield text
                    visible = False
                elif len(buf) > hold:
                    yield buf[: len(buf) - hold] if hold else buf
            # Continue watching request stops after the visible turn ends.
            buf = buf[-hold:] if hold else ""
        if visible and buf:
            if stats.finish_reason == "stop":
                buf = self._trim_stop_fragment(buf, req)
            if buf:
                yield buf

    async def _stream_content_deltas(
        self,
        req: ChatCompletionRequest,
        generation: AsyncIterator[str],
        stops: list[str],
        stop_hit: list[bool],
        stats: GenStats,
    ) -> AsyncIterator[DeltaMessage]:
        if req.ignore_eos:
            generation = self._clean_ignored_eos(req, generation, stop_hit, stats)
            stops = []
        if self._streaming_parser_factory is None:
            async for text in self._stream_plain_content(generation, stops, stop_hit):
                yield DeltaMessage(content=text)
            return
        parser = self._streaming_parser_factory()
        return_reasoning = self._return_reasoning(req)

        def on_stop():
            stop_hit[0] = True
            self._cancel_generation(generation)

        async def parsed():
            async for text in self._clean(generation, stops, on_stop=on_stop):
                for delta in parser.feed(text):
                    yield delta
            # Finalize only a successful logical EOF, including a drained stop.
            for delta in parser.finish():
                yield delta

        async for delta in parsed():
            visible = DeltaMessage(
                content=delta.content,
                reasoning_content=delta.reasoning_content if return_reasoning else None,
            )
            if visible.content or visible.reasoning_content:
                yield visible

    async def _stream_final_chunks(
        self,
        req: ChatCompletionRequest,
        stats: GenStats,
        use_tools: bool,
        tool_calls,
        reasoning: Optional[str],
        content: Optional[str],
        stopped: bool,
        chunk,
    ) -> AsyncIterator[str]:
        if use_tools:
            if reasoning:
                yield chunk(DeltaMessage(reasoning_content=reasoning))
            if content:
                yield chunk(DeltaMessage(content=content))
            for tool_call in tool_calls or []:
                yield chunk(DeltaMessage(tool_calls=[tool_call]))
            finish = self._finish_reason(
                req,
                stats.completion_tokens,
                tool_calls,
                stopped,
                stats.finish_reason,
            )
        else:
            finish = self._finish_reason(
                req,
                stats.completion_tokens,
                stopped=stopped,
                worker_finish=stats.finish_reason,
            )

        self._log_generation_stats(req.session_id, stats, finish)
        yield chunk(DeltaMessage(), finish=finish)

    async def _stream_tool_output(self, generation, stops, stop_hit, parts, parser):
        if parser is None:
            text, stopped = await self._collect_until_stop(generation, stops)
            parts.append(text)
            stop_hit[0] = stop_hit[0] or stopped
            return

        async def capture():
            async for token in generation:
                parts.append(token)
                yield token

        def stop():
            stop_hit[0] = True
            self._cancel_generation(generation)

        async for part in self._clean(capture(), stops, stop):
            for delta in parser.feed(part):
                yield delta
        for delta in parser.finish():
            yield delta

    async def _stream_tool_response(
        self, req, generation, stops, stop_hit, stats, response, tool_context
    ):
        parts, reasoning_parts = [], []
        parser = (
            self._tool_reasoning_parser_factory()
            if self._tool_reasoning_parser_factory and self._return_reasoning(req)
            else None
        )

        async for delta in self._stream_tool_output(
            generation, stops, stop_hit, parts, parser
        ):
            if delta.reasoning_content:
                reasoning_parts.append(delta.reasoning_content)
                yield DeltaMessage(reasoning_content=delta.reasoning_content)
        response.tool_calls, response.reasoning, response.content = (
            self._extract_response(
                req,
                "".join(parts),
                context=tool_context,
                matched_stop=stats.matched_stop,
                stopped=stop_hit[0] or stats.finish_reason == "stop",
                finish_reason=self._finish_reason(
                    req,
                    stats.completion_tokens,
                    stopped=stop_hit[0],
                    worker_finish=stats.finish_reason,
                ),
            )
        )
        response.remaining_reasoning = response.reasoning
        if parser is not None:
            emitted = "".join(reasoning_parts)
            response.remaining_reasoning = None
            if response.reasoning and response.reasoning.startswith(emitted):
                response.remaining_reasoning = (
                    response.reasoning[len(emitted) :] or None
                )
            if emitted and not (response.reasoning or "").startswith(emitted):
                logger.warning("Streamed reasoning differs from buffered extraction")
                response.reasoning = emitted

    async def _stream(  # noqa: C901 - response paths share one cancellation boundary
        self,
        req: ChatCompletionRequest,
        preamble: str,
        stops: list[str],
        generation: AsyncIterator[str],
        stats: GenStats,
        tool_context,
        observer: Optional[Turn] = None,
    ) -> AsyncIterator[str]:
        cid = _new_id("chatcmpl")

        def chunk(delta: DeltaMessage, finish=None) -> str:
            c = ChatCompletionChunk(
                id=cid,
                model=self._model_id,
                choices=[ChunkChoice(delta=delta, finish_reason=finish)],
                usage=(
                    self._usage(stats, ignore_eos=req.ignore_eos)
                    if finish is not None
                    and self.api_dialect == "openrouter"
                    and req.stream_options
                    and req.stream_options.include_usage
                    else None
                ),
            )
            return f"data: {c.model_dump_json(exclude_none=True)}\n\n"

        yield chunk(DeltaMessage(role="assistant"))
        error: Optional[Exception] = None
        use_tools = self._tools_active(req)
        tool_calls = None
        reasoning = None
        content = None
        remaining_tool_reasoning = None

        stop_hit = [False]  # set when a stop boundary is reached (forces finish="stop")
        # Per-path stop set from create(): for plain chat this includes the broad
        # content specials, so _clean cuts a leaked special out of the stream (and
        # the worker, given the same set, halts + omits ids -> non-resumable turn).
        try:
            if use_tools:
                # Stream safe reasoning, then validate complete tool calls.
                response = _ToolResponse()
                async for delta in self._stream_tool_response(
                    req, generation, stops, stop_hit, stats, response, tool_context
                ):
                    yield chunk(delta)
                tool_calls = response.tool_calls
                reasoning = response.reasoning
                content = response.content
                remaining_tool_reasoning = response.remaining_reasoning
            elif (
                self._reasoning_extractor is not None
                and self._streaming_parser_factory is None
            ):
                raw, stop_hit[0] = await self._collect_until_stop(
                    generation,
                    stops,
                )
                tool_calls, reasoning, content = self._extract_response(
                    req,
                    raw,
                    context=tool_context,
                    matched_stop=stats.matched_stop,
                    stopped=stop_hit[0] or stats.finish_reason == "stop",
                )
                if reasoning:
                    yield chunk(DeltaMessage(reasoning_content=reasoning))
                if content:
                    yield chunk(DeltaMessage(content=content))
            else:
                content_parts, reasoning_parts = [], []
                async for delta in self._stream_content_deltas(
                    req, generation, stops, stop_hit, stats
                ):
                    content_parts.append(delta.content or "")
                    reasoning_parts.append(delta.reasoning_content or "")
                    yield chunk(delta)
                content = "".join(content_parts)  # for the session fingerprint
                reasoning = "".join(reasoning_parts) or None
        except (
            Exception
        ) as e:  # noqa: BLE001 - emit a structured error event, never drop the socket
            error = e
        finally:
            await generation.aclose()

        if error is not None:
            api_error = self._generation_error(error)
            record_error(observer, api_error.code or api_error.err_type)
            err = self.error_body(api_error)
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"
            return
        self._transcript.record_assistant_turn(
            session_id=req.session_id,
            request_messages=req.messages,
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning or None,
            # Hidden post-EOS tokens must not be spliced into a later echo.
            generated_token_ids=(
                None
                if req.ignore_eos or tool_context.incomplete
                else stats.generated_token_ids
            ),
            prior_turns=sum(1 for m in req.messages if m.role == "assistant"),
            preamble=preamble,
        )

        async for final_chunk in self._stream_final_chunks(
            req,
            stats,
            use_tools,
            tool_calls,
            remaining_tool_reasoning if use_tools else reasoning,
            content,
            stop_hit[0],
            chunk,
        ):
            yield final_chunk
        if (
            req.stream_options
            and req.stream_options.include_usage
            and self.api_dialect != "openrouter"
        ):
            usage_chunk = ChatCompletionChunk(
                id=cid,
                model=self._model_id,
                choices=[],
                usage=self._usage(stats, ignore_eos=req.ignore_eos),
            )
            yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"
