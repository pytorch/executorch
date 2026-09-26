# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenAI/chat-template transcript state for token-ID warm resume.

Adapter-specific glue, not runtime infrastructure: it knows ChatMessages,
tool_calls, the ChatTemplate, sentinels, and assistant fingerprints (the runtime
only sees a PromptInput). It makes warm resume survive the chat template's lossy
re-render of prior assistant turns -- especially tool calls, which re-render from
parsed structure and don't re-tokenize to what the model generated.

Per session it stores, for each assistant turn it produced, the exact generated
token ids and a fingerprint of the response. On the next request each prior
assistant turn is replaced with a sentinel, the conversation is rendered once,
and the rendered text is split on the sentinels with the stored ids spliced back
in -- only for turns whose content/tool fingerprint and any supplied reasoning string
match the recorded response, and whose ids are present (a stop-trimmed turn is
left as text). An edit invalidates that turn and all later records. The worker's
exact-token prefix check separately protects KV reuse.
"""

import hashlib
import json
import re
import uuid
from typing import Optional

from .chat_template import ChatTemplate
from .protocol import ChatMessage
from .session_runtime import PromptInput

# The assistant header that precedes a turn's generation scaffold + content.
_ASSIST_HDR = "<|im_start|>assistant\n"
# A tail past an assistant-header match is template framing only if it is
# positively recognized: empty, or a recipient line closed by the message
# marker (e.g. ` to=user<|message|>`). Templates terminate turns in
# unbounded ways (`[/INST]`, `<|eot|>`, `<|im_end|>`, ...), so no blocklist
# of delimiters can enumerate turn structure exhaustively -- and anything
# unrecognized means the match sits inside message content (e.g. a user
# message quoting a header literal), where truncating would delete real
# conversation text. Unknown framing falls back to plain text (correct
# output, just no reuse).
_FRAMING_RE = re.compile(
    r"\A ?to=[A-Za-z0-9_.]{1,64}(?: constrain=[A-Za-z0-9_]+)?<\|message\|>\Z"
)
# A scaffold region is exactly empty (history strips it before the last user) or
# one of the Qwen3 think scaffolds (history preserves the empty block after the
# last user; the open form is the think-mode generation preamble). Anything else
# in that region is unrecognized -> the splice falls back to plain text.
_THINK_SCAFFOLD_RE = re.compile(r"\A(?:<think>\n\n</think>\n\n|<think>\n)?\Z")
_GEMMA_TOOL_CALL_START = "<|tool_call>"
_GEMMA_TOOL_CALL_END = "<tool_call|>"
_GEMMA_QUOTE = '<|"|>'


def _normalize_tool_args(args):
    """OpenAI tool-call ``arguments`` are JSON strings a client may reserialize
    with different whitespace or key order while preserving the same value (e.g.
    a server-emitted ``{"command": "x"}`` echoed back compact as
    ``{"command":"x"}``). Parse to an object so the fingerprint compares the
    semantic payload, not bytes -- the outer sort_keys dump then canonicalizes
    it. A non-JSON string (or already-structured args) is returned unchanged, so
    it stays byte-sensitive."""
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (ValueError, TypeError):
            return args
    return args


def _find_gemma_tool_call_span(rendered: str, search_pos: int):
    start = rendered.find(_GEMMA_TOOL_CALL_START, search_pos)
    if start == -1:
        return None
    i = start + len(_GEMMA_TOOL_CALL_START)
    in_string = False
    depth = 0
    saw_object = False
    while i < len(rendered):
        if rendered.startswith(_GEMMA_QUOTE, i):
            in_string = not in_string
            i += len(_GEMMA_QUOTE)
            continue
        if not in_string:
            if rendered.startswith(_GEMMA_TOOL_CALL_END, i):
                if saw_object and depth == 0:
                    return start, i + len(_GEMMA_TOOL_CALL_END)
            ch = rendered[i]
            if ch in "{[":
                depth += 1
                saw_object = saw_object or ch == "{"
            elif ch in "}]":
                if depth == 0:
                    return None
                depth -= 1
        i += 1
    return None


class OpenAITranscriptState:
    def __init__(self, template: ChatTemplate):
        self._template = template
        # Note: the method's mere presence says nothing about whether the
        # header is right for the template -- production adapters always
        # provide it, even when it returns the default for a template that
        # renders different framing. The tail-shape check below is the only
        # boundary verification.
        _header_fn = getattr(template, "assistant_header", None)
        self._assist_hdr = _header_fn() if _header_fn else _ASSIST_HDR
        # Keyed by assistant-turn index, including gaps after invalidation.
        self._turns: dict[str, dict[int, dict]] = {}

    @staticmethod
    def _assistant_fingerprint(content, tool_calls) -> str:
        """Stable fingerprint of an assistant turn's semantic payload (content +
        each tool call's name/arguments; the random call id is ignored). Used to
        confirm an incoming assistant message is the one we generated before
        splicing its stored token ids."""
        norm = []
        for tc in tool_calls or []:
            fn = getattr(tc, "function", None)
            if fn is not None:
                name, args = getattr(fn, "name", None), getattr(fn, "arguments", None)
            elif isinstance(tc, dict):
                f = tc.get("function", {})
                name, args = f.get("name"), f.get("arguments")
            else:
                continue
            norm.append([name, _normalize_tool_args(args)])
        blob = json.dumps([content or "", norm], sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def _reasoning_fingerprint(reasoning_content: Optional[str]) -> Optional[bytes]:
        if reasoning_content is None:
            return None
        return hashlib.sha256(reasoning_content.encode("utf-8")).digest()

    def _normalize_scaffold(self, text_chunk: str, preamble: str) -> Optional[str]:
        """Force the scaffold region (between the last assistant header in
        `text_chunk` and its end) to equal `preamble`, so the worker re-tokenizes
        the exact resident scaffold. The region is empty (history stripped it ->
        insert) or a think scaffold (history preserved it -> replace). Returns the
        adjusted text, or None if it isn't a recognized scaffold (-> text fallback)."""
        if not self._assist_hdr:
            return None
        h = text_chunk.rfind(self._assist_hdr)
        if h == -1:
            # Without a verified boundary, splicing can duplicate template framing.
            return None
        base = h + len(self._assist_hdr)
        if not preamble:
            # No generation scaffold: the worker prefills nothing ahead of the
            # turn's raw tokens, so the spliced ids begin immediately after the
            # header. Drop the template's re-rendered framing (recipient /
            # message markers), which describes the echoed message rather than
            # the raw generation -- e.g. a thinking turn was generated as
            # ` to=self<|message|>...` but re-renders as ` to=user<|message|>`,
            # and keeping that framing shifts every later token id (worker
            # "mismatch" -> full re-prefill). Also exact for non-thinking
            # turns, whose raw opening the spliced ids reproduce verbatim.
            tail = text_chunk[base:]
            if tail and not self._is_template_framing(tail):
                # Unverifiable boundary: the header match sits inside message
                # content (e.g. a user message quoting a header literal), so
                # truncating here would delete real conversation text while
                # splicing proceeds on the corrupted prompt. Fall back to
                # plain text (correct output, just no reuse).
                return None
            return text_chunk[:base]
        region = text_chunk[base:]
        if region == preamble:
            return text_chunk
        if not _THINK_SCAFFOLD_RE.match(region):
            return None
        return text_chunk[:base] + preamble

    def _is_template_framing(self, tail: str) -> bool:
        """Whether `tail` (text between an assistant-header match and the
        sentinel) is positively recognized template framing: empty, or a
        recipient line closed by the message marker. Anything else --
        message content after a quoted header literal, a second header,
        turn terminators of any shape -- falls back to plain text."""
        if not tail:
            return True
        return _FRAMING_RE.match(tail) is not None

    def _split_on_sentinels(
        self, rendered: str, sub: dict[str, dict]
    ) -> Optional[list[dict]]:
        """Split `rendered` on the sentinels into alternating {"text"} chunks and
        {"ids"} runs (each sentinel -> sub[sentinel] = {"ids", "preamble"}). The
        {text} chunk before each {ids} run has its assistant scaffold normalized
        to that turn's stored preamble. Returns None if any pre-sentinel scaffold
        region is ambiguous (caller falls back to plain text)."""
        pattern = re.compile("|".join(re.escape(s) for s in sub))
        segments: list[dict] = []
        pos = 0
        for mobj in pattern.finditer(rendered):
            norm = self._normalize_scaffold(
                rendered[pos : mobj.start()], sub[mobj.group()]["preamble"]
            )
            if norm is None:
                return None
            if norm:
                segments.append({"text": norm})
            segments.append({"ids": sub[mobj.group()]["ids"]})
            pos = mobj.end()
        if pos < len(rendered):
            segments.append({"text": rendered[pos:]})
        return segments

    def _split_gemma_tool_call_spans(
        self,
        rendered: str,
        messages: list[ChatMessage],
        splice: dict[int, dict],
    ) -> Optional[list[dict]]:
        segments: list[dict] = []
        pos = 0

        for i, msg in enumerate(messages):
            tool_calls = msg.tool_calls if msg.role == "assistant" else None
            if not tool_calls:
                continue

            start = None
            end = pos
            for _ in tool_calls:
                span = _find_gemma_tool_call_span(rendered, end)
                if span is None:
                    return None
                span_start, span_end = span
                if start is None:
                    start = span_start
                end = span_end
            if start is None:
                return None

            if i in splice:
                norm = self._normalize_scaffold(
                    rendered[pos:start], splice[i]["preamble"]
                )
                if norm is None:
                    return None
                if norm:
                    segments.append({"text": norm})
                segments.append({"ids": splice[i]["ids"]})
            else:
                segments.append({"text": rendered[pos:end]})
            pos = end

        if pos < len(rendered):
            segments.append({"text": rendered[pos:]})
        return segments

    def build_prompt_input(
        self,
        *,
        session_id: Optional[str],
        messages: list[ChatMessage],
        rendered_prompt: str,
        tools,
        template_kwargs,
    ) -> PromptInput:
        """Return a PromptInput: token-ID segments when this session has faithful
        stored ids for matching prior assistant turns, else the plain rendered
        text. Each incoming assistant turn is matched IN ORDER against the stored
        records and only spliced when its content/tool calls and any supplied
        reasoning string match what we returned, and we kept faithful ids for it.
        Omitted or null reasoning permits reuse; a string edit invalidates the tail.
        Falls back to text on a sentinel collision or a render that
        dropped/duplicated a sentinel."""
        stored = self._turns.get(session_id or "")
        if not stored:
            return PromptInput(text=rendered_prompt)
        # Missing records render as text without shifting later turn indices.
        positions = [i for i, m in enumerate(messages) if m.role == "assistant"]
        splice: dict[int, dict] = {}  # message index -> {"ids", "preamble"}
        for k, pos in enumerate(positions):
            record = stored.get(k)
            if record is None:
                continue
            m = messages[pos]
            if self._assistant_fingerprint(m.content, m.tool_calls) != record["fp"] or (
                m.reasoning_content is not None
                and self._reasoning_fingerprint(m.reasoning_content)
                != record["reasoning_fp"]
            ):
                # Discard the stale tail without shifting subsequent turn indices.
                self._turns[session_id or ""] = {
                    index: record for index, record in stored.items() if index < k
                }
                break
            if record["ids"] is not None:
                splice[pos] = {
                    "ids": record["ids"],
                    "preamble": record.get("preamble", ""),
                }
        if not splice:
            return PromptInput(text=rendered_prompt)
        tool_splice = {
            pos: data
            for pos, data in splice.items()
            if messages[pos].tool_calls and not (messages[pos].content or "")
        }
        if tool_splice and _GEMMA_TOOL_CALL_START in rendered_prompt:
            segments = self._split_gemma_tool_call_spans(
                rendered_prompt, messages, tool_splice
            )
            return (
                PromptInput(segments=segments)
                if segments is not None
                else PromptInput(text=rendered_prompt)
            )
        token = uuid.uuid4().hex
        sentinel_at = {pos: f"<<ETSEG{j}_{token}>>" for j, pos in enumerate(splice)}
        sub = {sentinel_at[pos]: splice[pos] for pos in splice}
        # A sentinel must not already occur in the rendered output.
        if any(s in rendered_prompt for s in sub):
            return PromptInput(text=rendered_prompt)
        modified = [
            (
                ChatMessage(role="assistant", content=sentinel_at[i])
                if i in sentinel_at
                else m
            )
            for i, m in enumerate(messages)
        ]
        rendered = self._template.render(
            modified, tools=tools, template_kwargs=template_kwargs
        )
        # Each sentinel must survive templating exactly once, else fall back.
        if any(rendered.count(s) != 1 for s in sub):
            return PromptInput(text=rendered_prompt)
        # Splice ids and normalize each turn's scaffold; None => ambiguous region.
        segments = self._split_on_sentinels(rendered, sub)
        if segments is None:
            return PromptInput(text=rendered_prompt)
        return PromptInput(segments=segments)

    def record_assistant_turn(
        self,
        *,
        session_id: Optional[str],
        content,
        tool_calls,
        generated_token_ids: list,
        prior_turns: int,
        preamble: str = "",
        reasoning_content: Optional[str] = None,
    ) -> None:
        """Record this turn's {fingerprint, generated ids, generation preamble} at
        `prior_turns` (the assistant-turn count of the request it answers).
        Records at/after that index are dropped first, so a regenerated/branched
        turn replaces stale records rather than shadowing later hits. ids is None
        when the worker omitted them (stop-trimmed -> non-resumable), kept for
        positional alignment. `reasoning_content` is the client-visible value,
        including None when the client opted out. `preamble` is the generation
        scaffold (e.g. the Qwen3 `<think>` block) reproduced ahead of the spliced
        ids next request."""
        if not session_id:
            return
        turns = self._turns.setdefault(session_id, {})
        for index in [index for index in turns if index >= prior_turns]:
            del turns[index]
        turns[prior_turns] = {
            "fp": self._assistant_fingerprint(content, tool_calls),
            "reasoning_fp": self._reasoning_fingerprint(reasoning_content),
            "ids": list(generated_token_ids) if generated_token_ids else None,
            "preamble": preamble,
        }

    def reset(self, session_id: str) -> None:
        self._turns.pop(session_id, None)

    def close(self, session_id: str) -> None:
        self._turns.pop(session_id, None)
