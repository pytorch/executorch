# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenAI/chat-template transcript state for token-ID warm resume (V2b.1.5).

This is the OpenAI-adapter-specific glue that makes warm resume work across the
chat template's lossy re-render of prior assistant turns (especially tool calls,
which re-render from parsed structure and don't re-tokenize to what the model
generated). It is NOT generic runtime infrastructure: it knows ChatMessages,
tool_calls, the ChatTemplate, sentinels, and assistant fingerprints. The runtime
(session_runtime) only sees PromptInput.

Per session we keep one record per assistant turn we produced, in order:
{"fp": fingerprint of the response we returned, "ids": exact generated token ids
| None}. On the next request each prior assistant turn is replaced with a unique
sentinel, the conversation is rendered once, and the rendered text is split on
the sentinels with the stored ids spliced back in -- but only for turns whose
fingerprint matches the incoming message (so an edited/branched history, or a
session reused for another conversation, is never substituted with stale ids)
and whose ids are present (a stop-trimmed turn has None and is left as text).
Everything is backstopped by the worker's exact-token prefix check.
"""

import hashlib
import json
import re
import uuid
from typing import Optional

from .chat_template import ChatTemplate
from .protocol import ChatMessage
from .session_runtime import PromptInput


class OpenAITranscriptState:
    def __init__(self, template: ChatTemplate):
        self._template = template
        # session_id -> [{"fp": str, "ids": list[int] | None}, ...] (one per
        # assistant turn we produced, in order). Cleared on reset/close.
        self._turns: dict[str, list[dict]] = {}

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
                norm.append([getattr(fn, "name", None), getattr(fn, "arguments", None)])
            elif isinstance(tc, dict):
                f = tc.get("function", {})
                norm.append([f.get("name"), f.get("arguments")])
        blob = json.dumps([content or "", norm], sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def _split_on_sentinels(rendered: str, sub: dict[str, list[int]]) -> list[dict]:
        """Split `rendered` on the sentinels into alternating {"text"} chunks and
        {"ids"} runs (each sentinel -> its stored id list)."""
        pattern = re.compile("|".join(re.escape(s) for s in sub))
        segments: list[dict] = []
        pos = 0
        for mobj in pattern.finditer(rendered):
            if mobj.start() > pos:
                segments.append({"text": rendered[pos : mobj.start()]})
            segments.append({"ids": sub[mobj.group()]})
            pos = mobj.end()
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
        records and only spliced when (a) its fingerprint matches what we returned
        (else the history diverged -> stop, splice nothing further) and (b) we
        kept faithful ids for it (a stop-trimmed turn's None -> rendered as text).
        Falls back to text on a sentinel collision or a render that
        dropped/duplicated a sentinel."""
        stored = self._turns.get(session_id or "")
        if not stored:
            return PromptInput(text=rendered_prompt)
        positions = [i for i, m in enumerate(messages) if m.role == "assistant"]
        splice: dict[int, list[int]] = {}  # message index -> exact ids
        diverged_at = None
        for k, pos in enumerate(positions):
            if k >= len(stored):
                break
            m = messages[pos]
            if self._assistant_fingerprint(m.content, m.tool_calls) != stored[k]["fp"]:
                diverged_at = k  # this stored turn and every later one are stale
                break
            if stored[k]["ids"] is not None:
                splice[pos] = stored[k]["ids"]
        if diverged_at is not None:
            # Drop the stale tail from the first mismatch so an edited/branched
            # earlier turn can't keep shadowing future requests; the matched
            # prefix [:diverged_at] is untouched and still splices. We have no
            # exact ids for the edited turn itself (the client authored it, we
            # didn't generate it), so warm resume for that turn and the ones after
            # it stays text until the session is reset/closed. Safe regardless:
            # stale ids are never spliced and the worker's exact-token prefix
            # check backstops correctness.
            del stored[diverged_at:]
        if not splice:
            return PromptInput(text=rendered_prompt)
        token = uuid.uuid4().hex
        sentinel_at = {pos: f"<<ETSEG{j}_{token}>>" for j, pos in enumerate(splice)}
        sub = {sentinel_at[pos]: ids for pos, ids in splice.items()}
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
        return PromptInput(segments=self._split_on_sentinels(rendered, sub))

    def record_assistant_turn(
        self,
        *,
        session_id: Optional[str],
        content,
        tool_calls,
        generated_token_ids: list,
        prior_turns: int,
    ) -> None:
        """Record this turn's {fingerprint, exact generated ids} at position
        `prior_turns` -- the count of assistant turns in the request this
        response answers. Stored records at/after that index are dropped first, so
        a regenerated or branched turn under the same session_id replaces stale
        records instead of leaving them to shadow future warm-resume hits with a
        stale fingerprint. ids is None when the worker omitted them (stop-trimmed
        turn) -- recorded as non-resumable but kept for positional alignment."""
        if not session_id:
            return
        turns = self._turns.setdefault(session_id, [])
        del turns[prior_turns:]
        turns.append(
            {
                "fp": self._assistant_fingerprint(content, tool_calls),
                "ids": list(generated_token_ids) if generated_token_ids else None,
            }
        )

    def reset(self, session_id: str) -> None:
        self._turns.pop(session_id, None)

    def close(self, session_id: str) -> None:
        self._turns.pop(session_id, None)
