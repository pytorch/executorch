# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Render OpenAI chat messages into a single prompt string.

The ExecuTorch runner tokenizes a plain prompt; chat formatting is the server's
job (control plane). We require the model's own Hugging Face ``chat_template``
(via ``--hf-tokenizer``) for correct, tool-aware, reasoning-aware formatting.
The generic ChatML fallback is opt-in only (``allow_fallback``): it is
approximate and cannot reproduce model-specific controls (e.g. enable_thinking),
so it must be a deliberate choice rather than a silent default.
"""

import json
import logging
from typing import Any, Optional

from .protocol import ChatMessage

logger = logging.getLogger(__name__)


_DEFAULT_SPECIAL_TOKENS = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>"]

# Chat turn terminators eligible to be used as generation stop strings. This is a
# deliberate allowlist of end-of-turn / end-of-text tokens -- NOT the tokenizer's
# full special-token set. Structural/tool delimiters (e.g. <tool_call>) must reach
# the tool parser, so they are intentionally excluded: using them as hard stops
# would truncate a tool call before it is ever parsed.
_TURN_TERMINATORS = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|end|>",
    "<|end_of_text|>",
    "<end_of_turn>",
    "</s>",
)


def _content_text(content) -> str:
    """Best-effort text for the ChatML fallback: a str as-is, or the concatenated
    text parts of an OpenAI list-content message (non-text parts dropped). Avoids
    rendering a Python repr of structured content. None -> empty string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(str(part.get("text", "")))
            elif isinstance(part, str):
                out.append(part)
        return "".join(out)
    return str(content or "")


def _decode_tool_call_arguments(messages: list[dict[str, Any]]) -> None:
    """In-place: parse each tool call's ``function.arguments`` from a JSON string
    into an object.

    OpenAI sends assistant tool-call arguments as a JSON-encoded string, but HF
    chat templates expect a mapping (e.g. Qwen renders ``arguments|items`` into
    ``<parameter=…>`` tags). Without this, a multi-turn tool conversation makes
    the template raise "Can only get item pairs from a mapping". Left as-is if
    the value isn't valid JSON, so a template that wants the raw string still works.
    """
    for m in messages:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args)
                except (ValueError, TypeError):
                    pass


class ChatTemplate:
    def __init__(
        self,
        hf_tokenizer_path: Optional[str] = None,
        default_template_kwargs: Optional[dict[str, Any]] = None,
        allow_fallback: bool = False,
    ):
        # Server-level defaults (e.g. {"enable_thinking": False}); per-request
        # chat_template_kwargs override these.
        self._defaults = default_template_kwargs or {}
        # Cache of the (deterministic) generation scaffold per resolved mode, so
        # warm-resume bookkeeping doesn't re-render a probe prompt every request.
        self._preamble_cache: dict[tuple, str] = {}
        self._hf = None
        if hf_tokenizer_path:
            from transformers import AutoTokenizer

            self._hf = AutoTokenizer.from_pretrained(hf_tokenizer_path)
            if self._hf.chat_template is None:
                self._hf = None
                if not allow_fallback:
                    raise ValueError(
                        f"HF tokenizer at {hf_tokenizer_path} has no chat_template; "
                        "pass an explicit fallback flag to use approximate ChatML."
                    )
                logger.warning(
                    "No chat_template at %s; using approximate ChatML.",
                    hf_tokenizer_path,
                )
        elif not allow_fallback:
            raise ValueError(
                "A chat template is required: pass --hf-tokenizer for the model's own "
                "template, or opt into approximate ChatML with --allow-chatml-fallback."
            )
        else:
            logger.warning(
                "No --hf-tokenizer; using approximate ChatML (no thinking control)."
            )

    def render(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[dict[str, Any]]] = None,
        template_kwargs: Optional[dict[str, Any]] = None,
    ) -> str:
        kwargs = {**self._defaults, **(template_kwargs or {})}
        if self._hf is not None:
            dumped = [m.model_dump(exclude_none=True) for m in messages]
            _decode_tool_call_arguments(dumped)
            return self._hf.apply_chat_template(
                dumped,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
                **kwargs,
            )
        return self._fallback(messages)

    def generation_preamble(
        self, template_kwargs: Optional[dict[str, Any]] = None
    ) -> str:
        """The deterministic text the generation prompt appends after the final
        ``<|im_start|>assistant\\n`` for this mode (Qwen3 no-think:
        ``<think>\\n\\n</think>\\n\\n``; think: ``<think>\\n``; ``""`` for
        templates that add no scaffold). The worker prefills this into resident
        KV, so warm-resume splicing must reproduce it ahead of a turn's generated
        ids. Computed by rendering a trivial prompt with the same mode resolution
        as :meth:`render` and taking the text after the final assistant header.
        Returns ``""`` for the fallback / no-scaffold templates (fix is a no-op).
        """
        if self._hf is None:
            return ""
        merged = {**self._defaults, **(template_kwargs or {})}
        key = tuple(sorted((k, repr(v)) for k, v in merged.items()))
        cached = self._preamble_cache.get(key)
        if cached is not None:
            return cached
        rendered = self.render(
            [ChatMessage(role="user", content="")],
            tools=None,
            template_kwargs=template_kwargs,
        )
        marker = "<|im_start|>assistant\n"
        idx = rendered.rfind(marker)
        preamble = rendered[idx + len(marker) :] if idx != -1 else ""
        self._preamble_cache[key] = preamble
        return preamble

    def chat_template_str(self) -> Optional[str]:
        """Raw chat-template string (for tool-format auto-detection), if available."""
        return (
            getattr(self._hf, "chat_template", None) if self._hf is not None else None
        )

    def count_tokens(self, prompt: str) -> Optional[int]:
        """Token count for the rendered prompt, or None if no tokenizer is available."""
        if self._hf is not None:
            # The prompt is already rendered (apply_chat_template includes the
            # control tokens), so encode without re-adding BOS/EOS — matching the
            # session/prefix-cache paths, so the count isn't inflated and
            # near-limit requests aren't falsely rejected under --max-context.
            return len(self._hf.encode(prompt, add_special_tokens=False))
        return None

    def turn_stop_sequences(self) -> list[str]:
        """Generation stop strings: model/template-specific *turn terminators*
        only -- the tokenizer's EOS plus known chat turn-end tokens -- NOT the
        full special-token set.

        Structural/tool delimiters (e.g. <tool_call>) are deliberately excluded:
        if a tokenizer registers them as special, using the whole special set as
        hard stops would halt generation at the delimiter and truncate the tool
        call before the parser ever sees it. Whitespace-only tokens are dropped.
        User-supplied request `stop` strings are handled separately and are not
        affected by this set.

        May return [] if the tokenizer has no eos_token and registers none of the
        known terminators as special; in that case end-of-turn detection relies
        entirely on the worker's EOS-by-token-id check (e.g. the Qwen engine adds
        <|im_end|> to eos_ids), so the string set here is only a backstop.
        """
        if self._hf is None:
            return list(_DEFAULT_SPECIAL_TOKENS)
        specials = {
            t
            for t in (getattr(self._hf, "all_special_tokens", []) or [])
            if isinstance(t, str) and t.strip()
        }
        out: list[str] = []
        eos = getattr(self._hf, "eos_token", None)
        if isinstance(eos, str) and eos.strip():
            out.append(eos)
        for t in _TURN_TERMINATORS:
            if t in specials and t not in out:
                out.append(t)
        return out

    def special_tokens(self) -> list[str]:
        """ALL special-token strings, for final content cleanup -- stripping any
        special token that leaked into visible output. Deliberately broad, and
        distinct from turn_stop_sequences(): this set must NOT be used as
        generation stops or pre-parse truncation (that would halt/cut a tool call
        at a structural delimiter), only to scrub trailing specials from the
        already-parsed visible content. Whitespace-only tokens are dropped so a
        stray '  ' token can't truncate content at the first double space.
        """
        if self._hf is not None:
            toks = list(getattr(self._hf, "all_special_tokens", []) or [])
            return [t for t in toks if isinstance(t, str) and t.strip()]
        return list(_DEFAULT_SPECIAL_TOKENS)

    @staticmethod
    def _fallback(messages: list[ChatMessage]) -> str:
        # Approximate ChatML, TEXT-ONLY. Provide --hf-tokenizer for model-correct
        # formatting (reasoning controls like enable_thinking, and structured
        # tool/multimodal turns, which this fallback cannot reproduce). This path
        # renders only text content: assistant `tool_calls` and a tool-role
        # `tool_call_id` are dropped, so it is NOT a correctness path for tool or
        # multimodal conversations -- use a real --hf-tokenizer for those.
        parts = []
        for m in messages:
            content = _content_text(m.content)
            parts.append(f"<|im_start|>{m.role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
