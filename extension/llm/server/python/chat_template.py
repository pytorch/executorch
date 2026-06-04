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

import logging
from typing import Any, Optional

from .protocol import ChatMessage

logger = logging.getLogger(__name__)


_DEFAULT_SPECIAL_TOKENS = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>"]


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
            return self._hf.apply_chat_template(
                [m.model_dump(exclude_none=True) for m in messages],
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
                **kwargs,
            )
        return self._fallback(messages)

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

    def special_tokens(self) -> list[str]:
        """Special-token strings whose appearance ends the visible content.

        From the HF tokenizer when available (model-accurate), else a default set
        covering common chat models.
        """
        if self._hf is not None:
            toks = list(getattr(self._hf, "all_special_tokens", []) or [])
            return [t for t in toks if isinstance(t, str) and t]
        return list(_DEFAULT_SPECIAL_TOKENS)

    @staticmethod
    def _fallback(messages: list[ChatMessage]) -> str:
        # Approximate ChatML. Provide --hf-tokenizer for model-correct formatting
        # (including reasoning controls like enable_thinking, which the fallback
        # cannot reproduce).
        parts = []
        for m in messages:
            content = m.content if isinstance(m.content, str) else str(m.content or "")
            parts.append(f"<|im_start|>{m.role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
