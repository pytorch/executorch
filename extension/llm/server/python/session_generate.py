# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-agnostic adapter from an LLMSession to the RunnerPool generate surface.

Drives any LLMSession-shaped object (from an LLMEngine's create_session())
plus a tokenizer through the uniform ``generate(prompt, config, token_cb,
stats_cb)`` / ``stop()`` surface RunnerPool expects, with no prefix reuse: encode
the prompt, prefill it, then loop decode_one(). It is the simple sibling of
PrefixCachingSession for models that cannot rewind by logical position (seek()
NotSupported) — RunnerPool wraps it in _StatelessRunner, which reset()s before
each request.

This helper is deliberately model-agnostic: no model imports, no backend
assumptions, no prefix-cache policy. A model launcher (e.g. an example serve.py)
constructs it with that model's engine/session and tokenizer.
"""

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MAX_NEW_TOKENS = 2048


def utf8_complete_prefix_len(buf: bytes) -> int:
    """Length of the longest prefix of `buf` that does not end in the middle of a
    UTF-8 multi-byte sequence.

    decode_one() returns raw token-piece bytes, and byte-level BPE can split a
    multi-byte character across pieces, so an incomplete trailing sequence is
    held until the next piece completes it.
    """
    i, n = 0, len(buf)
    while i < n:
        c = buf[i]
        if c < 0x80:
            length = 1
        elif c >> 5 == 0x6:
            length = 2
        elif c >> 4 == 0xE:
            length = 3
        elif c >> 3 == 0x1E:
            length = 4
        else:
            length = 1  # invalid lead byte; emit it and let "replace" handle it
        if i + length > n:
            break  # incomplete trailing sequence; wait for the next piece
        i += length
    return i


class _SessionStats:
    """The two fields RunnerPool's stats_callback reads off the runner Stats."""

    __slots__ = ("num_prompt_tokens", "num_generated_tokens")

    def __init__(self, num_prompt_tokens: int, num_generated_tokens: int):
        self.num_prompt_tokens = num_prompt_tokens
        self.num_generated_tokens = num_generated_tokens


class SessionGenerateAdapter:
    """Adapt an LLMSession + tokenizer to RunnerPool's generate() surface."""

    def __init__(
        self,
        session,
        tokenizer,
        add_special_tokens: bool = False,
        default_max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    ):
        # add_special_tokens defaults False: the server renders the chat template
        # to a string that already contains the control tokens as text, so a
        # second BOS must not be added.
        self._session = session
        self._tokenizer = tokenizer
        self._add_special_tokens = add_special_tokens
        self._default_max_new_tokens = default_max_new_tokens
        self._stop = False

    def reset(self) -> None:
        self._stop = False
        self._session.reset()

    def stop(self) -> None:
        # Token-boundary cooperative: checked before each decode_one(); does not
        # abort a step already running.
        self._stop = True
        self._session.stop()

    def generate(
        self,
        prompt: str,
        config,
        token_callback: Optional[Callable[[str], None]] = None,
        stats_callback: Optional[Callable[[object], None]] = None,
    ) -> None:
        ids = self._tokenizer.encode(
            prompt, add_special_tokens=self._add_special_tokens
        )
        num_prompt = len(ids)

        temperature = getattr(config, "temperature", -1.0)
        max_new = getattr(config, "max_new_tokens", -1)
        if not max_new or max_new <= 0:
            max_new = self._default_max_new_tokens

        # Pass the request temperature to prefill so backends that sample the
        # first token in-graph use it (not a stale default); decode-time samplers
        # ignore it.
        self._session.prefill_tokens(ids, temperature)

        buf = b""
        num_generated = 0
        for _ in range(max_new):
            if self._stop:
                break
            result = self._session.decode_one(temperature)
            if result["is_eos"]:
                break
            num_generated += 1
            buf += result["text"]
            cut = utf8_complete_prefix_len(buf)
            if cut and token_callback is not None:
                token_callback(buf[:cut].decode("utf-8", "replace"))
                buf = buf[cut:]
        if buf and token_callback is not None:
            token_callback(buf.decode("utf-8", "replace"))
        if stats_callback is not None:
            stats_callback(_SessionStats(num_prompt, num_generated))
