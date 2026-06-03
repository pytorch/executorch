# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Turn-to-turn KV prefix reuse over the Engine/Session API.

An agent re-sends a large, mostly-unchanged prompt every turn (system + tools +
context + history). The C++ KV cache is contiguous and position-indexed, so we
reuse it: find the longest token prefix shared with what's cached, seek() to it,
prefill only the new suffix, then drive decode_one().

Because decode_one() returns the EXACT sampled token ids, we track them
(`_cached = prompt_ids + generated_ids`) — so a follow-up turn that includes the
prior completion reuses that too (not just the static system prefix). This is
safe in a way re-tokenizing generated text is not (BPE encode(a)+encode(b) !=
encode(a+b)).

Constraints baked in:
  - Only position-0 prefixes are reusable (RoPE is position-dependent) — exactly
    the agent shape (system+tools+context+history at the front).
  - seek() refuses on sliding-window models; we catch that and fall back to a
    full reset + re-prefill.

Lifecycle: a session is conversation-scoped, not request-scoped. The pool keeps
it warm across requests (KV preserved) and routes follow-up turns back to it by
prefix affinity; it is reset only on an unrecoverable error or torn down at
shutdown — never reset per request, which would discard the cache reuse exists
to exploit.
"""

import codecs
import logging
import time
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


def longest_common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@runtime_checkable
class Session(Protocol):
    """The C++ LLMSession surface this driver uses."""

    def prefill_tokens(self, token_ids: list[int]) -> None: ...
    def decode_one(
        self, temperature: float = ...
    ) -> dict: ...  # {token_id, text:bytes, is_eos}
    def seek(self, pos: int) -> None: ...
    def position(self) -> int: ...
    def reset(self) -> None: ...
    def stop(self) -> None: ...


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


class _Stats:
    """Token counts handed to the pool's stats_callback (matches the C++ Stats
    attribute names the pool reads)."""

    __slots__ = ("num_prompt_tokens", "num_generated_tokens")

    def __init__(self, num_prompt_tokens: int, num_generated_tokens: int):
        self.num_prompt_tokens = num_prompt_tokens
        self.num_generated_tokens = num_generated_tokens


class PrefixCachingSession:
    """Drives one LLMSession with turn-to-turn prefix reuse, tracking the exact
    tokens resident in its KV cache (prompt + generated)."""

    def __init__(
        self,
        session: Session,
        tokenizer: Tokenizer,
        index: int = 0,
        max_context_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ):
        self._session = session
        self._tok = tokenizer
        self._cached: list[int] = []
        self._index = index
        self._fallbacks = 0
        self._stop = False
        self._max_context_len = max_context_len
        self._max_seq_len = max_seq_len

    @property
    def cached_tokens(self) -> list[int]:
        return self._cached

    def _encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def reuse_len(self, prompt_ids: list[int]) -> int:
        """Tokens reusable from cache, capped at the runner's resident position
        (never seek() past resident KV) and at len-1 (always prefill >=1 token)."""
        reuse = longest_common_prefix(self._cached, prompt_ids)
        position = getattr(self._session, "position", None)
        if position is not None:
            reuse = min(reuse, position())
        if reuse >= len(prompt_ids):
            reuse = len(prompt_ids) - 1
        return max(0, reuse)

    def _resolved_max_new_tokens(self, config) -> Optional[int]:
        max_new = getattr(config, "max_new_tokens", -1) if config is not None else -1
        if self._max_context_len is None:
            return None if max_new <= 0 else max_new
        position = self._session.position()
        # Match TextLLMRunner.generate(): sliding-window exports do not treat
        # position as consumed full-context capacity.
        if self._max_seq_len is not None and self._max_seq_len < self._max_context_len:
            position = 0
        if config is not None and hasattr(config, "resolve_max_new_tokens"):
            return config.resolve_max_new_tokens(self._max_context_len, position)
        if max_new <= 0:
            return max(0, self._max_context_len - position)
        return max(0, min(max_new, self._max_context_len - position))

    def generate(  # noqa: C901 - prefill/reuse + decode loop + fallbacks read clearest inline
        self, prompt: str, config, token_callback=None, stats_callback=None
    ) -> None:
        prompt_ids = self._encode(prompt)
        self._stop = False
        start = time.perf_counter()
        ttft = None

        # --- prefill: reuse the shared prefix, else (on failure) full prefill ---
        reuse = self.reuse_len(prompt_ids)
        fallback = False
        try:
            # seek(reuse) (reuse may be 0 for a cold session) repositions to the
            # shared prefix, discarding any stale KV beyond it; then prefill only
            # the suffix.
            self._session.seek(reuse)
            self._session.prefill_tokens(prompt_ids[reuse:])
        except Exception as e:  # noqa: BLE001 - reuse setup failed -> safe full path
            fallback = True
            self._fallbacks += 1
            reuse = 0
            logger.debug("prefix reuse setup failed (%s); full prefill", e)
            self._session.reset()
            self._cached = []
            try:
                self._session.prefill_tokens(prompt_ids)
            except Exception:
                self._session.reset()
                self._cached = []
                raise

        # --- decode loop: bounded by max_new_tokens; stop on EOS or stop() ---
        max_new = self._resolved_max_new_tokens(config)
        if max_new is not None and max_new <= 0:
            raise RuntimeError("No available context capacity for generation")
        temperature = (
            getattr(config, "temperature", -1.0) if config is not None else -1.0
        )
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        gen_ids: list[int] = []
        n = 0
        try:
            while (max_new is None or n < max_new) and not self._stop:
                step = self._session.decode_one(temperature)
                gen_ids.append(step["token_id"])
                n += 1
                piece = decoder.decode(step["text"])  # assemble UTF-8 from byte pieces
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    if token_callback:
                        token_callback(piece)
                if step["is_eos"]:
                    break
            tail = decoder.decode(b"", final=True)
            if tail and token_callback:
                token_callback(tail)
        except (
            Exception
        ):  # noqa: BLE001 - real decode error: reset + propagate (no retry)
            self._session.reset()
            self._cached = []
            raise

        # Track EXACT ids (prompt + generated) so the next turn can reuse the
        # completion too. seek() stays capped at position(), so this is safe.
        self._cached = prompt_ids + gen_ids
        if stats_callback:
            stats_callback(_Stats(len(prompt_ids), n))
        logger.info(
            "prefix-cache runner=%d reused=%d suffix=%d generated=%d fallback=%s "
            "fallbacks_total=%d ttft_ms=%.0f",
            self._index,
            reuse,
            len(prompt_ids) - reuse,
            n,
            fallback,
            self._fallbacks,
            (ttft or 0.0) * 1000,
        )

    def reset(self) -> None:
        self._session.reset()
        self._cached = []

    def stop(self) -> None:
        # Cooperative cancellation: the decode loop checks _stop each step.
        self._stop = True
        self._session.stop()
