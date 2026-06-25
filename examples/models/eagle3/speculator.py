# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""EAGLE-3 speculative-decoding module.

``Eagle3Speculator`` holds a shared target (any ``TapTarget`` — gemma4-31B is the
reference) and an EAGLE-3 draft head and exposes four methods: ``prefill``,
``target_verify``, ``draft_decode``, and
``decode``. The export (``export.py``) lowers only the first three — under the
shifted runner scheme (below) ``decode`` is unnecessary — sharing the target's KV
cache across ``prefill``/``target_verify`` and the draft's KV cache for
``draft_decode`` (``share_mutable_buffers`` deduplicates by tensor identity).
``decode`` is kept here for eager use / a non-speculative fallback.

The target methods return the *fused* draft feature — ``draft.fuse`` (the fc
projection) applied to the target's auxiliary taps — rather than the raw taps.
This gives the draft a single, uniform hidden-size feature whether it comes from
the target (confirmed positions) or from the draft's own recurrent output
(proposed positions), so ``draft_decode`` has one signature for both seeding and
stepping. It matches the eager reference, where confirmed positions use
``fuse(taps)`` and proposed positions use the midlayer output ``g`` — both
hidden-size.

The module only exposes the per-position greedy target ids a verifier needs
(argmax, not sampling). Acceptance, rejection, EOS truncation, and budget
clipping are the caller's responsibility; losslessness depends on the runner
applying the verification alignment (below) correctly.

Runner scheme (shifted, one target forward/round — matches vLLM EAGLE,
``vllm/v1/spec_decode/eagle.py`` ``set_inputs_first_pass``): the draft pairs the
target hidden state at position t with the token at t+1. So after verification,
the next chain reseeds the draft cache from the ``feature`` ``target_verify``
already produced for the accepted positions, paired with the next (corrected)
token — the corrected/bonus token never needs its own target forward, which is
why ``prefill`` + ``target_verify`` + ``draft_decode`` are sufficient for
multi-round decoding (no standalone target ``decode``). ``draft_decode`` permits
the overwrite (contiguous rollback). ``test_shifted_speculative_decode_is_lossless``
drives this loop through only those three methods and pins it to greedy.
"""

import torch
import torch.nn as nn

from executorch.examples.models.eagle3.draft import Eagle3Draft
from executorch.examples.models.eagle3.target import TapTarget


class Eagle3Speculator(nn.Module):
    def __init__(self, target: TapTarget, draft: Eagle3Draft):
        super().__init__()
        if not draft.config.has_own_embed:
            # The fallback (sourcing draft embeddings from the target) needs the
            # checkpoint's exact training-time embedding convention, which the
            # speculator format does not record; only owned-embedding heads are
            # supported here.
            raise ValueError(
                "Eagle3Speculator requires a draft head with its own "
                "embed_tokens (has_own_embed=True)"
            )
        self.target = target
        self.draft = draft
        # Wire the target's hidden-state taps to the draft's expected aux layers.
        target.set_eagle_tap_layers(draft.config.aux_hidden_state_layers)

    # ---------------- target methods (share the target KV cache) ----------------

    def prefill(
        self, tokens: torch.Tensor, input_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prompt prefill (T>=2). Populates the target KV cache.

        Returns:
            token:   (1, 1) int64 greedy next token after the prompt.
            feature: (1, T, hidden) fused draft feature for every prompt position.
        """
        logits, taps = self.target.forward_logits_taps(
            tokens, input_pos, last_logits_only=True
        )
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        return token, self.draft.fuse(taps)

    def target_verify(
        self, tokens: torch.Tensor, input_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Verify candidate tokens. Extends the target KV cache.

        Returns the greedy argmax (not full logits — ``vocab`` is 262144) at each
        fed position. Note the one-position shift: ``verify_ids[i]`` is the
        target's greedy token for the position *after* token ``i`` (it predicts
        ``input_pos[i] + 1``). So for proposals fed at positions L..L+K-1,
        proposal 0 is checked against the token from the preceding step
        (prefill/decode at L-1), proposal ``i>0`` against ``verify_ids[i-1]``,
        and ``verify_ids[-1]`` is the bonus token after the last candidate. The
        caller (not this module) performs acceptance with that alignment.

        Returns:
            verify_ids: (1, T) int64 greedy target token after each fed position.
            feature:    (1, T, hidden) fused draft feature for every position.
        """
        logits, taps = self.target.forward_logits_taps(
            tokens, input_pos, last_logits_only=False
        )
        return logits.argmax(dim=-1), self.draft.fuse(taps)

    def decode(
        self, tokens: torch.Tensor, input_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-token target decode (T=1). Same outputs as ``prefill``."""
        return self.prefill(tokens, input_pos)

    # ---------------- draft method (uses the draft KV cache) ----------------

    def draft_decode(
        self,
        tokens: torch.Tensor,
        feature: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft proposal over the KV cache (T>=1: seed with T>1, step with T=1).

        When seeding with T>1, only the *last* position's id is the next
        proposal after the seeded prefix; the earlier ids are per-prefix
        predictions and are not verification candidates. Single-step decode
        (T=1) returns the one proposal for the next position. Writes must be
        contiguous from position 0 (see ``Eagle3Draft.forward_cached``).

        Args:
            tokens:   (1, T) int64 token ids (target vocab) to embed.
            feature:  (1, T, hidden) per-position feature — fused target feature
                      for confirmed positions, recurrent ``g`` for proposed ones.
            input_pos: (T,) absolute positions for RoPE / draft KV cache.

        Returns:
            target_ids: (1, T) int64 proposed next tokens mapped to the target vocab.
            g:          (1, T, hidden) midlayer output — next-step recurrent feature.
        """
        emb = self.draft.embed(tokens)
        draft_logits, g = self.draft.forward_cached(emb, feature, input_pos)
        target_ids = self.draft.draft_to_target(draft_logits.argmax(dim=-1))
        return target_ids, g
