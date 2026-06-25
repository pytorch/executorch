# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end EAGLE-3 speculative-decoding smoke test (CPU, no export needed).

Builds tiny matching target + draft models and drives the shifted (vLLM-EAGLE)
method-flow the C++ runner uses -- prefill -> draft chain -> target_verify ->
accept -> reseed -> repeat -- checking the generated tokens equal greedy target
decoding (lossless by construction). Forced-acceptance cases pin the partial,
full, and accepted-EOS paths plus the one-token budget; the random-weight loop
alone can leave them uncovered.

This is CPU eager coverage of the decoding *algorithm*, not the C++ runner
itself: tokenizer integration, device buffers, CUDA-graph capture, and the real
CUDA/AOTI export are exercised manually (examples/models/eagle3/export.py + the
eagle3-cuda runner) and remain tracked as future automated CUDA coverage.
"""

import torch

from executorch.examples.models.eagle3.draft import Eagle3Config, Eagle3Draft
from executorch.examples.models.eagle3.speculator import Eagle3Speculator
from executorch.examples.models.gemma4_31b.model import Gemma4_31B, Gemma4_31BConfig

_TARGET_VOCAB = 128


def _build():
    torch.manual_seed(0)
    target = (
        Gemma4_31B(
            Gemma4_31BConfig(
                vocab_size=_TARGET_VOCAB,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=6,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                num_global_key_value_heads=1,
                global_head_dim=8,
                sliding_window=64,
                max_seq_len=128,
            )
        )
        .to(torch.float32)
        .eval()
    )
    draft = (
        Eagle3Draft(
            Eagle3Config(
                hidden_size=32,
                target_hidden_size=32,
                intermediate_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                draft_vocab_size=64,
                target_vocab_size=_TARGET_VOCAB,
                aux_hidden_state_layers=[0, 1, 3],
                max_seq_len=128,
                has_own_embed=True,
            )
        )
        .to(torch.float32)
        .eval()
    )
    return Eagle3Speculator(target, draft), target


def _toks(ids):
    return torch.tensor([ids], dtype=torch.long)


def _reset_kv(target):
    for name, buf in target.named_buffers():
        if ".kv_cache." in name:
            buf.zero_()


@torch.no_grad()
def _greedy(target, prompt, n):
    seq, out = list(prompt), []
    for _ in range(n):
        _reset_kv(target)
        logits, _ = target.forward_logits_taps(
            _toks(seq), torch.arange(len(seq)), last_logits_only=True
        )
        t = int(logits[:, -1].argmax())
        seq.append(t)
        out.append(t)
    return out


def _accept_len(proposals, verify_ids):
    """Greedy acceptance: count leading proposals matching the verifier ids."""
    a = 0
    for j, p in enumerate(proposals):
        if p != int(verify_ids[0, j]):
            break
        a += 1
    return a


def _truncate_at_eos(tokens, eos_ids):
    """Cut at the first stop token (inclusive); returns (tokens, hit_eos)."""
    for i, t in enumerate(tokens):
        if t in eos_ids:
            return tokens[: i + 1], True
    return tokens, False


@torch.no_grad()
def _speculative_decode(
    spec, prompt, K, num_gen, force=None, eos_ids=None, accept_out=None
):
    """The shifted one-target-forward-per-round loop the C++ runner implements.

    ``force(emitted) -> list[K]`` overrides the draft's proposal *values* (the
    draft chain is still run to reseed) so tests can pin the acceptance count.
    ``eos_ids`` truncates a round at the first emitted stop token (matching the
    runner). Per-round acceptance counts are appended to ``accept_out``.
    """
    target = spec.target
    _reset_kv(target)
    spec.draft.reset_cache()
    eos_ids = eos_ids or set()
    L = len(prompt)
    bonus, feat = spec.prefill(_toks(prompt), torch.arange(L))
    anchor, anchor_pos = int(bonus), L
    emitted = [anchor]
    if num_gen <= 1 or anchor in eos_ids:
        return emitted[:num_gen]  # prefill bonus suffices; no draft round runs

    def chain(seed_tokens, seed_feat, seed_pos):
        tids, g = spec.draft_decode(_toks(seed_tokens), seed_feat, seed_pos)
        proposals = [int(tids[0, -1])]
        last = int(seed_pos[-1])
        tok, f = tids[:, -1:], g[:, -1:]
        for k in range(1, K):
            tids, g = spec.draft_decode(tok, f, torch.tensor([last + k]))
            proposals.append(int(tids[0, 0]))
            tok, f = tids, g
        return proposals

    proposals = chain(prompt[1:] + [anchor], feat, torch.arange(L))
    if force is not None:
        proposals = force(emitted)
    while len(emitted) < num_gen:
        vids, vfeat = spec.target_verify(
            _toks([anchor] + proposals), torch.arange(anchor_pos, anchor_pos + K + 1)
        )
        a = _accept_len(proposals, vids)
        if accept_out is not None:
            accept_out.append(a)
        corrected = int(vids[0, a])
        new = (proposals[:a] + [corrected])[: num_gen - len(emitted)]
        new, hit_eos = _truncate_at_eos(new, eos_ids)
        emitted += new
        if hit_eos or len(emitted) >= num_gen:
            break
        proposals = chain(
            proposals[:a] + [corrected],
            vfeat[:, : a + 1],
            torch.arange(anchor_pos, anchor_pos + a + 1),
        )
        anchor, anchor_pos = corrected, anchor_pos + 1 + a
        if force is not None:
            proposals = force(emitted)
    return emitted[:num_gen]


_PROMPT = [2, 7, 3, 21, 9, 14]


def test_speculative_decode_matches_greedy_e2e():
    spec, target = _build()
    num_gen = 16
    got = _speculative_decode(spec, _PROMPT, K=4, num_gen=num_gen)
    assert len(got) == num_gen
    assert got == _greedy(target, _PROMPT, num_gen)


def test_full_acceptance_loop_is_lossless():
    # Force every round to fully accept (a == K) by proposing the target's own
    # greedy continuation. This deterministically exercises the a == K reseed and
    # the folded-bonus path across rounds, which a random-weight run may never hit.
    spec, target = _build()
    K, num_gen = 4, 16
    G = _greedy(target, _PROMPT, num_gen + K + 1)
    accepts = []
    got = _speculative_decode(
        spec,
        _PROMPT,
        K=K,
        num_gen=num_gen,
        force=lambda emitted: G[len(emitted) : len(emitted) + K],
        accept_out=accepts,
    )
    assert got == G[:num_gen]
    assert accepts and all(a == K for a in accepts)


def test_partial_acceptance_loop_is_lossless():
    # Force every round to accept K-1 (0 < a < K): greedy for the first K-1
    # proposals, then a deliberately wrong token. The corrected token must be the
    # greedy token at the mismatch, so the loop stays lossless.
    spec, target = _build()
    K, num_gen = 4, 16
    G = _greedy(target, _PROMPT, num_gen + K + 1)

    def force(emitted):
        e = len(emitted)
        good = G[e : e + K - 1]
        wrong = (G[e + K - 1] + 1) % _TARGET_VOCAB
        return good + [wrong]

    accepts = []
    got = _speculative_decode(
        spec, _PROMPT, K=K, num_gen=num_gen, force=force, accept_out=accepts
    )
    assert got == G[:num_gen]
    assert accepts and all(0 < a < K for a in accepts)


def test_accepted_proposal_eos_stops_emission():
    # An accepted proposal (not the prefill bonus or corrected token) that is a
    # stop token must end emission immediately, with nothing emitted after it.
    spec, target = _build()
    K, num_gen = 4, 16
    G = _greedy(target, _PROMPT, num_gen + K + 1)
    eos = {G[2]}  # the 3rd accepted token of the first full-acceptance round
    got = _speculative_decode(
        spec,
        _PROMPT,
        K=K,
        num_gen=num_gen,
        force=lambda emitted: G[len(emitted) : len(emitted) + K],
        eos_ids=eos,
    )
    assert len(got) >= 2  # reached an accepted proposal, not just the bonus
    assert got[-1] in eos  # stopped exactly on the stop token
    assert all(t not in eos for t in got[:-1])  # nothing emitted after EOS
    assert got == G[: len(got)]  # lossless prefix


def test_num_gen_one_returns_only_prefill_bonus():
    # A one-token request returns the free prefill bonus without a draft round.
    spec, target = _build()
    assert _speculative_decode(spec, _PROMPT, K=4, num_gen=1) == _greedy(
        target, _PROMPT, 1
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
