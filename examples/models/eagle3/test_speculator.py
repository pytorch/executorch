# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the EAGLE-3 speculator wrapper (the four exported methods).

Builds tiny matching target/draft models on CPU and checks each method against
the underlying model, plus that a KV-cached draft chain driven through
``prefill`` + ``draft_decode`` reproduces a stateless full recompute token for
token (the design's correctness crux before the AOTI export and C++ runner).
"""

import torch

from executorch.examples.models.eagle3.draft import Eagle3Config, Eagle3Draft
from executorch.examples.models.eagle3.speculator import Eagle3Speculator
from executorch.examples.models.gemma4_31b.model import Gemma4_31B, Gemma4_31BConfig

HIDDEN = 32
TAP_LAYERS = [0, 1, 3]
TARGET_VOCAB = 128
DRAFT_VOCAB = 64


def _build():
    torch.manual_seed(0)
    target = (
        Gemma4_31B(
            Gemma4_31BConfig(
                vocab_size=TARGET_VOCAB,
                hidden_size=HIDDEN,
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
                hidden_size=HIDDEN,
                target_hidden_size=HIDDEN,
                intermediate_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                draft_vocab_size=DRAFT_VOCAB,
                target_vocab_size=TARGET_VOCAB,
                aux_hidden_state_layers=TAP_LAYERS,
                max_seq_len=128,
                has_own_embed=True,
            )
        )
        .to(torch.float32)
        .eval()
    )
    return Eagle3Speculator(target, draft), target, draft


def _reset_target_kv(model):
    for name, buf in model.named_buffers():
        if ".kv_cache." in name:
            buf.zero_()


def _toks(ids):
    return torch.tensor([ids], dtype=torch.long)


def test_prefill_returns_greedy_token_and_fused_feature():
    spec, target, draft = _build()
    prompt = [3, 9, 1, 27, 5]
    pos = torch.arange(len(prompt))
    with torch.no_grad():
        _reset_target_kv(target)
        token, feat = spec.prefill(_toks(prompt), pos)
        _reset_target_kv(target)
        logits, taps = target.forward_logits_taps(
            _toks(prompt), pos, last_logits_only=True
        )
    assert token.shape == (1, 1)
    assert int(token) == int(logits[:, -1].argmax())
    assert feat.shape == (1, len(prompt), HIDDEN)
    torch.testing.assert_close(feat, draft.fuse(taps))


def test_target_verify_returns_greedy_ids_and_feature():
    spec, target, draft = _build()
    seq = [2, 8, 4, 19]
    pos = torch.arange(len(seq))
    with torch.no_grad():
        _reset_target_kv(target)
        ids, feat = spec.target_verify(_toks(seq), pos)
        _reset_target_kv(target)
        ref_logits, ref_taps = target.forward_logits_taps(
            _toks(seq), pos, last_logits_only=False
        )
    assert ids.shape == (1, len(seq))
    torch.testing.assert_close(ids, ref_logits.argmax(dim=-1))
    torch.testing.assert_close(feat, draft.fuse(ref_taps))


def test_target_verify_after_prefill_extends_cache():
    # Verify must reuse the prefilled target KV: greedy ids + feature at the
    # candidate positions must match a full recompute over prompt + candidates.
    spec, target, draft = _build()
    prompt, cands = [3, 9, 1, 27], [7, 2, 11, 4]
    L = len(prompt)
    with torch.no_grad():
        _reset_target_kv(target)
        ref_logits, ref_taps = target.forward_logits_taps(
            _toks(prompt + cands), torch.arange(L + len(cands)), last_logits_only=False
        )
        _reset_target_kv(target)
        spec.prefill(_toks(prompt), torch.arange(L))
        v_ids, v_feat = spec.target_verify(
            _toks(cands), torch.arange(L, L + len(cands))
        )
    torch.testing.assert_close(v_ids, ref_logits.argmax(dim=-1)[:, L:])
    torch.testing.assert_close(v_feat, draft.fuse(ref_taps)[:, L:])


def test_target_verify_alignment_with_prefill_token():
    # Pins the one-position shift: prefill token checks proposal 0; verify_ids[i]
    # checks proposal i+1; verify_ids[-1] is the bonus. Feeding the greedy
    # continuation as proposals makes the expected ids concrete.
    spec, target, _ = _build()
    prompt = [3, 9, 1, 27, 5]
    L, K = len(prompt), 4
    with torch.no_grad():
        seq, greedy = list(prompt), []
        for _ in range(K + 1):  # independent greedy continuation g0..gK
            _reset_target_kv(target)
            logits, _ = target.forward_logits_taps(
                _toks(seq), torch.arange(len(seq)), last_logits_only=True
            )
            t = int(logits[:, -1].argmax())
            greedy.append(t)
            seq.append(t)

        _reset_target_kv(target)
        prefill_token, _ = spec.prefill(_toks(prompt), torch.arange(L))
        verify_ids, _ = spec.target_verify(_toks(greedy[:K]), torch.arange(L, L + K))

    assert int(prefill_token) == greedy[0]  # checks proposal 0
    assert verify_ids[0].tolist() == greedy[1:]  # shifted; last entry is the bonus


def test_target_satisfies_tap_target_protocol():
    # The reference target (gemma4-31B) conforms to the TapTarget protocol the
    # speculator/export are written against, so any target implementing it works.
    from executorch.examples.models.eagle3.target import TapTarget

    _, target, _ = _build()
    assert isinstance(target, TapTarget)


def test_rejects_draft_without_own_embed():
    import pytest

    _, target, _ = _build()
    draft = Eagle3Draft(
        Eagle3Config(
            hidden_size=HIDDEN,
            target_hidden_size=HIDDEN,
            head_dim=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            draft_vocab_size=DRAFT_VOCAB,
            target_vocab_size=TARGET_VOCAB,
            aux_hidden_state_layers=TAP_LAYERS,
            has_own_embed=False,
        )
    )
    with pytest.raises(ValueError, match="has_own_embed"):
        Eagle3Speculator(target, draft)


def test_draft_decode_maps_to_target_vocab():
    spec, _, draft = _build()
    draft.d2t.copy_(torch.arange(DRAFT_VOCAB))  # target_id = 2 * draft_id
    tokens = _toks([5, 11, 2])
    feature = torch.randn(1, 3, HIDDEN)
    with torch.no_grad():
        draft.reset_cache()
        target_ids, g = spec.draft_decode(tokens, feature, torch.arange(3))
        draft.reset_cache()
        emb = draft.embed(tokens)
        dlogits, _ = draft.forward_cached(emb, feature, torch.arange(3))
    expected = dlogits.argmax(dim=-1)
    assert torch.equal(target_ids, expected + expected)
    assert g.shape == (1, 3, HIDDEN)


def test_draft_chain_via_methods_matches_full_recompute():
    # The crux: a KV-cached chain (prefill -> seed -> step) must equal the
    # stateless full-recompute draft chain the eager reference uses.
    spec, target, draft = _build()
    prompt = [7, 3, 21, 9, 14, 2]
    L, K = len(prompt), 4

    with torch.no_grad():
        # Reference: full recompute (mirrors eager_reference.draft_chain).
        _reset_target_kv(target)
        _, taps = target.forward_logits_taps(
            _toks(prompt), torch.arange(L), last_logits_only=False
        )
        feats = draft.fuse(taps)
        tokens, ref = list(prompt), []
        for _ in range(K):
            emb = draft.embed(_toks(tokens))
            dl, g = draft(emb, feats, torch.arange(len(tokens)))
            tid = int(draft.draft_to_target(dl[0, -1].argmax().reshape(1))[0])
            ref.append(tid)
            tokens.append(tid)
            feats = torch.cat([feats, g[:, -1:]], dim=1)

        # Incremental: prefill target, seed draft over the prompt, then step.
        _reset_target_kv(target)
        draft.reset_cache()
        _, feat_prompt = spec.prefill(_toks(prompt), torch.arange(L))
        tids, g = spec.draft_decode(_toks(prompt), feat_prompt, torch.arange(L))
        got = [int(tids[0, -1])]
        tok_step, feat_step = tids[:, -1:], g[:, -1:]
        for k in range(1, K):
            tids, g = spec.draft_decode(
                tok_step, feat_step, torch.arange(L + k - 1, L + k)
            )
            got.append(int(tids[0, 0]))
            tok_step, feat_step = tids, g

    assert got == ref


@torch.no_grad()
def _greedy(target, prompt, n):
    """Greedy target decode (the lossless reference)."""
    seq, out = list(prompt), []
    for _ in range(n):
        _reset_target_kv(target)
        logits, _ = target.forward_logits_taps(
            _toks(seq), torch.arange(len(seq)), last_logits_only=True
        )
        t = int(logits[:, -1].argmax())
        seq.append(t)
        out.append(t)
    return out


@torch.no_grad()
def _chain(spec, seed_tokens, seed_feat, seed_pos, K):
    """Seed the draft at seed_pos (last slot predicts p0), then K-1 recurrent steps."""
    tids, g = spec.draft_decode(seed_tokens, seed_feat, seed_pos)
    proposals = [int(tids[0, -1])]
    last = int(seed_pos[-1])
    tok, feat = tids[:, -1:], g[:, -1:]
    for k in range(1, K):
        tids, g = spec.draft_decode(tok, feat, torch.tensor([last + k]))
        proposals.append(int(tids[0, 0]))
        tok, feat = tids, g
    return proposals


@torch.no_grad()
def _shifted_spec_decode(spec, prompt, K, num_gen):
    """Shifted (vLLM-EAGLE) one-target-forward-per-round speculative decode.

    Mirrors the C++ SpeculativeTokenGenerator and uses ONLY the three exported
    methods (prefill, target_verify, draft_decode) — no standalone target decode.
    The draft pairs target hidden_state_t with token_{t+1} (vLLM
    ``set_inputs_first_pass``: input_ids shifted by one, hidden_states unshifted),
    so each new chain seeds from the last verified hidden state (verify_feat) plus
    the corrected token's embedding — the corrected token never needs its own
    target forward, which is what makes the 3-method artifact sufficient.
    """
    target, draft = spec.target, spec.draft
    _reset_target_kv(target)
    draft.reset_cache()
    L = len(prompt)
    bonus_t, feat_prompt = spec.prefill(_toks(prompt), torch.arange(L))
    anchor, anchor_pos = int(bonus_t), L
    emitted = [anchor]  # the prefill bonus (token at position L) is free
    # Seed shifted: draft slot p pairs feat_prompt[p] with token_{p+1}; last slot
    # pairs feat_prompt[L-1] with the bonus and predicts position L+1.
    proposals = _chain(
        spec, _toks(prompt[1:] + [anchor]), feat_prompt, torch.arange(L), K
    )

    while len(emitted) < num_gen:
        verify_ids, verify_feat = spec.target_verify(
            _toks([anchor] + proposals), torch.arange(anchor_pos, anchor_pos + K + 1)
        )
        a = 0
        for j in range(K):
            if proposals[j] == int(verify_ids[0, j]):
                a += 1
            else:
                break
        corrected = int(verify_ids[0, a])
        new = (proposals[:a] + [corrected])[: num_gen - len(emitted)]
        emitted += new
        if len(emitted) >= num_gen:
            break
        # Reseed draft slots anchor_pos..anchor_pos+a (shifted): slot anchor_pos+i
        # holds (verify_feat[i], token_{anchor_pos+i+1}); the last slot predicts the
        # next chain's p0. No separate forward for the corrected token's feature.
        reseed_tokens = proposals[:a] + [corrected]
        proposals = _chain(
            spec,
            _toks(reseed_tokens),
            verify_feat[:, : a + 1],
            torch.arange(anchor_pos, anchor_pos + a + 1),
            K,
        )
        anchor, anchor_pos = corrected, anchor_pos + 1 + a
    return emitted[:num_gen]


def test_shifted_speculative_decode_is_lossless():
    # The full runner loop (shifted, one target forward/round) over several rounds
    # must reproduce greedy decoding token for token, using only the three exported
    # methods. This exercises the cross-round reseed and proves the 3-method
    # artifact is sufficient for multi-round speculative decoding.
    spec, target, _ = _build()
    prompt = [7, 3, 21, 9, 14, 2]
    got = _shifted_spec_decode(spec, prompt, K=4, num_gen=16)
    ref = _greedy(target, prompt, len(got))
    assert got == ref


@torch.no_grad()
def _prefill_then_verify(spec, prompt, proposals):
    """Reset, prefill, then run one target_verify over [anchor] + proposals.

    Returns (anchor, accept_len, corrected) under greedy acceptance, mirroring
    the runner's verify step.
    """
    target = spec.target
    _reset_target_kv(target)
    spec.draft.reset_cache()
    L = len(prompt)
    bonus_t, _ = spec.prefill(_toks(prompt), torch.arange(L))
    anchor = int(bonus_t)
    K = len(proposals)
    verify_ids, _ = spec.target_verify(
        _toks([anchor] + proposals), torch.arange(L, L + K + 1)
    )
    a = 0
    for j in range(K):
        if proposals[j] == int(verify_ids[0, j]):
            a += 1
        else:
            break
    return anchor, a, int(verify_ids[0, a])


def test_target_verify_acceptance_paths_are_deterministic():
    # The lossless loop test above lets random weights pick the acceptance count,
    # so it can pass while only exercising a == 0. Here we force a == 0, 0 < a < K,
    # and a == K by building proposals from the target's own greedy continuation,
    # pinning the alignment contract: verify_ids[a] (the corrected token) is always
    # the greedy token after the last accepted position, and at a == K it is the
    # folded bonus that needs no separate target forward.
    spec, target, _ = _build()
    prompt = [7, 3, 21, 9, 14, 2]
    K = 4
    vocab = target.config.vocab_size
    greedy = _greedy(target, prompt, K + 2)

    anchor, a, corrected = _prefill_then_verify(spec, prompt, greedy[1 : 1 + K])
    assert anchor == greedy[0]  # prefill bonus is the first greedy token
    assert a == K
    assert corrected == greedy[1 + K]

    wrong = (greedy[3] + 1) % vocab
    _, a, corrected = _prefill_then_verify(spec, prompt, greedy[1:3] + [wrong, wrong])
    assert a == 2
    assert corrected == greedy[3]

    wrong0 = (greedy[1] + 1) % vocab
    _, a, corrected = _prefill_then_verify(spec, prompt, [wrong0] * K)
    assert a == 0
    assert corrected == greedy[1]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
