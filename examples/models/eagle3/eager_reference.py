# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""EAGLE-3 eager reference: greedy chain speculative decoding.

Loads a gemma4-31B target with EAGLE-3 hidden-state taps and an EAGLE-3 draft
head, proposes a fixed-length draft chain, verifies it with target logits, and
emits accepted draft tokens plus the target bonus token.

The script compares speculative output with greedy target decoding and reports
per-position acceptance rates ``n-alpha`` plus average emitted tokens per
verification round ``tau``. It recomputes full sequences instead of using a KV
cache.

Usage:
    python -m executorch.examples.models.eagle3.eager_reference \\
        --target /path/to/gemma4-31b-int4 \\
        --draft /path/to/eagle3-draft-head \\
        --prompt "Explain why the sky is blue." \\
        --num-gen 64 --chain 3
"""

import argparse
import os

import torch

from executorch.examples.models.eagle3.draft import Eagle3Draft
from executorch.examples.models.gemma4_31b.export import load_prequantized_model
from executorch.examples.models.gemma4_31b.inference import (
    _move_to_cuda,
    apply_chat_template,
)

EOS_TOKEN_IDS = {1, 50, 106}
BOS_TOKEN_ID = 2


def load_target(target_dir: str, max_seq_len: int, bf16: bool = False):
    """Load the gemma4-31B target from an INT4 directory or bf16 HF checkpoint."""
    if bf16:
        from executorch.examples.models.gemma4_31b.model import Gemma4_31B

        model, config = Gemma4_31B.from_hf_checkpoint(
            target_dir, max_seq_len=max_seq_len
        )
        _move_to_cuda(model, config)
        model.eval()
        return model

    model, config = load_prequantized_model(
        target_dir, max_seq_len=max_seq_len, backend="cuda"
    )
    _move_to_cuda(model, config)
    model.eval()
    import executorch.backends.cuda.int4_dispatch  # noqa: F401

    return model


class Target:
    """Wraps the gemma4-31B target: full-sequence forward returning logits + taps."""

    def __init__(self, model, tap_layers):
        self.model = model
        if tap_layers:
            model.set_eagle_tap_layers(tap_layers)

    @torch.no_grad()
    def forward(self, token_ids: list[int]):
        toks = torch.tensor([token_ids], dtype=torch.long, device="cuda")
        pos = torch.arange(len(token_ids), dtype=torch.long, device="cuda")
        # The verifier reads logits for every proposed-token position.
        logits, taps = self.model.forward_logits_taps(toks, pos, last_logits_only=False)
        return logits[0], taps[0]  # (L, vocab), (L, 3*hidden)


@torch.no_grad()
def embed_tokens(draft: Eagle3Draft, token_ids: list[int]) -> torch.Tensor:
    ids = torch.tensor(token_ids, dtype=torch.long, device="cuda")
    return draft.embed(ids)


@torch.no_grad()
def draft_chain(
    draft: Eagle3Draft,
    confirmed_ids: list[int],
    taps_confirmed: torch.Tensor,
    chain_len: int,
) -> list[int]:
    """Propose ``chain_len`` tokens with target taps followed by recurrent features."""
    feats = draft.fuse(taps_confirmed.unsqueeze(0))  # (1, L, hidden)
    tokens = list(confirmed_ids)
    proposals = []
    for _ in range(chain_len):
        emb = embed_tokens(draft, tokens).unsqueeze(0)  # (1, L, hidden)
        pos = torch.arange(len(tokens), dtype=torch.long, device="cuda")
        dlogits, g = draft(emb, feats, pos)
        draft_id = int(dlogits[0, -1].argmax())
        tgt_id = int(draft_id + draft.d2t[draft_id])
        proposals.append(tgt_id)
        tokens.append(tgt_id)
        feats = torch.cat([feats, g[:, -1:, :]], dim=1)
    return proposals


@torch.no_grad()
def speculative_decode(draft, target, prompt_ids, num_gen, chain_len):
    seq = list(prompt_ids)
    emitted = []
    reached = [0] * chain_len
    accepted = [0] * chain_len
    accept_lengths = []

    # This reference recomputes the whole sequence each round through the
    # stateful gemma target, whose sliding layers assert positions fit one ring
    # (2*sliding_window). It is a short-prompt correctness reference, not a
    # long-context path, so fail early with a clear message instead of letting
    # the RingKVCache assertion fire mid-run.
    max_ctx = 2 * target.model.config.sliding_window

    while len(emitted) < num_gen:
        L = len(seq)
        if L + chain_len > max_ctx:
            raise RuntimeError(
                f"eager reference is limited to 2*sliding_window={max_ctx} "
                f"positions (seq={L} + chain={chain_len} exceeds it); it "
                f"recomputes through the stateful RingKVCache and does not "
                f"support long context. Use a shorter prompt or smaller "
                f"--num-gen."
            )
        _, taps = target.forward(seq)
        proposals = draft_chain(draft, seq, taps, chain_len)

        vlogits, _ = target.forward(seq + proposals)
        a = 0
        for j in range(chain_len):
            reached[j] += 1
            tgt_tok = int(vlogits[L - 1 + j].argmax())
            if tgt_tok == proposals[j]:
                accepted[j] += 1
                a += 1
            else:
                break

        accepted_tokens = proposals[:a]
        eos_pos = next(
            (i for i, tok in enumerate(accepted_tokens) if tok in EOS_TOKEN_IDS),
            None,
        )
        if eos_pos is not None:
            new_tokens = accepted_tokens[: eos_pos + 1]
        else:
            corrected = int(vlogits[L - 1 + a].argmax())  # target's own greedy token
            new_tokens = accepted_tokens + [corrected]

        remaining = num_gen - len(emitted)
        new_tokens = new_tokens[:remaining]
        seq += new_tokens
        emitted += new_tokens
        accept_lengths.append(min(len(new_tokens), len(accepted_tokens)))
        if any(t in EOS_TOKEN_IDS for t in new_tokens):
            break

    n_alpha = [
        accepted[j] / reached[j] if reached[j] else 0.0 for j in range(chain_len)
    ]
    return emitted, n_alpha, accept_lengths


@torch.no_grad()
def greedy_decode(target, prompt_ids, num_gen):
    seq = list(prompt_ids)
    out = []
    while len(out) < num_gen:
        logits, _ = target.forward(seq)
        t = int(logits[-1].argmax())
        seq.append(t)
        out.append(t)
        if t in EOS_TOKEN_IDS:
            break
    return out


def main():
    p = argparse.ArgumentParser(description="EAGLE-3 eager reference (greedy chain).")
    p.add_argument("--target", required=True, help="gemma4-31B prequantized dir.")
    p.add_argument("--draft", required=True, help="EAGLE-3 draft head dir.")
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--prompt", default="Explain why the sky is blue.")
    p.add_argument("--raw-prompt", action="store_true")
    p.add_argument("--num-gen", type=int, default=64)
    p.add_argument("--chain", type=int, default=3)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument(
        "--bf16", action="store_true", help="Target is a bf16 HF checkpoint dir."
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        p.error("CUDA required.")
    if args.num_gen < 1 or args.chain < 1:
        p.error("--num-gen and --chain must be >= 1.")

    tok_path = args.tokenizer_path or os.path.join(args.target, "tokenizer.json")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tok_path)
    prompt_str = args.prompt if args.raw_prompt else apply_chat_template(args.prompt)
    prompt_ids = tokenizer.encode(prompt_str).ids
    if not prompt_ids or prompt_ids[0] != BOS_TOKEN_ID:
        prompt_ids = [BOS_TOKEN_ID] + prompt_ids

    print(f"Loading target from {args.target} (bf16={args.bf16}) ...")
    target_model = load_target(args.target, args.max_seq_len, bf16=args.bf16)
    draft, dcfg = Eagle3Draft.from_checkpoint(args.draft, device="cuda")
    target = Target(target_model, dcfg.aux_hidden_state_layers)

    print(
        f"\nPrompt: {args.prompt}\nPrompt tokens: {len(prompt_ids)}, chain={args.chain}"
    )
    print("-" * 60)

    emitted, n_alpha, accept_lengths = speculative_decode(
        draft, target, prompt_ids, args.num_gen, args.chain
    )
    greedy_out = greedy_decode(target, prompt_ids, len(emitted))

    n = min(len(emitted), len(greedy_out))
    lossless = emitted[:n] == greedy_out[:n]
    rounds = len(accept_lengths)
    tau = len(emitted) / rounds if rounds else 0.0
    avg_accepted = sum(accept_lengths) / rounds if rounds else 0.0

    print(tokenizer.decode(emitted))
    print("-" * 60)
    print(f"lossless (== greedy): {lossless}")
    if not lossless:
        for i in range(n):
            if emitted[i] != greedy_out[i]:
                print(
                    f"  first divergence at {i}: spec={emitted[i]} greedy={greedy_out[i]}"
                )
                break
    print(f"rounds: {rounds}, emitted: {len(emitted)}")
    print(f"tau (avg acceptance length, incl. bonus): {tau:.3f}")
    print(f"avg accepted draft tokens/round: {avg_accepted:.3f} / {args.chain}")
    for j, a in enumerate(n_alpha):
        print(f"  {j}-alpha: {a:.3f}")


if __name__ == "__main__":
    main()
