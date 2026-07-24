# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python implementation of the DFlash speculative decoding loop for Gemma4-31B.

Mirrors examples/models/qwen3/run_dflash.py's algorithm exactly (see that
file's module docstring for the four-step round structure and V1 scope
notes -- unchanged here). Tokenizer/chat-template handling differs
genuinely from Qwen3 -- see run_baseline.py's module docstring for why
(raw tokenizers.Tokenizer API, manual BOS prepend, three hardcoded EOS ids,
apply_chat_template imported directly from inference.py).
"""

import argparse
import time
from pathlib import Path

import torch

from executorch.backends.mlx.examples.llm.dflash_draft_model import load_dflash_config
from executorch.examples.models.gemma4_31b.inference import apply_chat_template
from executorch.runtime import Runtime, Verification
from tokenizers import Tokenizer

EOS_TOKEN_IDS = {1, 50, 106}
BOS_TOKEN_ID = 2


def first_mismatch(draft_ids, target_ids):
    """Returns the number of consecutive draft predictions that match the target."""
    for i in range(len(draft_ids)):
        if draft_ids[i] != target_ids[i]:
            return i
    return len(draft_ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--target-pte", default="gemma4_31b_dflash_exports_mlx/model.pte"
    )
    p.add_argument("--draft-pte", default="gemma4_31b_dflash_draft.pte")
    p.add_argument(
        "--draft-config-dir",
        default="./gemma-4-31B-it-DFlash",
        help="Local directory with the draft checkpoint's config.json "
        "(from `hf download z-lab/gemma-4-31B-it-DFlash --local-dir ...`).",
    )
    p.add_argument(
        "--tokenizer-path",
        default="./gemma-4-31B-it-HQQ-INT4/tokenizer.json",
    )
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--no-chat-template",
        dest="chat_template",
        action="store_false",
        default=True,
        help="Disable Gemma4's chat template. On by default (matches the "
        "draft model's training distribution).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-round timing/acceptance debug output.",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Override the draft checkpoint config's block_size -- needed when "
        "--draft-pte was exported with a different block_size than the "
        "z-lab checkpoint's native config.",
    )
    args = p.parse_args()

    config = load_dflash_config(Path(args.draft_config_dir))
    mask_id = config.mask_token_id
    block_size = args.block_size if args.block_size is not None else config.block_size

    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    rt = Runtime.get()
    target = rt.load_program(
        args.target_pte, verification=Verification.Minimal
    ).load_method("forward")
    draft = rt.load_program(
        args.draft_pte, verification=Verification.Minimal
    ).load_method("forward")

    prompt_str = args.prompt if not args.chat_template else apply_chat_template(args.prompt)
    input_ids = tokenizer.encode(prompt_str).ids
    if not input_ids or input_ids[0] != BOS_TOKEN_ID:
        input_ids = [BOS_TOKEN_ID] + input_ids
    prompt_ids = torch.tensor([input_ids], dtype=torch.long)
    prompt_len = prompt_ids.shape[1]

    input_pos = torch.arange(prompt_len, dtype=torch.long)
    logits, new_hidden_chunk = target.execute([prompt_ids, input_pos])
    new_hidden_chunk = new_hidden_chunk.float()
    draft_cached_len = 0
    pos = prompt_len
    last_token = int(logits[0, -1].argmax())

    generated = [last_token]
    rounds = 0
    accepted_total = 0
    emitted_total = 0
    t0 = time.time()

    while len(generated) < args.max_new_tokens:
        rounds += 1
        bs = block_size

        draft_input = torch.cat(
            [
                torch.tensor([[last_token]], dtype=torch.long),
                torch.full((1, bs - 1), mask_id, dtype=torch.long),
            ],
            dim=1,
        )
        draft_ctx_start_pos = torch.tensor([draft_cached_len], dtype=torch.long)
        _t0 = time.time()
        (draft_logits,) = draft.execute([draft_input, new_hidden_chunk, draft_ctx_start_pos])
        _draft_exec_time = time.time() - _t0
        draft_ids = draft_logits[0].argmax(-1).tolist()

        verify_input = torch.cat(
            [
                torch.tensor([[last_token]], dtype=torch.long),
                torch.tensor([draft_ids], dtype=torch.long),
            ],
            dim=1,
        )
        verify_pos = torch.arange(pos, pos + verify_input.shape[1], dtype=torch.long)
        _t1 = time.time()
        target_logits, new_hidden = target.execute([verify_input, verify_pos])
        _target_exec_time = time.time() - _t1
        target_ids = target_logits[0].argmax(-1).tolist()

        accepted = first_mismatch(draft_ids, target_ids)
        if args.verbose and rounds <= 10:
            print(
                f"  timing: draft_exec={_draft_exec_time*1000:.1f}ms "
                f"target_exec={_target_exec_time*1000:.1f}ms ctx_len={draft_cached_len}"
            )
        if args.verbose and rounds <= 5:
            print(
                f"round {rounds}: pos={pos} hidden_ctx={draft_cached_len} "
                f"draft_ids[:5]={draft_ids[:5]} target_ids[:5]={target_ids[:5]} accepted={accepted}"
            )
        new_tokens = draft_ids[:accepted] + [target_ids[accepted]]

        hit_eos = [t for t in new_tokens if t in EOS_TOKEN_IDS]
        if hit_eos:
            first_eos_pos = min(new_tokens.index(t) for t in hit_eos)
            new_tokens = new_tokens[: first_eos_pos + 1]
            accepted = min(accepted, len(new_tokens) - 1)

        accepted_total += accepted
        emitted_total += len(new_tokens)

        generated.extend(new_tokens)

        pos += len(new_tokens)
        last_token = new_tokens[-1]
        new_hidden_chunk = new_hidden[:, : len(new_tokens), :].float()
        draft_cached_len += new_hidden_chunk.shape[1]

        if last_token in EOS_TOKEN_IDS:
            break

    dt = time.time() - t0
    text = tokenizer.decode(generated)
    n = len(generated)
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated ({n} tokens): {text}")
    print("\n--stats--")
    print(f"rounds: {rounds}")
    print(f"avg accepted/round (draft-only): {accepted_total / rounds:.2f}")
    print(f"avg emitted/round (tau, incl. bonus): {emitted_total / rounds:.2f}")
    print(f"time: {dt:.2f}s   tokens/s: {n / dt:.2f}")


if __name__ == "__main__":
    main()
