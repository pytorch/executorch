# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DFlash speculative decoding loop, coordinating the target and draft
models. Instead of the target generating one token at a time, the draft
proposes a whole block of future tokens, the target verifies them all in a
single forward pass, and the accepted prefix (plus a bonus token from the
target) gets emitted. Each round:

    1. Build a draft block: [last_token, <mask>, <mask>, ...]
    2. Run the draft model to predict all masked tokens in parallel
    3. Verify with the target, keeping the matching prefix and replacing
       the first mismatch with the target's own prediction
    4. Advance to the newly accepted position and repeat

The draft keeps a persistent KV cache, so each round it only sees the context
confirmed since the last time it ran rather than reprojecting the full history.
"""

import argparse
import time

import torch

from executorch.backends.mlx.examples.llm.runtime_meta import (
    apply_chat_template,
    chunked_prefill,
    get_eos_token_ids,
    load_text_processor,
    read_const_int,
    read_model_limits,
)
from executorch.runtime import Runtime, Verification


def first_mismatch(draft_ids, target_ids):
    """Returns the number of consecutive draft predictions that match the target."""
    for i in range(len(draft_ids)):
        if draft_ids[i] != target_ids[i]:
            return i
    return len(draft_ids)


def compute_block_len(
    block_len, max_new_tokens, num_generated, max_ctx_len, position, draft_room=None
):
    """Compute the maximum proposal length allowed by the block, token and context limits.
    Also limit it to the remaining draft KV-cache capacity.
    """
    bs = min(block_len, max_new_tokens - num_generated, max_ctx_len - position)
    if draft_room is not None:
        bs = min(bs, draft_room)
    return bs


def truncate_at_eos(new_tokens, accepted, eos_ids):
    """Truncate right after the first EOS if present, and clamp accepted so it
    never claims more accepted draft tokens than survive truncation."""
    for i, token in enumerate(new_tokens):
        if token in eos_ids:
            new_tokens = new_tokens[: i + 1]
            return new_tokens, min(accepted, len(new_tokens) - 1)
    return new_tokens, accepted


def parse_args():
    """Parse the runner's CLI arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--pte", default="qwen3_4b_dflash.pte")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--no-chat-template",
        dest="chat_template",
        action="store_false",
        default=True,
        help="Disable the checkpoint's chat template. On by default (paper's eval setup).",
    )
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Qwen3 thinking mode, ignored by checkpoints whose template lacks it. Paper's Table 1 uses thinking mode DISABLED.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-round timing/acceptance debug output, including raw token IDs.",
    )
    p.add_argument(
        "--n-draft",
        type=int,
        default=None,
        help="Tokens to speculate per round. The draft is fed n_draft + 1 tokens "
        "(the last confirmed token plus n_draft masks) and predicts the masks. "
        "Defaults to the largest block the .pte was exported for, minus one.",
    )
    return p.parse_args()


def resolve_limits(program, pte_path, n_draft_arg):
    """Read the capacities baked into the .pte and derive the draft block length.

    Both caches are baked in at export time, so the program is the only source
    of truth for them -- the runner takes no capacity or block flags.
    Returns ``(max_ctx_len, prefill_chunk_size, block_len)``.
    """
    max_ctx_len, prefill_chunk_size = read_model_limits(program)
    if max_ctx_len is None:
        raise ValueError(
            f"{pte_path} publishes no get_max_ctx_len; re-export it with "
            "dflash/export.py."
        )
    if prefill_chunk_size is None:
        prefill_chunk_size = max_ctx_len

    # The draft consumes [last_token, <mask> * n_draft] and drops the first
    # position when projecting logits, so the block it is fed is n_draft + 1.
    max_block_len = read_const_int(program, "get_max_block_len")
    if max_block_len is None:
        raise ValueError(
            f"{pte_path} publishes no get_max_block_len; re-export it with "
            "dflash/export.py."
        )
    max_n_draft = max_block_len - 1
    n_draft = n_draft_arg if n_draft_arg is not None else max_n_draft
    if not 1 <= n_draft <= max_n_draft:
        raise ValueError(
            f"--n-draft {n_draft} must be in [1, {max_n_draft}]: {pte_path} was "
            f"exported for blocks of at most {max_block_len} tokens."
        )
    return max_ctx_len, prefill_chunk_size, n_draft + 1


def encode_prompt(tokenizer, args):
    """Tokenize the prompt, applying the chat template unless disabled."""
    if not args.chat_template:
        return tokenizer(args.prompt, return_tensors="pt").input_ids
    # The draft was trained on chat-formatted prompt/response pairs, so
    # matching that format at inference keeps acceptance rates high.
    return apply_chat_template(
        tokenizer, args.prompt, enable_thinking=args.enable_thinking
    )


def print_summary(
    args, tokenizer, generated, rounds, accepted_total, emitted_total, dt
):
    """Print the decoded text and the speculation stats for the run."""
    text = tokenizer.decode(generated, skip_special_tokens=True)
    n = len(generated)
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated ({n} tokens): {text}")
    if args.verbose:
        print(f"TOKEN_IDS: {generated}")
    print("\n--stats--")
    print(f"rounds: {rounds}")
    if rounds:
        print(f"avg accepted/round (draft-only): {accepted_total / rounds:.2f}")
        print(f"avg emitted/round (tau, incl. bonus): {emitted_total / rounds:.2f}")
    print(f"time: {dt:.2f}s   tokens/s: {n / dt:.2f}")


def main():
    args = parse_args()

    tokenizer = load_text_processor(args.tokenizer, local_files_only=True)
    eos_ids = get_eos_token_ids(
        tokenizer, model_id=args.tokenizer, local_files_only=True
    )
    if not eos_ids:
        raise ValueError(f"No EOS token id found for {args.tokenizer}")

    rt = Runtime.get()
    program = rt.load_program(args.pte, verification=Verification.Minimal)
    target = program.load_method("target")
    draft = program.load_method("draft")

    max_ctx_len, prefill_chunk_size, block_len = resolve_limits(
        program, args.pte, args.n_draft
    )
    mask_id = read_const_int(program, "get_mask_token_id")
    if mask_id is None:
        raise ValueError(
            f"{args.pte} publishes no get_mask_token_id; re-export it with "
            "dflash/export.py."
        )

    prompt_ids = encode_prompt(tokenizer, args)
    prompt_len = prompt_ids.shape[1]

    # Prefill the target once to obtain the first token and hidden states used by the draft.
    # Hidden states for the whole prompt feed the draft, so they are concatenated
    # across chunks rather than taken from the last step.
    logits, hidden = chunked_prefill(
        target, prompt_ids, prefill_chunk_size, concat_outputs=(1,)
    )
    pos = prompt_len
    last_token = int(logits[0, -1].argmax())

    # pending_ctx stores confirmed hidden states that have not yet been written
    # to the draft cache. It only flushes when the draft runs, since bs==1 rounds
    # skip speculation entirely.
    pending_ctx = hidden.float()
    prev_ctx_len = 0

    generated = [last_token]
    rounds = 0
    accepted_total = 0
    emitted_total = 0
    t0 = time.time()

    while len(generated) < args.max_new_tokens:
        rounds += 1
        draft_room = max_ctx_len - (prev_ctx_len + pending_ctx.shape[1])
        bs = compute_block_len(
            block_len,
            args.max_new_tokens,
            len(generated),
            max_ctx_len,
            pos,
            draft_room,
        )
        if bs <= 0:
            break

        if bs == 1:
            # No room to speculate, skip the draft, target-only step.
            draft_ids = []
            _draft_exec_time = 0.0
        else:
            draft_input = torch.cat(
                [
                    torch.tensor([[last_token]], dtype=torch.long),
                    torch.full((1, bs - 1), mask_id, dtype=torch.long),
                ],
                dim=1,
            )
            _t0 = time.time()
            cache_position = torch.arange(
                prev_ctx_len, prev_ctx_len + pending_ctx.shape[1], dtype=torch.long
            )
            assert prev_ctx_len + pending_ctx.shape[1] <= max_ctx_len, (
                f"draft cache overrun: writing up to "
                f"{prev_ctx_len + pending_ctx.shape[1]} but capacity is "
                f"{max_ctx_len}. Re-export with a larger "
                f"dflash/export.py --max-ctx-len."
            )
            (draft_logits,) = draft.execute(
                [draft_input, pending_ctx.contiguous(), cache_position.contiguous()]
            )
            prev_ctx_len += pending_ctx.shape[1]
            pending_ctx = pending_ctx.new_zeros((1, 0, pending_ctx.shape[-1]))
            _draft_exec_time = time.time() - _t0
            draft_ids = draft_logits[0].argmax(-1).tolist()  # bs - 1 tokens

        # Verify the entire proposal in one target forward pass.
        # target_ids includes one bonus token, so target_ids[accepted] is always valid.
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
        target_ids = target_logits[0].argmax(-1).tolist()  # bs tokens

        accepted = first_mismatch(draft_ids, target_ids)
        if args.verbose:
            print(
                f"round {rounds}: pos={pos} ctx_len={prev_ctx_len} "
                f"draft_exec={_draft_exec_time * 1000:.1f}ms target_exec={_target_exec_time * 1000:.1f}ms "
                f"draft_ids[:5]={draft_ids[:5]} target_ids[:5]={target_ids[:5]} accepted={accepted}"
            )
        new_tokens = draft_ids[:accepted] + [target_ids[accepted]]

        # Truncate at EOS before updating stats, so a round that hits EOS
        # doesn't over-count tokens that never actually get emitted.
        new_tokens, accepted = truncate_at_eos(new_tokens, accepted, eos_ids)
        accepted_total += accepted
        emitted_total += len(new_tokens)
        generated.extend(new_tokens)

        pos += len(new_tokens)
        last_token = new_tokens[-1]
        pending_ctx = torch.cat(
            [pending_ctx, new_hidden[:, : len(new_tokens), :].float()], dim=1
        )

        if eos_ids.intersection(new_tokens):
            break

    dt = time.time() - t0
    print_summary(args, tokenizer, generated, rounds, accepted_total, emitted_total, dt)


if __name__ == "__main__":
    main()
