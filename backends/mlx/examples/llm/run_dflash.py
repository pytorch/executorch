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

With --cached, the draft only ever sees newly-confirmed context each round
(via a persistent KV cache) instead of reprojecting the full history.
"""

import argparse
import time
from pathlib import Path

import torch

from executorch.backends.mlx.examples.llm.dflash_draft_model import load_dflash_config
from executorch.runtime import Runtime, Verification
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def first_mismatch(draft_ids, target_ids):
    """Returns the number of consecutive draft predictions that match the target."""
    for i in range(len(draft_ids)):
        if draft_ids[i] != target_ids[i]:
            return i
    return len(draft_ids)


def compute_block_len(
    block_size, max_new_tokens, num_generated, max_seq_len, position, draft_room=None
):
    """Compute the maximum proposal length allowed by the block, token and context limits.
    In cached mode, also limit it to the remaining draft KV-cache capacity.
    """
    bs = min(block_size, max_new_tokens - num_generated, max_seq_len - position)
    if draft_room is not None:
        bs = min(bs, draft_room)
    return bs


def truncate_at_eos(new_tokens, accepted, eos_id):
    """Truncate right after eos_id if present, and clamp accepted so it
    never claims more accepted draft tokens than survive truncation."""
    if eos_id in new_tokens:
        new_tokens = new_tokens[: new_tokens.index(eos_id) + 1]
        accepted = min(accepted, len(new_tokens) - 1)
    return new_tokens, accepted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-pte", default="qwen3_4b_dflash_target.pte")
    p.add_argument("--draft-pte", default="qwen3_4b_dflash_draft.pte")
    p.add_argument("--draft-model", default="z-lab/Qwen3-4B-DFlash-b16")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--no-chat-template",
        dest="chat_template",
        action="store_false",
        default=True,
        help="Disable Qwen3's chat template. On by default (paper's eval setup).",
    )
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Qwen3 thinking mode. Paper's Table 1 uses thinking mode DISABLED.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-round timing/acceptance debug output, including raw token IDs.",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Override the draft checkpoint config's block_size.",
    )
    p.add_argument(
        "--max-seq-len", type=int, default=4096, help="Target's static cache capacity."
    )
    p.add_argument(
        "--draft-max-ctx-len",
        type=int,
        default=None,
        help="Max context length used when the cached draft was exported. Required with --cached, defaults to --max-seq-len.",
    )
    p.add_argument(
        "--cached",
        action="store_true",
        help="Use the persistent draft KV cache -- draft-pte must have been "
        "exported with export_dflash_draft.py --cached.",
    )
    args = p.parse_args()

    config = load_dflash_config(
        Path(
            snapshot_download(
                args.draft_model, allow_patterns=["*.json"], local_files_only=True
            )
        )
    )
    mask_id = config.mask_token_id
    block_size = args.block_size if args.block_size is not None else config.block_size

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    eos_id = tokenizer.eos_token_id

    if tokenizer.chat_template is None:
        # Some transformers versions expect a standalone chat_template.jinja
        # and don't fall back to the embedded tokenizer_config.json field.
        import json

        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(args.tokenizer, "tokenizer_config.json")
        cfg = json.loads(Path(cfg_path).read_text())
        if "chat_template" in cfg:
            tokenizer.chat_template = cfg["chat_template"]

    rt = Runtime.get()
    target = rt.load_program(
        args.target_pte, verification=Verification.Minimal
    ).load_method("forward")
    draft = rt.load_program(
        args.draft_pte, verification=Verification.Minimal
    ).load_method("forward")

    if args.chat_template:
        # The draft was trained on chat-formatted prompt/response pairs, so
        # matching that format at inference keeps acceptance rates high.
        from executorch.backends.mlx.examples.llm.dflash_qwen3_adapter import (
            apply_qwen3_chat_template,
        )

        prompt_ids = apply_qwen3_chat_template(
            tokenizer, args.prompt, args.enable_thinking
        )
    else:
        prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]

    # Prefill the target once to obtain the first token and hidden states used by the draft. .contiguous() is required by the MLX backend.
    input_pos = torch.arange(prompt_len, dtype=torch.long)
    logits, hidden = target.execute([prompt_ids.contiguous(), input_pos.contiguous()])
    hidden = hidden.float()
    pos = prompt_len
    last_token = int(logits[0, -1].argmax())

    if args.cached:
        # pending_ctx stores confirmed hidden states that have not yet been written to the draft cache.
        # It only flushes when the draft runs, since bs==1 rounds skip speculation entirely.
        pending_ctx = hidden
        prev_ctx_len = 0
        draft_max_ctx = (
            args.draft_max_ctx_len
            if args.draft_max_ctx_len is not None
            else args.max_seq_len
        )

    generated = [last_token]
    rounds = 0
    accepted_total = 0
    emitted_total = 0
    t0 = time.time()

    while len(generated) < args.max_new_tokens:
        rounds += 1
        draft_room = None
        if args.cached:
            draft_room = draft_max_ctx - (prev_ctx_len + pending_ctx.shape[1])
        bs = compute_block_len(
            block_size,
            args.max_new_tokens,
            len(generated),
            args.max_seq_len,
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
            if args.cached:
                cache_position = torch.arange(
                    prev_ctx_len, prev_ctx_len + pending_ctx.shape[1], dtype=torch.long
                )
                assert prev_ctx_len + pending_ctx.shape[1] <= draft_max_ctx, (
                    f"draft cache overrun: writing up to "
                    f"{prev_ctx_len + pending_ctx.shape[1]} but capacity is "
                    f"{draft_max_ctx}. Re-export with larger --max-ctx-len "
                    f"or pass correct --draft-max-ctx-len."
                )
                (draft_logits,) = draft.execute(
                    [draft_input, pending_ctx.contiguous(), cache_position.contiguous()]
                )
                prev_ctx_len += pending_ctx.shape[1]
                pending_ctx = pending_ctx.new_zeros((1, 0, pending_ctx.shape[-1]))
            else:
                (draft_logits,) = draft.execute([draft_input, hidden])
            _draft_exec_time = time.time() - _t0
            draft_ids = draft_logits[0].argmax(-1).tolist()  # bs: 1 tokens

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
            ctx_len = prev_ctx_len if args.cached else hidden.shape[1]
            print(
                f"round {rounds}: pos={pos} ctx_len={ctx_len} "
                f"draft_exec={_draft_exec_time*1000:.1f}ms target_exec={_target_exec_time*1000:.1f}ms "
                f"draft_ids[:5]={draft_ids[:5]} target_ids[:5]={target_ids[:5]} accepted={accepted}"
            )
        new_tokens = draft_ids[:accepted] + [target_ids[accepted]]

        # Truncate at EOS before updating stats, so a round that hits EOS
        # doesn't over-count tokens that never actually get emitted.
        new_tokens, accepted = truncate_at_eos(new_tokens, accepted, eos_id)
        accepted_total += accepted
        emitted_total += len(new_tokens)
        generated.extend(new_tokens)

        pos += len(new_tokens)
        last_token = new_tokens[-1]
        if args.cached:
            pending_ctx = torch.cat(
                [pending_ctx, new_hidden[:, : len(new_tokens), :].float()], dim=1
            )
        else:
            hidden = torch.cat(
                [hidden, new_hidden[:, : len(new_tokens), :].float()], dim=1
            )

        if eos_id in new_tokens:
            break

    dt = time.time() - t0
    text = tokenizer.decode(generated)
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


if __name__ == "__main__":
    main()
