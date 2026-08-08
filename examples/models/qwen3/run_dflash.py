# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python implementation of the DFlash speculative decoding loop for the ExecuTorch MLX backend. 

This file coordinates the interaction between the target model and the draft model during inference. Instead of asking the target model to generate one token at a time, DFlash first lets the lightweight draft model predict a block of future tokens, then asks the target model to verify those predictions in a single forward pass. Any matching draft tokens are accepted, while the first incorrect prediction is replaced with the target model's token. The process then repeats from the updated position. 

Each speculation round consists of four steps: 
    1. Build a draft block: [last_token, <mask>, <mask>, ...]
    2. Run draft model to predict all masked tokens in parallel
    3. Verify those predictions with the target model, keeping matching prefix and replacing the first mismatch with target's prediction. 
    4. advance the sequence position to the newly accepted prefix and repeat. 

V1 scope (per the issue discussion): 
    - Greedy decoding
    - Single-batch inference
    - Chain drafting
    - Standard attention models
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


def compute_block_len(block_size, max_new_tokens, num_generated, max_seq_len, position, draft_room=None):
    """Bounded dynamic proposal length (review point 1):
        min(checkpoint block_size, remaining token budget, remaining context budget)
    draft_room, if given (cached mode), additionally bounds against the
    draft cache's own remaining capacity, so a cache write never overruns
    its fixed buffer.
    """
    bs = min(block_size, max_new_tokens - num_generated, max_seq_len - position)
    if draft_room is not None:
        bs = min(bs, draft_room)
    return bs


def truncate_at_eos(new_tokens, accepted, eos_id):
    """If eos_id appears in new_tokens, truncate right after it and clamp
    accepted so it never claims more accepted draft tokens than actually
    survive truncation."""
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
        help="Print per-round timing/acceptance debug output.",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Override the draft checkpoint config's block_size -- needed when "
        "--draft-pte was exported with a different block_size than the "
        "z-lab checkpoint's native config (e.g. our block_size=8 test export).",
    )
    p.add_argument("--max-seq-len", type=int, default=4096, help="Target's static cache capacity.")
    p.add_argument(
        "--draft-max-ctx-len", type=int, default=None,
        help="The --max-ctx-len the cached draft-pte was exported with. "
        "Required with --cached: block_len is bounded against it so the "
        "draft cache never overruns its buffer. Defaults to --max-seq-len.",
    )
    p.add_argument(
        "--cached",
        action="store_true",
        help="Use the persistent draft KV cache (review point 3) -- draft-pte "
        "must have been exported with export_dflash_draft.py --cached. "
        "Only newly-confirmed hidden states are sent to the draft each "
        "round, instead of the full accumulated context.",
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
        # Some transformers versions expect chat templates in a standalone
        # chat_template.jinja file and don't fall back to the legacy
        # embedded tokenizer_config.json field when that file is absent.
        import json
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(args.tokenizer, "tokenizer_config.json")
        cfg = json.loads(Path(cfg_path).read_text())
        if "chat_template" in cfg:
            tokenizer.chat_template = cfg["chat_template"]

    # The draft model was trained on Qwen3 chat-formatted prompt/response pairs,so applying the same chat template during inference keeps the input distribution consistent with training.
    # Using raw completion text noticeably reduces acceptance rates.

    rt = Runtime.get()
    target = rt.load_program(
        args.target_pte, verification=Verification.Minimal
    ).load_method("forward")
    draft = rt.load_program(
        args.draft_pte, verification=Verification.Minimal
    ).load_method("forward")

    if args.chat_template:
        messages = [{"role": "user", "content": args.prompt}]
        chat_out = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
            return_tensors="pt",
        )
        # Different Transformers versions return either a BatchEncoding or a tensor.
        # Normalize both cases to a tensor.
        prompt_ids = chat_out.input_ids if hasattr(chat_out, "input_ids") else chat_out
    else:
        prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]

    # Run the target model over the prompt once to initialize generation.
    # This produces the first next-token prediction and the hidden states that condition the draft model during speculative decoding.
    input_pos = torch.arange(prompt_len, dtype=torch.long)
    # .contiguous(): the MLX backend's native tensor conversion requires
    # standard C-contiguous strides and fails hard otherwise ("tensor is
    # not contiguous"). Seen on this environment's torch build even though
    # prompt_ids/input_pos are built via plain tokenizer output / arange --
    # a no-op if already contiguous, a real fix if either secretly isn't.
    logits, hidden = target.execute([prompt_ids.contiguous(), input_pos.contiguous()])
    hidden = hidden.float()
    pos = prompt_len
    last_token = int(logits[0, -1].argmax())

    if args.cached:
        # Hidden states not yet written to the draft's persistent cache.
        # Starts as the prompt's hidden states. Flushed into the cache (and
        # prev_ctx_len advanced) only on rounds where the draft actually
        # runs -- see the bs==1 skip below for why this can't just advance
        # unconditionally every round.
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
        # Dynamic block_len: shrink the proposal near the end of generation.
        bs = min(block_size, args.max_new_tokens - len(generated), args.max_seq_len - pos)
        if args.cached:
            # Bound against the DRAFT cache's own capacity too, so a cache
            # write never runs off the fixed buffer (silent corruption).
            draft_room = draft_max_ctx - (prev_ctx_len + pending_ctx.shape[1])
            bs = min(bs, draft_room)
        if bs <= 0:
            break

        if bs == 1:
            # No room to speculate -- skip the draft, target-only step.
            draft_ids = []
            _draft_exec_time = 0.0
        else:
            # 1. Build draft input block.
            draft_input = torch.cat(
                [
                    torch.tensor([[last_token]], dtype=torch.long),
                    torch.full((1, bs - 1), mask_id, dtype=torch.long),
                ],
                dim=1,
            )
            _t0 = time.time()
            if args.cached:
                # bs==1 rounds skip the draft entirely (see above), so
                # pending_ctx may hold more than one round's worth of
                # confirmed-but-unwritten context by the time the draft
                # actually runs again -- that's fine, it all gets written
                # (and prev_ctx_len advanced) together, right here.
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
            draft_ids = draft_logits[0].argmax(-1).tolist()  # bs - 1 tokens

        # 2. Verify the draft predictions. Target model predicts the next token after every position in the block in a single forward pass.
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

        # 3. Keep every drafting token that matches the target. At the first mismatch, stop accepting draft predictions and use the target model's token instead.
        # (target_ids has block_size entries vs draft_ids' block_size - 1, so
        # target_ids[accepted] is always in-bounds, including the all-accepted
        # bonus-token case.)
        accepted = first_mismatch(draft_ids, target_ids)
        if args.verbose:
            print(
                f"  timing: draft_exec={_draft_exec_time*1000:.1f}ms "
                f"target_exec={_target_exec_time*1000:.1f}ms "
                f"ctx_len={prev_ctx_len if args.cached else hidden.shape[1]}"
            )
        if args.verbose:
            print(
                f"round {rounds}: pos={pos} "
                f"hidden_ctx={prev_ctx_len if args.cached else hidden.shape[1]} "
                f"draft_ids[:5]={draft_ids[:5]} target_ids[:5]={target_ids[:5]} accepted={accepted}"
            )
        new_tokens = draft_ids[:accepted] + [target_ids[accepted]]

        # Stop generation once an EOS token becomes part of the accepted sequence.
        # Truncate before updating the running stats so a round that hits EOS
        # doesn't over-count tokens/acceptances that never actually get emitted.
        new_tokens, accepted = truncate_at_eos(new_tokens, accepted, eos_id)

        accepted_total += accepted
        emitted_total += len(new_tokens)

        generated.extend(new_tokens)

        # 4. Advance the accepted sequence. Rejected draft tokens are discarded, and the next round starts from the updated position.
        pos += len(new_tokens)
        last_token = new_tokens[-1]
        # Append the hidden states for the newly accepted tokens.
        if args.cached:
            # Always accumulate here, whether or not this round's draft call
            # actually ran (see the bs==1 skip and the flush-on-next-call
            # logic above) -- this is what guarantees prev_ctx_len only ever
            # advances by exactly what's genuinely been written to the cache.
            pending_ctx = torch.cat(
                [pending_ctx, new_hidden[:, : len(new_tokens), :].float()], dim=1
            )
        else:
            # The draft model conditions on the hidden states of the entire
            # sequence, so this context grows as generation progresses
            # rather than being replaced each round.
            hidden = torch.cat([hidden, new_hidden[:, : len(new_tokens), :].float()], dim=1)

        if eos_id in new_tokens:
            break

    dt = time.time() - t0
    text = tokenizer.decode(generated)
    n = len(generated)
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated ({n} tokens): {text}")
    print(f"TOKEN_IDS: {generated}")
    print("\n--stats--")
    print(f"rounds: {rounds}")
    if rounds:
        print(f"avg accepted/round (draft-only): {accepted_total / rounds:.2f}")
        print(f"avg emitted/round (tau, incl. bonus): {emitted_total / rounds:.2f}")
    print(f"time: {dt:.2f}s   tokens/s: {n / dt:.2f}")


if __name__ == "__main__":
    main()
