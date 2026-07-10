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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-pte", default="qwen3_4b_dflash_target.pte")
    p.add_argument("--draft-pte", default="qwen3_4b_dflash_draft.pte")
    p.add_argument("--draft-model", default="z-lab/Qwen3-4B-DFlash-b16")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--chat-template",
        action="store_true",
        default=True,
        help="Apply Qwen3's chat template (paper's eval setup). Default on.",
    )
    p.add_argument("--no-chat-template", dest="chat_template", action="store_false")
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
    logits, hidden = target.execute([prompt_ids, input_pos])
    hidden = hidden.float()
    pos = prompt_len
    last_token = int(logits[0, -1].argmax())

    generated = [last_token]
    rounds = 0
    accepted_total = 0
    emitted_total = 0
    t0 = time.time()

    while len(generated) < args.max_new_tokens:
        rounds += 1
        # The exported draft model expects a fixed input shape for its token block.
        # Although the hidden-state and position inputs support dynamic lengths, the token input does not.
        # Reducing the block size near the end of generation causes the runtime to reject the input.
        # Supporting truly dynamic block sizes would require exporting the draft model with a dynamic token dimension.
        bs = block_size

        # 1. Build draft input block.
        draft_input = torch.cat(
            [
                torch.tensor([[last_token]], dtype=torch.long),
                torch.full((1, bs - 1), mask_id, dtype=torch.long),
            ],
            dim=1,
        )
        draft_pos = torch.arange(hidden.shape[1] + bs, dtype=torch.long).unsqueeze(0)
        _t0 = time.time()
        (draft_logits,) = draft.execute([draft_input, hidden, draft_pos])
        _draft_exec_time = time.time() - _t0
        _t0b = time.time()
        draft_ids = draft_logits[0].argmax(-1).tolist()  # block_size - 1 tokens
        _draft_argmax_time = time.time() - _t0b

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
        _t1b = time.time()
        target_ids = target_logits[0].argmax(-1).tolist()  # block_size tokens
        _target_argmax_time = time.time() - _t1b

        # 3. Keep every drafting token that matches the target. At the first mismatch, stop accepting draft predictions and use the target model's token instead.
        _t2 = time.time()
        accepted = first_mismatch(draft_ids, target_ids)
        _fm_time = time.time() - _t2
        if args.verbose and rounds <= 10:
            print(
                f"  timing: draft_exec={_draft_exec_time*1000:.1f}ms draft_argmax={_draft_argmax_time*1000:.2f}ms "
                f"target_exec={_target_exec_time*1000:.1f}ms target_argmax={_target_argmax_time*1000:.2f}ms "
                f"first_mismatch={_fm_time*1000:.3f}ms ctx_len={hidden.shape[1]}"
            )
        if args.verbose and rounds <= 5:
            print(
                f"round {rounds}: pos={pos} hidden_ctx={hidden.shape[1]} "
                f"draft_ids[:5]={draft_ids[:5]} target_ids[:5]={target_ids[:5]} accepted={accepted}"
            )
        new_tokens = draft_ids[:accepted] + [target_ids[accepted]]
        accepted_total += accepted
        emitted_total += len(new_tokens)

        # Stop generation once an EOS token becomes part of the accepted sequence.
        if eos_id in new_tokens:
            new_tokens = new_tokens[: new_tokens.index(eos_id) + 1]

        generated.extend(new_tokens)

        # 4. Advance the accepted sequence. Rejected draft tokens are discarded, and the next round starts from the updated position.
        pos += len(new_tokens)
        last_token = new_tokens[-1]
        # Append the hidden states for the newly accepted tokens to the running target context.
        # The draft model conditions on the hidden states of the entire sequence, so this context grows as generation progresses rather than being replaced each round.
        _t3 = time.time()
        hidden = torch.cat([hidden, new_hidden[:, : len(new_tokens), :].float()], dim=1)
        _cat_time = time.time() - _t3
        if args.verbose and rounds <= 10:
            print(f"  timing: hidden_cat={_cat_time*1000:.2f}ms")

        if eos_id in new_tokens:
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
