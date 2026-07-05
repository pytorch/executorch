"""DFlash speculative decoding driver for the ExecuTorch MLX backend (Python).

Same four Phase 3 pieces as qwen3_dflash_engine.cpp, driven through the ET
Python runtime so it runs on machines that can't build the C++ core:
  1. draft block construction  [last_token, mask, mask, ...]
  2. target verification        run target on [last_token] + draft_tokens
  3. acceptance                 keep prefix up to first mismatch, + bonus token
  4. position-based rollback     pos += accepted + 1

V1 scope (per design doc): greedy, single batch, chain drafting, standard attn.
"""

import argparse
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from executorch.runtime import Runtime, Verification

from executorch.backends.mlx.examples.llm.dflash_draft_model import load_dflash_config


def first_mismatch(draft_ids, target_ids):
    """Number of leading draft tokens the target agrees with (greedy accept)."""
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
    p.add_argument("--chat-template", action="store_true", default=True,
                    help="Apply Qwen3's chat template (paper's eval setup). Default on.")
    p.add_argument("--no-chat-template", dest="chat_template", action="store_false")
    p.add_argument("--enable-thinking", action="store_true", default=False,
                    help="Qwen3 thinking mode. Paper's Table 1 uses thinking mode DISABLED.")
    p.add_argument("--block-size", type=int, default=None,
                    help="Override the draft checkpoint config's block_size -- needed when "
                         "--draft-pte was exported with a different block_size than the "
                         "z-lab checkpoint's native config (e.g. our block_size=8 test export).")
    args = p.parse_args()

    config = load_dflash_config(Path(snapshot_download(
        args.draft_model, allow_patterns=["*.json"], local_files_only=True)))
    mask_id = config.mask_token_id
    block_size = args.block_size if args.block_size is not None else config.block_size
    layer_ids = config.target_layer_ids

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    eos_id = tokenizer.eos_token_id

    # Paper's Table 1 evaluates with the chat template applied and thinking mode
    # disabled ("Q3-4B ... thinking mode disabled" -- Section 5.1), not raw
    # completion text. The draft was also trained on prompt+response pairs
    # (Section 5 "Datasets"; Figure 4 shows clean prompt p / response r), so
    # feeding it untemplated text is a real distribution mismatch, not a
    # cosmetic difference.

    rt = Runtime.get()
    target = rt.load_program(args.target_pte, verification=Verification.Minimal).load_method("forward")
    draft = rt.load_program(args.draft_pte, verification=Verification.Minimal).load_method("forward")

    if args.chat_template:
        messages = [{"role": "user", "content": args.prompt}]
        chat_out = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
            return_tensors="pt",
        )
        # Some transformers versions return a BatchEncoding (dict-like) here
        # instead of a raw tensor; normalize either way.
        prompt_ids = chat_out.input_ids if hasattr(chat_out, "input_ids") else chat_out
    else:
        prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]

    # --- Prefill: target over the full prompt -> (logits, hidden) ---
    input_pos = torch.arange(prompt_len, dtype=torch.long)
    logits, hidden = target.execute([prompt_ids, input_pos])
    hidden = hidden.float()
    pos = prompt_len
    last_token = int(logits[0, -1].argmax())

    generated = [last_token]
    rounds = 0
    accepted_total = 0
    t0 = time.time()

    while len(generated) < args.max_new_tokens:
        rounds += 1

        # 1. Draft block: [last_token, mask, mask, ...]
        draft_input = torch.cat(
            [torch.tensor([[last_token]], dtype=torch.long),
             torch.full((1, block_size - 1), mask_id, dtype=torch.long)], dim=1)
        draft_pos = torch.arange(hidden.shape[1] + block_size, dtype=torch.long).unsqueeze(0)
        _t0 = time.time()
        (draft_logits,) = draft.execute([draft_input, hidden, draft_pos])
        _draft_time = time.time() - _t0
        draft_ids = draft_logits[0].argmax(-1).tolist()  # block_size - 1 tokens

        # 2. Verify: target on [last_token] + draft_ids
        verify_input = torch.cat(
            [torch.tensor([[last_token]], dtype=torch.long),
             torch.tensor([draft_ids], dtype=torch.long)], dim=1)
        verify_pos = torch.arange(pos, pos + verify_input.shape[1], dtype=torch.long)
        _t1 = time.time()
        target_logits, new_hidden = target.execute([verify_input, verify_pos])
        _verify_time = time.time() - _t1
        if rounds <= 10:
            print(f"  timing: draft={_draft_time*1000:.1f}ms  verify={_verify_time*1000:.1f}ms  ctx_len={hidden.shape[1]}")
        target_ids = target_logits[0].argmax(-1).tolist()  # block_size tokens

        # 3. Accept: matching prefix + the target's bonus token at the mismatch
        accepted = first_mismatch(draft_ids, target_ids)
        if rounds <= 5:
            print(f"round {rounds}: pos={pos} hidden_ctx={hidden.shape[1]} "
                  f"draft_ids[:5]={draft_ids[:5]} target_ids[:5]={target_ids[:5]} accepted={accepted}")
        new_tokens = draft_ids[:accepted] + [target_ids[accepted]]
        accepted_total += accepted

        # Trim at EOS if it appears in the accepted run
        if eos_id in new_tokens:
            new_tokens = new_tokens[:new_tokens.index(eos_id) + 1]

        generated.extend(new_tokens)

        # 4. Position-based rollback
        pos += len(new_tokens)
        last_token = new_tokens[-1]
        # Accumulate: append this round's newly-generated tokens' hidden onto
        # the running context, don't replace it. The target context feature
        # must span the whole sequence generated so far (paper Figure 2), not
        # just the latest round.
        hidden = torch.cat([hidden, new_hidden[:, :len(new_tokens), :].float()], dim=1)

        if eos_id in new_tokens:
            break

    dt = time.time() - t0
    text = tokenizer.decode(generated)
    n = len(generated)
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated ({n} tokens): {text}")
    print(f"\n--- stats ---")
    print(f"rounds: {rounds}")
    print(f"avg accepted/round (tau proxy): {accepted_total / rounds:.2f}")
    print(f"time: {dt:.2f}s   tokens/s: {n / dt:.2f}")


if __name__ == "__main__":
    main()