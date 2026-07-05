"""Plain autoregressive baseline using the SAME target .pte, tokenizer, and
chat-template settings as run_dflash.py -- for an apples-to-apples comparison,
not the old Phase 0 number (measured under different conditions/quant pass).
"""
import argparse
import time

import torch
from transformers import AutoTokenizer
from executorch.runtime import Runtime, Verification


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-pte", default="qwen3_4b_dflash_target.pte")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--chat-template", action="store_true", default=True)
    p.add_argument("--no-chat-template", dest="chat_template", action="store_false")
    p.add_argument("--enable-thinking", action="store_true", default=False)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    eos_id = tokenizer.eos_token_id

    if args.chat_template:
        messages = [{"role": "user", "content": args.prompt}]
        chat_out = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            enable_thinking=args.enable_thinking, return_tensors="pt",
        )
        prompt_ids = chat_out.input_ids if hasattr(chat_out, "input_ids") else chat_out
    else:
        prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    rt = Runtime.get()
    target = rt.load_program(args.target_pte, verification=Verification.Minimal).load_method("forward")

    prompt_len = prompt_ids.shape[1]
    input_pos = torch.arange(prompt_len, dtype=torch.long)

    t0 = time.time()
    logits, _hidden = target.execute([prompt_ids, input_pos])
    pos = prompt_len
    token = int(logits[0, -1].argmax())
    generated = [token]

    while len(generated) < args.max_new_tokens:
        tok_input = torch.tensor([[token]], dtype=torch.long)
        pos_input = torch.tensor([pos], dtype=torch.long)
        logits, _hidden = target.execute([tok_input, pos_input])
        token = int(logits[0, -1].argmax())
        generated.append(token)
        pos += 1
        if token == eos_id:
            break

    dt = time.time() - t0
    text = tokenizer.decode(generated)
    n = len(generated)
    print(f"Prompt: {args.prompt}")
    print(f"Generated ({n} tokens): {text}")
    print(f"\n--- baseline stats ---")
    print(f"time: {dt:.2f}s   tokens/s: {n / dt:.2f}")


if __name__ == "__main__":
    main()
