"""Greedy autoregressive decoding used as the DFlash baseline. 

Reuses the same exported target model as DFlash for a fair comparison. Gemma4-specific prompt formatting is handled by inference.py's apply_chat_template(). 
"""

import argparse
import time

import torch
from executorch.examples.models.gemma4_31b.inference import apply_chat_template
from executorch.runtime import Runtime, Verification
from tokenizers import Tokenizer

EOS_TOKEN_IDS = {1, 50, 106}
BOS_TOKEN_ID = 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--target-pte", default="gemma4_31b_dflash_exports_mlx/model.pte"
    )
    p.add_argument(
        "--tokenizer-path",
        default="./gemma-4-31B-it-HQQ-INT4/tokenizer.json",
    )
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument(
        "--no-chat-template",
        dest="chat_template",
        action="store_false",
        default=True,
    )
    args = p.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    prompt_str = args.prompt if not args.chat_template else apply_chat_template(args.prompt)
    input_ids = tokenizer.encode(prompt_str).ids
    if not input_ids or input_ids[0] != BOS_TOKEN_ID:
        input_ids = [BOS_TOKEN_ID] + input_ids
    prompt_ids = torch.tensor([input_ids], dtype=torch.long)

    rt = Runtime.get()
    target = rt.load_program(
        args.target_pte, verification=Verification.Minimal
    ).load_method("forward")

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
        if token in EOS_TOKEN_IDS:
            break

    dt = time.time() - t0
    text = tokenizer.decode(generated)
    n = len(generated)
    print(f"Prompt: {args.prompt}")
    print(f"Generated ({n} tokens): {text}")
    print("\n--baseline stats--")
    print(f"time: {dt:.2f}s   tokens/s: {n / dt:.2f}")


if __name__ == "__main__":
    main()
