# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standard autoregressive decoding used as the baseline for the comparison.

Mirrors examples/models/qwen3/run_baseline.py's structure (same target .pte
reused for both baseline and DFlash -- see run_dflash.py's module docstring
for why), but Gemma4-31B's tokenizer/chat-template handling genuinely
differs from Qwen3's and is NOT a drop-in AutoTokenizer.apply_chat_template
call:

- Uses the raw `tokenizers.Tokenizer` API (`Tokenizer.from_file` +
  `.encode(...).ids` / `.decode(...)`), not transformers.AutoTokenizer --
  this checkpoint's tokenizer_config.json has no chat_template field at all.
- BOS is not part of the chat-template string; it must be prepended at the
  token-ID level or, per inference.py's docstring, "the model's logits
  collapse to a single high-frequency vocab token."
- Three stop tokens, not one: eos_token_ids = {1, 50, 106} (hardcoded in
  inference.py's main(), not derived from the tokenizer's own eos_token
  field).

apply_chat_template is imported directly from inference.py rather than
reimplemented, so this only has one place to go stale.
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
