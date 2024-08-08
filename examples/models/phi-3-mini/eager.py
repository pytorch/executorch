# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Script to run phi-3-mini model in eager mode.

import argparse
import time

import torch

from transformers import AutoTokenizer, Phi3ForCausalLM

from .phi_3_mini import Phi3Mini

end_of_text_token = 32000


def _generate_token(args, model, prompt_tokens):
    current_token = 0
    generated_tokens = []

    print("Generating tokens:", end="", flush=True)

    while current_token != end_of_text_token and len(generated_tokens) < args.seq_len:
        outputs = model.forward(input_ids=prompt_tokens)
        current_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
        print(f" {current_token}", end="", flush=True)
        generated_tokens.append(current_token)
        prompt_tokens = torch.cat(
            [prompt_tokens, torch.tensor([[current_token]], dtype=torch.long)], dim=-1
        )

    print("", flush=True)

    return generated_tokens


def _generate_token_with_kv_cache(args, model, prompt_tokens):
    print("Generating tokens:", end="", flush=True)

    model = Phi3Mini(model, 1, args.seq_len + prompt_tokens.shape[-1])
    result = model.forward(input_ids=prompt_tokens)

    current_token = torch.argmax(result, dim=-1).item()
    print(f" {current_token}", end="", flush=True)
    generated_tokens = [current_token]

    while current_token != end_of_text_token and len(generated_tokens) < args.seq_len:
        result = model.forward(
            input_ids=torch.tensor([[current_token]], dtype=torch.long),
        )
        current_token = torch.argmax(result, dim=-1).item()
        print(f" {current_token}", end="", flush=True)
        generated_tokens.append(current_token)

    print("", flush=True)

    return generated_tokens


def main(args):
    seed = 42
    torch.manual_seed(seed)
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    model = Phi3ForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokens = tokenizer.encode(args.prompt, return_tensors="pt")

    start = time.time()
    generated_tokens = (
        _generate_token_with_kv_cache(args, model, tokens)
        if args.use_kv_cache
        else _generate_token(args, model, tokens)
    )
    end = time.time()

    print(
        "Generated response: \n {}".format(
            tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ),
        flush=True,
    )
    print(f"Time spent: {end - start}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--seq_len",
        type=int,
        default=128,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "-kv",
        "--use_kv_cache",
        default=False,
        action="store_true",
        help="Whether or not to use KV cache",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="Tell me a story",
        help="Prompt as input for the model",
    )
    main(parser.parse_args())
