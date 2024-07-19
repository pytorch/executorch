# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Script to run phi-3-mini model in eager mode.

import argparse
import torch

from transformers import Phi3Config, Phi3ForCausalLM, AutoTokenizer

end_of_text_token = 32000

def _generate_token(args, model, prompt_tokens):
    current_token = 0
    generated_tokens = []

    print(f"Generating tokens:", end='')

    while current_token != end_of_text_token and len(generated_tokens) < args.seq_len:
        outputs = model.forward(input_ids=prompt_tokens)
        current_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
        print(f" {current_token}", end='')
        generated_tokens.append(current_token)
        prompt_tokens.append(current_token)

    return generated_tokens


def _generate_token_with_kv_cache(args, model, prompt_tokens):
    print(f"Generating tokens:", end='')

    result = model.forward(input_ids=prompt_tokens, use_cache=True, return_dict=True)

    current_token = torch.argmax(result.logits[:, -1, :], dim=-1).item()
    current_key_value = result.past_key_values

    print(f" {current_token}", end='')

    generated_tokens = [current_token]

    while current_token != end_of_text_token and len(generated_tokens) < args.seq_len:
        result = model.forward(
            input_ids=torch.tensor([[current_token]], dtype=torch.long),
            use_cache=True,
            return_dict=True,
            past_key_values=current_key_value)
        current_token = torch.argmax(result.logits[:, -1, :], dim=-1).item()
        current_key_value = result.past_key_values
        print(f" {current_token}", end='')
        generated_tokens.append(current_token)

    return generated_tokens


def main(args):
    seed = 42
    torch.manual_seed(seed)
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    model = Phi3ForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokens = tokenizer.encode(args, return_tensors="pt")

    generated_tokens = _generate_token_with_kv_cache(args, model, tokens) if args.use_kv_cach else _generate_token(args, model, tokens)

    print("Generated response: \n {}".format(
        tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s"
                        "--seq_len",
                        type=int,
                        default=128,
                        help="Maximum number tokens to generate"
                        )
    parser.add_argument("-kv"
                        "--use_kv_cache",
                        default=False,
                        action="store_true",
                        help="Whether or not to use KV cache")
    parser.add_argument("-p",
                        "--prompt",
                        type=str,
                        default="Hello!",
                        help="Prompt as input for the model"
                        )
    args = parser.parse_args()
