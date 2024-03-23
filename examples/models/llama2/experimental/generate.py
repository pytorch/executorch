# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
import argparse

from typing import Optional, Tuple

import torch

from executorch.examples.models.llama2.experimental.load_gguf_q4_0 import load_gguf_q4_0
from sentencepiece import SentencePieceProcessor


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def encode_tokens(tokenizer, string, bos=True, device="cpu"):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def decode_one_token(
    model: torch.nn.Module, x: torch.Tensor, **sampling_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = model(x)
    return sample(logits, **sampling_kwargs)


def prefill(model: torch.nn.Module, x: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    return decode_one_token(model, x, **sampling_kwargs)[0]


def decode_n_tokens(
    model: torch.nn.Module,
    cur_token: torch.Tensor,
    num_new_tokens: int,
    callback=lambda _: _,
    **sampling_kwargs,
):
    print(f"cur_token: {cur_token}")
    new_tokens, new_probs = [], []
    for _ in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token.view(1, -1), **sampling_kwargs
            )
            new_tokens.append(next_token.clone())
            # print(next_token)
            callback(next_token)
            new_probs.append(next_prob.clone())
            cur_token = torch.cat((cur_token.squeeze(), next_token), dim=0)
            # print(cur_token)

    return new_tokens, new_probs


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    interactive: bool,
    callback=lambda x: x,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)
    T_new = T + max_new_tokens
    # if interactive:
    #     max_seq_length = 350
    # else:
    #     max_seq_length = min(T_new, model.params.max_seq_len)

    device, dtype = prompt.device, prompt.dtype

    # with torch.device(device):
    #     model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    # input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt.view(1, -1), **sampling_kwargs)
    seq[T] = next_token
    callback(next_token)

    cur_tokens = torch.cat((prompt, next_token), dim=0)
    # input_pos = torch.tensor([T], device=device, dtype=torch.int)

    generated_tokens, _ = decode_n_tokens(
        model,
        cur_tokens.view(1, -1),
        # input_pos,
        max_new_tokens - 1,
        callback=callback,
        **sampling_kwargs,
    )
    seq[T + 1 :] = torch.cat(generated_tokens)

    return seq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gguf_file",
        type=str,
        help="The GGUF file to load.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="The tokenizer.model path.",
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello, my name is", help="Input prompt."
    )

    args = parser.parse_args()

    tokenizer = SentencePieceProcessor(model_file=str(args.tokenizer_path))
    encoded = encode_tokens(tokenizer, args.prompt, bos=True, device="cpu")

    pt_model = load_gguf_q4_0(args.gguf_file)

    max_new_tokens = 100
    buffer = [tokenizer.decode(encoded.tolist())]
    period_id = tokenizer.encode(".")[0]
    done_generating = False

    def callback(x):
        nonlocal done_generating
        if done_generating:
            return
        buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
        if x.item() == tokenizer.eos_id():
            done_generating = True
        if len(buffer) == 4 or done_generating:
            print("".join(buffer), end="", flush=True)
            buffer.clear()

    generate(
        pt_model,
        encoded,
        max_new_tokens,
        interactive=False,
        callback=callback,
        temperature=1.0,
        top_k=10,
    )


if __name__ == "__main__":
    main()
