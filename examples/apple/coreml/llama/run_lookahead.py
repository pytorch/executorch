# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import defaultdict

import sentencepiece as spm

import torch
from executorch.examples.apple.coreml.llama.llama_transformer import (
    InputManager,
    load_model,
)

from executorch.examples.models.llama.runner.generation import next_token
from executorch.examples.models.llama.tokenizer import tiktoken

from executorch.runtime import Runtime


class Tokenizer:
    def __init__(self, model_path: str):
        # Try sentence piece
        try:
            print("Trying to load sentencepiece")
            sp = spm.SentencePieceProcessor()
            sp.load(model_path)
            self.tokenizer = sp
        except:
            print("Trying to load tiktoken")
            self.tokenizer = tiktoken.Tokenizer(model_path)

    def encode(self, text, bos, eos):
        if isinstance(self.tokenizer, spm.SentencePieceProcessor):
            bos_string = "<s>" if bos else ""
            eos_string = "</s>" if eos else ""
            return self.tokenizer.encode(f"{bos_string}{text}{eos_string}")
        return self.tokenizer.encode(text, bos=bos, eos=eos)

    def decode(self, tokens):
        if isinstance(self.tokenizer, spm.SentencePieceProcessor):
            return self.tokenizer.decode(tokens)
        return self.tokenizer.decode(tokens)

    def decode_token(self, token):
        if isinstance(self.tokenizer, spm.SentencePieceProcessor):
            return f"{self.tokenizer.decode([token])} "
        return self.tokenizer.decode_token(token)

    def stop_tokens(self):
        if isinstance(self.tokenizer, spm.SentencePieceProcessor):
            return [self.tokenizer.eos_id()]
        return self.tokenizer.stop_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="model.pte",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        help="tokenizer.model path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time,",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--use_eager",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument("--dtype", type=str, choices=["fp16", "fp32"], default=None)
    parser.add_argument(
        "--seq_length",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=None,
    )
    # Lookahead decoding parameters
    parser.add_argument(
        "--ngram_size",
        type=int,
        default=3,
        help="Size of ngrams for lookahead decoding",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=4,
        help="Window size for lookahead decoding",
    )
    parser.add_argument(
        "--n_verifications",
        type=int,
        default=4,
        help="Number of verifications for lookahead decoding",
    )
    parser.add_argument(
        "--ngrams_seed",
        type=str,
        default=None,
        help="Seed for ngrams cache in lookahead decoding",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate",
    )

    args = parser.parse_args()

    tokenizer = Tokenizer(args.tokenizer)

    runtime = Runtime.get()
    if args.use_eager:
        assert args.params is not None
        assert args.checkpoint is not None
        assert args.dtype is not None
        assert args.max_seq_length is not None
        assert args.seq_length is not None

        max_seq_length = args.max_seq_length
        seq_length = args.seq_length
        model = load_model(
            args.checkpoint,
            args.params,
            max_seq_length=max_seq_length,
            use_cache_list=True,
        )
        n_layers = model.params.n_layers
        max_batch_size = model.params.max_batch_size
        n_kv_heads = model.params.n_kv_heads
        head_dim = model.params.head_dim
        cache_size = args.cache_size

        float_dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
        model.eval()
        model.to(float_dtype)
    else:
        program = runtime.load_program(args.model)
        method = program.load_method("forward")

        metadata = method.metadata
        print("Method metadata: ", metadata, "\n\n")

        assert (
            metadata.num_inputs() == 6
        ), "Do not export with --use_cache_list for use in pybindings"
        n_layers, max_batch_size, n_kv_heads, cache_size, head_dim = (
            metadata.input_tensor_meta(3).sizes()
        )
        float_dtype = {5: torch.float16, 6: torch.float32}[
            metadata.input_tensor_meta(3).dtype()
        ]

        seq_length, max_seq_length = metadata.input_tensor_meta(5).sizes()

    input_manager = InputManager(
        n_layers=n_layers,
        max_batch_size=max_batch_size,
        n_kv_heads=n_kv_heads,
        max_seq_length=max_seq_length,
        head_dim=head_dim,
        use_cache_list=True,
        seq_length=seq_length,
        dtype=float_dtype,
        minus_infinity=-30000.0,
        cache_size=cache_size,
        lookahead_enabled=True,
    )

    print(f"Prompt: {args.prompt}")
    tokens = tokenizer.encode(args.prompt, bos=True, eos=False)
    logits = None

    while len(tokens) > 0 and (input_manager.input_pos + seq_length < max_seq_length):
        inputs, remaining_tokens = input_manager.get_inputs_and_remaining_tokens(tokens)
        processed_tokens = len(tokens) - len(remaining_tokens)

        if args.use_eager:
            model_inputs = (
                inputs[0],  # tokens
                inputs[1],  # input_pos
                inputs[3],  # k_caches
                inputs[4],  # v_caches
                inputs[5],  # attn_mask
                inputs[2],  # input_length
            )
            logits, k, v = model(*model_inputs)
        else:
            logits, k, v = method.execute(inputs)

        input_manager.update(
            input_length=processed_tokens, new_k_caches=k, new_v_caches=v
        )
        tokens = remaining_tokens

    ngram_caches = None
    if args.ngrams_seed is not None:
        ngram_caches = defaultdict(
            lambda: InputManager.NGramCache(args.n_verifications)
        )
        seed_tokens = tokenizer.encode(args.ngrams_seed, bos=False, eos=False)
        for i in range(len(seed_tokens) - args.ngram_size + 1):
            key = seed_tokens[i]
            suffix = seed_tokens[i + 1 : i + args.ngram_size]
            ngram_caches[key].add(suffix)

    if input_manager.input_pos < max_seq_length and logits is not None:
        last_token_logits = logits[0, processed_tokens - 1, :]
        init_token = next_token(last_token_logits.unsqueeze(0), 0, 0)

        print("\nGenerating with lookahead decoding...")
        if args.use_eager:
            new_tokens = input_manager.lookahead_decode(
                model=model,
                init_token=init_token,
                n=args.max_tokens,
                ngram_size=args.ngram_size,
                window_size=args.window_size,
                n_verifications=args.n_verifications,
                stop_tokens=tokenizer.stop_tokens(),
                ngram_caches=ngram_caches,
            )
        else:
            new_tokens = input_manager.lookahead_decode(
                model=lambda *inputs: method.execute(inputs),
                init_token=init_token,
                n=args.max_tokens,
                ngram_size=args.ngram_size,
                window_size=args.window_size,
                n_verifications=args.n_verifications,
                stop_tokens=tokenizer.stop_tokens(),
                ngram_caches=ngram_caches,
            )

        print("\nGenerated text:")
        print(tokenizer.decode(new_tokens))
    else:
        print("Failed to generate text")


if __name__ == "__main__":
    main()
