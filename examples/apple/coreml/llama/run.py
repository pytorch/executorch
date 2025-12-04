# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import sentencepiece as spm

import torch
from executorch.examples.apple.coreml.llama.llama_transformer import (
    InputManager,
    load_model,
)

from executorch.examples.models.llama.runner.generation import next_token
from executorch.examples.models.llama.tokenizer import tiktoken

from executorch.runtime import Runtime


class ModelPieceManager:
    def __init__(self, runtime, model: str, n_layers: int, eager_model_pieces=None, use_eager=False):
        self.use_eager = use_eager
        self.n_layers = n_layers
        self.input_proj_method = runtime.load_program(f"{model}/input_block.pte").load_method("forward")
        self.output_proj_method = runtime.load_program(f"{model}/output_block.pte").load_method("forward")
        self.layer_methods = []

        max_batch_size = None
        n_kv_heads = None
        cache_size = None
        head_dim = None
        seq_length = None
        max_seq_length = None
        float_dtype = None

        for i in range(n_layers):
            method = runtime.load_program(f"{model}/transformer_block_{i}.pte").load_method("forward")
            self.layer_methods.append(method)
            if i == 0:
                metadata = method.metadata
                print("Method metadata (piece 0): ", metadata, "\n\n")
                assert metadata.num_inputs() == 6

                # k_cache input
                max_batch_size, n_kv_heads, cache_size, head_dim = metadata.input_tensor_meta(3).sizes()
                float_dtype = {5: torch.float16, 6: torch.float32}[metadata.input_tensor_meta(3).dtype()]

                # mask input
                seq_length, max_seq_length = metadata.input_tensor_meta(5).sizes()

        self.max_batch_size = max_batch_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.cache_size = cache_size
        self.seq_length = seq_length
        self.max_seq_length = max_seq_length
        self.float_dtype = float_dtype
        self.eager_model_pieces = eager_model_pieces

    def run(self, runtime, inputs):
        """
        Run the piecewise model.

        inputs: (tokens, input_pos, input_length, k_caches, v_caches, attn_mask)
        returns: (logits, new_k_caches, new_v_caches)
        """
        tokens, input_pos, input_length, k_caches, v_caches, attn_mask = inputs

        # Input block
        h, freqs_cos, freqs_sin = self.input_proj_method.execute((tokens, input_pos))

        if self.eager_model_pieces is not None:
            h_eager, freqs_cos_eager, freqs_sin_eager = self.eager_model_pieces[0].forward(tokens, input_pos)
            assert torch.allclose(h, h_eager, atol=1e-4)
            assert torch.allclose(freqs_cos, freqs_cos_eager, atol=1e-4)
            assert torch.allclose(freqs_sin, freqs_sin_eager, atol=1e-4)
            rmse = torch.sqrt(torch.mean((h - h_eager) ** 2))
            print("RMSE h (input_block)", rmse)
            rmse = torch.sqrt(torch.mean((freqs_cos - freqs_cos_eager) ** 2))
            print("RMSE freqs_cos (input_block)", rmse)
            rmse = torch.sqrt(torch.mean((freqs_sin - freqs_sin_eager) ** 2))
            print("RMSE freqs_sin (input_block)", rmse)

            if self.use_eager:
                h = h_eager
                freqs_cos = freqs_cos_eager
                freqs_sin = freqs_sin_eager

        new_ks = []
        new_vs = []

        # Transformer blocks
        for i in range(self.n_layers):
            method = self.layer_methods[i]
            h_new, new_k, new_v = method.execute((h, freqs_cos, freqs_sin, k_caches[i], v_caches[i], attn_mask))

            if self.eager_model_pieces is not None:
                print("CHECK LAYER", i)
                h_eager, new_k_eager, new_v_eager = self.eager_model_pieces[i + 1].forward(
                    h, freqs_cos, freqs_sin, k_caches[i], v_caches[i], attn_mask
                )

                rmse_h = torch.sqrt(torch.mean((h_new - h_eager) ** 2))
                print(
                    "RMSE h",
                    rmse_h,
                    torch.max(h_eager),
                    torch.min(h_eager),
                    torch.quantile(h_eager.to(torch.float32), torch.tensor([0.1, 0.25, 0.5, 0.74, 0.9])),
                )

                rmse_k = torch.sqrt(torch.mean((new_k - new_k_eager) ** 2))
                print(
                    "RMSE new_k",
                    rmse_k,
                    torch.quantile(new_k_eager.to(torch.float32), torch.tensor([0.1, 0.25, 0.5, 0.74, 0.9])),
                )

                rmse_v = torch.sqrt(torch.mean((new_v - new_v_eager) ** 2))
                print(
                    "RMSE new_v",
                    rmse_v,
                    torch.quantile(new_v_eager.to(torch.float32), torch.tensor([0.1, 0.25, 0.5, 0.74, 0.9])),
                )

                if self.use_eager:
                    h_new = h_eager
                    new_k = new_k_eager
                    new_v = new_v_eager

            h = h_new
            new_ks.append(new_k)
            new_vs.append(new_v)

        # Output block – this should return logits
        logits, = self.output_proj_method.execute((h, input_length))

        if self.eager_model_pieces is not None:
            logits_eager = self.eager_model_pieces[-1].forward(h, input_length)
            rmse_out = torch.sqrt(torch.mean((logits - logits_eager) ** 2))
            print(
                "RMSE logits OUT",
                rmse_out,
                torch.max(logits_eager),
                torch.min(logits_eager),
            )
            if self.use_eager:
                logits = logits_eager

        return logits, new_ks, new_vs


class Tokenizer:
    def __init__(self, model_path: str):
        # Try sentence piece
        try:
            print("Trying to load sentencepiece")
            sp = spm.SentencePieceProcessor()
            sp.load(model_path)
            self.tokenizer = sp
        except Exception:
            print("Trying to load tiktoken")
            self.tokenizer = tiktoken.Tokenizer(model_path)

    def encode(self, text, bos, eos):
        if isinstance(self.tokenizer, spm.SentencePieceProcessor):
            bos_string = "<s>" if bos else ""
            eos_string = "</s>" if eos else ""
            return self.tokenizer.encode(f"{bos_string}{text}{eos_string}")
        return self.tokenizer.encode(text, bos=bos, eos=eos)

    def decode_token(self, token):
        if isinstance(self.tokenizer, spm.SentencePieceProcessor):
            return f"{self.tokenizer.decode(token)} "
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
        help="Either a single model.pte OR a directory containing input_block.pte / transformer_block_*.pte / output_block.pte",
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

    args = parser.parse_args()

    tokenizer = Tokenizer(args.tokenizer)

    runtime = Runtime.get()

    # These will be set depending on monolithic vs pieces
    model = None
    method = None
    piece_manager = None

    n_layers = None
    max_batch_size = None
    n_kv_heads = None
    cache_size = args.cache_size
    head_dim = None
    seq_length = None
    max_seq_length = None
    float_dtype = None

    ### PIECES: detect directory of piecewise exports
    if os.path.isdir(args.model):
        # Infer number of layers from transformer_block_*.pte files
        n_layers = 0
        while os.path.exists(os.path.join(args.model, f"transformer_block_{n_layers}.pte")):
            n_layers += 1
        if n_layers == 0:
            raise RuntimeError(f"No transformer_block_*.pte found in directory: {args.model}")

        print(f"Detected piecewise model with {n_layers} layers")

        piece_manager = ModelPieceManager(
            runtime=runtime,
            model=args.model,
            n_layers=n_layers,
            eager_model_pieces=None,  # wire in eager pieces if/when you want debugging
            use_eager=args.use_eager,
        )

        # Pull shapes/dtype from piece manager
        max_batch_size = piece_manager.max_batch_size
        n_kv_heads = piece_manager.n_kv_heads
        cache_size = piece_manager.cache_size
        head_dim = piece_manager.head_dim
        seq_length = piece_manager.seq_length
        max_seq_length = piece_manager.max_seq_length
        float_dtype = piece_manager.float_dtype

    else:
        # Original monolithic path
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
                use_cache_list=False,
            )
            n_layers = model.params.n_layers
            max_batch_size = model.params.max_batch_size
            n_kv_heads = model.params.n_kv_heads
            head_dim = model.params.head_dim

            float_dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
            model.eval()
            model.to(float_dtype)
        else:
            program = runtime.load_program(args.model)
            method = program.load_method("forward")

            metadata = method.metadata
            print("Method metadata: ", metadata, "\n\n")

            assert metadata.num_inputs() == 6, "Do not export with --use_cache_list for use in pybindings"

            # k_cache input
            n_layers, max_batch_size, n_kv_heads, cache_size, head_dim = metadata.input_tensor_meta(3).sizes()
            float_dtype = {5: torch.float16, 6: torch.float32}[metadata.input_tensor_meta(3).dtype()]

            # mask input
            seq_length, max_seq_length = metadata.input_tensor_meta(5).sizes()

    # Common InputManager setup
    input_manager = InputManager(
        n_layers=n_layers,
        max_batch_size=max_batch_size,
        n_kv_heads=n_kv_heads,
        max_seq_length=max_seq_length,
        head_dim=head_dim,
        use_cache_list=piece_manager is not None,
        seq_length=seq_length,
        dtype=float_dtype,
        minus_infinity=-30000.0,
        cache_size=cache_size,
    )

    print(args.prompt, end="")
    tokens = tokenizer.encode(args.prompt, bos=True, eos=False)
    while input_manager.input_pos + seq_length < max_seq_length:
        while len(tokens) > 0 and (input_manager.input_pos + seq_length < max_seq_length):
            inputs, remaining_tokens = input_manager.get_inputs_and_remaining_tokens(tokens)
            processed_tokens = len(tokens) - len(remaining_tokens)

            ### PIECES: choose correct execution path
            if piece_manager is not None:
                logits, k, v = piece_manager.run(runtime, inputs)
            elif args.use_eager:
                logits, k, v = model(*inputs)
            else:
                logits, k, v = method.execute(inputs)

            input_manager.update(input_length=processed_tokens, new_k_caches=k, new_v_caches=v)
            tokens = remaining_tokens

        tokens = [next_token(logits, args.temperature, args.top_p)]

        if tokens[-1] in tokenizer.stop_tokens():
            break
        print(tokenizer.decode_token(tokens[-1]), end="", flush=True)


if __name__ == "__main__":
    main()
