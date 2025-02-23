import argparse
import sys

import sentencepiece as spm

import torch

from executorch.runtime import Runtime


sys.path.insert(0, ".")
from executorch.examples.models.llama.runner.generation import next_token
from executorch.examples.models.llama.tokenizer import tiktoken
from llama_transformer import InputManager, load_model


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
            use_cache_list=False,
        )
        n_layers = model.params.n_layers
        max_batch_size = model.params.max_batch_size
        n_kv_heads = model.params.n_kv_heads
        head_dim = model.params.head_dim
        cache_size = args.cache_size

        float_dtype = {"fp16": torch.float16, "fp32": torch.float32}[
            args.dtype
        ]  # dtype for model/inputs
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
        # k_cache input
        n_layers, max_batch_size, n_kv_heads, cache_size, head_dim = (
            metadata.input_tensor_meta(3).sizes()
        )
        float_dtype = {5: torch.float16, 6: torch.float32}[
            metadata.input_tensor_meta(3).dtype()
        ]

        # mask input
        seq_length, max_seq_length = metadata.input_tensor_meta(5).sizes()

    input_manager = InputManager(
        n_layers=n_layers,
        max_batch_size=max_batch_size,
        n_kv_heads=n_kv_heads,
        max_seq_length=max_seq_length,
        head_dim=head_dim,
        use_cache_list=False,
        seq_length=seq_length,
        dtype=float_dtype,
        minus_infinity=-30000.0,
        cache_size=cache_size,
    )

    print(args.prompt, end="")
    tokens = tokenizer.encode(args.prompt, bos=True, eos=False)
    while input_manager.input_pos + seq_length < max_seq_length:
        while len(tokens) > 0 and (
            input_manager.input_pos + seq_length < max_seq_length
        ):
            inputs, remaining_tokens = input_manager.get_inputs_and_remaining_tokens(
                tokens
            )
            processed_tokens = len(tokens) - len(remaining_tokens)
            if args.use_eager:
                logits, k, v = model(*inputs)
            else:
                logits, k, v = method.execute(inputs)

            input_manager.update(
                input_length=processed_tokens, new_k_caches=k, new_v_caches=v
            )
            tokens = remaining_tokens

        tokens = [next_token(logits, args.temperature, args.top_p)]

        if tokens[-1] in tokenizer.stop_tokens():
            break
        print(tokenizer.decode_token(tokens[-1]), end="", flush=True)


if __name__ == "__main__":
    main()
