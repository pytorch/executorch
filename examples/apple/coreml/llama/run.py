import argparse
import sys

import torch

sys.path.insert(0, "..")
import json

import sentencepiece as spm
from executorch.runtime import Runtime

from llama.llama_transformer import InputManager, ModelArgs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="model.pte",
    )
    parser.add_argument(
        "-p",
        "--params",
        help="config.json",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        help="tokenizer.model path",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=1,  # set to 1 for decode
        help="length sequence to evaluate",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="maximum length sequence to evaluate",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time,",
    )

    args = parser.parse_args()
    params_path = args.params

    # Load model args
    with open(params_path, "r") as f:
        params = json.loads(f.read())

    model_args = ModelArgs(
        max_seq_len=args.max_seq_length,
        generate_full_logits=False,
        **params,
    )

    input_manager = InputManager(
        model_args=model_args,
        seq_length=args.seq_length,
        dtype=torch.float16,
        minus_infinity=-30000,
    )

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    runtime = Runtime.get()
    program = runtime.load_program(args.model)
    method = program.load_method("forward")
    generated_tokens = []
    tokens = sp.encode(args.prompt)
    generated_tokens.extend(tokens)
    while (
        input_manager.input_pos + args.seq_length
        < args.max_seq_length - args.seq_length
    ):
        while len(tokens) > 0:
            inputs, processed_tokens, remaining_tokens = (
                input_manager.get_inputs_and_remaining_tokens(tokens)
            )
            logits, k, v = method.execute(inputs)
            input_manager.update(
                input_length=processed_tokens, new_k_cache=k, new_v_cache=v
            )
            tokens = remaining_tokens

        tokens = [logits.argmax(-1).item()]
        generated_tokens.extend(tokens)
        print(sp.decode(generated_tokens[-1]), end=" ", flush=True)

    print("\n\nFull text:")
    print(sp.decode(generated_tokens))


if __name__ == "__main__":
    main()
