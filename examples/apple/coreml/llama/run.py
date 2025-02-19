import argparse
from multiprocessing import process
import sys

import torch
from pathlib import Path


sys.path.insert(0, "..")
import json

import sentencepiece as spm
import tiktoken
from tiktoken.load import load_tiktoken_bpe

from executorch.runtime import Runtime

from llama.llama_transformer import InputManager, ModelArgs

class Tokenizer:
    def __init__(self, model_path: str):
        # Try sentence piece
        try:
            print("Trying to load sentencepiece")
            sp = spm.SentencePieceProcessor()
            sp.load(model_path)
            self.tokenizer = sp
        except:
            print("Trying to tiktoken")
            self.num_reserved_special_tokens = 256
            self.pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

            mergeable_ranks = load_tiktoken_bpe(model_path)
            num_base_tokens = len(mergeable_ranks)
            special_tokens = [
                "<|begin_of_text|>",
                "<|end_of_text|>",
                "<|reserved_special_token_0|>",
                "<|reserved_special_token_1|>",
                "<|reserved_special_token_2|>",
                "<|reserved_special_token_3|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|reserved_special_token_4|>",
                "<|eot_id|>",  # end of turn
            ] + [
                f"<|reserved_special_token_{i}|>"
                for i in range(5, self.num_reserved_special_tokens - 5)
            ]
            self.special_tokens = {
                token: num_base_tokens + i for i, token in enumerate(special_tokens)
            }
            self.tokenizer = tiktoken.Encoding(
                name=Path(model_path).name,
                pat_str=self.pat_str,
                mergeable_ranks=mergeable_ranks,
                special_tokens=self.special_tokens,
            )
    
    def encode(self, text):
        return self.tokenizer.encode(text)
    
    def encode_prompt(self, text):
        if isinstance(self.tokenizer, spm.SentencePieceProcessor):
            return self.tokenizer.encode(text)

        get_prompt = lambda x: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{x}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        return self.tokenizer.encode(get_prompt(text), allowed_special={"<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"})

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
    
    def stop_tokens(self):
        if isinstance(self.tokenizer, spm.SentencePieceProcessor):
            return [self.tokenizer.eos_id()]
        if isinstance(self.tokenizer, tiktoken.Encoding):
            return [
                self.tokenizer.encode("<|eot_id|>", allowed_special={"<|eot_id|>"})[0],
                self.tokenizer.encode("<|end_of_text|>", allowed_special={"<|end_of_text|>"})[0],
            ]

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
    parser.add_argument(
        "--n_steps",
        type=int,
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=None,
        help="Cache size.  Old items are evicted from cache",
    )

    args = parser.parse_args()
    params_path = args.params

    # Load model args
    with open(params_path, "r") as f:
        params = json.loads(f.read())

    model_args = ModelArgs(
        max_seq_len=args.max_seq_length,
        generate_full_logits=False,
        use_cache_list=False, # cache_list does not work in pybindings
        **params,
    )

    input_manager = InputManager(
        model_args=model_args,
        seq_length=args.seq_length,
        dtype=torch.float16,
        minus_infinity=-30000,
        cache_size=args.cache_size,
    )

    tokenizer = Tokenizer(args.tokenizer)

    runtime = Runtime.get()
    program = runtime.load_program(args.model)
    method = program.load_method("forward")
    generated_tokens = []
    tokens = tokenizer.encode_prompt(args.prompt)
    generated_tokens.extend(tokens)
    while input_manager.input_pos < args.n_steps:
        while len(tokens) > 0:
            inputs, remaining_tokens = (
                input_manager.get_inputs_and_remaining_tokens(tokens)
            )
            processed_tokens = len(tokens) - len(remaining_tokens)
            logits, k, v = method.execute(inputs)
            input_manager.update(
                input_length=processed_tokens, new_k_caches=k, new_v_caches=v
            )
            tokens = remaining_tokens

        tokens = [logits.argmax(-1).item()]
        generated_tokens.extend(tokens)
        if tokens[-1] in tokenizer.stop_tokens():
            break
        print(tokenizer.decode([generated_tokens[-1]]), end=" ", flush=True)

    print("\n\nFull text:")
    print(tokenizer.decode(generated_tokens))


if __name__ == "__main__":
    main()
