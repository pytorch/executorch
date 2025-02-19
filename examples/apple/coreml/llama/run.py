import sys

import torch

sys.path.insert(0, "..")
import json

from llama.llama_transformer import InputManager, ModelArgs, Transformer

params_path = "/Users/scroy/models/stories110M/params.json"
max_seq_length = 512
seq_length = 64
# Load model args
with open(params_path, "r") as f:
    params = json.loads(f.read())

args = ModelArgs(
    max_seq_len=max_seq_length,
    generate_full_logits=False,
    **params,
)
input_manager = InputManager(
    model_args=args,
    seq_length=seq_length,
    dtype=torch.float16,
    minus_infinity=-30000,
)


filename = "/Users/scroy/Desktop/model.pte"

# Test PTE
from executorch.runtime import Runtime

from transformers import AutoTokenizer

# Load the tokenizer for LLaMA 3
tokenizer = AutoTokenizer.from_pretrained("neuralmagic/llama2.c-stories110M-pruned50")

text = "Once upon a time,"
runtime = Runtime.get()
program = runtime.load_program(filename)
method = program.load_method("forward")
print(text)
tokens = tokenizer.encode(text)
while input_manager.input_pos + len(tokens) < max_seq_length:
    inputs = input_manager.get_inputs(torch.tensor(tokens, dtype=torch.long))
    logits, k, v = method.execute(inputs)
    input_manager.update(input_length=len(tokens), new_k_cache=k, new_v_cache=v)

    new_token = logits.argmax(-1).item()
    tokens = [new_token]
    decoded_text = tokenizer.decode(tokens)
    print(decoded_text, end=" ")
