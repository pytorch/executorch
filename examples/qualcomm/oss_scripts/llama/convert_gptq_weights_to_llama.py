# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
from typing import List, Union

import torch
from tqdm import tqdm
from gptqmodel import GPTQModel
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.utils.backend import BACKEND
import argparse
from transformers import AutoTokenizer

NUM_SHARDS = {
    "1B": 1,
    "7B": 1,
    "8B": 1,
    "13B": 2,
    "34B": 4,
    "30B": 4,
    "65B": 8,
    "70B": 8,
}


def write_model(model_path, model_size, output_base_path):
    dtype = torch.bfloat16

    params = json.load(open(os.path.join(output_base_path, "params.json"), "r"))
    num_shards = NUM_SHARDS[model_size]
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    llama_version = 3 if params.get("vocab_size") == 128256 else 2

    if "n_kv_heads" in params:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_local_key_value_heads = num_key_value_heads // num_shards
        key_value_dim = dims_per_head * num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim

    # instead of load state_dict directly
    # load state_dict from gptqmodel
    # to deal with GPTQ v1 -> GPTQ v2 transformation
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPTQModel.from_quantized(
        model_path,
        low_cpu_mem_usage=True,
        backend=BACKEND.TORCH,
    )
    messages = [
        {"role": "system", "content": "You are a helpful and harmless assistant. You should think step-by-step."},
        {"role": "user", "content": "How can I design a data structure in C++ to store the top 5 largest integer numbers?"},
    ]
    # input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    # outputs = model.generate(input_ids=input_tensor.to(model.device), max_new_tokens=32)
    # result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    # print(result)

    loaded = model.model.state_dict()

    # permute for sliced rotary
    def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
        return (
            w.view(n_heads, 2, dim1 // n_heads // 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
        )

    state_dict = [{} for _ in range(num_shards)]

    def insert(name: str, tensor: Union[List, torch.Tensor]):
        for i in range(num_shards):
            state_dict[i][name] = (
                tensor[i].clone() if isinstance(tensor, list) else tensor
            )

    def insert_chunk(name: str, tensor: torch.Tensor, dim: int):
        tensors = tensor.chunk(num_shards, dim=dim)
        for i, tensor in enumerate(tensors):
            state_dict[i][name] = tensor.clone()

    def insert_quantized(name: str, original_name: str):
        insert(
            f"{name}.qweight",
            loaded[f"{original_name}.qweight"],
        )
        insert(
            f"{name}.scales",
            loaded[f"{original_name}.scales"],
        )
        insert(
            f"{name}.qzeros",
            loaded[f"{original_name}.qzeros"],
        )
        insert(
            f"{name}.g_idx",
            loaded[f"{original_name}.g_idx"],
        )

    concat_dim = 0 if llama_version == 3 else 1
    insert_chunk(
        "tok_embeddings.weight", loaded["model.embed_tokens.weight"], concat_dim
    )
    insert("norm.weight", loaded["model.norm.weight"])
    insert_chunk("output.weight", loaded["lm_head.weight"], 0)

    for layer_i in tqdm(range(n_layers), desc="Converting layers"):

        # deal with hf permute in static_llama.py as it's hard to permute quantized weights
        insert_quantized(
            f"layers.{layer_i}.attention.wq",
            f"model.layers.{layer_i}.self_attn.q_proj",
        )
        insert_quantized(
            f"layers.{layer_i}.attention.wk",
            f"model.layers.{layer_i}.self_attn.k_proj",
        )
        insert_quantized(
            f"layers.{layer_i}.attention.wv",
            f"model.layers.{layer_i}.self_attn.v_proj",
        )
        insert_quantized(
            f"layers.{layer_i}.attention.wo",
            f"model.layers.{layer_i}.self_attn.o_proj",
        )
        insert_quantized(
            f"layers.{layer_i}.feed_forward.w1",
            f"model.layers.{layer_i}.mlp.gate_proj",
        )
        insert_quantized(
            f"layers.{layer_i}.feed_forward.w2",
            f"model.layers.{layer_i}.mlp.down_proj",
        )
        insert_quantized(
            f"layers.{layer_i}.feed_forward.w3",
            f"model.layers.{layer_i}.mlp.up_proj",
        )
        insert(
            f"layers.{layer_i}.attention_norm.weight",
            loaded[f"model.layers.{layer_i}.input_layernorm.weight"],
        )
        insert(
            f"layers.{layer_i}.ffn_norm.weight",
            loaded[f"model.layers.{layer_i}.post_attention_layernorm.weight"],
        )
    if llama_version != 3:
        base = 10000.0
        inv_freq = (
            1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
        ).to(dtype)
        insert("rope.freqs", inv_freq)

    for i in tqdm(range(num_shards), desc="Saving checkpoint shards"):
        torch.save(
            state_dict[i], os.path.join(output_base_path, f"consolidated.{i:02d}.pth")
        )


def main(
    model_path: str,
    model_size: str,
    output_dir: str,
):
    """Convert llama weights from huggingface format to consolidated format.
    params:
    model_path: model name or path to the model directory.
    model_size: Llama model size, one of 7B, 13B, 34B, 30B, 65B, 70B.
    output_dir: directory to save Llama weights, should contains params.json.
    """
    assert model_size in NUM_SHARDS, f"Unknown model size {model_size}"
    params_path = os.path.join(output_dir, "params.json")
    assert os.path.isfile(params_path), f"{params_path} does not exist"

    write_model(model_path, model_size, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        required=True,
        help="Path to the model directory.",
    )
    parser.add_argument(
        "-s",
        "--model-size",
        type=str,
        required=True,
        choices=NUM_SHARDS.keys(),
        help="Llama model size, one of 7B, 13B, 34B, 30B, 65B, 70B.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save Llama weights, should contains params.json.",
    )
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        model_size=args.model_size,
        output_dir=args.output_dir,
    )
