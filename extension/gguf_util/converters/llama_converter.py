# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Mapping

import torch
import torch.nn as nn
from executorch.examples.models.llama2.llama_transformer import (
    ModelArgs as LlamaModelArgs,
    Transformer as LlamaTransformer,
)
from executorch.extension.gguf_util.load_gguf import GGUFModelArgs, GGUFWeights


def _create_pt_model(
    gguf_model_args: GGUFModelArgs,
) -> nn.Module:
    llama_model_args = LlamaModelArgs(
        dim=gguf_model_args.embedding_length,
        n_layers=gguf_model_args.block_count,
        n_heads=gguf_model_args.attention.head_count,
        n_kv_heads=gguf_model_args.attention.head_count_kv,
        vocab_size=gguf_model_args.vocab_size,
        norm_eps=gguf_model_args.attention.layer_norm_rms_epsilon,
        hidden_dim=gguf_model_args.feed_forward_length,
        rope_freq_base=gguf_model_args.rope.freq_base,
    )
    pt_model = LlamaTransformer(llama_model_args)
    pt_model.eval()
    return pt_model


_name_replacements = [
    ("blk", "layers"),
    ("token_embd", "tok_embeddings"),
    ("attn_q", "attention.wq"),
    ("attn_k", "attention.wk"),
    ("attn_v", "attention.wv"),
    ("attn_output", "attention.wo"),
    ("attn_norm", "attention_norm"),
    ("output_norm.weight", "norm.weight"),
    ("ffn_down", "feed_forward.w2"),
    ("ffn_gate", "feed_forward.w1"),
    ("ffn_up", "feed_forward.w3"),
]


def _convert_gguf_tensor_name_to_llama_nn(gguf_name: str) -> str:
    result = copy.deepcopy(gguf_name)
    for gguf_string, replacement in _name_replacements:
        result = result.replace(gguf_string, replacement)
    return result


def _convert_to_state_dict(gguf_weights: GGUFWeights) -> Mapping[str, Any]:

    state_dict = {}
    for tensor in gguf_weights.tensors:
        gguf_tensor_name = tensor.name
        nn_tensor_name = _convert_gguf_tensor_name_to_llama_nn(gguf_tensor_name)
        # gguf is reversed
        reversed_shape = tensor.shape[::-1]
        new_tensor = tensor.data.reshape(reversed_shape)
        state_dict[nn_tensor_name] = torch.from_numpy(new_tensor)

    return state_dict


def _load_weights_into_nn(
    pt_model: nn.Module, gguf_model_args: GGUFModelArgs, gguf_weights: GGUFWeights
):

    state_dict: Mapping[str, Any] = _convert_to_state_dict(gguf_weights)

    # We need to fake initialize the mask, to match with the llama_transformer.py
    for id in range(gguf_model_args.block_count):
        mask_name = f"layers.{id}.attention.mask"
        mask = torch.full(
            (1, 1, pt_model.params.max_seq_len, pt_model.params.max_seq_len),
            float("-inf"),
        )
        mask = torch.triu(mask, diagonal=1)
        state_dict[mask_name] = mask

    pt_model.load_state_dict(state_dict)
    return


def _create_pte_program(pt_model: nn.Module) -> bytes:
    # TODO (mnachin): Export
    return


def convert_to_pte(gguf_model_args: GGUFModelArgs, gguf_weights: GGUFWeights) -> bytes:
    """Convert a GGUF model into an ExecuTorch program.

    Args:
        gguf_model_args: The arguments for the GGUF model.
        gguf_weights: The weights of the GGUF model.
    """

    assert (
        gguf_model_args.arch == "llama"
    ), "Only LLaMa models are supported by this converter."

    # Step 1: Create the PyTorch model
    print("Create the PyTorch model")
    pt_model = _create_pt_model(
        gguf_model_args,
    )

    # Step 2: Load the weights into the PyTorch model
    print("Load the weights into the PyTorch model")
    _load_weights_into_nn(pt_model, gguf_model_args, gguf_weights)

    # Step 3: Export to ExecuTorch
    print("Exporting to ExecuTorch.")
    pte_program = _create_pte_program(pt_model)
    return pte_program
