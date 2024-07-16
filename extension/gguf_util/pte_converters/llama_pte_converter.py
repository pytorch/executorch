# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Mapping

import torch
from executorch.extension.gguf_util.load_gguf import GGUFModelArgs
from executorch.examples.models.llama2.llama_transformer import (
    ModelArgs as LlamaModelArgs,
    Transformer as LlamaTransformer,
)

from torch import nn


def _load_weights_into_nn(
    pt_model: nn.Module,
    state_dict: Mapping[str, Any],
    gguf_model_args: GGUFModelArgs,
):

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


def _create_pt_model(
    gguf_model_args: GGUFModelArgs,
) -> nn.Module:
    """
    Creates reference nn.Module that corresponds to the architecutre
    """

    # NOTE: Currently it is using the Llama2 model in the executorch/examples/ directory
    # But in the future, should point to a generic/reference implementation instead.
    #
    # Currently, we are doing this so that it is exportable.

    # Step 3: Transform the PyTorch nn.Module to another PyTorch nn.Module that
    # is compatible with the quantized weights.
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


def convert_to_pte(
    pt_model: nn.Module,
    state_dict: Mapping[str, Any],
    gguf_model_args: GGUFModelArgs,
) -> bytes:
    """Convert a GGUF model into an ExecuTorch program.

    Args:
        model_args: The arguments for the GGUF model.
        weights: The weights of the GGUF model.
    """

    print("Load the weights into the PyTorch model")
    _load_weights_into_nn(pt_model, state_dict, gguf_model_args)

    # Step 3: Export to ExecuTorch
    print("Exporting to ExecuTorch.")
    pte_program = _create_pte_program(pt_model)
    return pte_program
