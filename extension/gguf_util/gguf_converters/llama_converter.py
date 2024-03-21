# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO(mnachin): Move this file to torchao

import copy
from typing import Any, Mapping

import torch
from executorch.extension.gguf_util.load_gguf import GGUFWeights


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


def convert_to_state_dict(gguf_weights: GGUFWeights) -> Mapping[str, Any]:

    state_dict = {}
    for tensor in gguf_weights.tensors:
        gguf_tensor_name = tensor.name
        nn_tensor_name = _convert_gguf_tensor_name_to_llama_nn(gguf_tensor_name)
        # gguf is reversed
        reversed_shape = tensor.shape[::-1]
        new_tensor = tensor.data.reshape(reversed_shape)
        state_dict[nn_tensor_name] = torch.from_numpy(new_tensor)

    return state_dict
