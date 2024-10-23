# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import numpy as np

import torch


def prune_output_vocab(
    model: torch.nn.Module,
    token_map: Dict[int, int],
    output_layer_name: str = "output",
) -> torch.nn.Module:
    """Prune the model output linear layer while keeping the tokens in the token map.

    Note: Pruning is performed in-place.

    Args:
        model: The model to prune.
        token_map: A dictionary mapping from new token ids to the old token ids to preserve.
            e.g. {0: 221, 1: 1325, 2: 1542, 3: 1728, 4: 18243}
        output_layer_name: name of the output layer to prune

    Returns:
        The pruned model.
    """
    assert hasattr(
        model, output_layer_name
    ), f"Model does not have {output_layer_name} layer"
    output_layer = getattr(model, output_layer_name)
    assert isinstance(
        output_layer, torch.nn.Linear
    ), "Output layer is not a linear layer"
    original_shape = output_layer.weight.shape
    input_features = original_shape[1]
    num_pruned_tokens = len(token_map)
    has_bias = output_layer.bias is not None
    weight_dtype = output_layer.weight.dtype
    pruned_layer = torch.nn.Linear(input_features, num_pruned_tokens, bias=has_bias)
    pruned_layer.to(dtype=weight_dtype)
    pruned_layer_weights = np.zeros(pruned_layer.weight.shape, dtype=np.float32)
    pruned_layer_bias = None
    if has_bias:
        pruned_layer_bias = np.zeros(pruned_layer.bias.shape, dtype=np.float32)
    for i, token_id in token_map.items():
        # Copy the weights and biases from the original layer to the pruned layer
        pruned_wt = output_layer.weight[token_id].detach()
        if weight_dtype == torch.bfloat16:
            pruned_wt = pruned_wt.float()
        pruned_layer_weights[i] = pruned_wt.numpy()
        if has_bias:
            pruned_bias = output_layer.bias[token_id].detach()
            if weight_dtype == torch.bfloat16:
                pruned_bias = pruned_bias.float()
            pruned_layer_bias[i] = pruned_bias.numpy()
    with torch.no_grad():
        pruned_layer.weight.copy_(
            torch.tensor(pruned_layer_weights, dtype=weight_dtype)
        )
        if has_bias:
            pruned_layer.bias.copy_(torch.tensor(pruned_layer_bias, dtype=weight_dtype))

    # Replace the original layer with the pruned layer
    setattr(model, output_layer_name, pruned_layer)

    return model


def prune_input_vocab(
    model: torch.nn.Module,
    token_map: Dict[int, int],
    imput_layer_name: str = "tok_embeddings",
) -> torch.nn.Module:
    """Prune the model input embedding layer while keeping the tokens in the token map.

    Note: Pruning is performed in-place.

    Args:
        model: The model to prune.
        token_map: A dictionary mapping from new token ids to the old token ids to preserve.
            e.g. {0: 221, 1: 1325, 2: 1542, 3: 1728, 4: 18243}
        imput_layer_name: name of the input embedding layer to prune

    Returns:
        The pruned model.
    """
    assert hasattr(
        model, imput_layer_name
    ), f"Model does not have {imput_layer_name} layer"
    input_layer = getattr(model, imput_layer_name)
    assert isinstance(
        input_layer, torch.nn.Embedding
    ), "Input layer is not an Embedding layer"
    original_shape = input_layer.weight.shape
    num_pruned_tokens = len(token_map)
    weight_dtype = input_layer.weight.dtype
    pruned_layer = torch.nn.Embedding(num_pruned_tokens, original_shape[1])
    pruned_layer.to(dtype=weight_dtype)
    pruned_layer_weights = np.zeros(pruned_layer.weight.shape, dtype=np.float32)
    for i, token_id in token_map.items():
        # Copy the weights from the original layer to the pruned layer
        pruned_wt = input_layer.weight[token_id].detach()
        if weight_dtype == torch.bfloat16:
            pruned_wt = pruned_wt.float()
        pruned_layer_weights[i] = pruned_wt.numpy()
    with torch.no_grad():
        pruned_layer.weight.copy_(
            torch.tensor(pruned_layer_weights, dtype=weight_dtype)
        )

    # Replace the original layer with the pruned layer
    setattr(model, imput_layer_name, pruned_layer)

    return model
