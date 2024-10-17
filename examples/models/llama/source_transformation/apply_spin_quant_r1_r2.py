# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing

import torch


def rotate_embeddings(model, R1: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in [model.tok_embeddings]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device="cpu", dtype=torch.float32)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, R1) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.attention.wq, layer.attention.wk, layer.attention.wv]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device="cpu", dtype=torch.float32)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, R1) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.attention.wo
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cpu", dtype=torch.float32)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device="cpu", dtype=torch.float32)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, R1):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.feed_forward.w3, layer.feed_forward.w1]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device="cpu", dtype=torch.float32)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, R1):
    # Rotate the MLP output weights and bias.
    W = layer.feed_forward.w2
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cpu", dtype=torch.float32)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)

    if W.bias is not None:
        b = W.bias.data.to(device="cpu", dtype=torch.float32)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, R1: torch.Tensor) -> None:
    # Rotate the head.
    W = model.output
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cpu", dtype=torch.float32)
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_dim, R2=None):
    W = layer.attention.wv
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cpu", dtype=torch.float32).t()
    transposed_shape = W_.shape
    temp = W_.reshape(-1, transposed_shape[-1] // head_dim, head_dim)
    temp = temp.to(torch.float32) @ R2
    W_ = temp.reshape(transposed_shape).t()
    W.weight.data = W_.to(device="cpu", dtype=dtype)

    W = layer.attention.wo
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cpu", dtype=torch.float32)
    init_shape = W_.shape
    temp = W_.reshape(-1, init_shape[-1] // head_dim, head_dim)
    temp = temp.to(torch.float32) @ R2
    W_ = temp.reshape(init_shape)
    W.weight.data = W_.to(device="cpu", dtype=dtype)


def cleanup_memory() -> None:
    """Run GC and clear GPU memory."""
    import gc

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()


def get_model_with_r1_r2(optimized_rotation_path: str):
    return lambda model: apply_spin_quant_r1_r2(model, optimized_rotation_path)


def apply_spin_quant_r1_r2(model: torch.nn.Module, optimized_rotation_path: str):
    optimized_rotation = torch.load(optimized_rotation_path, weights_only=True)
    R1 = optimized_rotation["R1"].to(torch.float32)
    config = model.params
    num_heads = config.n_heads
    head_dim = config.dim // num_heads

    rotate_embeddings(model, R1)
    rotate_head(model, R1)
    cleanup_memory()

    for idx, layer in enumerate(model.layers):
        key = f"model.layers.{idx}.self_attn.R2"
        R2 = optimized_rotation[key].to(torch.float32)
        rotate_attention_inputs(layer, R1)
        rotate_attention_output(layer, R1)
        rotate_mlp_input(layer, R1)
        rotate_mlp_output(layer, R1)
        rotate_ov_proj(layer, head_dim, R2=R2)
    return model


def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.to(dtype=torch.float32)
        linear.weight.data = (W_ * layernorm.weight.to(dtype=torch.float32)).to(
            linear_dtype
        )

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float32)
                )
            linear.bias.data = linear.bias.data.to(dtype=torch.float32) + torch.matmul(
                W_, layernorm.bias.to(dtype=torch.float32)
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model: torch.nn.Module):
    # Embedding fusion
    for W in [model.tok_embeddings]:
        W_ = W.weight.data.to(dtype=torch.float32)
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in model.layers:
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(layer.ffn_norm, [layer.feed_forward.w3, layer.feed_forward.w1])
        fuse_ln_linear(
            layer.attention_norm,
            [
                layer.attention.wq,
                layer.attention.wk,
                layer.attention.wv,
            ],
        )

        W_norm = layer.ffn_norm.weight.data
        layer.ffn_norm.weight.data = torch.ones_like(W_norm, dtype=torch.float32)
        W_norm = layer.attention_norm.weight.data
        layer.attention_norm.weight.data = torch.ones_like(W_norm, dtype=torch.float32)

    fuse_ln_linear(
        model.norm,
        [model.output],
    )
    W_norm = model.norm.weight.data
    model.norm.weight.data = torch.ones_like(W_norm, dtype=torch.float32)

    return model
