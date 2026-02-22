# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Import custom op defined in op_sdpa_aot.cpp. Those ops are using PyTorch
# C++ APIs for registration so here we need to import the shared library.
# This is only needed for OSS.

# pyre-unsafe

import logging

from typing import Tuple

import torch

from torch._inductor.lowering import lowerings as L, register_lowering

from torch.library import impl

aten = torch.ops.aten

try:
    op = torch.ops.llama.sdpa_with_kv_cache.default
    assert op is not None
    op2 = torch.ops.llama.fast_hadamard_transform.default
    assert op2 is not None
except:
    # This is needed to ensure that custom ops are registered
    from executorch.extension.pybindings import portable_lib  # noqa # usort: skip

    # Ideally package is installed in only one location but usage of
    # PYATHONPATH can result in multiple locations.
    # ATM this is mainly used in CI for qnn runner. Will need to revisit this
    from pathlib import Path

    package_path = Path(__file__).parent.resolve()
    logging.info(f"Looking for libcustom_ops_aot_lib.so in {package_path}")

    libs = list(package_path.glob("**/*custom_ops_aot_lib.*"))

    assert len(libs) == 1, f"Expected 1 library but got {len(libs)}"
    logging.info(f"Loading custom ops library: {libs[0]}")
    torch.ops.load_library(libs[0])
    op = torch.ops.llama.sdpa_with_kv_cache.default
    assert op is not None
    op2 = torch.ops.llama.fast_hadamard_transform.default
    assert op2 is not None

custom_ops_lib = torch.library.Library("llama", "IMPL")


def _validate_params(
    query,
    key,
    value,
    key_cache,
    value_cache,
    start_pos,
    seq_len,
    attn_mask,
    drpout_p,
    is_causal,
    scale,
):
    assert (
        query.dim() == 4
    ), f"Expected query to be 4 dimensional but got {query.dim()} dimensions."
    assert (
        key.dim() == 4
    ), f"Expected key to be 4 dimensional but got {key.dim()} dimensions."
    assert (
        value.dim() == 4
    ), f"Expected value to be 4 dimensional but got {value.dim()} dimensions."

    assert (
        query.dtype == torch.float32
    ), f"Expected query to be float32 but got {query.dtype}"
    assert key.dtype == torch.float32, f"Expected key to be float32 but got {key.dtype}"
    assert (
        value.dtype == torch.float32
    ), f"Expected value to be float32 but got {value.dtype}"

    assert (
        key_cache.dim() == 4
    ), f"Expected key_cache to be 4 dimensional but got {key_cache.dim()}"
    assert (
        value_cache.dim() == 4
    ), f"Expected value_cache to be 4 dimensional but got {value_cache.dim()}"

    assert (
        key_cache.dtype == torch.float32
    ), f"Expected key_cache to be float32 but got {key_cache.dtype}"
    assert (
        value_cache.dtype == torch.float32
    ), f"Expected value_cache to be float32 but got {value_cache.dtype}"

    assert (
        key_cache.size() == value_cache.size()
    ), f"Key cache and value cache must have same size but got {key_cache.size()} and {value_cache.size()}"

    # These asserts are real but they require me to add constrain_as_size/value calls to the model and I dont want to do that right now
    # assert start_pos < key_cache.size(
    #     1
    # ), f"Start position {start_pos} must be less than sequence length {key_cache.size(2)}"
    # assert (start_pos + seq_len) < key_cache.size(
    #     1
    # ), f"Start position  + length = {start_pos + seq_len} must be less than sequence length {key_cache.size(2)}"

    if attn_mask is not None:
        assert (
            attn_mask.dim() == 2
        ), f"Expected attn_mask to be 2 dimensional but got {attn_mask.dim()} dimensions."
        assert (attn_mask.dtype == torch.float32) or (
            attn_mask.dtype == torch.float16
        ), f"Expected attn_mask to be float but got {attn_mask.dtype}"


@impl(custom_ops_lib, "sdpa_with_kv_cache", "Meta")
def sdpa_with_kv_cache_meta(
    query,
    key,
    value,
    key_cache,
    value_cache,
    start_pos,
    seq_len,
    attn_mask=None,
    drpout_p=0.0,
    is_causal=False,
    scale=None,
):
    _validate_params(
        query,
        key,
        value,
        key_cache,
        value_cache,
        start_pos,
        seq_len,
        attn_mask,
        drpout_p,
        is_causal,
        scale,
    )

    return torch.empty_like(query)


@impl(custom_ops_lib, "fast_hadamard_transform", "Meta")
def fast_hadamard_transform_meta(mat):
    # assert(mat.strides[-1] == 1, "input matrix must be contiguous in the last dimension!")
    # assert(mat.shape[-1] == 128 or mat.shape[-1] == 14336, "unexpected input size for llama3 demo!")
    # assert(mat.is_contiguous(), "input matrix must be contiguous currently!")
    return torch.empty_like(mat)


@impl(custom_ops_lib, "custom_sdpa", "Meta")
def custom_sdpa(
    query,
    key_cache,
    value_cache,
    start_pos,
    attn_mask=None,
    drpout_p=0.0,
    is_causal=False,
    scale=None,
):
    seq_len = query.size(1)
    _validate_params(
        query,
        key_cache,
        value_cache,
        key_cache,
        value_cache,
        start_pos,
        seq_len,
        attn_mask,
        drpout_p,
        is_causal,
        scale,
    )

    return torch.empty_like(query)


def _validate_update_cache_params(
    value,
    cache,
    start_pos,
    indices=None,
):
    seq_len = value.size(1)
    assert (
        value.dim() == 4
    ), f"Expected value to be 4 dimensional but got {value.dim()} dimensions."

    assert (
        value.dtype == cache.dtype
    ), f"Expected value and cache to be of the same type but got value type {value.dtype} and cache type {cache.dtype}"

    for i in [0, 2, 3]:
        assert value.size(i) == cache.size(
            i
        ), f"Expected value and cache to have same size in dimension {i} but got {value.size(i)} and {cache.size(i)}"

    torch._check_is_size(start_pos)
    if indices is None:
        torch._check(start_pos < cache.size(1))
        assert start_pos < cache.size(
            1
        ), f"Start position {start_pos} must be less than sequence length {cache.size(1)}"

        torch._check((start_pos + seq_len) <= cache.size(1))
        assert (start_pos + seq_len) <= cache.size(
            1
        ), f"Start position  + length = {start_pos + seq_len} must be less than sequence length {cache.size(1)}"

    if indices is not None:
        assert (
            indices.dim() == 2
        ), f"Expected indices to be 2 dimensional but got {indices.dim()} dimensions."
        assert (
            indices.dtype == torch.int64
        ), f"Expected indices to be int64 but got {indices.dtype}"
        assert indices.size(0) == value.size(
            0
        ), f"Expected indices batch dimension to match value batch dimension but got {indices.size(0)} and {value.size(0)}"
        assert indices.size(1) == value.size(
            1
        ), f"Expected indices sequence length dimension to match value sequence length dimension but got {indices.size(1)} and {value.size(1)}"


@impl(custom_ops_lib, "update_cache", "Meta")
def update_cache_meta(
    value,
    cache,
    start_pos,
):
    _validate_update_cache_params(
        value,
        cache,
        start_pos,
    )

    # Update cache doesnt really return anything but I dont know a better
    # workaround. Should we just return cache instead? But I am afraid that
    # will result in extra memory allocation
    return torch.empty((1,), dtype=value.dtype, device="meta")


@impl(custom_ops_lib, "update_cache_with_indices", "Meta")
def update_cache_with_indices_meta(
    value,
    cache,
    start_pos,
    indices,
):
    _validate_update_cache_params(
        value,
        cache,
        start_pos,
        indices,
    )

    # Update cache doesnt really return anything but I dont know a better
    # workaround. Should we just return cache instead? But I am afraid that
    # will result in extra memory allocation
    return torch.empty((1,), dtype=value.dtype, device="meta")


def _validate_quantized_sdpa_params(
    query,
    key,
    value,
    start_pos,
    seq_len,
    attn_mask,
    drpout_p,
    is_causal,
    scale,
    q_scale,
    q_zero_point,
    k_scale,
    k_zero_point,
    v_scale,
    v_zero_point,
    is_seq_at_dim_2,
):
    assert (
        query.dim() == 4
    ), f"Expected query to be 4 dimensional but got {query.dim()} dimensions."
    assert (
        key.dim() == 4
    ), f"Expected key to be 4 dimensional but got {key.dim()} dimensions."
    assert (
        value.dim() == 4
    ), f"Expected value to be 4 dimensional but got {value.dim()} dimensions."

    assert (q_scale is not None) and (
        q_zero_point is not None
    ), "q_scale and q_zero_point must be provided"
    assert (k_scale is not None) and (
        k_zero_point is not None
    ), "k_scale and k_zero_point must be provided"
    assert (v_scale is not None) and (
        v_zero_point is not None
    ), "v_scale and v_zero_point must be provided"

    assert query.dtype == torch.int8, f"Expected query to be int8 but got {query.dtype}"
    assert key.dtype == torch.int8, f"Expected key to be int8 but got {key.dtype}"
    assert value.dtype == torch.int8, f"Expected value to be int8 but got {value.dtype}"

    assert (
        q_scale.dtype == torch.float32
    ), f"Expected q_scale to be float32 but got {q_scale.dtype}"
    assert (
        q_zero_point.dtype == torch.int8
    ), f"Expected q_zero_point to be int8 but got {q_zero_point.dtype}"
    assert (
        k_scale.dtype == torch.float32
    ), f"Expected k_scale to be float32 but got {k_scale.dtype}"
    assert (
        k_zero_point.dtype == torch.int8
    ), f"Expected k_zero_point to be int8 but got {k_zero_point.dtype}"
    assert (
        v_scale.dtype == torch.float32
    ), f"Expected v_scale to be float32 but got {v_scale.dtype}"
    assert (
        v_zero_point.dtype == torch.int8
    ), f"Expected v_zero_point to be int8 but got {v_zero_point.dtype}"

    assert (
        query.size()[:-1] == q_scale.size()[:-1]
    ), f"Expected query and q_scale to have same size except last dimensions but got {query.size()} and {q_scale.size()}"
    assert (
        query.size()[:-1] == q_zero_point.size()[:-1]
    ), f"Expected query and q_zero_point to have same size except last dimensions but got {query.size()} and {q_zero_point.size()}"

    assert (
        key.size()[:-1] == k_scale.size()[:-1]
    ), f"Expected key and k_scale to have same size except last dimensions but got {key.size()} and {k_scale.size()}"
    assert (
        key.size()[:-1] == k_zero_point.size()[:-1]
    ), f"Expected key and k_zero_point to have same size except last dimensions but got {key.size()} and {k_zero_point.size()}"

    assert (
        value.size()[:-1] == v_scale.size()[:-1]
    ), f"Expected value and v_scale to have same size except last dimensions but got {value.size()} and {v_scale.size()}"
    assert (
        value.size()[:-1] == v_zero_point.size()[:-1]
    ), f"Expected value and v_zero_point to have same size except last dimensions but got {value.size()} and {v_zero_point.size()}"


@impl(custom_ops_lib, "custom_quantized_sdpa", "Meta")
def custom_quantized_sdpa_meta(
    query,
    key,
    value,
    start_pos,
    attn_mask=None,
    drpout_p=0.0,
    is_causal=False,
    scale=None,
    q_zero_point=None,
    q_scale=None,
    k_zero_point=None,
    k_scale=None,
    v_zero_point=None,
    v_scale=None,
    is_seq_at_dim_2=False,
):
    seq_len = query.size(1)
    _validate_quantized_sdpa_params(
        query,
        key,
        value,
        start_pos,
        seq_len,
        attn_mask,
        drpout_p,
        is_causal,
        scale,
        q_scale,
        q_zero_point,
        k_scale,
        k_zero_point,
        v_scale,
        v_zero_point,
        is_seq_at_dim_2,
    )

    return torch.empty(query.size(), dtype=torch.float32, device="meta")


# 1) Define the custom op in the "executorch" namespace with name "alias"
@torch.library.custom_op("executorch::alias", mutates_args=())
def custom_alias(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # no copies, just pass-through
    return x, y


# 2) FakeTensor kernel: describes output metadata for compile-time
@custom_alias.register_fake
def _(x, y):
    # For this op, outputs have exactly the same shape/dtype/device as inputs.
    # We just need *dummy* tensors with that metadata.
    out_x = torch.empty_like(x)
    out_y = torch.empty_like(y)
    return out_x, out_y


@register_lowering(torch.ops.executorch.alias.default)
def lowering_custom_alias(x, y):
    # x, y here are IR values (Inductor's internal representation).
    # Alias is logically a no-op – just pass them through.
    return x, y


# Expecting cache shape: (B, H, S_max, D), value shape (B, H, S, D) where S <= S_max
def _validate_cross_attn_cache_params(value: torch.Tensor, cache: torch.Tensor):
    torch._assert(value.dim() == 4, "value must be 4D")
    torch._assert(cache.dim() == 4, "cache must be 4D")
    # Cache shape: (B, H, S_max, D)
    # Value shape: (B, H, S, D)
    torch._assert(
        value.size(2) <= cache.size(2),
        f"value sequence length {value.size(2)} exceeds cache size {cache.size(2)}",
    )
    torch._assert(value.size(0) == cache.size(0), "batch size mismatch")
    torch._assert(value.size(1) == cache.size(1), "num heads mismatch")
    torch._assert(value.size(3) == cache.size(3), "head dim mismatch")
    torch._assert(value.dtype == cache.dtype, "dtype mismatch")


# Intentionally declaring no mutations to enable use inside torch.cond branches,
# which require pure functions. torch.cond requires branch functions to be mutation-free.
# We omit `cache` from `mutates_args` to satisfy this constraint, accepting the
# mutation for inference use.
@torch.library.custom_op("executorch::update_cross_attn_cache", mutates_args=[])
def _update_cross_attn_cache(value: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    """
    Update cross-attention KV cache with new values.

    Copies the value tensor into the beginning of the cache tensor along the
    sequence dimension. This is used for cross-attention caching where the
    encoder outputs are computed once and reused across decoding steps.

    Args:
        value: New values to store in cache. Shape: [B, H, S, D] where
            B = batch size, H = num heads, S = sequence length, D = head dim.
        cache: Pre-allocated cache tensor to update. Shape: [B, H, S_max, D]
            where S_max >= S.

    Returns:
        A clone of the updated cache tensor. Note that this is different from
        inductor lowering which returns the cache tensor itself. The reason is
        that if we return input buffer directly, we will fail torch check in
        higher order ops.

    Note:
        The cache is mutated in-place, but we return a clone to avoid aliasing
        issues with the exported program.
    """
    _validate_cross_attn_cache_params(value, cache)
    cache[:, :, : value.size(2), :].copy_(value)
    return cache.clone()


# Register the fake (meta) kernel
@_update_cross_attn_cache.register_fake
def _update_cross_attn_cache_fake(
    value: torch.Tensor, cache: torch.Tensor
) -> torch.Tensor:
    _validate_cross_attn_cache_params(value, cache)
    return torch.empty_like(cache)


# Register Inductor lowering
@register_lowering(torch.ops.executorch.update_cross_attn_cache)
def _update_cross_attn_cache_lowering(value, cache):
    # cache shape: [B, H, S_max, D]
    # value shape: [B, H, S, D]

    # We need to slice the cache along dim 2 (sequence length)
    # slice(self, dim, start, end, step=1)
    seq_len = value.get_size()[2]
    cache_slice = L[aten.slice.Tensor](cache, 2, 0, seq_len, 1)

    # Copy value into the slice
    L[aten.copy_.default](cache_slice, value)

    return cache
