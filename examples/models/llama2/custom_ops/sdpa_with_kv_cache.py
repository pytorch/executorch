# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.library import impl, impl_abstract

custom_ops_lib = torch.library.Library("llama", "DEF")
custom_ops_lib.define(
    "sdpa_with_kv_cache(Tensor query, Tensor key, Tensor value, Tensor key_cache, "
    "Tensor value_cache, int layer_id, SymInt start_pos, SymInt seq_len, Tensor? attn_mask=None, "
    "float drpout_p=0.0, bool is_causal=False, float? scale=None) -> Tensor"
)

custom_ops_lib.define(
    "sdpa_with_kv_cache.out(Tensor query, Tensor key, Tensor value, Tensor key_cache, "
    "Tensor value_cache, int layer_id, SymInt start_pos, SymInt seq_len, Tensor? attn_mask=None, "
    "float drpout_p=0.0, bool is_causal=False, float? scale=None, *, Tensor(a!) out) -> Tensor(a!)"
)


def _validate_params(
    query,
    key,
    value,
    key_cache,
    value_cache,
    layer_id,
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
        key_cache.dim() == 5
    ), f"Expected key_cache to be 5 dimensional but got {key_cache.dim()}"
    assert (
        value_cache.dim() == 5
    ), f"Expected value_cache to be 5 dimensional but got {value_cache.dim()}"

    assert (
        key_cache.dtype == torch.float32
    ), f"Expected key_cache to be float32 but got {key_cache.dtype}"
    assert (
        value_cache.dtype == torch.float32
    ), f"Expected value_cache to be float32 but got {value_cache.dtype}"

    assert (
        key_cache.size() == value_cache.size()
    ), f"Key cache and value cache must have same size but got {key_cache.size()} and {value_cache.size()}"

    assert start_pos < key_cache.size(
        2
    ), f"Start position {start_pos} must be less than sequence length {key_cache.size(2)}"
    assert (start_pos + seq_len) < key_cache.size(
        2
    ), f"Start position  + length = {start_pos + seq_len} must be less than sequence length {key_cache.size(2)}"

    assert seq_len == 1, "Only support seq_len = 1 for now."

    assert layer_id < key_cache.size(
        0
    ), f"Layer id {layer_id} must be less than number of layers {key_cache.size(0)}"

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
    layer_id,
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
        layer_id,
        start_pos,
        seq_len,
        attn_mask,
        drpout_p,
        is_causal,
        scale,
    )

    return torch.empty_like(query)


@impl(custom_ops_lib, "sdpa_with_kv_cache", "CompositeExplicitAutograd")
def sdpa_with_kv_cache(
    query,
    key,
    value,
    key_cache,
    value_cache,
    layer_id,
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
        layer_id,
        start_pos,
        seq_len,
        attn_mask,
        drpout_p,
        is_causal,
        scale,
    )

    if attn_mask is not None:
        attn_mask = attn_mask[start_pos].view((1, -1))
        attn_mask = attn_mask[:, : start_pos + seq_len]
    q = query.transpose(1, 2)
    key_cache[layer_id, :, start_pos] = key
    value_cache[layer_id, :, start_pos] = value

    sliced_k_cache = key_cache[layer_id]
    sliced_v_cache = value_cache[layer_id]
    sliced_k_cache = sliced_k_cache[:, : start_pos + seq_len, :, :]
    sliced_v_cache = sliced_v_cache[:, : start_pos + seq_len, :, :]
    sliced_k_cache = sliced_k_cache.transpose(1, 2)
    sliced_v_cache = sliced_v_cache.transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(
        q, sliced_k_cache, sliced_v_cache, attn_mask=attn_mask
    )
    out = out.transpose(1, 2)
    return out


@impl_abstract("llama::sdpa_with_kv_cache.out")
def sdpa_with_kv_cache_out(
    query,
    key,
    value,
    key_cache,
    value_cache,
    layer_id,
    start_pos,
    seq_len,
    attn_mask,
    drpout_p,
    is_causal,
    scale,
    out,
):
    out = sdpa_with_kv_cache_meta(
        query,
        key,
        value,
        key_cache,
        value_cache,
        layer_id,
        start_pos,
        seq_len,
        attn_mask,
        drpout_p,
        is_causal,
        scale,
    )
    return out
