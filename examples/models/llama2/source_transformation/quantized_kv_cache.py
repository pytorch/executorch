# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum

import torch
import torch.nn as nn
from executorch.examples.models.llama2.llama_transformer import KVCache
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401


"""
 Heavily "inspired" by AO's implementation of the same in torchao/_models/llama/model.py
"""


# Doesnt have to abide by affine quantizaiton laws
# However, if we do implement quantized sdpa, then this might be handy
class QuantizedCacheType(Enum):
    AffineSymmetric = 0
    AffineAsymmetric = 1
    AffineSymmetricGroupWise = 1
    AffineAsymmetricGroupWise = 2


class QuantizedKVCache(nn.Module):
    def __init__(
        self,
        max_batch_size,
        max_seq_length,
        n_heads,
        head_dim,
        cache_type: QuantizedCacheType = QuantizedCacheType.AffineSymmetric,
        tranposed=False,
        enable_dynamic_shape=False,
    ):
        super().__init__()
        if cache_type not in (
            QuantizedCacheType.AffineSymmetric,
            QuantizedCacheType.AffineAsymmetric,
        ):

            raise ValueError(
                f"Only affine symmetric and asymmetric cache types are supported: got {cache_type}"
            )

        # For now supporting int8 only
        self.quantized_cache_dtype = torch.int8
        self.cache_fp_type = torch.float32
        self.is_transposed = tranposed
        self.enable_dynamic_shape = enable_dynamic_shape
        if self.is_transposed:
            cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
            scale_shape = (max_batch_size, n_heads, max_seq_length, 1)
        else:
            cache_shape = (max_batch_size, max_seq_length, n_heads, head_dim)
            scale_shape = (max_batch_size, max_seq_length, n_heads, 1)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=torch.int8))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=torch.int8))
        self.register_buffer(
            "k_cache_scales", torch.ones(scale_shape, dtype=torch.double)
        )
        self.register_buffer(
            "v_cache_scales", torch.ones(scale_shape, dtype=torch.double)
        )
        if cache_type == QuantizedCacheType.AffineAsymmetric:
            self.register_buffer(
                "k_cache_zero_points", torch.ones(scale_shape, dtype=torch.int64)
            )
            self.register_buffer(
                "v_cache_zero_points", torch.ones(scale_shape, dtype=torch.int64)
            )

    def update(self, input_pos, k_val, v_val):
        # quantize current k_val and store it in the cache
        k_scales, k_zero_points = (
            torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric.default(
                k_val, torch.int8  # no other value is supported by this op anyway
            )
        )
        quantized_k_val = torch.ops.quantized_decomposed.quantize_per_token(
            k_val,
            k_scales,
            k_zero_points,
            torch.iinfo(torch.int8).min,
            torch.iinfo(torch.int8).max,
            torch.int8,
        )

        v_scales, v_zero_points = (
            torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric(
                v_val, torch.int8
            )
        )
        quantized_v_val = torch.ops.quantized_decomposed.quantize_per_token(
            v_val,
            v_scales,
            v_zero_points,
            torch.iinfo(torch.int8).min,
            torch.iinfo(torch.int8).max,
            torch.int8,
        )

        if self.is_transposed:
            # We cannot use update_cache op at the moment
            # if the cache is transposed
            # Also note that we shold not need separate paths
            # for dynamic shape vs !
            # Only reason it is done this way is to accommodate
            # for lowering pains of backends that work better
            # with index_put op.
            if self.enable_dynamic_shape:
                start_pos = input_pos[0].item()
                torch._check_is_size(start_pos)
                if self.is_transposed:
                    dim_to_slice = 2
                else:
                    dim_to_slice = 1
                torch._check(start_pos < self.k_cache.size(dim_to_slice))
                seq_length = k_val.size(dim_to_slice)
                narrowed_k = self.k_cache.narrow(dim_to_slice, start_pos, seq_length)
                narrowed_k_scales = self.k_cache_scales.narrow(
                    dim_to_slice, start_pos, seq_length
                )
                narrowed_k_zp = self.k_cache_zero_points.narrow(
                    dim_to_slice, start_pos, seq_length
                )
                narrowed_k.copy_(quantized_k_val)
                narrowed_k_scales.copy_(k_scales)
                narrowed_k_zp.copy_(k_zero_points)
                # pyre-ignore: Incompatible parameter type [6]
                narrowed_v = self.v_cache.narrow(dim_to_slice, start_pos, seq_length)
                narrowed_v_scales = self.v_cache_scales.narrow(
                    dim_to_slice, start_pos, seq_length
                )
                narrowed_v_zp = self.v_cache_zero_points.narrow(
                    dim_to_slice, start_pos, seq_length
                )
                narrowed_v.copy_(quantized_v_val)
                narrowed_v_scales.copy_(v_scales)
                narrowed_v_zp.copy_(v_zero_points)
            else:
                self.k_cache[:, :, input_pos] = quantized_k_val
                self.k_cache_scales[:, :, input_pos] = k_scales
                self.k_cache_zero_points[:, :, input_pos] = k_zero_points
                self.v_cache[:, :, input_pos] = quantized_v_val
                self.v_cache_scales[:, :, input_pos] = v_scales
                self.v_cache_zero_points[:, :, input_pos] = v_zero_points
        else:
            # Right now using custom ops on this path.
            # In future we can update custom op to handle transposed cache
            # as well.
            # Note that we may have to revert this change if other ET
            # backends such as QNN want to use quantized cache, with dynamic shape,
            # instead of quantizing on their own.
            # But until this opting for code simplicity
            start_pos = input_pos[0].item()
            _ = torch.ops.llama.update_quantized_cache(
                quantized_k_val, self.k_cache, start_pos
            )
            _ = torch.ops.llama.update_quantized_cache(
                k_scales, self.k_cache_scales, start_pos
            )
            _ = torch.ops.llama.update_quantized_cache(
                k_zero_points, self.k_cache_zero_points, start_pos
            )
            _ = torch.ops.llama.update_quantized_cache(
                quantized_v_val, self.v_cache, start_pos
            )
            _ = torch.ops.llama.update_quantized_cache(
                v_scales, self.v_cache_scales, start_pos
            )
            _ = torch.ops.llama.update_quantized_cache(
                v_zero_points, self.v_cache_zero_points, start_pos
            )

        k_out = torch.ops.quantized_decomposed.dequantize_per_token(
            self.k_cache,
            self.k_cache_scales,
            self.k_cache_zero_points,
            torch.iinfo(torch.int8).min,
            torch.iinfo(torch.int8).max,
            self.quantized_cache_dtype,
            self.cache_fp_type,
        )
        v_out = torch.ops.quantized_decomposed.dequantize_per_token(
            self.v_cache,
            self.v_cache_scales,
            self.v_cache_zero_points,
            torch.iinfo(torch.int8).min,
            torch.iinfo(torch.int8).max,
            self.quantized_cache_dtype,
            self.cache_fp_type,
        )
        return k_out, v_out

    @classmethod
    def from_float(cls, kv_cache, cache_type: QuantizedCacheType):
        cache_shape = kv_cache.k_cache.shape
        if kv_cache.is_tranposed:
            max_batch_size, n_heads, max_seq_length, head_dim = cache_shape
        else:
            max_batch_size, max_seq_length, n_heads, head_dim = cache_shape
        return cls(
            max_batch_size,
            max_seq_length,
            n_heads,
            head_dim,
            cache_type,
            kv_cache.is_tranposed,
            kv_cache.enable_dynamic_shape,
        )


def replace_kv_cache_with_quantized_kv_cache(module):
    logging.warning(
        "Replacing KVCache with QuantizedKVCache. This modifies the model in place."
    )
    for name, child in module.named_children():
        if isinstance(child, KVCache):
            setattr(
                module,
                name,
                QuantizedKVCache.from_float(child, QuantizedCacheType.AffineAsymmetric),
            )
        else:
            replace_kv_cache_with_quantized_kv_cache(child)
    return module
