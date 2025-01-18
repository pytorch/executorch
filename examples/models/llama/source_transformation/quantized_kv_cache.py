# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Tuple

import torch
import torch.nn as nn
from executorch.examples.models.llama.llama_transformer import KVCache

from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401


"""
 Heavily "inspired" by AO's implementation of the same in torchao/_models/llama/model.py
"""


# Doesnt have to abide by affine quantizaiton laws
# However, if we do implement quantized sdpa, then this might be handy
class QuantizedCacheType(Enum):
    AffineSymmetric = 0
    AffineAsymmetric = 1
    AffineSymmetricGroupWise = 2
    AffineAsymmetricGroupWise = 3


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
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=self.quantized_cache_dtype)
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=self.quantized_cache_dtype)
        )
        self.register_buffer(
            "k_cache_scales", torch.ones(scale_shape, dtype=torch.float64)
        )
        self.register_buffer(
            "v_cache_scales", torch.ones(scale_shape, dtype=torch.float64)
        )
        if cache_type == QuantizedCacheType.AffineAsymmetric:
            self.register_buffer(
                "k_cache_zero_points", torch.ones(scale_shape, dtype=torch.int64)
            )
            self.register_buffer(
                "v_cache_zero_points", torch.ones(scale_shape, dtype=torch.int64)
            )

    def _quantize(self, value):
        (
            scales,
            zero_points,
        ) = torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric.default(
            value, self.quantized_cache_dtype
        )
        quantized_value = torch.ops.quantized_decomposed.quantize_per_token(
            value,
            scales,
            zero_points,
            torch.iinfo(self.quantized_cache_dtype).min,
            torch.iinfo(self.quantized_cache_dtype).max,
            self.quantized_cache_dtype,
        )
        return quantized_value, scales, zero_points

    def update(self, input_pos, k_val, v_val):
        # quantize current k_val and store it in the cache
        quantized_k_val, k_scales, k_zero_points = self._quantize(k_val)

        quantized_v_val, v_scales, v_zero_points = self._quantize(v_val)

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
                dim_to_slice = 2 if self.is_transposed else 1
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
            _ = torch.ops.llama.update_cache(quantized_k_val, self.k_cache, start_pos)
            _ = torch.ops.llama.update_cache(k_scales, self.k_cache_scales, start_pos)
            _ = torch.ops.llama.update_cache(
                k_zero_points, self.k_cache_zero_points, start_pos
            )
            _ = torch.ops.llama.update_cache(quantized_v_val, self.v_cache, start_pos)
            _ = torch.ops.llama.update_cache(v_scales, self.v_cache_scales, start_pos)
            _ = torch.ops.llama.update_cache(
                v_zero_points, self.v_cache_zero_points, start_pos
            )

        k_out = torch.ops.quantized_decomposed.dequantize_per_token(
            self.k_cache,
            self.k_cache_scales,
            self.k_cache_zero_points,
            torch.iinfo(self.quantized_cache_dtype).min,
            torch.iinfo(self.quantized_cache_dtype).max,
            self.quantized_cache_dtype,
            self.cache_fp_type,
        )
        v_out = torch.ops.quantized_decomposed.dequantize_per_token(
            self.v_cache,
            self.v_cache_scales,
            self.v_cache_zero_points,
            torch.iinfo(self.quantized_cache_dtype).min,
            torch.iinfo(self.quantized_cache_dtype).max,
            self.quantized_cache_dtype,
            self.cache_fp_type,
        )

        if self.is_transposed:
            if self.enable_dynamic_shape:
                start_pos = input_pos[0].item()
                torch._check_is_size(start_pos)
                dim_to_slice = 2 if self.is_transposed else 1
                torch._check(start_pos < self.k_cache.size(dim_to_slice))
                seq_length = k_val.size(dim_to_slice)
                narrowed_k = k_out.narrow(dim_to_slice, start_pos, seq_length)
                narrowed_k.copy_(k_val)
                narrowed_v = v_out.narrow(dim_to_slice, start_pos, seq_length)
                narrowed_v.copy_(v_val)
            else:
                k_out[:, :, input_pos] = k_val
                v_out[:, :, input_pos] = v_val
        else:
            start_pos = input_pos[0].item()
            _ = torch.ops.llama.update_cache(k_val, k_out, start_pos)
            _ = torch.ops.llama.update_cache(v_val, v_out, start_pos)

        return k_out, v_out

    @classmethod
    def from_float(cls, kv_cache, cache_type: QuantizedCacheType):
        cache_shape = kv_cache.k_cache.shape
        if kv_cache.is_transposed:
            max_batch_size, n_heads, max_seq_length, head_dim = cache_shape
        else:
            max_batch_size, max_seq_length, n_heads, head_dim = cache_shape
        return cls(
            max_batch_size,
            max_seq_length,
            n_heads,
            head_dim,
            cache_type,
            kv_cache.is_transposed,
            kv_cache.enable_dynamic_shape,
        )


def replace_kv_cache_with_quantized_kv_cache(module):
    try:
        op = torch.ops.quantized_decomposed.quantize_per_token.out
        assert op is not None
    except:
        import glob

        import executorch
        from executorch.extension.pybindings import portable_lib  # noqa # usort: skip

        # Ideally package is installed in only one location but usage of
        # PYATHONPATH can result in multiple locations.
        # ATM this is mainly used in CI for qnn runner. Will need to revisit this
        executorch_package_path = executorch.__path__[-1]
        libs = list(
            glob.glob(
                f"{executorch_package_path}/**/libquantized_ops_aot_lib.*",
                recursive=True,
            )
        )
        assert len(libs) == 1, f"Expected 1 library but got {len(libs)}"
        logging.info(f"Loading custom ops library: {libs[0]}")
        torch.ops.load_library(libs[0])
        op = torch.ops.quantized_decomposed.quantize_per_token.out
        assert op is not None
    # This is needed to ensure that custom ops are registered
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

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


class CustomKVCache(nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype=torch.float32,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        cache_shape = (max_batch_size, max_seq_length, n_heads, head_dim)

        self.max_batch_size = max_batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )

    def update(
        self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_pos: [S], k_val: [B, S, H, D]
        start_pos = input_pos[0].item()
        _ = torch.ops.llama.update_cache(k_val, self.k_cache, start_pos)
        _ = torch.ops.llama.update_cache(v_val, self.v_cache, start_pos)
        return self.k_cache, self.v_cache


def replace_kv_cache_with_custom_kv_cache(module):
    r"""
    Replace KVCache with CustomKVCache. This modifies the model in place.
    At the moment custom kv cache only supports cache with shape
    [B, S, H, D] as opposed to [B, H, S, D]
    This is because the custom op treats second dim as sequence dim.
    Future work: support [B, H, S, D]
    """
    logging.warning(
        "Replacing KVCache with CustomKVCache. This modifies the model in place."
    )
    for name, child in module.named_children():
        if isinstance(child, KVCache):
            cache_shape = child.k_cache.shape
            cache_dtype = child.k_cache.dtype
            assert (
                child.is_transposed is False
            ), "CustomKVCache does not support transposed cache"
            max_batch_size, max_seq_length, n_heads, head_dim = cache_shape
            setattr(
                module,
                name,
                CustomKVCache(
                    max_batch_size,
                    max_seq_length,
                    n_heads,
                    head_dim,
                    dtype=cache_dtype,
                ),
            )
        else:
            replace_kv_cache_with_custom_kv_cache(child)
    return module
