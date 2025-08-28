# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn
from executorch.examples.models.llama.attention import (
    _create_causal_mask_for_ring_buffer,
    CachePositionsManager,
    KVCache,
    RingKVCache,
)

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
        max_context_length,
        n_heads,
        head_dim,
        cache_type: QuantizedCacheType = QuantizedCacheType.AffineSymmetric,
        use_custom_update_cache_op: bool = False,
        return_float_values: bool = True,
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
        self.use_custom_update_cache_op = use_custom_update_cache_op
        self.quantized_cache_dtype = torch.int8
        self.cache_fp_type = torch.float32
        self.return_float_values = return_float_values
        self.max_context_length = max_context_length
        cache_shape = (max_batch_size, max_context_length, n_heads, head_dim)
        scale_shape = (max_batch_size, max_context_length, n_heads, 1)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=self.quantized_cache_dtype)
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=self.quantized_cache_dtype)
        )
        self.register_buffer(
            "k_cache_scales", torch.ones(scale_shape, dtype=torch.float32)
        )
        self.register_buffer(
            "v_cache_scales", torch.ones(scale_shape, dtype=torch.float32)
        )
        if cache_type == QuantizedCacheType.AffineAsymmetric:
            self.register_buffer(
                "k_cache_zero_points", torch.ones(scale_shape, dtype=torch.int8)
            )
            self.register_buffer(
                "v_cache_zero_points", torch.ones(scale_shape, dtype=torch.int8)
            )
        self.cache_type = cache_type

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

    def _quantize_and_update(self, input_pos, k_val, v_val, indices=None):
        quantized_k_val, k_scales, k_zero_points = self._quantize(k_val)
        quantized_v_val, v_scales, v_zero_points = self._quantize(v_val)

        k_scales = k_scales.to(torch.float32)
        k_zero_points = k_zero_points.to(self.quantized_cache_dtype)
        v_scales = v_scales.to(torch.float32)
        v_zero_points = v_zero_points.to(self.quantized_cache_dtype)

        if self.use_custom_update_cache_op:
            start_pos = input_pos[0].item()
            if indices is not None:
                _ = torch.ops.llama.update_cache_with_indices(
                    quantized_k_val, self.k_cache, start_pos, indices
                )
                _ = torch.ops.llama.update_cache_with_indices(
                    k_scales, self.k_cache_scales, start_pos, indices
                )
                _ = torch.ops.llama.update_cache_with_indices(
                    k_zero_points, self.k_cache_zero_points, start_pos, indices
                )
                _ = torch.ops.llama.update_cache_with_indices(
                    quantized_v_val, self.v_cache, start_pos, indices
                )
                _ = torch.ops.llama.update_cache_with_indices(
                    v_scales, self.v_cache_scales, start_pos, indices
                )
                _ = torch.ops.llama.update_cache_with_indices(
                    v_zero_points, self.v_cache_zero_points, start_pos, indices
                )
            else:
                _ = torch.ops.llama.update_cache(
                    quantized_k_val, self.k_cache, start_pos
                )
                _ = torch.ops.llama.update_cache(
                    k_scales, self.k_cache_scales, start_pos
                )
                _ = torch.ops.llama.update_cache(
                    k_zero_points, self.k_cache_zero_points, start_pos
                )
                _ = torch.ops.llama.update_cache(
                    quantized_v_val, self.v_cache, start_pos
                )
                _ = torch.ops.llama.update_cache(
                    v_scales, self.v_cache_scales, start_pos
                )
                _ = torch.ops.llama.update_cache(
                    v_zero_points, self.v_cache_zero_points, start_pos
                )
        else:
            assert indices is None, "Indices not supported for this path"
            # Following is also broken because in prefill input_pos = [0]
            # but we need to update some slice of cache
            self.k_cache[:, input_pos] = quantized_k_val
            self.k_cache_scales[:, input_pos] = k_scales
            self.k_cache_zero_points[:, input_pos] = k_zero_points
            self.v_cache[:, input_pos] = quantized_v_val
            self.v_cache_scales[:, input_pos] = v_scales
            self.v_cache_zero_points[:, input_pos] = v_zero_points

    def _update_and_return_float_values(self, input_pos, k_val, v_val, indices=None):
        self._quantize_and_update(input_pos, k_val, v_val, indices)

        k_out = torch.ops.quantized_decomposed.dequantize_per_token(
            self.k_cache,
            self.k_cache_scales.to(torch.float64),
            self.k_cache_zero_points.to(torch.int64),
            torch.iinfo(self.quantized_cache_dtype).min,
            torch.iinfo(self.quantized_cache_dtype).max,
            self.quantized_cache_dtype,
            self.cache_fp_type,
        )
        v_out = torch.ops.quantized_decomposed.dequantize_per_token(
            self.v_cache,
            self.v_cache_scales.to(torch.float64),
            self.v_cache_zero_points.to(torch.int64),
            torch.iinfo(self.quantized_cache_dtype).min,
            torch.iinfo(self.quantized_cache_dtype).max,
            self.quantized_cache_dtype,
            self.cache_fp_type,
        )

        # When returning float values we just use the last value
        # instead of dequantized value.
        start_pos = input_pos[0].item()
        if self.use_custom_update_cache_op:
            if indices is not None:
                _ = torch.ops.llama.update_cache_with_indices(
                    k_val, k_out, start_pos, indices
                )
                _ = torch.ops.llama.update_cache_with_indices(
                    v_val, v_out, start_pos, indices
                )
            else:
                _ = torch.ops.llama.update_cache(k_val, k_out, start_pos)
                _ = torch.ops.llama.update_cache(v_val, v_out, start_pos)
        else:
            k_out[:, input_pos] = k_val
            v_out[:, input_pos] = v_val

        return k_out, v_out

    def _update_and_return_quantized_values(
        self, input_pos, k_val, v_val, indices=None
    ):
        self._quantize_and_update(input_pos, k_val, v_val, indices)

        return self.k_cache, self.v_cache

    def update(self, input_pos, k_val, v_val, indices=None):
        """
        k_val, v_val: [B, H, S, D]
        return: [B, H, S, D]
        However the storage is [B, S, H, D] so we incur transpose in, transpose out
        This shall be removed by subsequent post-export graph pass
        """

        k_val = k_val.transpose(1, 2)
        v_val = v_val.transpose(1, 2)

        if self.return_float_values:
            k_out, v_out = self._update_and_return_float_values(
                input_pos, k_val, v_val, indices
            )
        else:
            k_out, v_out = self._update_and_return_quantized_values(
                input_pos, k_val, v_val, indices
            )
        return k_out.transpose(1, 2), v_out.transpose(1, 2)

    @classmethod
    def from_float(
        cls,
        kv_cache,
        cache_type: QuantizedCacheType,
        use_custom_update_cache_op: bool = False,
    ):
        max_batch_size, n_heads, max_context_length, head_dim = kv_cache.k_cache.shape
        if isinstance(kv_cache, CustomKVCache):
            # If replacing custom kv cache, then the shape is [B, S, H, D]
            max_batch_size, max_context_length, n_heads, head_dim = (
                kv_cache.k_cache.shape
            )
        return cls(
            max_batch_size,
            max_context_length,
            n_heads,
            head_dim,
            cache_type,
            use_custom_update_cache_op,
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
                f"{executorch_package_path}/**/*quantized_ops_aot_lib.*",
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

    logging.info(
        "Replacing KVCache with QuantizedKVCache. This modifies the model in place."
    )
    return _replace_kv_cache_with_quantized_kv_cache(module)


def _replace_kv_cache_with_quantized_kv_cache(module):
    for name, child in module.named_children():
        if isinstance(child, KVCache) or isinstance(child, CustomKVCache):
            setattr(
                module,
                name,
                QuantizedKVCache.from_float(
                    child,
                    QuantizedCacheType.AffineAsymmetric,
                    use_custom_update_cache_op=True,
                ),
            )
        else:
            _replace_kv_cache_with_quantized_kv_cache(child)
    return module


class CustomKVCache(nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_context_length: int,
        n_heads: int,
        head_dim: int,
        dtype=torch.float32,
    ):
        super().__init__()
        self.max_context_length = max_context_length
        cache_shape = (max_batch_size, max_context_length, n_heads, head_dim)

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
        self,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_pos: [S], k_val: [B, H, S, D]
        k_val = k_val.transpose(1, 2)
        v_val = v_val.transpose(1, 2)
        start_pos = input_pos[0].item()

        if indices is not None:
            _ = torch.ops.llama.update_cache_with_indices(
                k_val, self.k_cache, start_pos, indices
            )
            _ = torch.ops.llama.update_cache_with_indices(
                v_val, self.v_cache, start_pos, indices
            )
        else:
            _ = torch.ops.llama.update_cache(k_val, self.k_cache, start_pos)
            _ = torch.ops.llama.update_cache(v_val, self.v_cache, start_pos)

        return (
            self.k_cache.transpose(1, 2),
            self.v_cache.transpose(1, 2),
        )


def replace_kv_cache_with_custom_kv_cache(module):
    """
    Replace KVCache with CustomKVCache. This modifies the model in place.
    At the moment custom kv cache only supports cache with shape
    [B, S, H, D] as opposed to [B, H, S, D]
    This is because the custom op treats second dim as sequence dim.
    Future work: support [B, H, S, D]
    """
    logging.info(
        "Replacing KVCache with CustomKVCache. This modifies the model in place."
    )
    return _replace_kv_cache_with_custom_kv_cache(module)


def _replace_kv_cache_with_custom_kv_cache(module):
    for name, child in module.named_children():
        if isinstance(child, KVCache):
            cache_shape = child.k_cache.shape
            cache_dtype = child.k_cache.dtype
            max_batch_size, n_heads, max_context_length, head_dim = cache_shape
            setattr(
                module,
                name,
                CustomKVCache(
                    max_batch_size,
                    max_context_length,
                    n_heads,
                    head_dim,
                    dtype=cache_dtype,
                ),
            )
        else:
            _replace_kv_cache_with_custom_kv_cache(child)
    return module


class QuantizedRingKVCache(QuantizedKVCache):
    def __init__(
        self,
        max_batch_size,
        max_context_length,
        n_heads,
        head_dim,
        cache_type: QuantizedCacheType = QuantizedCacheType.AffineSymmetric,
        use_custom_update_cache_op: bool = False,
        return_float_values: bool = True,
    ):
        # Look at attention.py for explanation on why max_context_length * 2
        super().__init__(
            max_batch_size,
            max_context_length * 2,
            n_heads,
            head_dim,
            cache_type,
            use_custom_update_cache_op,
            return_float_values,
        )
        self.cache_positions_manager = CachePositionsManager(self.max_context_length)
        self.is_ring_buffer = True
        self.window_size = max_context_length

    def create_causal_mask_for_ring_buffer(self, start_pos, seq_len):
        cache_positions = self.cache_positions_manager.cache_positions
        return _create_causal_mask_for_ring_buffer(
            cache_positions, self.window_size, start_pos, seq_len
        )

    def update(self, input_pos, k_val, v_val):
        """
        k_val, v_val: [B, H, S, D]
        return: [B, H, S, D]
        However the storage is [B, S, H, D] so we incur transpose in, transpose out
        This shall be removed by subsequent post-export graph pass
        """
        # Need to transpose for two reasons
        # 1. kv cache is stored as [B, S, H, D]
        # 2. If seq_len = k_val.size(2), we wont be able be able to optimize
        #    away transpose at the output of k, v projection
        seq_len = k_val.transpose(1, 2).size(1)
        assert seq_len <= self.k_cache.size(
            1
        ), f"Update sequence length({seq_len}) for kv cache must be smaller than the cache size({self.k_cache.size(2)})"
        indices = self.cache_positions_manager.calculate_positions_and_update_indices(
            input_pos, seq_len
        )
        indices = indices.unsqueeze(0)

        return super().update(input_pos, k_val, v_val, indices)

    @classmethod
    def from_quantized_kv_cache(
        cls,
        kv_cache,
        sliding_window_size,
    ):
        assert isinstance(
            kv_cache, QuantizedKVCache
        ), "For QuantizedRingKVCache expect QuantizedKVCache as input kv_cache"
        max_batch_size, _, n_heads, head_dim = kv_cache.k_cache.shape
        return cls(
            max_batch_size,
            sliding_window_size,
            n_heads,
            head_dim,
            kv_cache.cache_type,
            kv_cache.use_custom_update_cache_op,
            kv_cache.return_float_values,
        )


class CustomRingKVCache(CustomKVCache):
    def __init__(
        self,
        max_batch_size,
        max_context_length,
        n_heads,
        head_dim,
        dtype=torch.float32,
    ):
        # Look at attention.py for explanation on why max_context_length * 2
        super().__init__(
            max_batch_size, max_context_length * 2, n_heads, head_dim, dtype
        )
        self.cache_positions_manager = CachePositionsManager(self.max_context_length)
        self.is_ring_buffer = True
        self.window_size = max_context_length

    def create_causal_mask_for_ring_buffer(self, start_pos, seq_len):
        cache_positions = self.cache_positions_manager.cache_positions
        return _create_causal_mask_for_ring_buffer(
            cache_positions, self.window_size, start_pos, seq_len
        )

    def update(self, input_pos, k_val, v_val):
        """
        k_val, v_val: [B, H, S, D]
        return: [B, H, S, D]
        However the storage is [B, S, H, D] so we incur transpose in, transpose out
        This shall be removed by subsequent post-export graph pass
        """
        # Need to transpose for two reasons
        # 1. kv cache is stored as [B, S, H, D]
        # 2. If seq_len = k_val.size(2), we wont be able be able to optimize
        #    away transpose at the output of k, v projection
        seq_len = k_val.transpose(1, 2).size(1)
        assert seq_len <= self.k_cache.size(
            1
        ), f"Update sequence length({seq_len}) for kv cache must be smaller than the cache size({self.k_cache.size(2)})"
        indices = self.cache_positions_manager.calculate_positions_and_update_indices(
            input_pos, seq_len
        )
        indices = indices.unsqueeze(0)

        return super().update(input_pos, k_val, v_val, indices)

    @classmethod
    def from_custom_kv_cache(
        cls,
        kv_cache,
        sliding_window_size,
    ):
        max_batch_size, n_heads, _, head_dim = kv_cache.k_cache.shape
        if isinstance(kv_cache, CustomKVCache):
            # If replacing custom kv cache, then the shape is [B, S, H, D]
            max_batch_size, _, n_heads, head_dim = kv_cache.k_cache.shape
        return cls(
            max_batch_size,
            sliding_window_size,
            n_heads,
            head_dim,
            dtype=kv_cache.k_cache.dtype,
        )


def _replace_kv_cache_with_ring_kv_cache(attention, layer_size):
    sliding_window_size = layer_size
    assert (
        getattr(attention, "kv_cache", None) is not None
    ), "Attention module must have kv_cache module"
    kv_cache = attention.kv_cache
    if isinstance(kv_cache, KVCache):
        attention.kv_cache = RingKVCache(
            kv_cache.max_batch_size,
            sliding_window_size,
            kv_cache.n_heads,
            kv_cache.head_dim,
            kv_cache.enable_dynamic_shape,
            kv_cache.k_cache.dtype,
        )
    elif isinstance(kv_cache, CustomKVCache):
        attention.kv_cache = CustomRingKVCache.from_custom_kv_cache(
            kv_cache, layer_size
        )
    elif isinstance(kv_cache, QuantizedKVCache):
        attention.kv_cache = QuantizedRingKVCache.from_quantized_kv_cache(
            kv_cache, layer_size
        )


def replace_kv_cache_with_ring_kv_cache(module, layer_sizes):
    # This is needed to ensure that custom ops are registered
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

    assert len(module.layers) >= len(
        layer_sizes
    ), f"Length of layer sizes {len(layer_sizes)} must match the number of layers in the module {len(module.layers)}."
    multiplier = len(module.layers) // len(layer_sizes)
    modulo = len(module.layers) % len(layer_sizes)
    assert (
        modulo == 0
    ), f"num layers specified must be multiple of model layers in order to specify pattern. pattern: {layer_sizes} model's num layers {len(module.layers)}"
    layer_sizes = layer_sizes * multiplier
    logging.info(
        f"Applying local sliding window attention with following pattern {layer_sizes}."
    )
    assert len(layer_sizes) == len(
        module.layers
    ), f"Length of layer sizes {len(layer_sizes)} must match the number of layers in the module {len(module.layers)}."
    for i, transformer_block in enumerate(module.layers):
        sliding_window_size = layer_sizes[i]
        if sliding_window_size == 0:
            continue
        assert (
            getattr(transformer_block, "attention", None) is not None
        ), f"Transfomer block must have attention module. Transformer block {transformer_block}"
        attention = transformer_block.attention
        _replace_kv_cache_with_ring_kv_cache(attention, sliding_window_size)
        # if attention's sdpa is custom sdpa then we have to make sure
        # it is not doing causal attention
        if "SDPACustom" in attention.SDPA.__class__.__name__:
            attention.SDPA.use_attention_mask = True
        # QuantizedSDPA has to store kv_cache in order to obtrain
        # scales and zero points for k and v cache.
        # So if we replcaed attention module's quantized kv cache with
        # QuantizedRingKVCache then we also have to replace attention's
        # SDPA module kv_cache so that it refers to the same kv_cache
        if "QuantizedSDPA" in attention.SDPA.__class__.__name__:
            attention.SDPA.use_attention_mask = True
            attention.SDPA.kv_cache = attention.kv_cache
    return module
