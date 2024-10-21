# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import operator

from executorch.backends.vulkan._passes.custom_ops_defs import (  # noqa
    conv_with_clamp_op,
    grid_priors_op,
)

from executorch.exir.dialects._ops import ops as exir_ops


class OpFeatures:
    __slots__ = ["supports_texture", "supports_buffer", "supports_dynamic_shape"]

    def __init__(
        self,
        supports_dynamic_shape: bool = False,
        supports_buffer: bool = False,
        supports_texture: bool = True,
    ):
        self.supports_dynamic_shape = supports_dynamic_shape
        self.supports_texture = supports_texture
        self.supports_buffer = supports_buffer


class OpList:
    def __init__(self):
        self._ops = {}

    def __getitem__(self, op):
        if op not in self._ops:
            self._ops[op] = OpFeatures()
        return self._ops[op]

    def __contains__(self, op):
        return op in self._ops


PRIM_OPS = [
    operator.getitem,
    # Quantization related ops will be fused via graph passes
    exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
    exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
]

SUPPORTS_DYNAMIC_SHAPE = [
    # Binary broadcasting
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.sub.Tensor,
    exir_ops.edge.aten.minimum.default,
    exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.div.Tensor,
    exir_ops.edge.aten.div.Tensor_mode,
    exir_ops.edge.aten.pow.Tensor_Tensor,
    # Unary elementwise
    exir_ops.edge.aten.abs.default,
    exir_ops.edge.aten.clamp.default,
    exir_ops.edge.aten.cos.default,
    exir_ops.edge.aten.exp.default,
    exir_ops.edge.aten.gelu.default,
    exir_ops.edge.aten.hardshrink.default,
    exir_ops.edge.aten.hardtanh.default,
    exir_ops.edge.aten.neg.default,
    exir_ops.edge.aten.relu.default,
    exir_ops.edge.aten.sigmoid.default,
    exir_ops.edge.aten.sin.default,
    exir_ops.edge.aten.sqrt.default,
    exir_ops.edge.aten.tanh.default,
    exir_ops.edge.aten._to_copy.default,
    # Matrix Multiplication
    exir_ops.edge.aten.bmm.default,
    exir_ops.edge.aten.mm.default,
    exir_ops.edge.aten.addmm.default,
    exir_ops.edge.aten.linear.default,
    exir_ops.edge.et_vk.linear_weight_int4.default,
    exir_ops.edge.aten._weight_int8pack_mm.default,
    # Reduction
    exir_ops.edge.aten._log_softmax.default,
    exir_ops.edge.aten._softmax.default,
    # 2D Pooling
    exir_ops.edge.aten.avg_pool2d.default,
    exir_ops.edge.aten.max_pool2d_with_indices.default,
    # Convolution
    exir_ops.edge.aten.convolution.default,
    exir_ops.edge.et_vk.conv_with_clamp.default,
    # Llama ops
    "llama::sdpa_with_kv_cache",
    exir_ops.edge.et_vk.apply_rotary_emb.default,
]

NO_DYNAMIC_SHAPE = [
    # Reduction
    exir_ops.edge.aten.mean.dim,
    exir_ops.edge.aten.sum.dim_IntList,
    # Normalization
    exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
    exir_ops.edge.aten.native_layer_norm.default,
    # Shape Manipulation
    exir_ops.edge.aten.squeeze_copy.dims,
    exir_ops.edge.aten.unsqueeze_copy.default,
    exir_ops.edge.aten.view_copy.default,
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.edge.aten.t_copy.default,
    # Indexing and lookup
    exir_ops.edge.aten.embedding.default,
    exir_ops.edge.aten.flip.default,
    exir_ops.edge.aten.index_select.default,
    exir_ops.edge.aten.select_copy.int,
    exir_ops.edge.aten.slice_copy.Tensor,
    # Tensor combination
    exir_ops.edge.aten.cat.default,
    exir_ops.edge.aten.split_with_sizes_copy.default,
    exir_ops.edge.aten.split.Tensor,
    exir_ops.edge.aten.repeat.default,
    # Tensor creation
    exir_ops.edge.aten.arange.start_step,
    exir_ops.edge.aten.clone.default,
    exir_ops.edge.aten.constant_pad_nd.default,
    exir_ops.edge.aten.full.default,
    exir_ops.edge.aten.full_like.default,
    exir_ops.edge.aten.ones.default,
    exir_ops.edge.aten.ones_like.default,
    exir_ops.edge.aten.upsample_nearest2d.vec,
    exir_ops.edge.aten.zeros.default,
    exir_ops.edge.aten.zeros_like.default,
    exir_ops.edge.et_vk.grid_priors.default,
]


def enumerate_supported_ops():
    ops = OpList()

    # Register in order of least to most capabilities

    for op in NO_DYNAMIC_SHAPE:
        ops[op].supports_dynamic_shape = False

    for op in SUPPORTS_DYNAMIC_SHAPE:
        ops[op].supports_dynamic_shape = True

    for op in PRIM_OPS:
        ops[op].supports_texture = True
        ops[op].supports_buffer = True
        ops[op].supports_dynamic_shape = True

    return ops
