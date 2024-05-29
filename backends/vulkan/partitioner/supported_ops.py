# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

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
]

BINARY_OPS = [
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.sub.Tensor,
    exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.div.Tensor,
    exir_ops.edge.aten.div.Tensor_mode,
    exir_ops.edge.aten.pow.Tensor_Tensor,
]

UNARY_OPS = [
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
]

MATMUL_OPS = [
    exir_ops.edge.aten.bmm.default,
    exir_ops.edge.aten.mm.default,
    exir_ops.edge.aten.addmm.default,
    exir_ops.edge.aten.linear.default,
]

POOLING_OPS = [
    exir_ops.edge.aten.max_pool2d_with_indices.default,
]

CONVOLUTION_OPS = [
    exir_ops.edge.aten.convolution.default,
]

REDUCTION_OPS = [
    exir_ops.edge.aten.sum.dim_IntList,
    exir_ops.edge.aten._softmax.default,
    exir_ops.edge.aten._log_softmax.default,
]

NORMALIZATION_OPS = [
    exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
    exir_ops.edge.aten.native_layer_norm.default,
]

SHAPE_MANIPULATION_OPS = [
    exir_ops.edge.aten.unsqueeze_copy.default,
    exir_ops.edge.aten.view_copy.default,
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.edge.aten.t_copy.default,
]

INDEXING_OPS = [
    exir_ops.edge.aten.embedding.default,
    exir_ops.edge.aten.index_select.default,
    exir_ops.edge.aten.select_copy.int,
    exir_ops.edge.aten.slice_copy.Tensor,
]

ORCHESTRATION_OPS = [
    exir_ops.edge.aten.cat.default,
    exir_ops.edge.aten.split_with_sizes_copy.default,
    exir_ops.edge.aten.split.Tensor,
    exir_ops.edge.aten.repeat.default,
]

CREATION_OPS = [
    exir_ops.edge.aten.clone.default,
    exir_ops.edge.aten.full.default,
]


def register_prim_ops(ops: OpList):
    for op in PRIM_OPS:
        ops[op].supports_texture = True
        ops[op].supports_buffer = True
        ops[op].supports_dynamic_shape = True


def register_no_dynamic_shape_ops(ops: OpList):
    for op in [
        *REDUCTION_OPS,
        *NORMALIZATION_OPS,
        *SHAPE_MANIPULATION_OPS,
        *INDEXING_OPS,
        *ORCHESTRATION_OPS,
        *CREATION_OPS,
    ]:
        ops[op].supports_dynamic_shape = False


def register_dynamic_shape_ops(ops: OpList):
    for op in [
        *BINARY_OPS,
        *UNARY_OPS,
        *MATMUL_OPS,
        *POOLING_OPS,
        *CONVOLUTION_OPS,
    ]:
        ops[op].supports_dynamic_shape = True


def enumerate_supported_ops():
    ops = OpList()
    register_prim_ops(ops)
    register_no_dynamic_shape_ops(ops)
    register_dynamic_shape_ops(ops)
    return ops
