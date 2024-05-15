# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Any, Dict, final, List, Optional

import executorch.backends.vulkan.serialization.vulkan_graph_schema as vk_graph_schema

import torch
from executorch.backends.vulkan.vulkan_preprocess import VulkanBackend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from executorch.exir.dialects._ops import ops as exir_ops

from torch._subclasses.fake_tensor import FakeTensor
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

from torch.fx.passes.operator_support import OperatorSupportBase


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
    exir_ops.edge.aten.gelu.default,
    exir_ops.edge.aten.hardtanh.default,
    exir_ops.edge.aten.relu.default,
    exir_ops.edge.aten.sigmoid.default,
    exir_ops.edge.aten.sqrt.default,
    exir_ops.edge.aten.tanh.default,
]

MATMUL_OPS = [
    exir_ops.edge.aten.bmm.default,
    exir_ops.edge.aten.mm.default,
    exir_ops.edge.aten.addmm.default,
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
    exir_ops.edge.aten.native_layer_norm.default,
]

SHAPE_MANIPULATION_OPS = [
    exir_ops.edge.aten.unsqueeze_copy.default,
    exir_ops.edge.aten.view_copy.default,
    exir_ops.edge.aten.permute_copy.default,
]

INDEXING_OPS = [
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


def enumerate_ops():
    ops = OpList()
    register_prim_ops(ops)
    register_no_dynamic_shape_ops(ops)
    register_dynamic_shape_ops(ops)
    return ops


class VulkanSupportedOperators(OperatorSupportBase):
    _ops = enumerate_ops()

    def __init__(self, require_dynamic_shape: bool = False):
        super().__init__()
        self.require_dynamic_shapes = require_dynamic_shape

    def node_val_is_compatible(self, node: torch.fx.Node) -> bool:
        node_val = node.meta.get("val", None)
        # Skip nodes that don't have a value
        if node_val is None:
            return True

        # TODO(ssjia) support symbolic ints
        if isinstance(node_val, torch.SymInt):
            return False

        if isinstance(node_val, FakeTensor):
            if len(node_val.shape) > 4:
                return False

        return True

    def all_args_compatible(self, node: torch.fx.Node) -> bool:
        if not self.node_val_is_compatible(node):
            return False

        for arg in node.args:
            if not isinstance(arg, torch.fx.Node):
                continue

            if not self.node_val_is_compatible(node):
                return False

        return True

    def is_linear_permute(self, node: torch.fx.Node) -> bool:
        if node.target not in [
            exir_ops.edge.aten.t_copy.default,
            exir_ops.edge.aten.permute_copy.default,
        ]:
            return False

        if len(node.users) != 1:
            return False

        if list(node.users.keys())[0].target in [
            exir_ops.edge.aten.mm.default,
            exir_ops.edge.aten.addmm.default,
        ]:
            return True

        return False

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        if self.is_linear_permute(node):
            return True

        if node.target not in VulkanSupportedOperators._ops:
            return False

        features = VulkanSupportedOperators._ops[node.target]

        if self.require_dynamic_shapes and not features.supports_dynamic_shape:
            return False

        return self.all_args_compatible(node)


def parse_compile_options(compile_options: Dict[str, Any]) -> List[CompileSpec]:
    compile_specs = []

    for key, value in compile_options.items():
        if isinstance(
            value, (vk_graph_schema.VkStorageType, vk_graph_schema.VkMemoryLayout)
        ):
            value_bytes = int(value).to_bytes(4, byteorder="little")
            compile_specs.append(CompileSpec(key, value_bytes))

        # Unhandled options are ignored

    return compile_specs


@final
class VulkanPartitioner(Partitioner):
    def __init__(self, compile_options: Optional[Dict[str, Any]] = None) -> None:
        self.options: Dict[str, Any] = {}
        if compile_options is not None:
            self.options = compile_options

        compile_spec = parse_compile_options(self.options)
        self.delegation_spec = DelegationSpec(VulkanBackend.__name__, compile_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        partition_tags = {}

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            VulkanSupportedOperators(self.options.get("require_dynamic_shapes", False)),
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()
        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        tag_constant_data(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
