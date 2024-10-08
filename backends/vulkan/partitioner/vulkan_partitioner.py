# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Any, Callable, Dict, final, List, Mapping, Optional, Tuple

import executorch.backends.vulkan.serialization.vulkan_graph_schema as vk_graph_schema

import torch

from executorch.backends.vulkan.partitioner.supported_ops import (
    enumerate_supported_ops,
    OpList,
)
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

# pyre-ignore
ops_not_to_decompose = [
    torch.ops.aten.upsample_nearest2d.vec,
]

logger: logging.Logger = logging.getLogger("")
logger.setLevel(logging.INFO)


class VulkanSupportedOperators(OperatorSupportBase):
    _ops: OpList = enumerate_supported_ops()

    def __init__(self, require_dynamic_shape: bool = False) -> None:
        super().__init__()
        self.require_dynamic_shapes = require_dynamic_shape
        # The tensor dim limit is to guard against tensors with one or more
        # large dimensions, which cannot be represented by an image texture due
        # to the texture axis limits.
        self.tensor_dim_limit = 16384

    # pyre-ignore
    def node_val_is_compatible(self, node_val: Any) -> bool:
        # Skip nodes that don't have a value
        if node_val is None:
            return True

        # TODO(ssjia) support symbolic ints
        if isinstance(node_val, torch.SymInt):
            return False

        if isinstance(node_val, FakeTensor):
            # Vulkan currently only supports tensors of up to 4D
            if len(node_val.shape) > 4:
                return False

            # bool dtype not currently supported
            if node_val.dtype == torch.bool:
                return False

            for dim in node_val.shape:
                if dim > self.tensor_dim_limit:
                    return False

        if isinstance(node_val, (list, tuple)):
            for item in node_val:
                if not self.node_val_is_compatible(item):
                    return False

        return True

    def all_args_compatible(self, node: torch.fx.Node) -> bool:
        node_val = node.meta.get("val", None)
        if not self.node_val_is_compatible(node_val):
            return False

        for arg in node.args:
            if not isinstance(arg, torch.fx.Node):
                continue

            arg_val = arg.meta.get("val", None)
            if not self.node_val_is_compatible(arg_val):
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

        first_user = list(node.users.keys())[0]
        if first_user.target in [
            exir_ops.edge.aten.mm.default,
            exir_ops.edge.aten.addmm.default,
        ]:
            # Only mark this node if the overall linear op is valid
            if self.all_args_compatible(first_user):
                return True

        return False

    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        r = self._is_node_supported(submodules, node)
        if not r and node.op == "call_function":
            logger.info(f"Skipping node in Vulkan partitioning: {node.format_node()}")
        return r

    def _is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
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

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        return (ops_not_to_decompose, None)

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

        pl = len(partition_list)
        if pl == 0:
            logger.warning("No Vulkan subgraphs can be partitioned!")
        else:
            logger.info(f"Found {pl} Vulkan subgraphs to be partitioned.")

        tag_constant_data(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
