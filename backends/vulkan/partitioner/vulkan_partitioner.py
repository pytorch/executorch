# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Any, Callable, Dict, final, List, Mapping, Optional, Tuple

import executorch.backends.vulkan.utils as utils

import torch

from executorch.backends.vulkan.op_registry import (
    get_op_features,
    has_impl,
    OpFeatures,
    vulkan_supported_ops,
)

from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
    VkMemoryLayout,
    VkStorageType,
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
    def __init__(
        self,
        texture_limits: utils.ImageExtents,
        buffer_limit: int,
        require_dynamic_shape: bool = False,
    ) -> None:
        super().__init__()
        self.texture_limits: utils.ImageExtents = texture_limits
        self.buffer_limit = buffer_limit
        self.require_dynamic_shapes = require_dynamic_shape

    def op_node_is_compatible(
        self, node: torch.fx.Node, features: Optional[OpFeatures] = None
    ) -> Tuple[bool, str]:
        """
        Check if a given node is compatible with the Vulkan delegate's implementation
        of the operator called by the node. Each tensor argument participating in the
        operator call must be able to be represented with a (storage type, memory layout)
        combination that is supported by the operator implementation.
        """
        target = node.target
        # Account for custom operators
        if node.target == torch.ops.higher_order.auto_functionalized:
            first_arg = node.args[0]
            assert isinstance(first_arg, torch._ops.OpOverload)
            target = first_arg.name()

        # Extract the features for the node's operator, if no override was provided
        if features is None:
            if not has_impl(target):
                return False, "no operator implementation"
            features = get_op_features(target)

        valid_texture_layouts = utils.possible_node_memory_layouts(
            node, self.texture_limits
        )

        can_use_buffers = utils.within_buffer_limit(node, self.buffer_limit)
        for i, arg in enumerate(node.args):
            if (
                isinstance(arg, torch.fx.Node)
                and utils.is_tensor_node(arg)
                and i not in features.skip_limits_check
            ):
                arg_texture_layouts = utils.possible_node_memory_layouts(
                    arg, self.texture_limits
                )
                valid_texture_layouts = valid_texture_layouts.intersection(
                    arg_texture_layouts
                )
                can_use_buffers = can_use_buffers and utils.within_buffer_limit(
                    arg, self.buffer_limit
                )

        # If there are no valid texture memory layouts, then buffer storage must be
        # supported by the operator implementation.
        if len(valid_texture_layouts) == 0:
            if not can_use_buffers:
                return (
                    False,
                    f"op requires buffers that exceed the buffer limit ({self.buffer_limit})",
                )

            compatible = VkStorageType.BUFFER in features.supported_storage_types()
            reason = "op is compatible"
            if not compatible:
                reason = "op requires buffers which is not supported by op impl"
            return compatible, reason

        op_available_layouts = features.supported_memory_layouts(
            VkStorageType.TEXTURE_3D
        )

        is_compatible = any(
            layout in op_available_layouts for layout in valid_texture_layouts
        )
        if not is_compatible:
            return False, "Required texutre memory layout not supported"

        return is_compatible, "Op is compatible"

    def node_is_compatible(
        self, node: torch.fx.Node, features: Optional[OpFeatures] = None
    ) -> Tuple[bool, str]:
        # TODO(ssjia) support symbolic ints
        if utils.is_symint_node(node):
            return False, "symint node not supported yet"
        elif utils.is_tensor_node(node):
            return self.op_node_is_compatible(node, features=features)

        return False, f"Unsupported node type: {node.format_node()}"

    def is_linear_permute(self, node: torch.fx.Node) -> Tuple[bool, bool]:
        """
        Detect if a node is a permute/transpose that precedes a call to a `mm` or
        `addmm` operator. This node can be fused with the `mm` or `addmm` to produce a
        `linear` operator.

        This function returns two bool values:
        1. The first indicates if this node can be fused into a linear node
        2. The second indicates if the overall linear op can be executed with Vulkan

        The node will be partitioned only if both are true.
        """
        if node.target not in [
            exir_ops.edge.aten.t_copy.default,
            exir_ops.edge.aten.permute_copy.default,
        ]:
            return False, False

        if len(node.users) != 1:
            return False, False

        first_user = list(node.users.keys())[0]
        if first_user.target in [
            exir_ops.edge.aten.mm.default,
            exir_ops.edge.aten.addmm.default,
        ]:
            # Only mark this node if the target linear op is valid
            if self.node_is_compatible(first_user)[0]:
                return True, True
            else:
                return True, False

        return False, False

    def is_in_local_scalar_dense_chain(self, node: torch.fx.Node) -> Tuple[bool, bool]:
        """
        Scalar tensors are usually converted to scalar values in the graph via`
        scalar_tensor[0].item()` in Python, which translates to a chain of
        `local_scalar_dense(torch.select.int(scalar_tensor, 0, 0))` in the graph.
        This function marks the entire chain as supported by the Vulkan delegate.

        Later, within vulkan_preprocess there will be a graph transform which replaces
        the chain with passing in the scalar tensor directly.

        Similar to the `is_linear_permute` function, this function has 2 return values.
        """
        if node.target == exir_ops.edge.aten.select_copy.int:
            if len(node.users) != 1:
                return False, False
            # pyre-ignore
            if node.args[0].meta["val"].numel() != 1:
                return False, False

            local_scalar_dense = list(node.users.keys())[0]
            if local_scalar_dense.target != torch.ops.aten._local_scalar_dense.default:
                return False, False

            return self.is_in_local_scalar_dense_chain(local_scalar_dense)

        if node.target == torch.ops.aten._local_scalar_dense.default:
            return True, all(self.node_is_compatible(user)[0] for user in node.users)

        return False, False

    def log_skip(self, node: torch.fx.Node, reason: str) -> None:
        if node.op == "call_function":
            logger.info(
                f"[Vulkan Partitioner] Due to [{reason}], skipping {node.format_node()}"
            )

    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        r = self._is_node_supported(node)
        return r

    def _is_node_supported(self, node: torch.fx.Node) -> bool:
        target = node.target
        if node.target == torch.ops.higher_order.auto_functionalized:
            first_arg = node.args[0]
            assert isinstance(first_arg, torch._ops.OpOverload)
            target = first_arg.name()

        is_linear_permute, target_linear_is_compatible = self.is_linear_permute(node)
        if is_linear_permute and target_linear_is_compatible:
            return True
        elif is_linear_permute:
            # Skip so that the permute can be fused into a linear by another backend
            self.log_skip(node, "permute node of non compatible linear node")
            return False

        is_in_local_scalar_dense_chain, dst_node_is_compatible = (
            self.is_in_local_scalar_dense_chain(node)
        )
        if is_in_local_scalar_dense_chain and dst_node_is_compatible:
            return True
        elif is_in_local_scalar_dense_chain:
            self.log_skip(node, "local scalar dense of incompatible op node")
            return False

        if target not in vulkan_supported_ops:
            self.log_skip(node, "no operator implementation")
            return False

        features = vulkan_supported_ops[target]

        if not features.check_node_fn(node):
            self.log_skip(node, "op args not supported")
            return False

        if self.require_dynamic_shapes and not features.resize_fn:
            self.log_skip(node, "no dynamic shape support")
            return False

        is_compatible, reason = self.node_is_compatible(node, features=features)
        if not is_compatible:
            self.log_skip(node, reason)

        return is_compatible


def parse_compile_options(compile_options: Dict[str, Any]) -> List[CompileSpec]:
    compile_specs = []

    for key, value in compile_options.items():
        if isinstance(value, (VkStorageType, VkMemoryLayout)):
            value_bytes = int(value).to_bytes(4, byteorder="little")
            compile_specs.append(CompileSpec(key, value_bytes))

        if isinstance(value, bool):
            value_bytes = value.to_bytes(1, byteorder="little")
            compile_specs.append(CompileSpec(key, value_bytes))

        if key == "texture_limits":
            compile_specs.append(
                CompileSpec(
                    "texture_limits_x", int(value[0]).to_bytes(4, byteorder="little")
                )
            )
            compile_specs.append(
                CompileSpec(
                    "texture_limits_y", int(value[1]).to_bytes(4, byteorder="little")
                )
            )
            compile_specs.append(
                CompileSpec(
                    "texture_limits_z", int(value[2]).to_bytes(4, byteorder="little")
                )
            )

        # Unhandled options are ignored

    return compile_specs


@final
class VulkanPartitioner(Partitioner):
    def __init__(
        self,
        compile_options: Optional[Dict[str, Any]] = None,
    ) -> None:
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

        texture_limits: utils.ImageExtents = self.options.get(
            "texture_limits", utils.DEFAULT_TEXTURE_LIMITS
        )
        buffer_limit: int = self.options.get("buffer_limit", utils.DEFAULT_BUFFER_LIMIT)
        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            VulkanSupportedOperators(
                texture_limits,
                buffer_limit,
                require_dynamic_shape=self.options.get("require_dynamic_shapes", False),
            ),
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
