# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Any, Callable, Dict, final, List, Mapping, Optional, Set, Tuple

import executorch.backends.vulkan.patterns as vk_patterns
import executorch.backends.vulkan.utils as utils

import torch

from executorch.backends.vulkan.op_registry import (
    get_op_features,
    has_impl,
    OpFeatures,
    OpKey,
    vulkan_supported_ops,
)

from executorch.backends.vulkan.patterns import PatternMatch

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
from executorch.exir.backend.utils import tag_constant_data, tag_mutated_buffer
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
        skip_bool_tensors: bool = False,
        operator_blocklist: Optional[Set[OpKey]] = None,
        operator_allowlist: Optional[Set[OpKey]] = None,
        fusable_subgraphs: Optional[List[PatternMatch]] = None,
        nn_module_blocklist: Optional[Set[str]] = None,
        nn_module_allowlist: Optional[Set[str]] = None,
    ) -> None:
        super().__init__()
        self.texture_limits: utils.ImageExtents = texture_limits
        self.buffer_limit = buffer_limit
        self.require_dynamic_shapes = require_dynamic_shape
        self.skip_bool_tensors = skip_bool_tensors
        self.operator_blocklist: Set[OpKey] = (
            operator_blocklist if operator_blocklist is not None else set()
        )
        self.operator_allowlist = operator_allowlist
        self.fusable_subgraphs: List[PatternMatch] = (
            fusable_subgraphs if fusable_subgraphs is not None else []
        )
        # Create a set of all nodes that are part of fusable subgraphs for quick lookup
        self.fusable_nodes: Set[torch.fx.Node] = set()
        for match in self.fusable_subgraphs:
            self.fusable_nodes.update(match.all_nodes)

        self.nn_module_blocklist = nn_module_blocklist
        self.nn_module_allowlist = nn_module_allowlist

    def op_node_is_compatible(  # noqa: C901: Function is too complex
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
        if (
            node.target == torch.ops.higher_order.auto_functionalized
            or node.target == torch.ops.higher_order.auto_functionalized_v2
        ):
            first_arg = node.args[0]
            assert isinstance(first_arg, torch._ops.OpOverload)
            target = first_arg.name()

        # Operator allow list is only used for torch ops
        if (
            utils.is_torch_op_node(node)
            and (self.operator_allowlist is not None)
            and (target not in self.operator_allowlist)
        ):
            return False, "op is not in allowlist"

        if target in self.operator_blocklist:
            return False, "op is in blocklist"

        # Extract the features for the node's operator, if no override was provided
        if features is None:
            if not has_impl(target):
                return False, "no operator implementation"
            features = get_op_features(target)

        # bool tensors are internally represented with int8 buffers, which may not be
        # supported by some GPUs. Therefore, provide the option to skip these tensors.
        if self.skip_bool_tensors and utils.op_contains_bool_tensor(node):
            return False, f"op {utils.node_io_str(node)} contains bool tensor"

        # Get the possible tensor representations for each tensor participating in the
        # this operator. Then check that all tensors are representable as either a
        # buffer or texture.
        op_repsets: utils.OpRepSets = features.make_op_repsets(
            node, self.texture_limits
        )

        if op_repsets.any_is_empty():
            return (
                False,
                f"no valid representations for op {utils.node_io_str(node)}",
            )

        return True, "Op is compatible"

    def node_is_compatible(
        self, node: torch.fx.Node, features: Optional[OpFeatures] = None
    ) -> Tuple[bool, str]:
        if utils.is_tensor_node(node):
            return self.op_node_is_compatible(node, features=features)
        # For non-tensor nodes, just check if the op is registered
        elif hasattr(node, "target"):
            return node.target in vulkan_supported_ops, "Op is compatible"

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

    def log_skip(self, node: torch.fx.Node, reason: str) -> None:
        if node.op == "call_function":
            logger.info(
                f"[Vulkan Partitioner] Due to [{reason}], skipping {utils.node_io_str(node)}"
            )

    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        r = self._is_node_supported(node)
        return r

    def _is_node_supported(self, node: torch.fx.Node) -> bool:  # noqa: C901
        # Check if tensor node dtype is supported by vulkan
        if utils.is_tensor_node(node) and not utils.io_dtypes_are_supported(node):
            self.log_skip(node, "dtype not supported")
            return False

        if node.op == "call_function":
            # Apply nn module allowlist and blocklist
            if self.nn_module_allowlist is not None:
                if not utils.node_comes_from_any_nn_module_in_set(
                    node, self.nn_module_allowlist
                ):
                    self.log_skip(node, "source nn.Module is not in allowlist")
                    return False

            if self.nn_module_blocklist is not None:
                if utils.node_comes_from_any_nn_module_in_set(
                    node, self.nn_module_blocklist
                ):
                    self.log_skip(node, "source nn.Module is in blocklist")
                    return False

            # Check if this node is part of a fusable subgraph
            if node in self.fusable_nodes:
                return True

        target = node.target
        if (
            node.target == torch.ops.higher_order.auto_functionalized
            or node.target == torch.ops.higher_order.auto_functionalized_v2
        ):
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

        features = None
        if target not in vulkan_supported_ops:
            # For some ops, i.e. custom ops the name is registered instead of the
            # OpOverload object.
            if hasattr(target, "name") and target.name() in vulkan_supported_ops:
                features = vulkan_supported_ops[target.name()]
            else:
                self.log_skip(node, "no operator implementation")
                return False
        else:
            features = vulkan_supported_ops[target]

        assert features is not None

        if not features.are_node_inputs_supported_fn(node):
            self.log_skip(node, "op args not supported")
            return False

        if self.require_dynamic_shapes and not features.supports_resize:
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
        operator_blocklist: Optional[List[OpKey]] = None,
        operator_allowlist: Optional[List[OpKey]] = None,
        nn_module_blocklist: Optional[List[str]] = None,
        nn_module_allowlist: Optional[List[str]] = None,
    ) -> None:
        self.options: Dict[str, Any] = {}
        if compile_options is not None:
            self.options = compile_options

        compile_spec = parse_compile_options(self.options)
        self.delegation_spec = DelegationSpec(VulkanBackend.__name__, compile_spec)

        self.operator_blocklist: Set[OpKey] = set()
        if operator_blocklist is not None:
            for entry in operator_blocklist or []:
                self.operator_blocklist.add(entry)

        self.operator_allowlist: Optional[Set[OpKey]] = None
        if operator_allowlist is not None:
            self.operator_allowlist = set()
            for entry in operator_allowlist:
                assert self.operator_allowlist is not None
                self.operator_allowlist.add(entry)

        self.nn_module_blocklist: Optional[Set[str]] = None
        if nn_module_blocklist is not None:
            self.nn_module_blocklist = set()
            for entry in nn_module_blocklist or []:
                assert self.nn_module_blocklist is not None
                self.nn_module_blocklist.add(entry)

        self.nn_module_allowlist: Optional[Set[str]] = None
        if nn_module_allowlist is not None:
            self.nn_module_allowlist = set()
            for entry in nn_module_allowlist:
                assert self.nn_module_allowlist is not None
                self.nn_module_allowlist.add(entry)

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        def filter_fn(node: torch.fx.Node) -> bool:
            return True

        return (ops_not_to_decompose, filter_fn)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        partition_tags = {}

        # Get all fusable subgraphs from fuse_patterns
        fusable_subgraphs = vk_patterns.get_all_fusable_subgraphs(
            exported_program.graph_module
        )

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
                skip_bool_tensors=self.options.get("skip_bool_tensors", False),
                operator_blocklist=self.operator_blocklist,
                operator_allowlist=self.operator_allowlist,
                fusable_subgraphs=fusable_subgraphs,
                nn_module_blocklist=self.nn_module_blocklist,
                nn_module_allowlist=self.nn_module_allowlist,
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
        tag_mutated_buffer(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
