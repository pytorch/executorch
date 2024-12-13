# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from copy import deepcopy
from typing import Set

import executorch.backends.vulkan.utils as utils

import torch

from executorch.backends.vulkan.op_registry import get_op_features, has_impl

from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
    VkMemoryLayout,
    VkStorageType,
)

from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.pass_base import ExportPass, PassResult

from torch.fx.passes.tools_common import NodeList
from torch.fx.passes.utils.fuser_utils import topo_sort

logger: logging.Logger = logging.getLogger("")
logger.setLevel(logging.INFO)


def set_memory_metadata(
    node: torch.fx.Node, storage: VkStorageType, layout: VkMemoryLayout
) -> None:
    utils.set_node_spec_attr(node, "vk_storage_type", storage)
    utils.set_node_spec_attr(node, "vk_memory_layout", layout)


def insert_transition_node(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
    arg: torch.fx.Node,
    storage: VkStorageType,
    layout: VkMemoryLayout,
) -> None:
    """
    Insert a clone node to copy the original tensor to a tensor with the desired storage
    type and memory layout.
    """
    with graph_module.graph.inserting_before(node):
        clone_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.aten.clone.default,
            (arg,),
        )
        clone_node.meta["val"] = arg.meta["val"]
        clone_node.meta["spec"] = deepcopy(arg.meta["spec"])
        clone_node.meta["spec"].const = False
        set_memory_metadata(clone_node, storage, layout)
        arg.replace_all_uses_with(clone_node, lambda x, y=node: x == y)


class TagMemoryMetaPass(ExportPass):
    """
    There are a variety of ways that tensors can be represented in Vulkan. The two main
    descriptors for how a tensor is laid out in memory is:

    1. Storage Type (buffer or texture)
    2. Memory Layout (which dim is packed along a texel / has a stride of 1, etc.)

    Due to the differences between buffers and textures, and the differences between
    different memory layouts, an implementation for an operator may only support a
    specific set of (storage type, memory layout) combinations.

    Furthermore, if an operator implementation supports multiple (storage type, memory
    layout) combinations, there may be a "preferred" setting which results in optimal
    performance.

    This pass is responsible for ensuring that all tensors participating in an operator
    call have a valid/optimal (storage type, memory layout) setting, and insert
    transition operators to transfer input tensors to the correct memory settings when
    necessary.
    """

    def __init__(
        self,
        texture_limits: utils.ImageExtents,
        default_storage_type: VkStorageType = VkStorageType.TEXTURE_3D,
        default_memory_layout: VkMemoryLayout = VkMemoryLayout.TENSOR_WIDTH_PACKED,
    ):
        super().__init__()
        self.default_storage: VkStorageType = default_storage_type
        self.default_layout: VkMemoryLayout = default_memory_layout
        self.texture_limits = texture_limits

    def propose_node_storage(
        self,
        node: torch.fx.Node,
    ) -> VkStorageType:
        """
        Uses the operator registry to determine the storage type that should be used for
        a given node. The storage type is determined with the following priorities:
        1. In some cases, a tensor involved in the computation may be too large to be
           represented as a texture. If this is the case, the node is "opinionated" and
           buffer representation must be used.
        1. If the operator called by the node indicates an optimal storage type, or only
           supports a single storage type, use that storage type. If either is true,
           then the node is considered to be opinionated as well. If multiple storage
           and no preferred storage type is indicated, then the node is not opinionated;
           go to the next step.
        2. If the node's arguments already have memory metadata annotations, then
           preserve the settings of the first argument. Otherwise, proceed to the next
           step.
        3. Recursively search the node's uses to see if any subsequent uses are
           opinionated; inherit the settings of the first opinionated node. If no
           opinionated user can be found, then proceed to the last step.
        4. Use the default storage type setting.
        """
        # The node may have an input/output tensor that is too big to be stored in a
        # texture. In this case, buffer storage must be used. Note that the partitioner
        # has already checked for the fact that buffer storage is supported by the
        # operator.
        if len(utils.possible_node_memory_layouts(node, self.texture_limits)) == 0:
            return VkStorageType.BUFFER

        valid_storage_types: Set[VkStorageType] = utils.all_storage_types

        # pyre-ignore
        if has_impl(node.target):
            # pyre-ignore
            features = get_op_features(node.target)
            valid_storage_types = features.supported_storage_types()
            storage = features.propose_storage_type()
            if storage is not None:
                return storage

        for arg in node.args:
            if isinstance(arg, torch.fx.Node) and utils.is_tensor_node(arg):
                storage = utils.get_node_storage_type(arg)
                if storage is not None and storage in valid_storage_types:
                    return storage

        # If no storage type has been resolved yet, assume the optimal storage type of
        # the first opinionated user. This search is recursive.
        for user in node.users:
            optimal_storage = self.propose_node_storage(user)
            if optimal_storage is not None:
                return optimal_storage

        if self.default_storage in valid_storage_types:
            return self.default_storage
        else:
            return next(iter(valid_storage_types))

    def propose_node_layout(
        self,
        node: torch.fx.Node,
        storage: VkStorageType,
    ) -> VkMemoryLayout:
        """
        Performs the same steps as propose_node_storage, but detects the memory layout
        that should be used for the specific storage type. The same prioritization logic
        is applied.
        """
        valid_layouts: Set[VkMemoryLayout] = utils.all_memory_layouts
        # pyre-ignore
        if has_impl(node.target):
            # pyre-ignore
            features = get_op_features(node.target)
            valid_layouts = features.supported_memory_layouts(storage)
            layout = features.propose_memory_layout(storage)
            if layout is not None:
                return layout

        for arg in node.args:
            if isinstance(arg, torch.fx.Node) and utils.is_tensor_node(arg):
                layout = utils.get_node_memory_layout(arg)
                if layout is not None and layout in valid_layouts:
                    return layout

        # If no storage type has been resolved yet, assume the optimal storage type of
        # the first opinionated user. This search is recursive.
        for user in node.users:
            optimal_storage = self.propose_node_layout(user, storage)
            if optimal_storage is not None:
                return optimal_storage

        # As a last resort, return the default storage type that should be used.
        if self.default_layout in valid_layouts:
            return self.default_layout
        else:
            return next(iter(valid_layouts))

    def should_annotate(self, node) -> bool:
        if not isinstance(node, torch.fx.Node):
            return False

        if not utils.is_tensor_node(node):
            return False

        # Storage type and memory layout for tensorref will be determined at runtime
        # so there's no use in setting those attributes ahead of time.
        if node.meta.get("vkdg_tensorref", False):
            return False

        # Skip annotating output node. The output tensors should be annotated by the
        # time the output node is observed.
        if node.op == "output":
            return False

        return True

    def should_delay_annotation(self, node: torch.fx.Node) -> bool:
        # For prepack nodes, delay setting the storage type and memory layout as long as
        # possible. This is to minimize the number of transitions, since it can be
        # difficult to predict what storage type and memory layout should be used at the
        # time the prepack node is observed.
        return node.target == exir_ops.edge.et_vk.prepack.default

    # noqa
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        sorted_nodes: NodeList = topo_sort(list(graph_module.graph.nodes))

        for node in sorted_nodes:
            if not self.should_annotate(node) or self.should_delay_annotation(node):
                continue

            storage = self.propose_node_storage(node)
            layout = self.propose_node_layout(node, storage)

            set_memory_metadata(node, storage, layout)

            inserting_transitions_for_node = False
            for i, arg in enumerate(node.args):
                if not self.should_annotate(arg):
                    continue

                assert isinstance(arg, torch.fx.Node)

                arg_storage = utils.get_node_storage_type(arg)
                arg_layout = utils.get_node_memory_layout(arg)

                if arg_storage is None:
                    utils.set_node_spec_attr(arg, "vk_storage_type", storage)
                    arg_storage = storage
                if arg_layout is None:
                    utils.set_node_spec_attr(arg, "vk_memory_layout", layout)
                    arg_layout = layout

                if arg_storage == storage and arg_layout == layout:
                    continue

                if not inserting_transitions_for_node:
                    inserting_transitions_for_node = True
                    logger.info(
                        f"[Vulkan Delegate] Inserting transition(s) for {node.format_node()}:"
                    )

                insert_transition_node(graph_module, node, arg, storage, layout)

                logger.info(
                    f"   args {i} ({arg}): ({arg_storage}, {arg_layout}) -> ({storage}, {layout})"
                )

        return PassResult(graph_module, True)
