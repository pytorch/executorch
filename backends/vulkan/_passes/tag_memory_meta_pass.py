# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator
from typing import Any

import executorch.backends.vulkan.utils as utils
import torch
from executorch.backends.vulkan.op_registry import get_op_features, has_impl, OpFeatures
from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
    VkMemoryLayout,
    VkStorageType,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.tensor import TensorSpec

logger: logging.Logger = logging.getLogger("")
logger.setLevel(logging.INFO)


def insert_transition_node(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
    arg: torch.fx.Node,
    arg_node_repr: utils.TensorRepr,
) -> None:
    """
    Insert a clone node to transition the tensor associated with `arg` to a tensor with
    the requested representation `arg_node_repr`, and use the cloned node as an argument
    to `node` instead of `arg`.
    """
    with graph_module.graph.inserting_before(node):
        clone_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.aten.clone.default,
            (arg,),
        )
        clone_node.meta["val"] = arg.meta["val"]
        clone_node.meta["spec"] = TensorSpec.from_tensor(clone_node.meta["val"])
        clone_node.meta["spec"].const = False
        utils.set_node_repr(clone_node, arg_node_repr)
        arg.replace_all_uses_with(clone_node, lambda x, y=node: x == y)


def set_arg_node_repr_or_transition(
    graph_module: torch.fx.GraphModule,
    op_node: torch.fx.Node,
    arg_i: int,
    arg_node_repr: utils.TensorRepr,
    dirty: bool,
) -> bool:
    """
    Does one of following:
    1. Sets the `node_repr` of the argument at `arg_i` of `op_node` if the argument node
       does not currently have a `node_repr`
    2. No-op if the current `node_repr` is already the same as the requested represetnation.
    3. Insert a transition node to create a copy of the argument with the desired `node_repr`
       if the current `node_repr` is different than what is needed.
    """
    arg_node = op_node.args[arg_i]

    def single_node_impl(node: torch.fx.Node) -> bool:
        # Case where the arg node has not been touched yet; in this case, simply set it and
        # return.
        if not utils.has_node_repr(node):
            utils.set_node_repr(node, arg_node_repr)
            return False

        # Case where the current node representation is the same as the new one.
        cur_node_repr = utils.get_node_repr(node)
        assert isinstance(cur_node_repr, utils.TensorRepr)

        if cur_node_repr == arg_node_repr:
            return False

        if not dirty:
            logger.info(
                f"[Vulkan Delegate] Inserting transition(s) for {op_node.format_node()}:"
            )

        # Existing node representation is different; insert a transition node
        # Currently, the transition node insertion logic can only handle single tensor nodes
        assert utils.is_single_tensor_node(node)
        insert_transition_node(graph_module, op_node, node, arg_node_repr)

        logger.info(f"   arg {arg_i} ({node}): ({cur_node_repr}) -> ({arg_node_repr})")

        return True

    if isinstance(arg_node, torch.fx.Node):
        return single_node_impl(arg_node)
    elif isinstance(arg_node, (list, tuple)):
        ret: bool = False
        for n in arg_node:
            assert isinstance(n, torch.fx.Node)
            assert utils.is_single_tensor_node(n)
            ret = single_node_impl(n) or ret

        return ret

    raise NotImplementedError(f"Unhandled node type {arg_node}")


class TagMemoryMetaPass(ExportPass):
    """
    Operator implementations in the Vulkan delegate may require that input and output
    tensors use a specific representation. Representation in this case refers to a
    combination of storage type (buffer or texture) and memory layout (width, height, or
    channels packed).

    The tag memory metadata pass is responsible for marking each tensor in the graph
    with the appropriate representation to use. It is also responsible for inserting
    operators to transition argument tensors to a required/compatible representation if
    a mismatch has been detected.
    """

    def __init__(
        self,
        texture_limits: utils.ImageExtents,
        default_storage_type: VkStorageType = VkStorageType.TEXTURE_3D,
        default_memory_layout: VkMemoryLayout = VkMemoryLayout.TENSOR_WIDTH_PACKED,
        force_fp16: bool = False,
    ):
        super().__init__()
        self.default_storage: VkStorageType = default_storage_type
        self.default_layout: VkMemoryLayout = default_memory_layout
        self.texture_limits = texture_limits
        self.force_fp16 = force_fp16

        # Magic number to limit "lookahead" when tracing through users of an operator
        # to constrain the representation of its arguments/outputs.
        self.max_trace_search_depth = None

    def is_valid_op_node(self, node: Any) -> bool:
        """
        Fails the check for:
        * nodes that are not associated with a tensor
        * nodes that are associated with a constant tensor
        * nodes that are not associated with a supported operator
        """
        if not isinstance(node, torch.fx.Node) or not utils.is_tensor_node(node):
            return False
        if node.meta.get("etvk_tensorref", False):
            return False
        if not has_impl(node.target):
            return False

        return True

    def is_non_constant_tensor_node(self, node: Any) -> bool:
        """
        Fails the check for:
        * Nodes that are not associated with tensor values
        * Nodes associated with constant tensors
        *
        """
        if isinstance(node, torch.fx.Node):
            if not utils.is_tensor_node(node):
                return False
            if node.meta.get("etvk_tensorref", False):
                return False
            return True

        if isinstance(node, (tuple, list)):
            for n in node:
                if not isinstance(n, torch.fx.Node):
                    return False
                if not self.is_non_constant_tensor_node(n):
                    return False

            return True

        # Return false by default
        return False

    def get_node_cached_repsets(self, op_node: torch.fx.Node) -> utils.OpRepSets:
        """
        Implements a cache layer for getting the OpRepSets for a given operator node.
        """
        assert self.is_valid_op_node(op_node)

        if "etvk_node_repsets" in op_node.meta:
            op_repsets = op_node.meta["etvk_node_repsets"]
            assert isinstance(op_repsets, utils.OpRepSets)
            return op_repsets
        else:
            # Special case for getitem - set the input and output to the repset of the
            # tensor value being extracted
            if op_node.target == operator.getitem:
                src_node = op_node.args[0]
                assert isinstance(src_node, torch.fx.Node)
                idx = op_node.args[1]
                assert isinstance(idx, int)

                arg_node_repsets = self.get_node_cached_repsets(src_node)
                out_tensor_repset = arg_node_repsets.get_out_repset(idx)

                op_repsets = utils.OpRepSets(
                    utils.TensorRepSetList(out_tensor_repset),
                    utils.TensorRepSetList(out_tensor_repset),
                    op_node,
                    self.texture_limits,
                )
            else:
                features: OpFeatures = get_op_features(op_node.target)  # noqa
                op_repsets = features.make_op_repsets(op_node, self.texture_limits)

            op_node.meta["etvk_node_repsets"] = op_repsets
            return op_repsets

    def get_arg_tensor_source_repset(
        self, op_node: torch.fx.Node, arg_i: int
    ) -> utils.TensorRepSet:
        """
        Get the "source RepSet" for the tensor argument at index `arg_i` of `op_node`.
        The source repset is obtained in one of two ways:

        1. If the tensor argument already has a representation determined for it, return
           a repset that contains that representation.
        2. Otherwise, return the output repset of the operator that produces the tensor
        """
        arg_node = op_node.args[arg_i]

        # For non-tensor arguments, return ALL_STORAGES_REPSET so that the respset does
        # not appear to be empty.
        if not utils.is_tensor_arg_node(arg_node):
            return utils.ALL_STORAGES_REPSET

        # Special case for cat - use the first tensor in the list as representative
        if isinstance(arg_node, list):
            arg_node = arg_node[0]

        if utils.has_node_repr(arg_node):
            arg_node_repr = utils.get_node_repr(arg_node)
            assert isinstance(arg_node_repr, utils.TensorRepr)
            return utils.make_tensor_repset(arg_node_repr)
        elif self.is_valid_op_node(arg_node):
            # Special case for getitem - propagate the node representation of the original node
            if op_node.target == operator.getitem:
                src_node = op_node.args[0]
                assert isinstance(src_node, torch.fx.Node)
                idx = op_node.args[1]
                assert isinstance(idx, int)

                src_node_repsets = self.get_node_cached_repsets(src_node)
                return src_node_repsets.get_out_repset(idx)

            src_node_repsets = self.get_node_cached_repsets(arg_node)
            return src_node_repsets.get_out_repset(0)

        # default return
        return utils.ANY_STORAGE

    def constrain_repset_with_user(
        self,
        current_node: torch.fx.Node,
        arg_i: int,
        arg_repset: utils.TensorRepSet,
        search_depth: int = 0,
    ) -> utils.TensorRepSet:
        """
        Attempts to constrain `arg_repset` based on the required repset of the argument
        at index `arg_i` of `current_node`. This tries to find a representation for the
        argument that can be used for as long as possible without needing a transition.
        """
        # The repset is already constrained; return it
        if arg_repset.is_constrained():
            return arg_repset

        # The current node is not a valid op node, so no OpRepSets object can be created
        # for it.
        if not self.is_valid_op_node(current_node):
            return arg_repset

        cur_node_repsets = self.get_node_cached_repsets(current_node)

        # Intersect with the repset required by the current operator; otherwise, return
        # since a transition will be required anyways
        req_arg_repset = cur_node_repsets.get_arg_repset(arg_i)
        if req_arg_repset.any_in_common(arg_repset):
            arg_repset = arg_repset.make_intersect(req_arg_repset)
        else:
            return arg_repset

        # Check if the argument at `arg_i` will influence the output representation of
        # the current operator.
        repset_propagates_to_output = cur_node_repsets.sync_primary_io_repr and (
            cur_node_repsets.sync_args_repr or arg_i == cur_node_repsets.primary_arg_idx
        )

        # If not, then no point in continuing to trace the users of the current node
        if not repset_propagates_to_output:
            return arg_repset

        return self.trace_node_users_to_constrain_repset(
            current_node, arg_repset, search_depth
        )

    def trace_node_users_to_constrain_repset(
        self,
        origin_node: torch.fx.Node,
        repset: utils.TensorRepSet,
        search_depth: int = 0,
    ) -> utils.TensorRepSet:
        """
        For an ambiguous repset, try to constrain the repset by tracing the required
        repsets of the users of `origin_node`. The idea is to try to find a representation
        that can be used the longest without needing user nodes to insert a transition
        for its arguments.
        """
        # Optionally limit the search depth to improve export time
        if self.max_trace_search_depth is not None:
            if search_depth > self.max_trace_search_depth:
                return repset

        users_to_trace = origin_node.users

        sync_outs_repr = True
        if self.is_valid_op_node(origin_node):
            sync_outs_repr = self.get_node_cached_repsets(origin_node).sync_outs_repr

        if utils.num_tensors_in_node(origin_node) > 1 and not sync_outs_repr:
            users_to_trace = []
            for usage_node in origin_node.users:
                if usage_node.target == operator.getitem and usage_node.args[1] == 1:
                    users_to_trace.append(usage_node)

        for usage_node in users_to_trace:
            arg_i_in_user = None
            for i in range(len(usage_node.args)):
                if origin_node == usage_node.args[i]:
                    arg_i_in_user = i
                    break

            if arg_i_in_user is not None:
                repset = self.constrain_repset_with_user(
                    usage_node, arg_i_in_user, repset, search_depth + 1
                )

            if repset.is_constrained():
                return repset

        return repset

    def constrain_op_arg_repset(self, arg_i: int, op_repsets: utils.OpRepSets) -> None:
        """
        Attempts to constrain the repset of the argument at index `arg_i` of the op
        associated with `op_repsets`. Does this with two stages:

        1. First, account for any existing representation that has already been determined
           for the argument. If no existing representation has been determined, then use
           the output repset of the operator that produces the argument.
        2. Then, try to trace through the users of the argument to find a representation
           that can be used for as long as possible without needing a transition.
        """
        # If forcing fp16, then try to use texture storage whenever possible. This is
        # a temporary stopgap measure until all buffer implementations properly account
        # for potential overflow of fp16 representation range when doing math in fp16.
        if self.force_fp16:
            op_repsets.try_constrain_with_arg_repset(arg_i, utils.ANY_TEXTURE)

        arg_source_repset = self.get_arg_tensor_source_repset(op_repsets.op_node, arg_i)
        op_repsets.try_constrain_with_arg_repset(arg_i, arg_source_repset)

        arg_repset = op_repsets.get_arg_repset(arg_i)
        if arg_repset.is_constrained():
            return

        arg_node = op_repsets.op_node.args[arg_i]

        if isinstance(arg_node, list):
            arg_node = arg_node[0]

        arg_repset = self.trace_node_users_to_constrain_repset(arg_node, arg_repset)
        op_repsets.try_constrain_with_arg_repset(arg_i, arg_repset)

    def constrain_op_out_repset(self, op_repsets: utils.OpRepSets) -> None:
        """
        Similar to the `constrain_op_arg_repset` function, but for the output repset of
        the operator.
        """
        out_repset = op_repsets.get_out_repset(0)
        if out_repset.is_constrained():
            return

        op_node = op_repsets.op_node
        out_respset = self.trace_node_users_to_constrain_repset(op_node, out_repset)

        op_repsets.try_constrain_with_out_repset(out_respset)

    def constrain_op_repsets(self, op_repsets: utils.OpRepSets) -> None:
        # For most ops, constraining the argument repsets will also contrain the output
        # repset due to OpRepSets maintaining synchronization rules.
        for i in range(len(op_repsets.op_node.args)):
            if utils.is_tensor_arg_node(op_repsets.op_node.args[i]):
                self.constrain_op_arg_repset(i, op_repsets)

        # However, some operators do not sync input and output representations and also
        # define ambiguous repsets for the output tensor(s). In those cases we will need
        # to execute additional logic to constrain the output repsets separately from
        # the input repsets.
        if not op_repsets.sync_primary_io_repr and op_repsets.sync_outs_repr:
            self.constrain_op_out_repset(op_repsets)

    def set_op_node_tensor_reprs(
        self, graph_module: torch.fx.GraphModule, op_node: torch.fx.Node
    ) -> None:
        """
        For an operator representated by `op_node`, get the OpRepSets associated with
        the operation and try to constrain the repsets by accounting for existing
        representations and tracing through the users of the operator.

        Then, determine a tensor representation for all tensors participating in the
        operation and mark it in the node metadata. If the requested representation is
        different than an already determined representation, then insert a transition
        node to create a copy of the tensor with the desired representation.
        """
        if not self.is_valid_op_node(op_node):
            return

        # Special case for getitem - propagate the node representation of the original node
        if op_node.target == operator.getitem:
            src_node = op_node.args[0]
            assert isinstance(src_node, torch.fx.Node)
            idx = op_node.args[1]
            assert isinstance(idx, int)

            arg_node_repr = utils.get_node_repr(src_node)
            assert isinstance(arg_node_repr, list)
            utils.set_node_repr(op_node, arg_node_repr[idx])
            return

        # Get a "fresh" OpRepSets object instead of using the cache. Do this because this
        # class instance will go through the constraining process which may modify it.
        features: OpFeatures = get_op_features(op_node.target)
        op_repsets = features.make_op_repsets(op_node, self.texture_limits)

        self.constrain_op_repsets(op_repsets)

        args_repr_list, outs_repr_list = op_repsets.pick_representations()

        if len(outs_repr_list) == 1:
            utils.set_node_repr(op_node, outs_repr_list[0])
        else:
            utils.set_node_repr(op_node, outs_repr_list)

        transitions_inserted = False
        for i, arg_node in enumerate(op_node.args):
            if not self.is_non_constant_tensor_node(arg_node):
                continue

            arg_node_repr = args_repr_list[i]

            if isinstance(arg_node, torch.fx.Node):
                transitions_inserted = (
                    set_arg_node_repr_or_transition(
                        graph_module, op_node, i, arg_node_repr, transitions_inserted
                    )
                    or transitions_inserted
                )
            elif isinstance(arg_node, (list, tuple)):
                for n in arg_node:
                    assert isinstance(n, torch.fx.Node)
                    assert utils.is_single_tensor_node(n)
                    transitions_inserted = (
                        set_arg_node_repr_or_transition(
                            graph_module,
                            op_node,
                            i,
                            arg_node_repr,
                            transitions_inserted,
                        )
                        or transitions_inserted
                    )

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            self.set_op_node_tensor_reprs(graph_module, node)

        return PassResult(graph_module, True)
