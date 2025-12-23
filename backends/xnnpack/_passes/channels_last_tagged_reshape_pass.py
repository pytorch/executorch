# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from enum import Enum
from typing import Optional, Tuple

import torch
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.utils.quant_utils import (
    is_dequant,
    is_dynamic_qdq,
    is_tagged_as_implicit_q_dq,
    tag_as_implicit_q_dq,
)
from executorch.backends.xnnpack.utils.utils import is_param_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


class InputDimOrder(Enum):
    NCHW = 1
    NHWC = 2


# TODO(T151254305) use subgraph_rewriter
class ChannelsLastTaggedReshapePass(XNNPACKPass):
    """
    This pass is Internal to XNNPACK only! It is meant to give a new representation
    of the edge graph to be consumed by XNNPACK Preprocess. All added operators
    will be consumed by delegate and turned to delegate blobs.

    Edge IR graph pass to add operator stubs that signal a change in
    memory format from contiguous to channels last. This is to help with
    XNNPACK Delegate to add transpose nodes to change input memory format
    at runtime and run operators in Channels Last Format.

    During this pass, nhwc nodes are not converted back to nchw immediately.
    Instead, they are tagged as nhwc, and this tag is propagated to their
    children which are also kept in nwhc format until a node which requires
    converting back to nchw is encountered.

    For example, Convolution requires inputs to be  NHWC so at runtime if
    the input tensor is NCHW then we need to transpose it to NHWC before
    feeding it to convolution.

    Before pass:
        Input(NCHW) --> Conv --> Output(NCHW)

    After pass:
        Input(NCHW) --> Transpose(NHWC) -->  Conv --> (NHWC)Transpose --> Output(NCHW)
    """

    # Set of ops that require memory format to be channels last (NHWC)
    memory_sensitive_ops_nhwc = {
        exir_ops.edge.aten.convolution.default,
        exir_ops.edge.aten.upsample_bilinear2d.vec,
        exir_ops.edge.aten.mean.dim,
        exir_ops.edge.aten.max_pool2d.default,
        exir_ops.edge.aten.amax.default,
        exir_ops.edge.aten.max.dim,
        exir_ops.edge.aten.avg_pool2d.default,
        exir_ops.edge.aten.prelu.default,
    }

    # Set of ops that require memory format to be NCHW
    memory_sensitive_ops_nchw = {
        exir_ops.edge.aten.squeeze_copy.dim,
        exir_ops.edge.aten.unsqueeze_copy.default,
        exir_ops.edge.aten.linear.default,
    }

    # Tag which is added to a node's meta to indicate that it uses NHWC format.
    # A constant data tensor with this tag assigned for use in a particular
    # format in one place cannot be used in other places in the other format
    # (The only case where we can have this tag set to True for a node is when
    # that node is a constant data tensor and it is being used in NCHW format
    # somewhere)
    XNN_NHWC_NODE = "XNN_NHWC_NODE"

    # A node's partner node is the node in the graph gives the same output but
    # in opposing memory format. i.e. An nhwc node's partner node uses nchw
    # format and vice versa. This key is used to store partner nodes in nodes'
    # meta so if we ever need to conver the same node's memory format multiple
    # times, rather than having multiple copy nodes we only need to have one.
    # We remove the partner node and key from the node's meta after this pass
    # is done
    PARTNER_NODE = "XNN_CHANNELS_LAST_TAGGED_RESHAPE_PARTNER_NODE"

    @staticmethod
    def mark_as_nhwc_node(node: torch.fx.Node) -> None:
        node.meta[ChannelsLastTaggedReshapePass.XNN_NHWC_NODE] = True

    @staticmethod
    def mark_as_nchw_node(node: torch.fx.Node) -> None:
        node.meta[ChannelsLastTaggedReshapePass.XNN_NHWC_NODE] = False

    def tag_node(self, node: torch.fx.Node) -> None:
        if node.kwargs["memory_format"] == torch.channels_last:
            self.mark_as_nhwc_node(node)
        else:
            self.mark_as_nchw_node(node)

    @staticmethod
    def is_nhwc_node(node: torch.fx.Node) -> bool:
        if is_dequant(node) and len(node.all_input_nodes) > 0:
            quantize_node = node.args[0]
            if len(quantize_node.all_input_nodes) > 0:
                actual_node = quantize_node.args[0]
                if actual_node.op == "placeholder":
                    return ChannelsLastTaggedReshapePass._is_nhwc_tensor(
                        actual_node.meta["val"][0]
                    )
                else:
                    return actual_node.meta.get(
                        ChannelsLastTaggedReshapePass.XNN_NHWC_NODE, False
                    )

        return node.meta.get(ChannelsLastTaggedReshapePass.XNN_NHWC_NODE, False)

    @staticmethod
    def is_nchw_node(node: torch.fx.Node) -> bool:
        if is_dequant(node) and len(node.all_input_nodes) > 0:
            quantize_node = node.args[0]
            if len(quantize_node.all_input_nodes) > 0:
                actual_node = quantize_node.args[0]
                if actual_node.op == "placeholder":
                    return not ChannelsLastTaggedReshapePass._is_nhwc_tensor(
                        actual_node.meta["val"][0]
                    )
                else:
                    return not actual_node.meta.get(
                        ChannelsLastTaggedReshapePass.XNN_NHWC_NODE, False
                    )

        return not ChannelsLastTaggedReshapePass.is_nhwc_node(node)

    @staticmethod
    def _is_nhwc_tensor(tensor: torch.Tensor) -> bool:
        nhwc = tensor.is_contiguous(memory_format=torch.channels_last)
        nchw = tensor.is_contiguous()
        # if both are true false
        # if  both nchw and nhwc are true
        #     then we want to see this is nchw hence return false
        # if either of nchw or nhwc is false, then just rely on hwc
        # if both are false, mayb channels_last_3d, then return nhwc
        #    however this should not happen here
        # return (not (nchw and nhwc)) and nhwc
        # Readable version
        if nchw and nhwc:
            return False
        else:
            return nhwc

    def _is_nhwc(self, tensor: torch.Tensor) -> bool:
        return ChannelsLastTaggedReshapePass._is_nhwc_tensor(tensor)

    def requires_nhwc_input(self, node: torch.fx.Node) -> bool:
        return node.target in self.memory_sensitive_ops_nhwc

    def requires_nchw_inputs(self, node: torch.fx.Node) -> bool:
        if node.target == exir_ops.edge.aten.view_copy.default:
            return True

        return node.target in self.memory_sensitive_ops_nchw

    def can_be_converted_to_nhwc(self, node: torch.fx.Node) -> bool:
        # There are two conditions that must be met for a node to be able to
        # be converted to NHWC:
        # 1) It must be 4-dimensional, and
        # 2) It must not be a constant data tensor which is already used
        #    in NCHW format somewhere
        is_4d = ("val" in node.meta) and (len(node.meta["val"].shape) == 4)
        is_nchw_constant = (
            is_param_node(self.exported_program, node)
            and (ChannelsLastTaggedReshapePass.XNN_NHWC_NODE in node.meta)
            and (ChannelsLastTaggedReshapePass.is_nchw_node(node))
        )
        return is_4d and not is_nchw_constant

    def make_partners(self, node_a, node_b):
        node_a.meta[ChannelsLastTaggedReshapePass.PARTNER_NODE] = node_b
        node_b.meta[ChannelsLastTaggedReshapePass.PARTNER_NODE] = node_a

    def create_call_function_node(
        self,
        graph_module: torch.fx.GraphModule,
        target: torch.fx.node.Target,
        args: Tuple[torch.fx.node.Argument, ...],
        memory_format: Optional[torch.memory_format] = None,
    ):
        return graph_module.graph.create_node(
            "call_function",
            target=target,
            args=args,
            kwargs=(  # pyre-fixme[6]
                {"memory_format": memory_format} if memory_format is not None else {}
            ),
        )

    def insert_copy_q_dq(
        self,
        graph_module: torch.fx.GraphModule,
        before: torch.fx.Node,
        after: torch.fx.Node,
        copy: torch.fx.Node,
        q_params: Tuple,
    ) -> None:
        with graph_module.graph.inserting_after(copy):
            q = self.create_call_function_node(
                graph_module=graph_module,
                target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                args=(copy,) + q_params,
            )
            q.meta = copy.meta.copy()

        with graph_module.graph.inserting_after(q):
            dq = self.create_call_function_node(
                graph_module=graph_module,
                target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                args=(q,) + q_params,
            )
            dq.meta = q.meta.copy()

            # Always tag q as implicit
            tag_as_implicit_q_dq(q)

            # Tag relevant q/ dq nodes
            # Ex: Original: G = conv -> q1 (Tag) -> dq1 (No Tag) -> output
            #     Insert (copy q dq pattern), G = conv -> q1 -> dq1 -> (copy q2 dq2)-> output
            #     if dq1 is not tagged as implicit, then tag dq2 and swap the dq1 and dq2 to simulate
            #        the pattern: G = conv -> q1 (Tag) -> (dq2 (Tag) copy q2 (Tag))-> dq1 (No Tag) -> output

            if is_dequant(before) and is_tagged_as_implicit_q_dq(before):
                tag_as_implicit_q_dq(dq)
            if is_dequant(before):
                tag_as_implicit_q_dq(before)

            before.replace_all_uses_with(dq)
            copy.replace_input_with(dq, before)

    def insert_dq_copy_q(
        self,
        graph_module: torch.fx.GraphModule,
        before: torch.fx.Node,
        after: torch.fx.Node,
        copy: torch.fx.Node,
        q_params: Tuple,
    ) -> None:
        with graph_module.graph.inserting_after(before):
            dq = self.create_call_function_node(
                graph_module=graph_module,
                target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                args=(before,) + q_params,
            )
            dq.meta = before.meta.copy()

        with graph_module.graph.inserting_after(copy):
            q = self.create_call_function_node(
                graph_module=graph_module,
                target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                args=(copy,) + q_params,
            )
            q.meta = copy.meta.copy()

            # Always tag q/dq as implicit
            tag_as_implicit_q_dq(dq)
            tag_as_implicit_q_dq(q)

            copy.replace_input_with(before, dq)
            after.replace_input_with(before, q)

    def insert_copy_and_assign_partner_nodes_quantization_sensitive(
        self,
        graph_module: torch.fx.GraphModule,
        original_input: torch.fx.Node,
        copy_node: torch.fx.Node,
        target_node: torch.fx.Node,
    ) -> None:
        """
        Wrapper for calling inserting copy between original_input and
        target_node, but also with inserting quantization nodes or assigning
        partner nodes if applicable. We don't assign partner nodes for
        quantization nodes because using the same quantization node multiple
        times causes issues with matching dq -> op -> q patterns

        Three cases
        1)
            original_node is dq, ex.
                dq -> conv
            we need to insert (copy q dq), ex.
                dq -> (copy q dq) -> conv

        2)
            original node is a q, ex.
                q -> output
            we need to insert (dq copy q), ex.
                q -> (dq copy q) -> output

        3)
            original node is neither, ex.
                conv -> output
            we only need to insert copy, ex
                conv -> (copy) -> output
        """
        if (
            original_input.target
            == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
        ):
            # Case 1
            self.insert_copy_q_dq(
                graph_module,
                original_input,
                target_node,
                copy_node,
                original_input.args[1:],
            )
        elif (
            original_input.target
            == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
        ):
            # Case 2
            self.insert_dq_copy_q(
                graph_module,
                original_input,
                target_node,
                copy_node,
                original_input.args[1:],
            )
        else:
            # Case 3
            target_node.replace_input_with(original_input, copy_node)

            # This may be redundant if copy node was obtained from
            # original input's meta's PARTNER_NODE_KEY but doesn't do any harm
            # in that case
            self.make_partners(original_input, copy_node)

    def input_dim_order(
        self, input_node: torch.fx.Node, input_order: InputDimOrder
    ) -> bool:
        if input_node.op == "placeholder":
            is_nhwc = self._is_nhwc(input_node.meta["val"])
            return not is_nhwc if input_order == InputDimOrder.NCHW else is_nhwc
        else:
            return (
                ChannelsLastTaggedReshapePass.is_nchw_node(input_node)
                if input_order == InputDimOrder.NCHW
                else ChannelsLastTaggedReshapePass.is_nhwc_node(input_node)
            )

    def input_to_nhwc(
        self,
        graph_module: torch.fx.GraphModule,
        input_node: torch.fx.Node,
        target_node: torch.fx.Node,
    ) -> None:
        if is_param_node(self.exported_program, input_node):
            if (
                ChannelsLastTaggedReshapePass.XNN_NHWC_NODE in input_node.meta
                and ChannelsLastTaggedReshapePass.is_nchw_node(input_node)
            ):
                # This constant data tensor has been used somewhere else
                # in NCHW format so we can't use it here in NHWC format
                raise AssertionError(
                    "The same constant data tensor can't be used in NCHW format in one place and NHWC in another"
                )
            # Mark the constant data tensor to be converted to NHWC when
            # serializing graph, but don't do anything else here
            self.mark_as_nhwc_node(input_node)

        if input_node.op == "placeholder":
            if self._is_nhwc(input_node.meta["val"][0]):
                return
        elif ChannelsLastTaggedReshapePass.is_nhwc_node(input_node):
            return

        if self.input_dim_order(input_node, InputDimOrder.NHWC):
            return

        if not self.can_be_converted_to_nhwc(input_node):
            raise AssertionError(
                "Attempting to convert non-NHWC compatible node to NHWC"
            )

        if ChannelsLastTaggedReshapePass.PARTNER_NODE in input_node.meta:
            # Already has an associated NHWC node
            input_node_nhwc = input_node.meta[
                ChannelsLastTaggedReshapePass.PARTNER_NODE
            ]
        else:
            # Need to create NHWC node
            # Check if input uses dynamic quantization
            is_dynamic_input = is_dynamic_qdq(input_node)

            if is_dynamic_input:
                # Trace back to original source node
                while getattr(input_node, "args", None):
                    input_node = input_node.args[0]

            with graph_module.graph.inserting_after(input_node):
                input_node_nhwc = self.create_call_function_node(
                    graph_module=graph_module,
                    target=exir_ops.edge.aten._to_copy.default,
                    args=(input_node,),
                    memory_format=torch.channels_last,
                )
                # Use static method for consistency
                ChannelsLastTaggedReshapePass.mark_as_nhwc_node(input_node_nhwc)

            if is_dynamic_input:
                # Replace downstream input_nodes with NHWC node
                input_node.replace_all_uses_with(input_node_nhwc)
                input_node_nhwc.args = (input_node,)

        self.insert_copy_and_assign_partner_nodes_quantization_sensitive(
            graph_module=graph_module,
            original_input=input_node,
            copy_node=input_node_nhwc,
            target_node=target_node,
        )

    def input_to_nchw(
        self,
        graph_module: torch.fx.GraphModule,
        input_node: torch.fx.Node,
        target_node: torch.fx.Node,
    ) -> None:
        if is_param_node(self.exported_program, input_node):
            if (
                ChannelsLastTaggedReshapePass.XNN_NHWC_NODE in input_node.meta
                and ChannelsLastTaggedReshapePass.is_nhwc_node(input_node)
            ):
                # This constant data tensor has been used somewhere else
                # in NHWC format so we can't use it here in NCHW format
                raise AssertionError(
                    "The same constant data tensor can't be used in NHWC format in one place and NCHW in another"
                )
            # Mark the constant data tensor as being used in NCHW format so
            # we know not to try to use it in NHWC format elsewhere, but don't
            # do anything else here
            self.mark_as_nchw_node(input_node)

        if input_node.op == "placeholder":
            if not self._is_nhwc(input_node.meta["val"]):
                return
        elif ChannelsLastTaggedReshapePass.is_nchw_node(input_node):
            return

        if self.input_dim_order(input_node, InputDimOrder.NCHW):
            return

        if ChannelsLastTaggedReshapePass.PARTNER_NODE in input_node.meta:
            # Already has an associated NCHW node
            input_node_nchw = input_node.meta[
                ChannelsLastTaggedReshapePass.PARTNER_NODE
            ]
        else:
            # Need to create NCHW node
            with graph_module.graph.inserting_after(input_node):
                input_node_nchw = self.create_call_function_node(
                    graph_module=graph_module,
                    target=exir_ops.edge.aten._to_copy.default,
                    args=(input_node,),
                    memory_format=torch.contiguous_format,
                )
                ChannelsLastTaggedReshapePass.mark_as_nchw_node(input_node_nchw)

        self.insert_copy_and_assign_partner_nodes_quantization_sensitive(
            graph_module=graph_module,
            original_input=input_node,
            copy_node=input_node_nchw,
            target_node=target_node,
        )

    def call(self, graph_module: torch.fx.GraphModule):  # noqa: C901
        graph = graph_module.graph
        original_nodes = list(graph.nodes)
        for node in original_nodes:
            if len(node.all_input_nodes) == 0:
                # This node has no inputs so we don't need to change anything, but still need to tag input nodes
                if (
                    "val" in node.meta
                    and isinstance(node.meta["val"], torch.Tensor)
                    and len(node.meta["val"].shape) == 4
                ):
                    if self._is_nhwc(node.meta["val"]):
                        self.mark_as_nhwc_node(node)
                    else:
                        self.mark_as_nchw_node(node)
                continue

            # Need special case for output node because it can have multiple output dim orders as we can output a tuple multiple nodes
            if node.op == "output":
                out_tuple = node.args[0]
                for out_node in out_tuple:
                    if not self._is_nhwc(out_node.meta["val"]):
                        self.input_to_nchw(graph_module, out_node, node)
                    else:
                        self.input_to_nhwc(graph_module, out_node, node)
            elif self.requires_nhwc_input(node):
                # Nodes which enter this branch are ones that require their
                # first input to be nhwc. This makes this node's output nhwc too
                self.input_to_nhwc(graph_module, node.args[0], node)
                for input_node in node.all_input_nodes[1:]:
                    if (
                        input_node.op == "placeholder"
                        and ChannelsLastTaggedReshapePass.is_nhwc_node(input_node)
                    ):
                        raise AssertionError(
                            f"Expected {input_node} to be NCHW in channels last reshape pass"
                        )
                self.mark_as_nhwc_node(node)
            elif self.requires_nchw_inputs(node):
                # The node requires nchw inputs
                for input_node in node.all_input_nodes:
                    self.input_to_nchw(graph_module, input_node, node)
            elif node.target == exir_ops.edge.aten._to_copy.default:
                self.tag_node(node)
            else:
                # The node can have inputs in any format (but all must be the
                # same format)
                is_or_isnt_nhwc_node = [
                    ChannelsLastTaggedReshapePass.is_nhwc_node(input_node)
                    for input_node in node.all_input_nodes
                ]
                if all(is_or_isnt_nhwc_node):
                    # All inputs are nhwc so this node's output is nhwc too
                    self.mark_as_nhwc_node(node)
                elif any(is_or_isnt_nhwc_node):
                    # There is at least one node of each format, so we need to
                    # convert some of them to another format such that all
                    # are the same.
                    # If all nodes can be converted to nhwc, then convert them
                    # all to nhwc. Otherwise, convert all to nchw
                    if all(
                        self.can_be_converted_to_nhwc(input_node)
                        for input_node in node.all_input_nodes
                    ):
                        for input_node in node.all_input_nodes:
                            self.input_to_nhwc(graph_module, input_node, node)
                        self.mark_as_nhwc_node(node)
                    else:
                        for input_node in node.all_input_nodes:
                            self.input_to_nchw(graph_module, input_node, node)

        graph_module.recompile()

        for node in graph_module.graph.nodes:
            if ChannelsLastTaggedReshapePass.PARTNER_NODE in node.meta:
                node.meta.pop(ChannelsLastTaggedReshapePass.PARTNER_NODE)

        # Since we are overriding "call", we need to call the parent's "call"
        # to retrace the graph and regenerate metadata
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
