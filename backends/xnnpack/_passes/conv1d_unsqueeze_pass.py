# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.utils.quant_utils import (
    is_dequant,
    is_quant,
    tag_as_implicit_q_dq,
)
from executorch.backends.xnnpack.utils.utils import get_param_tensor, is_param_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult
from torch._ops import OpOverload


class Conv1dUnsqueezePass(XNNPACKPass):
    """
    This pass is used to change conv1d ops into conv2d since xnnpack only
    supports 2d convolution. This is done by modifying the graph to do the
    following:
    1) unsqueeze the convolution's input from 3d to 4d
    2) perform a conv2d (with a modified version of the original conv1d args)
    3) squeeze the output back down to 3d.
    """

    def create_node(
        self,
        graph: torch.fx.Graph,
        op_target: OpOverload,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ):
        return graph.create_node(
            "call_function",
            op_target,
            args=args,
            kwargs=kwargs or {},
        )

    def insert_q_dq_pair(
        self,
        graph: torch.fx.Graph,
        anchor: torch.fx.Node,
        q_params: tuple,
    ):
        with graph.inserting_after(anchor):
            q = self.create_node(
                graph=graph,
                op_target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                args=(),  # We add the argument last
            )
            q.meta = anchor.meta.copy()

            # Tag q as implicit
            tag_as_implicit_q_dq(q)

        with graph.inserting_after(q):
            dq = self.create_node(
                graph=graph,
                op_target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                args=(q,) + q_params,
            )
            dq.meta = q.meta.copy()

            # Tag dq as implicit
            tag_as_implicit_q_dq(dq)

        anchor.replace_all_uses_with(dq)
        # We add this last so the replace all uses above does not replace the quqntized
        # node's first use
        q.args = (anchor,) + q_params

    def unsqueeze_kernel_weights(self, kernel_node):
        """
        Unsqueezes the weights of a conv1d to make it 4 dimensional.

        Args:
            kernel_node: the weights of conv1d node to be unsqueezed
        """
        kernel_param_3d = get_param_tensor(self.exported_program, kernel_node)
        if kernel_param_3d is None:
            raise AssertionError("Expected param tensor for the kernel node")

        kernel_param_4d = torch.nn.Parameter(
            data=kernel_param_3d.data.contiguous().unsqueeze(dim=-1),
            requires_grad=False,
        )

        if torch._export.utils.is_param(self.exported_program, kernel_node):
            parameter_name = self.exported_program.graph_signature.inputs_to_parameters[
                kernel_node.name
            ]
            self.exported_program.state_dict[parameter_name] = kernel_param_4d
            kernel_node.meta["val"] = kernel_param_4d.data.contiguous()
        elif torch._export.utils.is_buffer(self.exported_program, kernel_node):
            buffer_name = self.exported_program.graph_signature.inputs_to_buffers[
                kernel_node.name
            ]
            self.exported_program.state_dict[buffer_name] = kernel_param_4d
            kernel_node.meta["val"] = kernel_param_4d.data.contiguous()
        elif torch._export.utils.is_lifted_tensor_constant(
            self.exported_program, kernel_node
        ):
            buffer_name = (
                self.exported_program.graph_signature.inputs_to_lifted_tensor_constants[
                    kernel_node.name
                ]
            )
            self.exported_program.constants[buffer_name] = kernel_param_4d
            kernel_node.meta["val"] = kernel_param_4d.data.contiguous()
        else:
            setattr(
                kernel_node.graph.owning_module,
                kernel_node.target,
                kernel_param_4d,
            )

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        node_list = list(graph.nodes)
        for node in node_list:
            if node.op == "call_function":
                if node.target.__name__ == "aten.convolution.default":
                    stride = list(node.args[3])
                    if len(stride) != 1:
                        # skip conv if it is not 1d
                        continue

                    kernel_node = node.args[1]
                    if is_dequant(kernel_node) and not is_quant(kernel_node.args[0]):
                        kernel_node = kernel_node.args[0]

                    if not is_param_node(self.exported_program, kernel_node):
                        raise AssertionError(
                            "Expected op for convolution weight node to be a get_attr node or a parameter"
                        )

                    # Modify graph such that the conv changes from 1d to 2d
                    self.unsqueeze_kernel_weights(kernel_node)

                    # (b) Extend stride, padding, and dilation for extra dim
                    node.args = (
                        node.args[0],
                        node.args[1],
                        node.args[2],
                        node.args[3] + [1],  # stride
                        node.args[4] + [0],  # padding
                        node.args[5] + [1],  # dilation
                        node.args[6],
                        node.args[7] + [0],
                        node.args[8],
                    )

                    # c. Add unsqueeze to input (3d -> 4d) and squeeze to output (4d -> 3d)
                    # unsqueeze -> conv2d -> squeeze
                    input_node = node.args[0]
                    if (
                        isinstance(input_node, torch.fx.Node)
                        and input_node.target
                        == exir_ops.edge.aten.constant_pad_nd.default
                        and len(input_node.users) == 1
                    ):
                        pad_value = (
                            input_node.args[2] if len(input_node.args) > 2 else 0
                        )
                        pad_amounts = list(input_node.args[1])
                        temporal_only = len(pad_amounts) <= 2 or all(
                            amount == 0 for amount in pad_amounts[2:]
                        )
                        if (
                            pad_value == 0
                            and len(pad_amounts) % 2 == 0
                            and all(amount >= 0 for amount in pad_amounts)
                            and temporal_only
                        ):
                            time_before = pad_amounts[0] if len(pad_amounts) > 0 else 0
                            time_after = pad_amounts[1] if len(pad_amounts) > 1 else 0
                            conv_time_padding = node.args[4][0]
                            node.meta["xnnpack_input_padding"] = [
                                conv_time_padding + time_before,
                                0,
                                conv_time_padding + time_after,
                                0,
                            ]
                            node.meta["xnnpack_conv1d_folded_pad"] = True
                            pad_node = input_node
                            input_node = pad_node.args[0]
                            node.replace_input_with(pad_node, input_node)
                            graph.erase_node(pad_node)

                    with graph.inserting_before(node):
                        unsqueeze_before = self.create_node(
                            graph, exir_ops.edge.aten.unsqueeze_copy.default
                        )
                        unsqueeze_before.args = (
                            input_node,  # Input is node's original input
                            -1,  # Last Dimension
                        )
                        node.replace_input_with(input_node, unsqueeze_before)

                    # If Quantized we must insert unsqueeze --> q --> dq --> node
                    if is_dequant(input_node):
                        q_params = input_node.args[1:]
                        self.insert_q_dq_pair(graph, unsqueeze_before, q_params)

                    with graph.inserting_after(node):
                        squeeze_after = self.create_node(
                            graph,
                            exir_ops.edge.aten.squeeze_copy.dim,
                        )
                        squeeze_after.args = (
                            node,  # Input is the conv node
                            -1,  # Last dimension
                        )
                        original_users = [
                            user for user in node.users if user != squeeze_after
                        ]
                        for user in original_users:
                            user.replace_input_with(node, squeeze_after)

                        # If quantized, insert q --> dq --> squeeze
                    if all(is_quant(original_user) for original_user in original_users):
                        q_params = original_users[0].args[1:]
                        self.insert_q_dq_pair(graph, node, q_params)

        graph_module.recompile()
        # Since we are overriding "call", we need to call the parent's "call"
        # to retrace the graph and regenerate metadata
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)


class Conv1dFoldedPadMetaPass(XNNPACKPass):
    """
    Restores metadata for Conv1d nodes whose explicit pad was folded into
    asymmetric XNNPACK input padding.

    Later passes retrace the graph using symmetric ATen convolution args, which
    cannot represent even-kernel same padding exactly. Serialization, however,
    needs tensor metadata that matches the asymmetric XNNPACK padding fields.
    """

    def _resize_meta_val(self, node: torch.fx.Node, shape: tuple):
        if "val" not in node.meta:
            return

        val = node.meta["val"]
        if hasattr(val, "new_empty"):
            node.meta["val"] = val.new_empty(shape)
        else:
            node.meta["val"] = torch.empty(shape)

    def _set_users_meta(
        self,
        node: torch.fx.Node,
        shape_4d: tuple,
        shape_3d: tuple,
    ):
        visited = set()
        stack = list(node.users)
        while stack:
            user = stack.pop()
            if user in visited or user.op != "call_function":
                continue
            visited.add(user)

            if user.target == exir_ops.edge.aten.squeeze_copy.dim:
                self._resize_meta_val(user, shape_3d)
                for squeeze_user in user.users:
                    if squeeze_user.op == "call_function":
                        self._resize_meta_val(squeeze_user, shape_3d)
                continue

            self._resize_meta_val(user, shape_4d)
            stack.extend(user.users)

    def call(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if (
                node.op != "call_function"
                or node.target != exir_ops.edge.aten.convolution.default
                or "xnnpack_input_padding" not in node.meta
                or "xnnpack_conv1d_folded_pad" not in node.meta
            ):
                continue

            input_shape = tuple(node.args[0].meta["val"].shape)
            kernel_shape = tuple(node.args[1].meta["val"].shape)
            stride = node.args[3]
            dilation = node.args[5]
            padding_top, padding_right, padding_bottom, padding_left = node.meta[
                "xnnpack_input_padding"
            ]

            kernel_h = kernel_shape[2]
            kernel_w = kernel_shape[3]
            out_h = (
                input_shape[2]
                + padding_top
                + padding_bottom
                - dilation[0] * (kernel_h - 1)
                - 1
            ) // stride[0] + 1
            out_w = (
                input_shape[3]
                + padding_left
                + padding_right
                - dilation[1] * (kernel_w - 1)
                - 1
            ) // stride[1] + 1

            shape_4d = (input_shape[0], kernel_shape[0], out_h, out_w)
            shape_3d = (input_shape[0], kernel_shape[0], out_h)
            self._resize_meta_val(node, shape_4d)
            self._set_users_meta(node, shape_4d, shape_3d)

        graph_module.recompile()
        return PassResult(graph_module, True)
