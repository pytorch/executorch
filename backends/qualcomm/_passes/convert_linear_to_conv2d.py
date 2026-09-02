# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from executorch.backends.qualcomm._passes.utils import copy_meta
from executorch.backends.qualcomm.builders.node_visitor import dq_ops
from executorch.backends.qualcomm.builders.utils import (
    get_attr_from_target,
    get_parameter,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule
from torchao.quantization.pt2e.utils import get_new_attr_name_with_prefix


def _pad_list_to_4(lst):
    return lst + [1] * (4 - len(lst)) if len(lst) < 4 else lst[:4]


class ConvertLinearToConv2d(ExportPass):
    """
    Replace linear with an equivalent 1x1 convolution using call_function nodes.
    """

    def __init__(
        self, edge_program: Optional[torch.export.ExportedProgram] = None
    ) -> None:
        super().__init__()
        self.edge_program = edge_program
        self.aten_linear_target = torch.ops.aten.linear.default
        self.edge_linear_target = exir_ops.edge.aten.linear.default
        self._op_table = {
            # (is_edge) -> {role: target}
            True: {
                "view": exir_ops.edge.aten.view_copy.default,
                "permute": exir_ops.edge.aten.permute_copy.default,
                "conv": exir_ops.edge.aten.convolution.default,
                "linear": exir_ops.edge.aten.linear.default,
            },
            False: {
                "view": torch.ops.aten.reshape.default,
                "permute": torch.ops.aten.permute.default,
                "conv": torch.ops.aten.conv2d.default,
                "linear": torch.ops.aten.linear.default,
            },
        }

    def _rewrite_module_stack(self, node, suffix: str):
        """
        Append a suffix to nn_module_stack so the quant recipe can differentiate
        the converted node. Without this, the conv inherits the linear's
        nn_module_stack verbatim, which makes regex targeting in the recipe
        ambiguous (e.g. ``model.lm_head`` vs ``model.lm_head.conv``).
        """
        stack = node.meta.get("nn_module_stack")
        if not stack:
            return
        stack = dict(stack)
        last_key = next(reversed(stack))
        path, qualname = stack[last_key]
        stack[last_key] = (path + suffix, qualname)
        node.meta["nn_module_stack"] = stack

    def _register_tensor(
        self,
        graph_module: torch.fx.GraphModule,
        weight_placeholder_node: torch.fx.Node,
        weight_val: torch.Tensor,
    ) -> torch.fx.Node:
        buffer_name = get_new_attr_name_with_prefix(weight_placeholder_node.name)(
            graph_module
        )
        graph_module.register_buffer(buffer_name, weight_val)

        with graph_module.graph.inserting_after(weight_placeholder_node):
            get_attr_node = graph_module.graph.get_attr(buffer_name)
            get_attr_node.meta = copy_meta(weight_placeholder_node.meta)
            fake_mode = detect_fake_mode(weight_placeholder_node.meta["val"])
            converter = fake_mode.fake_tensor_converter
            get_attr_node.meta["val"] = converter.from_real_tensor(
                fake_mode, weight_val
            )

        return get_attr_node

    def _create_node(
        self,
        graph_module: torch.fx.GraphModule,
        target: EdgeOpOverload,
        args: tuple,
        node_meta_val: FakeTensor,
        meta_source_node: torch.fx.Node = None,
    ) -> torch.fx.Node:
        anchor_node = meta_source_node if meta_source_node else args[0]
        with graph_module.graph.inserting_after(anchor_node):
            inserted_node = graph_module.graph.create_node(
                "call_function",
                target,
                args,
            )
            inserted_node.meta = copy_meta(
                anchor_node.meta, lambda m: {**m, "val": node_meta_val}
            )

        return inserted_node

    def _reshape_weight_for_all_users(
        self,
        graph_module: torch.fx.GraphModule,
        weight_placeholder_node: torch.fx.Node,
        ops: dict,
    ) -> torch.fx.Node:
        weight_val = (
            get_attr_from_target(graph_module, weight_placeholder_node.target)
            if weight_placeholder_node.op == "get_attr"
            else get_parameter(weight_placeholder_node, self.edge_program)
        )
        assert weight_val is not None, "Cannot get the weight in linear node."

        weight_val = weight_val.reshape(*weight_val.shape, 1, 1).contiguous().detach()
        get_attr_node = self._register_tensor(
            graph_module, weight_placeholder_node, weight_val
        )
        for user in list(weight_placeholder_node.users):
            if user.target is not ops["linear"]:
                # Add a safety check to prevent replacing weights of non-linear nodes with updated weights.
                continue
            user.replace_input_with(weight_placeholder_node, get_attr_node)

        return get_attr_node

    def call(self, graph_module: GraphModule):
        graph = graph_module.graph
        # The set tracks whether the weights associated with a linear node have been preprocessed,
        # and is used in cases where multiple linear nodes share the same weights.
        preprocessed_linear_weights_set = set()

        for node in graph.nodes:
            if node.target in {self.aten_linear_target, self.edge_linear_target}:
                is_edge = node.target == self.edge_linear_target
                ops = self._op_table[is_edge]
                input_node = node.args[0]
                weight_placeholder_node = node.args[1]
                assert weight_placeholder_node.target not in dq_ops, (
                    "ConvertLinearToConv2d does not handle quantized weights. "
                    "A quantized flow must convert during the annotation "
                    "pipeline, before q/dq nodes exist. Please set "
                    "quantizer.set_convert_linear_to_conv2d(True)."
                )
                bias_arg = node.args[2] if len(node.args) > 2 else None

                input_meta_val = input_node.meta["val"]
                output_meta_val = node.meta["val"]
                bias_meta_val = bias_arg.meta["val"] if bias_arg else None
                rank = input_meta_val.ndim
                cur_meta_val = input_meta_val

                # Step 1: calibrate input
                # rank = 2: Reshape (dim, C) to (1, dim, C, 1), and permute to (1, C, 1, dim)
                # rank = 3: Reshape (N, dim, C) to (N, dim, C, 1), and permute to (N, C, 1, dim)
                # rank = 4: Permute (N, H, W, C) to (N, C, H, W)
                if rank <= 3:
                    shape = (
                        (1, *input_meta_val.shape, 1)
                        if rank == 2
                        else (*input_meta_val.shape, 1)
                    )
                    cur_meta_val = cur_meta_val.reshape(shape)
                    reshape_node = self._create_node(
                        graph_module,
                        ops["view"],
                        (input_node, shape),
                        cur_meta_val,
                    )

                order = (0, 3, 1, 2) if rank == 4 else (0, 2, 3, 1)
                cur_meta_val = cur_meta_val.permute(order)
                permute_node = self._create_node(
                    graph_module,
                    ops["permute"],
                    (reshape_node, order) if rank <= 3 else (input_node, order),
                    cur_meta_val,
                )

                # Step 2: reshape weight
                if weight_placeholder_node.name not in preprocessed_linear_weights_set:
                    weight_arg = self._reshape_weight_for_all_users(
                        graph_module, weight_placeholder_node, ops
                    )
                    # Add the name of the preprocessed weights to the list.
                    preprocessed_linear_weights_set.add(weight_arg.name)
                else:
                    # Skip preprocessing as the weights have already been processed due to shared weights.
                    weight_arg = node.args[1]

                weight_meta_val = weight_arg.meta["val"]

                stride = [1, 1]
                padding = [0, 0]
                dilation = [1, 1]
                transposed = False
                output_padding = [0, 0]
                groups = 1
                """
                The two dialects take different argument lists:

                Spec for `aten.convolution` (https://docs.pytorch.org/docs/stable/torch.compiler_ir.html)
                convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding,
                            SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor

                aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=[1, 1],
                             SymInt[2] padding=[0, 0], SymInt[2] dilation=[1, 1], SymInt groups=1) -> Tensor

                conv2d has no transposed / output_padding, so passing the 9-arg
                convolution list to it raises "expected at most 7 argument(s)".
                """
                conv_args = (
                    (
                        permute_node,
                        weight_arg,
                        bias_arg,
                        stride,
                        padding,
                        dilation,
                        transposed,
                        output_padding,
                        groups,
                    )
                    if is_edge
                    else (
                        permute_node,
                        weight_arg,
                        bias_arg,
                        stride,
                        padding,
                        dilation,
                        groups,
                    )
                )
                # Shape inference always goes through the 9-arg convolution;
                # conv2d is just the narrower spelling of the same op.
                cur_meta_val = exir_ops.edge.aten.convolution.default(
                    cur_meta_val,
                    weight_meta_val,
                    bias_meta_val,
                    stride,
                    padding,
                    dilation,
                    transposed,
                    output_padding,
                    groups,
                )
                conv_node = self._create_node(
                    graph_module,
                    ops["conv"],
                    conv_args,
                    cur_meta_val,
                    meta_source_node=node,
                )

                # Step 3: restore original output of linear node
                # rank = 2: Permute (1, C, 1, dim) to (1, dim, C, 1), and reshape to (dim, C)
                # rank = 3: Permute (N, C, 1, dim) to (N, dim, C, 1), and reshape to (N, dim, C)
                # rank = 4: Permute (N, C, H, W) to (N, H, W, C)
                order = (0, 2, 3, 1) if rank == 4 else (0, 3, 1, 2)
                cur_meta_val = cur_meta_val.permute(order)
                permute_node = self._create_node(
                    graph_module,
                    ops["permute"],
                    (conv_node, order),
                    cur_meta_val,
                )
                if rank <= 3:
                    target_shape = output_meta_val.shape
                    cur_meta_val = cur_meta_val.reshape(target_shape)
                    reshape_node = self._create_node(
                        graph_module,
                        ops["view"],
                        (permute_node, target_shape),
                        cur_meta_val,
                    )
                    node.replace_all_uses_with(reshape_node)
                else:
                    node.replace_all_uses_with(permute_node)

                graph.erase_node(node)
                self._rewrite_module_stack(conv_node, ".conv")

        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
