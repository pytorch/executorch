# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.qualcomm._passes.utils import append_qdq, copy_meta
from executorch.backends.qualcomm.builders.utils import get_parameter, set_parameter
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule
from torchao.quantization.pt2e.utils import get_new_attr_name_with_prefix


def _pad_list_to_4(lst):
    return lst + [1] * (4 - len(lst)) if len(lst) < 4 else lst[:4]


class ConvertLinearToConv2d(ExportPass):
    """
    Replace aten.linear.default with equivalent 1x1 conv2d using call_function nodes.
    """

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super().__init__()
        self.edge_program = edge_program
        self.per_block_dq = torch.ops.torchao.dequantize_affine.default

    def _register_tensor(
        self,
        gm: torch.fx.GraphModule,
        node: torch.fx.Node,
        tensor_constant: torch.Tensor,
    ) -> torch.fx.Node:
        new_node_name = get_new_attr_name_with_prefix(node.name)(gm)
        gm.register_buffer(new_node_name, tensor_constant)

        with gm.graph.inserting_before(node):
            get_attr_node = gm.graph.get_attr(new_node_name)
            get_attr_node.meta["val"] = tensor_constant
        return get_attr_node

    def _append_dq(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        qdq_node: torch.fx.Node,
    ):
        q_op = torch.ops.quantized_decomposed.quantize_per_tensor.default
        dq_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default

        if qdq_node.target not in {q_op, dq_op}:
            return node

        with graph_module.graph.inserting_after(node):
            dq_args = (node, *qdq_node.args[1:])
            dq_node = graph_module.graph.create_node("call_function", dq_op, dq_args)
            dq_node.meta = copy_meta(node.meta)
        return dq_node

    def _create_node(
        self, graph_module, target, args, meta_node, new_meta_val, qdq_node
    ):
        new_node = graph_module.graph.call_function(target, args)
        new_node.meta = copy_meta(
            meta_node.meta,
            lambda m, new_meta_val=new_meta_val: {
                **m,
                "val": new_meta_val,
            },
        )
        dq_node = append_qdq(
            graph_module=graph_module,
            node=new_node,
            qdq_node=qdq_node,
        )
        return dq_node

    def _reshape_weight(self, graph_module, weight_node, dq_node):
        # After export, constant node will be placeholder from edge_program
        weight_val = get_parameter(weight_node, self.edge_program)
        assert weight_val is not None, "Cannot get the weight in linear node."

        weight_val = weight_val.reshape(*weight_val.shape, 1, 1)
        # Create the new weight node when several node share the same weight
        # such as embedding and lm_head in LLM.
        if len(list(weight_node.users)) > 1:
            weight_node = self._register_tensor(graph_module, weight_node, weight_val)
            dq_node = self._append_dq(graph_module, weight_node, dq_node)
        else:
            set_parameter(
                (
                    torch.nn.Parameter(weight_val)
                    if weight_val.dtype == torch.float
                    else weight_val
                ),
                weight_node,
                self.edge_program,
            )

        # Update node meta val
        weight_node.meta["val"] = weight_node.meta["val"].reshape(weight_val.shape)
        dq_node.meta["val"] = dq_node.meta["val"].reshape(weight_val.shape)
        # Update block size for per-block quant
        if dq_node.target == self.per_block_dq:
            new_args = list(dq_node.args)
            # pad block size
            new_args[1] = _pad_list_to_4(list(new_args[1]))
            dq_node.args = tuple(new_args)

        return dq_node

    def call(self, graph_module: GraphModule):
        graph = graph_module.graph

        for node in list(graph.nodes):
            if node.target == torch.ops.aten.linear.default:
                input_node = node.args[0]
                # In quantization flow, weight_arg will be dq node.
                weight_arg = node.args[1]
                weight_node = (
                    weight_arg if weight_arg.op == "placeholder" else weight_arg.args[0]
                )
                bias_arg = node.args[2] if len(node.args) > 2 else None

                input_meta_val = input_node.meta["val"]
                output_meta_val = node.meta["val"]
                if bias_arg:
                    bias_meta_val = bias_arg.meta["val"]

                rank = input_meta_val.ndim
                with graph.inserting_before(node):
                    # Step 1: reshape input
                    # rank = 2: (dim, C) -> (1, C, 1, dim)
                    # rank = 3: (N, dim, C) -> (N, C, 1, dim)
                    # rank = 4: (N, H, W, C) -> (N, C, H, W)
                    order = (0, 3, 1, 2)
                    if rank <= 3:
                        # (dim, C) -> (1, C, 1, dim)
                        # (N, dim, C) -> (N, C, 1, dim)
                        shape = (
                            (1, *input_meta_val.shape, 1)
                            if rank == 2
                            else (*input_meta_val.shape, 1)
                        )
                        x_meta_val = input_meta_val.reshape(shape)
                        input_node = self._create_node(
                            graph_module,
                            torch.ops.aten.reshape.default,
                            (input_node, shape),
                            node,
                            x_meta_val,
                            input_node,
                        )
                        order = (0, 2, 3, 1)

                    x_meta_val = x_meta_val.permute(order)
                    x = self._create_node(
                        graph_module,
                        torch.ops.aten.permute.default,
                        (input_node, order),
                        node,
                        x_meta_val,
                        input_node,
                    )

                    # Step 2: reshape weight
                    weight_arg = self._reshape_weight(
                        graph_module, weight_node, weight_arg
                    )
                    weight_meta_val = weight_arg.meta["val"]

                    conv_args = [x, weight_arg]
                    conv_args_meta_val = [x_meta_val, weight_meta_val]
                    if bias_arg:
                        conv_args.append(bias_arg)
                        conv_args_meta_val.append(bias_meta_val)
                    else:
                        conv_args.append(None)
                        conv_args_meta_val.append(None)

                    conv_args.extend(
                        [[1, 1], [0, 0], [1, 1], 1]
                    )  # stride, padding, dilation, groups
                    conv_node_val = torch.nn.functional.conv2d(
                        *conv_args_meta_val,
                        stride=(1, 1),
                        padding=(0, 0),
                        dilation=(1, 1),
                        groups=1,
                    )
                    conv_node = self._create_node(
                        graph_module,
                        torch.ops.aten.conv2d.default,
                        tuple(conv_args),
                        node,
                        conv_node_val,
                        list(node.users)[0],
                    )

                    # Step 3: restore shape
                    # rank = 2: (1, C, 1, dim) -> (dim, C)
                    # rank = 3: (N, C, 1, dim) -> (N, dim C)
                    # rank = 4: (N, C, H, W) -> (N, H, W, C)
                    order = (0, 2, 3, 1) if rank == 4 else (0, 3, 1, 2)
                    y_meta_val = conv_node_val.permute(order)
                    y = self._create_node(
                        graph_module,
                        torch.ops.aten.permute.default,
                        (conv_node, order),
                        node,
                        y_meta_val,
                        list(node.users)[0],
                    )
                    if rank <= 3:
                        target_shape = output_meta_val.shape
                        y_meta_val = y_meta_val.reshape(target_shape)
                        y = self._create_node(
                            graph_module,
                            torch.ops.aten.reshape.default,
                            (y, target_shape),
                            node,
                            y_meta_val,
                            list(node.users)[0],
                        )

                    node.replace_all_uses_with(y)
                    graph.erase_node(node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
