# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import _operator
from typing import List, Tuple

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.sym_util import eval_shape


class LayoutTransform(ExportPass):
    """
    QNN delegate requires channel last layout format, this pass aims to
    help generate the correct transformation by inserting fewest ammount of
    'permute' operators in the graph.
    """

    layout_sensitive_ops = {
        exir_ops.edge.aten.convolution.default,
        exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
        exir_ops.edge.aten.max_pool2d_with_indices.default,
        exir_ops.edge.aten.avg_pool2d.default,
        exir_ops.edge.aten.upsample_bilinear2d.default,
        exir_ops.edge.aten.pixel_shuffle.default,
    }

    layout_agnostic_ops = {
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.cat.default,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.hardtanh.default,
        exir_ops.edge.aten.hardswish.default,
        exir_ops.edge.aten.mean.dim,
        exir_ops.edge.aten.linear.default,
        exir_ops.edge.aten.clamp.default,
        exir_ops.edge.aten._to_copy.default,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.div.Tensor,
        exir_ops.edge.aten.ceil.default,
        exir_ops.edge.aten._softmax.default,
        exir_ops.edge.aten.constant_pad_nd.default,
        exir_ops.edge.aten.bmm.default,
        exir_ops.edge.aten.full.default,
        exir_ops.edge.aten.embedding.default,
        _operator.getitem,
    }

    layout_transformed_tag = "axis_order"
    inserted_permute_tag = "qnn_permute"

    layout_type = {
        1: ("N", "N"),
        2: ("NC", "NC"),
        3: ("NCW", "NWC"),
        4: ("NCHW", "NHWC"),
        5: ("NCDHW", "NDHWC"),
    }

    q_ops = {
        torch.ops.quantized_decomposed.quantize_per_channel.default,
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
    }

    @classmethod
    def get_axis_order(cls, size: List[int], reverse=False) -> Tuple[int]:
        old_layout, new_layout = cls.layout_type[len(size)]
        if reverse:
            old_layout, new_layout = new_layout, old_layout
        return tuple(old_layout.find(x) for x in new_layout)

    def __init__(self, insert_permute=False):
        super(LayoutTransform, self).__init__()
        self.insert_permute = insert_permute

    def mark_as_transformed(self, node: torch.fx.Node) -> None:
        if isinstance(node.meta["val"], (tuple, list)):
            getitem_node = list(node.users.keys())[0]
            if getitem_node.target.__name__ != "getitem":
                raise AssertionError(
                    "Expected node's user to be getitem, "
                    f"got {getitem_node.target.__name__}"
                )
            index = getitem_node.args[1]
            node.meta[self.layout_transformed_tag] = self.get_axis_order(
                eval_shape(node.meta["val"][index].shape)
            )
        else:
            node.meta[self.layout_transformed_tag] = self.get_axis_order(
                eval_shape(node.meta["val"].shape)
            )

    def is_transformed_node(self, node: torch.fx.Node) -> bool:
        if not hasattr(node, "meta"):
            return False
        return self.layout_transformed_tag in node.meta

    def is_layout_sensitive(self, node: torch.fx.Node) -> bool:
        return node.target in self.layout_sensitive_ops

    def is_layout_agnostic(self, node: torch.fx.Node) -> bool:
        if node.target == exir_ops.edge.aten.mean.dim:
            # if dimemsion is not kept, we'll have no clue how to do layout transform
            if len(node.args) < 3 or not node.args[2]:
                return False
        return node.target in self.layout_agnostic_ops

    def is_edge_condition(self, node):
        if not isinstance(node, torch.fx.Node):
            return True

        if any(
            [
                self.is_transformed_node(node),
                node.op == "get_attr",
                (
                    node.target == exir_ops.edge.aten.permute_copy.default
                    and node.meta.get(self.inserted_permute_tag, False)
                ),
                (
                    node.op != "output"
                    and not isinstance(node.meta["val"], tuple)
                    and len(node.meta["val"].shape) == 0
                ),
            ]
        ):
            return True

        return False

    def insert_node(self, graph_module, node, revert_layout: bool) -> None:
        if not self.insert_permute:
            return
        with graph_module.graph.inserting_after(node):
            users = node.users.copy()
            if isinstance(node.meta["val"], tuple):
                getitem_node = list(node.users.keys())[0]
                if getitem_node.target.__name__ != "getitem":
                    raise AssertionError(
                        f"Expected bn node's user to be getitem, got {getitem_node.target.__name__}"
                    )
                index = getitem_node.args[1]
                tensor = node.meta["val"][index]
            else:
                tensor = node.meta["val"]

            permute = self.create_call_function_node(
                graph_module,
                exir_ops.edge.aten.permute_copy.default,
                (
                    node,
                    self.get_axis_order(eval_shape(tensor.shape), revert_layout),
                ),
            )
            permute.meta["val"] = tensor
            permute.meta["quant_attrs"] = node.meta.get("quant_attrs")
            # we need this to check the annotation boundary
            permute.meta[self.inserted_permute_tag] = True

            for user in users:
                user.replace_input_with(node, permute)

    def create_call_function_node(
        self,
        graph_module: torch.fx.GraphModule,
        target: torch.fx.node.Target,
        args: Tuple[torch.fx.node.Argument, ...],
    ):
        return graph_module.graph.create_node(
            "call_function",
            target=target,
            args=args,
        )

    def traverse(self, node: torch.fx.Node, graph_module: torch.fx.GraphModule) -> None:
        for arg in node.args:
            self.annotate_layout(arg, graph_module, revert_layout=False)

        node_users = set(node.users.keys())
        for user in node_users:
            self.annotate_layout(user, graph_module, revert_layout=True)

    def annotate_layout(
        self, node: torch.fx.Node, graph_module: torch.fx.GraphModule, revert_layout
    ) -> None:

        if self.is_edge_condition(node):
            return
        elif self.is_layout_agnostic(node) or self.is_layout_sensitive(node):
            self.mark_as_transformed(node)
            self.traverse(node, graph_module)
        else:

            def check_arg(arg):
                if self.is_transformed_node(arg):
                    self.insert_node(graph_module, arg, revert_layout=revert_layout)

            if not revert_layout:
                self.insert_node(graph_module, node, revert_layout=revert_layout)
            else:
                for args in node.args:
                    if isinstance(args, torch.fx.immutable_collections.immutable_list):
                        for arg in args:
                            check_arg(arg)
                    else:
                        check_arg(args)

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        sensitive_nodes = [
            node for node in graph.nodes if self.is_layout_sensitive(node)
        ]
        for node in sensitive_nodes:
            if not self.is_transformed_node(node):
                self.mark_as_transformed(node)
                self.traverse(node, graph_module)

        graph_module.recompile()
        if not self.insert_permute:
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
