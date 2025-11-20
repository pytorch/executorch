# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import _operator
from typing import List, Tuple

import torch

from executorch.backends.qualcomm.builders.utils import is_parameter
from executorch.backends.qualcomm.utils.constants import (
    QCOM_AXIS_ORDER,
    QCOM_INSERTED_PERMUTE,
    QCOM_LAYOUT_CHANGE,
    QCOM_QUANT_ATTRS,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.sym_util import eval_shape


class LayoutTransform(ExportPass):
    """
    QNN delegate requires channel last layout format, this pass aims to
    help generate the correct transformation by inserting fewest amount of
    'permute' operators in the graph.
    Please notice that permute op is inserted during qnn_preprocess.

    Operations are divided into 3 categories: sensitive_layout, agnostic_layout, and pytorch_layout.
    sensitive_layout: These ops must be lowered to QNN in NHWC format. A permute(NCHW->NHWC) op will be inserted in front of the sensitive_layout op.
    agnostic_layout: These ops are agnostic to layout format, which means it can be passed to QNN in either NCHW or NHWC format.
    pytorch_layout: These ops must be lowered to QNN in NCHW format. A permute(NHWC->NCHW) op will be inserted in front of the pytorch_layout op.

    For optimization purposes, permute is only inserted when it is necessary to switch between sensitive_layout and pytorch_layout.
    For example, a model consists of three kinds of operations: conv(sensitive_layout), relu(agnostic_layout), and unsqueeze(pytorch_layout)
    If a graph originally looks like : in -> conv -> relu -> conv -> relu -> unsqueeze -> out
    After layout_transform pass: in -> permute(NCHW->NHWC) -> conv -> relu -> conv -> relu -> permute(NHWC->NCHW) -> unsqueeze -> out
    The reason for inserting the 1st permute is because conv is layout sensitive. Since relu is agnostic to layout, it doesn't matter what format is used.
    This format works fine until unsqueeze is encountered, which is a pytorch_format operation, so a 2nd permute is necessary to convert it back to pytorch format.
    """

    layout_sensitive_ops = {
        exir_ops.edge.aten.adaptive_avg_pool2d.default,
        exir_ops.edge.aten._adaptive_avg_pool3d.default,
        exir_ops.edge.aten.adaptive_max_pool2d.default,
        exir_ops.edge.aten.avg_pool2d.default,
        exir_ops.edge.aten.avg_pool3d.default,
        exir_ops.edge.aten.convolution.default,
        exir_ops.edge.aten.grid_sampler_2d.default,
        exir_ops.edge.aten.grid_sampler_3d.default,
        exir_ops.edge.aten.instance_norm.default,
        exir_ops.edge.aten.max_pool2d_with_indices.default,
        exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
        exir_ops.edge.aten._native_batch_norm_legit.no_stats,
        exir_ops.edge.aten.native_group_norm.default,
        exir_ops.edge.aten.pixel_shuffle.default,
        exir_ops.edge.aten.pixel_unshuffle.default,
        exir_ops.edge.aten.upsample_bicubic2d.default,
        exir_ops.edge.aten.upsample_bicubic2d.vec,
        exir_ops.edge.aten.upsample_bilinear2d.default,
        exir_ops.edge.aten.upsample_bilinear2d.vec,
        exir_ops.edge.aten.upsample_nearest2d.default,
        exir_ops.edge.aten.upsample_nearest2d.vec,
    }

    layout_agnostic_ops = {
        exir_ops.edge.aten.abs.default,
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.amax.default,
        exir_ops.edge.aten.amin.default,
        exir_ops.edge.aten.asin.default,
        exir_ops.edge.aten.atan.default,
        exir_ops.edge.aten.bitwise_or.Tensor,
        exir_ops.edge.aten.bitwise_xor.Tensor,
        exir_ops.edge.aten.bmm.default,
        exir_ops.edge.aten.bitwise_and.Tensor,
        exir_ops.edge.aten.cat.default,
        exir_ops.edge.aten.ceil.default,
        exir_ops.edge.aten.clamp.default,
        exir_ops.edge.aten.constant_pad_nd.default,
        exir_ops.edge.aten.cumsum.default,
        exir_ops.edge.aten.div.Tensor,
        exir_ops.edge.aten.elu.default,
        exir_ops.edge.aten.eq.Tensor,
        exir_ops.edge.aten.exp.default,
        exir_ops.edge.aten.flip.default,
        exir_ops.edge.aten.floor.default,
        exir_ops.edge.aten.floor_divide.default,
        exir_ops.edge.aten.full.default,
        exir_ops.edge.aten.full_like.default,
        exir_ops.edge.aten.ge.Tensor,
        exir_ops.edge.aten.gelu.default,
        exir_ops.edge.aten.gt.Tensor,
        exir_ops.edge.aten.hardswish.default,
        exir_ops.edge.aten.hardsigmoid.default,
        exir_ops.edge.aten.hardtanh.default,
        exir_ops.edge.aten.le.Tensor,
        exir_ops.edge.aten.linear.default,
        exir_ops.edge.aten.log.default,
        exir_ops.edge.aten.logical_and.default,
        exir_ops.edge.aten.logical_not.default,
        exir_ops.edge.aten.lt.Scalar,
        exir_ops.edge.aten.lt.Tensor,
        exir_ops.edge.aten.max.dim,
        exir_ops.edge.aten.maximum.default,
        exir_ops.edge.aten.mean.dim,
        exir_ops.edge.aten.min.dim,
        exir_ops.edge.aten.minimum.default,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.ne.Scalar,
        exir_ops.edge.aten.ne.Tensor,
        exir_ops.edge.aten.neg.default,
        exir_ops.edge.aten.pow.Tensor_Scalar,
        exir_ops.edge.aten.prelu.default,
        exir_ops.edge.aten.repeat.default,
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.round.default,
        exir_ops.edge.aten.sigmoid.default,
        exir_ops.edge.aten.sign.default,
        exir_ops.edge.aten.slice_copy.Tensor,
        exir_ops.edge.aten.split_with_sizes.default,
        exir_ops.edge.aten.split_with_sizes_copy.default,
        exir_ops.edge.aten.sqrt.default,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.sum.dim_IntList,
        exir_ops.edge.aten.stack.default,
        exir_ops.edge.aten.topk.default,
        exir_ops.edge.aten._to_copy.default,
        exir_ops.edge.aten.unbind.int,
        exir_ops.edge.aten.where.self,
        _operator.getitem,
        torch.ops.aten.scalar_tensor.default,
    }

    layout_type = {
        1: ("N", "N"),
        2: ("NC", "NC"),
        3: ("NCW", "NWC"),
        4: ("NCHW", "NHWC"),
        5: ("NCDHW", "NDHWC"),
    }

    @classmethod
    def get_axis_order(cls, size: List[int], reverse=False) -> Tuple[int]:
        old_layout, new_layout = cls.layout_type[len(size)]
        if reverse:
            old_layout, new_layout = new_layout, old_layout
        return tuple(old_layout.find(x) for x in new_layout)

    def __init__(
        self, edge_program: torch.export.ExportedProgram, insert_permute=False
    ):
        super(LayoutTransform, self).__init__()
        self.edge_program = edge_program
        self.insert_permute = insert_permute
        self.transformed_tag = QCOM_AXIS_ORDER

    def mark_as_transformed(self, node: torch.fx.Node) -> None:
        if isinstance(node.meta["val"], (tuple, list)):
            getitem_node = list(node.users.keys())[0]
            if getitem_node.target.__name__ != "getitem":
                raise AssertionError(
                    "Expected node's user to be getitem, "
                    f"got {getitem_node.target.__name__}"
                )
            index = getitem_node.args[1]
            node.meta[self.transformed_tag] = self.get_axis_order(
                eval_shape(node.meta["val"][index].shape)
            )
        else:
            node.meta[self.transformed_tag] = self.get_axis_order(
                eval_shape(node.meta["val"].shape)
            )

    def is_transformed_node(self, node: torch.fx.Node) -> bool:
        if not hasattr(node, "meta"):
            return False
        return self.transformed_tag in node.meta

    def is_layout_sensitive(self, node: torch.fx.Node) -> bool:
        return node.target in self.layout_sensitive_ops

    def is_layout_agnostic(self, node: torch.fx.Node) -> bool:
        if node.target in {
            exir_ops.edge.aten.max.dim,
            exir_ops.edge.aten.mean.dim,
            exir_ops.edge.aten.min.dim,
            exir_ops.edge.aten.sum.dim_IntList,
            exir_ops.edge.aten.amax.default,
        }:
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
                    and node.meta.get(QCOM_INSERTED_PERMUTE, False)
                ),
                (
                    node.op != "output"
                    and not isinstance(node.meta["val"], (tuple, list))
                    and len(node.meta["val"].shape) == 0
                ),
                is_parameter(node, self.edge_program),
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
            permute.meta[QCOM_QUANT_ATTRS] = node.meta.get(QCOM_QUANT_ATTRS)
            # we need this to check the annotation boundary
            permute.meta[QCOM_INSERTED_PERMUTE] = True

            # this is the case when residual connection happened:
            # e.g. consider following graph
            # x --> permute --> layer_norm --> permute --> conv2d --> add
            #               └-------------------------------------┙
            # we should have premute node to be correctly inserted as:
            # x --> permute --> layer_norm --> permute --> qnn_permute --> conv2d --> add
            #               └--------------------------------------> qnn_premute -┙
            # i.e. insert permute by condition between user and current node
            #      if there are multiple users included
            is_node_transformed = self.is_transformed_node(node)
            for user in users:
                is_user_transformed = (
                    self.is_transformed_node(user) or QCOM_LAYOUT_CHANGE in user.meta
                )
                # insert permute only in exclusive condition
                if is_node_transformed != is_user_transformed:
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
            if isinstance(arg, list):
                for arg_node in arg:
                    self.annotate_layout(arg_node, graph_module, revert_layout=False)
            else:
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

    def conditional_sensitive_check(self, node):
        # For softmax and log_softmax, we must ensure axis == -1 since thats the only axis supported by QNN.
        # Softmax and log_softmax is treated as pytorch_layout in default, and will be treated as sensitive_layout when axis is not given as last dim.
        target_nodes = [
            exir_ops.edge.aten._softmax.default,
            exir_ops.edge.aten._log_softmax.default,
        ]
        if node.target in target_nodes:
            dim = node.args[1]
            if dim < 0:
                dim = dim % node.meta["val"].dim()
            if dim != node.meta["val"].dim() - 1:
                return True
        return False

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        sensitive_nodes = [
            node
            for node in graph.nodes
            if self.is_layout_sensitive(node) or self.conditional_sensitive_check(node)
        ]

        # perform first run traversal for identifying nodes subjected to layout changes
        if self.insert_permute:
            self.insert_permute, self.transformed_tag = False, QCOM_LAYOUT_CHANGE
            for node in sensitive_nodes:
                if not self.is_transformed_node(node):
                    self.mark_as_transformed(node)
                    self.traverse(node, graph_module)
            self.insert_permute, self.transformed_tag = True, QCOM_AXIS_ORDER

        for node in sensitive_nodes:
            if not self.is_transformed_node(node):
                self.mark_as_transformed(node)
                self.traverse(node, graph_module)

        graph_module.recompile()
        if not self.insert_permute:
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
