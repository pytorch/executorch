# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import cast, List

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from torch._subclasses.fake_tensor import FakeTensorMode


def add_fake_tensor_to_node(padding_node, input_shape, padding_args, dtype):
    fake_mode = FakeTensorMode()

    with fake_mode:
        batch, channels, height, width = input_shape
        pad_left, pad_right, pad_top, pad_bottom = padding_args
        output_shape = (
            batch,
            channels,
            height + pad_top + pad_bottom,
            width + pad_left + pad_right,
        )
        fake_output = torch.empty(output_shape, dtype=dtype)
        if not hasattr(padding_node, "meta"):
            padding_node.meta = {}
        padding_node.meta["val"] = fake_output

        return fake_output


class RecomposePadMaxPool2d(ExportPass):
    """
    The padding value used in max_pool2d operations differs between PyTorch and QNN implementations.
    PyTorch uses negative infinity, while QNN uses zero. To ensure consistent max_pool2d output across both frameworks,
    we handle this by padding tensor with constant in advance then doing max_pool2d without constant padding.
    Note that for the quantization flow, we set quant_min as the padding value. If, at runtime, there is a value smaller than quant_min,
    it could result in an accuracy drop.
    """

    def __init__(self):
        super(RecomposePadMaxPool2d, self).__init__()
        self.getitem = operator.getitem
        self.max_pool2d = exir_ops.edge.aten.max_pool2d_with_indices.default
        self.pad_op = exir_ops.edge.aten.constant_pad_nd.default

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in graph.nodes:
            num_args = len(node.args)
            if (
                node.op == "call_function"
                and node.target == self.max_pool2d
                and num_args > 3
            ):
                padding = cast(List[int], node.args[3])
                if len(padding) == 1:
                    padding *= 2
                if padding[0] == 0 and padding[1] == 0:
                    continue
                # create padding info for constant_pad_nd
                padding = cast(List[int], node.args[3])
                if len(padding) == 1:
                    padding *= 4
                elif len(padding) == 2:
                    padding = [padding[1], padding[1], padding[0], padding[0]]

                input_node = node.args[0]
                # kernel info
                filter_size = cast(List[int], node.args[1])
                if len(filter_size) == 1:
                    filter_size *= 2
                # stride info
                stride = cast(List[int], node.args[2])
                if len(stride) == 1:
                    stride *= 2
                # dilation info
                dilation = [1, 1]
                if num_args > 4:
                    dilation = cast(List[int], node.args[4])
                    if len(padding) == 1:
                        dilation *= 2

                ceil_mode = node.args[5] if num_args > 5 else False

                # We need to know the minimum value of input tensor of max_pool2d.
                padding_value = float("-inf")
                if quant_attrs := node.meta.get("quant_attrs"):
                    padding_value = quant_attrs.get("quant_min")
                pad_value = padding_value
                if quant_attrs:
                    pad_value = (
                        padding_value - quant_attrs["zero_point"]
                    ) * quant_attrs["scale"]
                with graph_module.graph.inserting_after(input_node):
                    padding_node = graph.create_node(
                        "call_function",
                        self.pad_op,
                        (
                            input_node,
                            padding,
                            pad_value,
                        ),
                    )
                    add_fake_tensor_to_node(
                        padding_node,
                        input_node.meta["val"].shape,
                        padding,
                        input_node.meta["val"].dtype,
                    )
                    if quant_attrs:
                        padding_node.meta["quant_attrs"] = node.meta["quant_attrs"]

                    with graph_module.graph.inserting_after(padding_node):
                        # max_pool2d
                        maxpool2d_args = (
                            padding_node,
                            filter_size,
                            stride,
                            (0, 0),
                            dilation,
                            ceil_mode,
                        )
                        maxpool2d_node_tuple = graph.create_node(
                            "call_function",
                            self.max_pool2d,
                            maxpool2d_args,
                        )
                        if quant_attrs:
                            maxpool2d_node_tuple.meta["quant_attrs"] = node.meta[
                                "quant_attrs"
                            ]
                        maxpool2d_node_tuple.meta["val"] = [None, None]
                        maxpool2d_node_tuple.meta["val"][0] = padding_node.meta["val"]

                        for user in node.users.copy():
                            user.replace_input_with(node, maxpool2d_node_tuple)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
