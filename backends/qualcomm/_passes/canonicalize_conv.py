# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import IntEnum
from typing import cast, Tuple

import torch

from executorch.backends.qualcomm.builders.node_visitor import dq_ops
from executorch.backends.qualcomm.builders.utils import get_parameter
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from torch._guards import detect_fake_mode
from torchao.quantization.pt2e.utils import get_new_attr_name_with_prefix

from .utils import copy_meta


class ConvParamIdx(IntEnum):
    """
    Spec for `aten.convolution` (https://docs.pytorch.org/docs/stable/torch.compiler_ir.html)
    convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding,
                SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor
    """

    INPUT = 0
    WEIGHT = 1
    BIAS = 2
    STRIDE = 3
    PADDING = 4
    DILATION = 5
    TRANSPOSED = 6
    OUTPUT_PADDING = 7
    GROUPS = 8


class CanonicalizeConv(ExportPass):
    """
    1. QNN does not support dilation on TransposeConvND
       Dilate the kernel manually for math-equivalent operation
    2. Conv1d is not supported by QNN.
       Change it to input -> unsqueeze -> conv2d -> squeeze -> output
    """

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super(CanonicalizeConv, self).__init__()
        self.edge_program = edge_program

    def _dilate(self, tensor, dilation):
        # e.g.
        # for 3x3 kernel with dilation == (2, 3)
        #             1, 0, 0, 2, 0, 0, 3
        # 1, 2, 3     0, 0, 0, 0, 0, 0, 0
        # 4, 5, 6 --> 4, 0, 0, 5, 0, 0, 6
        # 7, 8, 9     0, 0, 0, 0, 0, 0, 0
        #             7, 0, 0, 8, 0, 0, 9
        i, o, *k = tensor.shape
        new_k = [dim + (dim - 1) * (s - 1) for s, dim in zip(dilation, k)]
        new_tensor = torch.zeros((i, o, *new_k), dtype=tensor.dtype)
        indexing = (...,) + tuple([slice(None, None, d) for d in dilation])
        new_tensor[indexing] = tensor
        return new_tensor

    def _replace_1d_conv(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        input_node: torch.fx.Node,
    ):
        graph = graph_module.graph
        with graph_module.graph.inserting_after(node):
            unsqueeze_op = exir_ops.edge.aten.unsqueeze_copy.default
            unsqueeze_node = graph.create_node(
                "call_function",
                unsqueeze_op,
                (
                    input_node,
                    2,
                ),
            )
            # This pass is scheduled after the `FoldQDQ` pass. After copying the metadata
            # from the input node, the quantization attributes are also propagated to
            # the corresponding `unsqueeze` node.
            unsqueeze_node.meta = copy_meta(
                input_node.meta, lambda m: {**m, "val": m["val"].unsqueeze(2)}
            )

            with graph_module.graph.inserting_after(unsqueeze_node):
                conv_args = (
                    unsqueeze_node,
                    node.args[ConvParamIdx.WEIGHT],
                    node.args[ConvParamIdx.BIAS],
                    [1] + node.args[ConvParamIdx.STRIDE],
                    [0] + node.args[ConvParamIdx.PADDING],
                    [1] + node.args[ConvParamIdx.DILATION],
                    node.args[ConvParamIdx.TRANSPOSED],
                    [0] + node.args[ConvParamIdx.OUTPUT_PADDING],
                    node.args[ConvParamIdx.GROUPS],
                )
                conv2d_node = graph.create_node(
                    "call_function",
                    exir_ops.edge.aten.convolution.default,
                    conv_args,
                )
                conv2d_node.meta = copy_meta(
                    node.meta, lambda m: {**m, "val": m["val"].unsqueeze(2)}
                )

                with graph_module.graph.inserting_after(conv2d_node):
                    squeeze_op = exir_ops.edge.aten.squeeze_copy.dims
                    squeeze_node = graph.create_node(
                        "call_function",
                        squeeze_op,
                        (
                            conv2d_node,
                            [2],
                        ),
                    )
                    squeeze_node.meta = copy_meta(node.meta)

        for user in node.users.copy():
            user.replace_input_with(node, squeeze_node)

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in graph.nodes:
            # After decomposition to Core ATen, all variants within the conv and conv_transpose family
            # are lowered to `aten.convolution`.
            if node.target is exir_ops.edge.aten.convolution.default:
                stride = cast(Tuple[int], node.args[ConvParamIdx.STRIDE])
                dilation = cast(Tuple[int], node.args[ConvParamIdx.DILATION])
                is_conv_transpose = node.args[ConvParamIdx.TRANSPOSED]
                is_1d_op = len(stride) == 1
                has_conv_transpose_dilation = is_conv_transpose and any(
                    val != 1 for val in dilation
                )
                has_filter_update = is_1d_op or has_conv_transpose_dilation
                if not has_filter_update:
                    continue

                input_node = node.args[ConvParamIdx.INPUT]
                filter_placeholder_node = (
                    # FP graph
                    node.args[ConvParamIdx.WEIGHT]
                    if node.args[ConvParamIdx.WEIGHT].op == "placeholder"
                    # QDQ graph
                    else node.args[ConvParamIdx.WEIGHT].args[0]
                )
                filter_tensor = (
                    get_parameter(filter_placeholder_node, self.edge_program)
                    .contiguous()
                    .detach()
                )

                if has_conv_transpose_dilation:
                    filter_tensor = self._dilate(filter_tensor, dilation)
                    node.args = (
                        *node.args[: ConvParamIdx.DILATION],
                        [1] * len(dilation),
                        *node.args[ConvParamIdx.TRANSPOSED :],
                    )
                if is_1d_op:
                    filter_tensor = filter_tensor.unsqueeze(2)

                if has_filter_update:
                    buffer_name = get_new_attr_name_with_prefix(node.name)(graph_module)
                    graph_module.register_buffer(buffer_name, filter_tensor)

                    with graph_module.graph.inserting_after(filter_placeholder_node):
                        get_attr_node = graph_module.graph.get_attr(buffer_name)
                        fake_mode = detect_fake_mode(
                            filter_placeholder_node.meta["val"]
                        )
                        converter = fake_mode.fake_tensor_converter
                        get_attr_node.meta["val"] = converter.from_real_tensor(
                            fake_mode, filter_tensor
                        )

                        if node.args[ConvParamIdx.WEIGHT].target in dq_ops:
                            filter_dequant_node = node.args[ConvParamIdx.WEIGHT]
                            # Reuse the dequant node and replace its input with the get_attr node.
                            # The quant attrs of the get_attr node are populated by the `AnnotateGetAttr` pass.
                            filter_dequant_node.replace_input_with(
                                filter_placeholder_node, get_attr_node
                            )
                        else:
                            node.replace_input_with(
                                filter_placeholder_node, get_attr_node
                            )

                if is_1d_op:
                    self._replace_1d_conv(graph_module, node, input_node)

        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
