# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import copy_meta


class DecomposeColIm(ExportPass):
    """
    Decompose im2col(unfold) to pixel_unshuffle + view_copy
    Decompose col2im(fold) to view_copy + pixel_shuffle
    """

    def __init__(self):
        super(DecomposeColIm, self).__init__()
        self.im2col_op = exir_ops.edge.aten.im2col.default
        self.col2im_op = exir_ops.edge.aten.col2im.default
        self.pixel_unshuffle_op = exir_ops.edge.aten.pixel_unshuffle.default
        self.pixel_shuffle_op = exir_ops.edge.aten.pixel_shuffle.default
        self.view_copy_op = exir_ops.edge.aten.view_copy.default

    def _decompose_im2col(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.target == self.im2col_op:
                input_node = node.args[0]
                kernel_size = node.args[1]
                stride = node.args[4]
                batch_size = node.meta["val"].shape[0]
                assert (
                    stride == kernel_size
                ), "im2col can only be converted when stride == kernel_size"
                assert (
                    input_node.meta["val"].dim() == 4
                ), "im2col can only be converted when input dims == 4"
                assert (
                    kernel_size[0] == kernel_size[1]
                ), "im2col can only be converted when kernel height == width"
                users = list(node.users.keys())
                with graph_module.graph.inserting_after(input_node):
                    pixel_unshuffle_node = graph_module.graph.create_node(
                        "call_function",
                        self.pixel_unshuffle_op,
                        (input_node, kernel_size[0]),
                    )
                    pixel_unshuffle_node.meta = copy_meta(node.meta)
                    orig_height = input_node.meta["val"].shape[2]
                    orig_width = input_node.meta["val"].shape[3]

                    pixel_unshuffle_node.meta["val"] = pixel_unshuffle_node.meta[
                        "val"
                    ].reshape(
                        batch_size,
                        -1,
                        orig_height // kernel_size[0],
                        orig_width // kernel_size[1],
                    )

                    with graph_module.graph.inserting_after(pixel_unshuffle_node):
                        view_copy_node = graph_module.graph.create_node(
                            "call_function",
                            self.view_copy_op,
                            (pixel_unshuffle_node, tuple(node.meta["val"].shape)),
                        )
                        view_copy_node.meta = copy_meta(node.meta)
                        for user in users:
                            user.replace_input_with(node, view_copy_node)

    def _decompose_col2im(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.target == self.col2im_op:
                input_node = node.args[0]
                output_size = node.args[1]
                kernel_size = node.args[2]
                stride = node.args[5]
                batch_size = node.meta["val"].shape[0]
                assert (
                    stride == kernel_size
                ), "col2im can only be converted when stride == kernel_size"
                assert (
                    node.meta["val"].dim() == 4
                ), "col2im can only be converted when output dims == 4"
                assert (
                    kernel_size[0] == kernel_size[1]
                ), "col2im can only be converted when kernel height == width"
                users = list(node.users.keys())
                with graph_module.graph.inserting_after(input_node):
                    view_tensor = input_node.meta["val"].reshape(
                        batch_size,
                        -1,
                        output_size[0] // kernel_size[0],
                        output_size[1] // kernel_size[1],
                    )
                    view_copy_node = graph_module.graph.create_node(
                        "call_function",
                        self.view_copy_op,
                        (input_node, tuple(view_tensor.shape)),
                    )
                    view_copy_node.meta = copy_meta(node.meta)
                    view_copy_node.meta["val"] = view_tensor

                    with graph_module.graph.inserting_after(view_copy_node):
                        pixel_shuffle_node = graph_module.graph.create_node(
                            "call_function",
                            self.pixel_shuffle_op,
                            (view_copy_node, kernel_size[0]),
                        )
                        pixel_shuffle_node.meta = copy_meta(node.meta)

                        for user in users:
                            user.replace_input_with(node, pixel_shuffle_node)

    def call(self, graph_module: torch.fx.GraphModule):
        self._decompose_im2col(graph_module)
        self._decompose_col2im(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)
