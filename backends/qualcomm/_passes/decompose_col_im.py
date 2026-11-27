# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass

from .utils import copy_meta, create_const_node


class DecomposeColIm(ExportPass):
    """
    Decompose im2col(unfold) to pad + index_select + space_to_depth + view_copy
    Decompose col2im(fold) to view_copy + pixel_shuffle
    """

    # When stride != kernel_size, the index_select gather in
    # _extend_for_space_to_depth duplicates elements. Skip decomposition if the
    # expansion_factor is too large to avoid unbounded memory consumption;
    # raise the value if a real case would benefit from more overlap.
    _MAX_GATHER_EXPANSION_FACTOR = 16

    def __init__(self):
        super(DecomposeColIm, self).__init__()
        self.im2col_op = exir_ops.edge.aten.im2col.default
        self.col2im_op = exir_ops.edge.aten.col2im.default
        self.pixel_unshuffle_op = exir_ops.edge.aten.pixel_unshuffle.default
        self.pixel_shuffle_op = exir_ops.edge.aten.pixel_shuffle.default
        self.view_copy_op = exir_ops.edge.aten.view_copy.default
        self.index_select_op = exir_ops.edge.aten.index_select.default
        self.space_to_depth_op = exir_ops.edge.qnn_custom.space_to_depth.default
        self.pad_op = exir_ops.edge.aten.constant_pad_nd.default

    def _emit_space_to_depth(
        self,
        graph: torch.fx.Graph,
        meta,
        input_node: torch.fx.Node,
        kernel_size: list[int],
    ):
        kernel_height, kernel_width = kernel_size
        input_val = input_node.meta["val"]
        val = self.space_to_depth_op(input_val, kernel_height, kernel_width)
        node = graph.create_node(
            "call_function",
            self.space_to_depth_op,
            (input_node, kernel_height, kernel_width),
        )
        node.meta = copy_meta(meta)
        node.meta["val"] = val
        return node

    def _extend_for_space_to_depth(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        input_node: torch.fx.Node,
        kernel_size,
        stride,
        padding,
    ):
        graph = graph_module.graph
        if any(p != 0 for p in padding):
            # aten.im2col zero-pads symmetrically on both sides
            # of each spatial dim before extracting windows;
            # constant_pad_nd takes amounts last-dim-first:
            # [width_left, width_right, height_top, height_bottom].
            pad_amount = [padding[1], padding[1], padding[0], padding[0]]
            pad_val = self.pad_op(input_node.meta["val"], pad_amount, 0.0)
            pad_node = graph.create_node(
                "call_function", self.pad_op, (input_node, pad_amount, 0.0)
            )
            pad_node.meta = copy_meta(node.meta)
            pad_node.meta["val"] = pad_val
            input_node = pad_node
        # im2col with stride != kernel_size: gather each spatial dim (one 1-D
        # index_select per dim) so the result tiles exactly into
        # output_block_height*kernel_height by output_block_width*kernel_width -- i.e.
        # what space_to_depth expects when stride == kernel_size, matching
        # torch.nn.functional.unfold's output layout:
        #   x.index_select(2, height_idx).index_select(3, width_idx)

        for kernel_value, stride_value, dim in zip(kernel_size, stride, [2, 3]):
            input_val = input_node.meta["val"]
            input_dim_value = input_val.shape[dim]
            if kernel_value != stride_value:
                num_output_block = (input_dim_value - kernel_value) // stride_value + 1
                expand_idx = [
                    j * stride_value + i
                    for j in range(num_output_block)
                    for i in range(kernel_value)
                ]
                idx_node = create_const_node(
                    graph,
                    graph_module,
                    f"{node.name}_dim_{dim}_expand_idx",
                    expand_idx,
                    input_node,
                    const_dtype=torch.int32,
                    static_shapes=True,  # use static or it become symbolic shape
                )
                gather_node = graph.create_node(
                    "call_function", self.index_select_op, (input_node, dim, idx_node)
                )
                gather_node.meta = copy_meta(node.meta)
                gather_node.meta["val"] = input_val.index_select(
                    dim, idx_node.meta["val"]
                )
                input_node = gather_node

        return input_node

    def _decompose_im2col(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.target == self.im2col_op:
                input_node = node.args[0]
                kernel_size = node.args[1]
                dilation = node.args[2]
                padding = node.args[3]
                stride = node.args[4]
                if input_node.meta["val"].dim() != 4:
                    logging.warning(
                        f"{node}: im2col input must be 4-D to be decomposed, got "
                        f"dim={input_node.meta['val'].dim()}. Skipping decomposition."
                    )
                    continue
                if any(d != 1 for d in dilation):
                    logging.warning(
                        f"{node}: im2col can only be decomposed when dilation == "
                        f"(1, 1), got dilation={dilation}. Skipping decomposition."
                    )
                    continue

                # Given a array length L with kernel K and stride S, the
                # expanded output is |1...K|1+S...K+S|...|1+(N-1)*S...K+(N-1)*S|
                # the expanded output length is K*N, where N = (L-K)/S+1,
                # expansion_factor = K*N/L = K*(L-K+S)/(L*S)
                # e.g., L=9, K=5, S=2, then expanded output is
                # |1...5|3...7|5...9|, expansion_factor = 5*(9-5+2)/(9*2) = 30/18
                # Assume L is largely greater than K and S, simplify
                # expansion_factor to K/S
                expansion_factor = (
                    kernel_size[0] / stride[0] * kernel_size[1] / stride[1]
                )
                if expansion_factor > self._MAX_GATHER_EXPANSION_FACTOR:
                    logging.warning(
                        f"{node}: im2col decomposition would expand the gather "
                        f"by {expansion_factor}x (kernel_size={kernel_size}, "
                        f"stride={stride}), exceeding the "
                        f"{self._MAX_GATHER_EXPANSION_FACTOR}x limit imposed to "
                        "avoid excessive memory consumption. Skipping "
                        "decomposition -- raise "
                        "DecomposeColIm._MAX_GATHER_EXPANSION_FACTOR if a real "
                        "case would benefit from loosening this threshold."
                    )
                    continue
                graph = graph_module.graph
                users = list(node.users.keys())
                x = input_node
                with graph.inserting_before(node):
                    # given input [1,1,4,4] with kernel=[2,2] and stride=[1,1]
                    # e.g.,input
                    # input = [[ 0,  1,  2,  3],
                    #          [ 4,  5,  6,  7],
                    #          [ 8,  9, 10, 11],
                    #          [12, 13, 14, 15]]
                    # since kernel is not equal to stride
                    # it will be extend to a input shape with kernel=stride=[2,2]
                    # e.g.,
                    # input = [[ 0,  1,  1,  2,  2,  3],
                    #          [ 4,  5,  5,  6,  6,  7],
                    #          [ 4,  5,  5,  6,  6,  7],
                    #          [ 8,  9,  9, 10, 10, 11],
                    #          [ 8,  9,  9, 10, 10, 11],
                    #          [12, 13, 13, 14, 14, 15]]
                    x = self._extend_for_space_to_depth(
                        graph_module, node, x, kernel_size, stride, padding
                    )

                    s2d_node = self._emit_space_to_depth(
                        graph, node.meta, x, kernel_size
                    )

                    final_view_node = graph.create_node(
                        "call_function",
                        self.view_copy_op,
                        (s2d_node, tuple(node.meta["val"].shape)),
                    )
                    final_view_node.meta = copy_meta(node.meta)

                for user in users:
                    user.replace_input_with(node, final_view_node)

    def _decompose_col2im(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.target == self.col2im_op:
                input_node = node.args[0]
                output_size = node.args[1]
                kernel_size = node.args[2]
                dilation = node.args[3]
                padding = node.args[4]
                stride = node.args[5]
                batch_size = node.meta["val"].shape[0]
                if stride != kernel_size:
                    logging.warning(
                        f"{node}: col2im can only be decomposed when stride == "
                        f"kernel_size, got stride={stride}, kernel_size="
                        f"{kernel_size}. Skipping decomposition."
                    )
                    continue
                if node.meta["val"].dim() != 4:
                    logging.warning(
                        f"{node}: col2im output must be 4-D to be decomposed, got "
                        f"dim={node.meta['val'].dim()}. Skipping decomposition."
                    )
                    continue
                if kernel_size[0] != kernel_size[1]:
                    logging.warning(
                        f"{node}: col2im can only be decomposed when kernel "
                        f"height == width, got kernel_size={kernel_size}. "
                        f"Skipping decomposition."
                    )
                    continue
                if any(d != 1 for d in dilation):
                    logging.warning(
                        f"{node}: col2im can only be decomposed when dilation == "
                        f"(1, 1), got dilation={dilation}. Skipping decomposition."
                    )
                    continue
                if any(p != 0 for p in padding):
                    logging.warning(
                        f"{node}: col2im can only be decomposed when padding == "
                        f"(0, 0), got padding={padding}. Skipping decomposition."
                    )
                    continue

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
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
