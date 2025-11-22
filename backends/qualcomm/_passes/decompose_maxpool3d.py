# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import cast, List

import torch
import torch.nn as nn
from executorch.exir import to_edge
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import merge_decomposed_graph


class ModelMaxPool3D(torch.nn.Module):
    def __init__(
        self, filter_size, stride, padding, dilation, return_indices, ceil_mode
    ):
        super().__init__()

        self.pool2d_hw = nn.MaxPool2d(
            kernel_size=[1, filter_size[2]],  # (H, W) part
            stride=[1, stride[2]],
            padding=[0, padding[2]],
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )
        self.pool2d_dh = nn.MaxPool2d(
            kernel_size=filter_size[:2],  # (D, H) part
            stride=stride[:2],
            padding=padding[:2],
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, D, H, W = x.shape
        x_ = x.permute(0, 1, 4, 2, 3)
        x1_1d = x_.reshape(N * C, W, D, H)
        # first pool over (D, H)
        out_pool1d_0 = self.pool2d_dh(x1_1d)
        D_out = out_pool1d_0.shape[2]
        # NC, W, D, H-> NC, D, H, W
        x1b = out_pool1d_0.permute(0, 2, 3, 1)
        # second pool over (H, W)
        out4d = self.pool2d_hw(x1b)
        H_out2 = out4d.shape[2]
        W_out = out4d.shape[3]
        out = out4d.reshape(N, C, D_out, H_out2, W_out)
        return out


class DecomposeMaxPool3d(ExportPass):
    # The max_pool3d is not supported yet by QNN.
    # Decompose: input -> permute -> reshape -> max_pool2d -> permute -> max_pool2d -> reshape -> output

    def __init__(self, quantization_capture=False) -> None:
        super().__init__()
        self.quantization_capture = quantization_capture

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if node.op == "call_function" and "max_pool3d" in str(node.target):
                # kernel info
                filter_size = cast(List[int], node.args[1])
                if len(filter_size) == 1:
                    filter_size *= 3

                num_args = len(node.args)

                # stride info
                stride = filter_size
                if num_args > 2:
                    stride = cast(List[int], node.args[2])
                    if len(stride) == 1:
                        stride *= 3

                # padding info
                padding = [0, 0, 0]
                if num_args > 3:
                    padding = cast(List[int], node.args[3])
                    if len(padding) == 1:
                        padding *= 3

                # dilation info
                dilation = [1, 1, 1]
                if num_args > 4:
                    dilation = cast(List[int], node.args[4])
                    if len(padding) == 1:
                        dilation *= 3

                ceil_mode = node.args[5] if num_args > 5 else False
                return_indices = node.args[6] if num_args > 6 else False
                if return_indices:
                    warnings.warn(
                        "[QNN Delegate Op Builder]: The case return_indices=True is not be support, fallback",
                        stacklevel=1,
                    )
                    return

                model = ModelMaxPool3D(
                    filter_size, stride, padding, dilation, return_indices, ceil_mode
                )
                if self.quantization_capture:
                    decomposed_module = torch.export.export(
                        model, (node.args[0].meta["val"],), strict=True
                    ).module()
                else:
                    edge_mgr = to_edge(
                        torch.export.export(
                            model, (node.args[0].meta["val"],), strict=True
                        )
                    )
                    decomposed_module = edge_mgr.exported_program()

                with graph.inserting_before(node):
                    # remap is used to map original node values to new node values,
                    # which ensures that reference to nodes are correctly updated in the new graph
                    remap = {"x": node.args[0]}
                    merge_decomposed_graph(
                        remap=remap,
                        target_node=node,
                        target_graph=graph,
                        decomposed_graph_module=decomposed_module,
                    )
                    graph.erase_node(node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
