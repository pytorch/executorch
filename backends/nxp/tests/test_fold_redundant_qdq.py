# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.edge_passes.fold_redundant_qdq_pass import (
    FoldRedundantDequantizeQuantizePass,
)
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program

ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate


class ConvDropoutConvModule(torch.nn.Module):
    """Two conv clusters separated by an eval-mode dropout (an identity).

    In eval mode the dropout is a no-op, but the quantizer still wraps it in a
    shared-spec ``dequantize -> quantize`` cluster (identical scale/zero-point on
    both sides). Lowering eliminates the dropout itself, yet leaves that wrapping
    quantization behind as a redundant ``dequantize -> quantize`` pair with no
    compute node between the two conv clusters. The Neutron partitioner cannot
    delegate a compute-free QDQ island, so it splits ``conv1`` and ``conv2`` into
    two separate subgraphs -- the same split that
    ``FoldRedundantDequantizeQuantizePass`` was written to prevent. Folding that
    pair away lets both convs delegate as a single subgraph.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, kernel_size=2, bias=False)
        self.dropout = torch.nn.Dropout(0.5)
        self.conv2 = torch.nn.Conv2d(8, 8, kernel_size=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


def _count_delegates(edge_program) -> int:
    graph = edge_program.exported_program().graph_module.graph
    return sum(
        1
        for node in graph.nodes
        if node.op == "call_function" and node.target == ExecutorchDelegateCall
    )


INPUT_SHAPE = (1, 4, 8, 8)


def test_fold_pass_present_merges_into_single_delegate():
    # The fold pass is part of the default NeutronEdgePassManager.
    edge_program = to_quantized_edge_program(ConvDropoutConvModule(), INPUT_SHAPE)

    num_delegates = _count_delegates(edge_program)
    assert (
        num_delegates == 1
    ), f"expected a single delegate with the fold pass, got {num_delegates}"


def test_fold_pass_removes_redundant_qdq():
    graph = torch.fx.Graph()
    quantized_input = graph.placeholder("quantized_input")
    qparams = (0.25, 3, -128, 127, torch.int8)
    dequantize = graph.call_function(
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        args=(quantized_input, *qparams),
    )
    quantize = graph.call_function(
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        args=(dequantize, *qparams),
    )
    graph.output(quantize)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    result = FoldRedundantDequantizeQuantizePass().run(graph_module)

    assert result.modified
    remaining_nodes = list(result.graph_module.graph.nodes)
    assert [node.op for node in remaining_nodes] == ["placeholder", "output"]
    assert remaining_nodes[-1].args == (quantized_input,)
