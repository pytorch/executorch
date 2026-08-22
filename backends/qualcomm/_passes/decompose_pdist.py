# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass

from .utils import merge_decomposed_graph


class PDist(torch.nn.Module):
    def __init__(self, N):
        super().__init__()
        # Precompute row and column indices for upper triangle pairs
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        self.register_buffer(
            "rows", torch.tensor([i for i, _ in pairs], dtype=torch.int32)
        )
        self.register_buffer(
            "cols", torch.tensor([j for _, j in pairs], dtype=torch.int32)
        )

    def forward(self, x):
        # Select paired rows directly — only N*(N-1)/2 differences computed
        diff = torch.index_select(x, 0, self.rows) - torch.index_select(x, 0, self.cols)

        # Square differences
        sq_diff = diff**2

        # Sum of squares
        sum_sq_diff = sq_diff.sum(dim=-1)

        # Square root
        return torch.sqrt(sum_sq_diff)


class DecomposePDist(ExportPass):
    """
    Decompose aten.pdist and aten._pdist_forward into supported primitives.

    torch.pdist(x, p=2) computes pairwise Euclidean distances between all
    row pairs of a 2D input tensor x of shape [N, M], returning a condensed
    1D vector of length N*(N-1)/2 (upper triangle, row-major order).

    Decomposition (p=2 only):
        1. diff = x[rows] - x[cols]           # index_select paired rows: [N*(N-1)/2, M]
        2. sq_diff = diff ** 2                # element-wise square
        3. sum_sq  = sq_diff.sum(dim=-1)      # sum over feature dim -> [N*(N-1)/2]
        4. out    = sqrt(sum_sq)              # Euclidean distances [N*(N-1)/2]

    Row and column indices for upper triangle pairs are precomputed at
    decomposition time (requires static N). This avoids materializing the
    full [N, N, M] difference tensor.

    Only p=2 is supported.
    """

    pdist_targets = {
        torch.ops.aten.pdist.default,
        torch.ops.aten._pdist_forward.default,
    }

    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in list(graph.nodes):
            if node.target in self.pdist_targets:
                # Check p value - only p=2 is supported
                if len(node.args) > 1:
                    p = node.args[1]
                elif "p" in node.kwargs:
                    p = node.kwargs["p"]
                else:
                    p = 2.0
                assert (
                    p == 2 or p == 2.0
                ), "Currently only p=2 is supported for PDist Decomposition"

                input_val = node.args[0].meta["val"]
                N = input_val.shape[0]
                assert isinstance(N, int), (
                    f"DecomposePDist requires static shapes but got symbolic "
                    f"dimension N={N}. Dynamic shapes are not supported."
                )

                model = PDist(N)
                decomposed_module = torch.export.export(
                    model,
                    (input_val,),
                    strict=True,
                ).module()

                rows_attr = f"_pdist_rows_{node.name}"
                cols_attr = f"_pdist_cols_{node.name}"
                graph_module.register_buffer(rows_attr, model.rows)
                graph_module.register_buffer(cols_attr, model.cols)

                for decomposed_node in decomposed_module.graph.nodes:
                    if decomposed_node.op == "get_attr":
                        if decomposed_node.target == "rows":
                            decomposed_node.target = rows_attr
                        elif decomposed_node.target == "cols":
                            decomposed_node.target = cols_attr

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

        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
