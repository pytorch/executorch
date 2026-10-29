# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm._passes.rewrite_argmax_pass import RewriteArgmaxPass
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops


class ArgmaxBranches(torch.nn.Module):
    def __init__(self, dim, keyword_dim):
        super().__init__()
        self.dim = dim
        self.keyword_dim = keyword_dim

    def forward(self, tensor):
        if self.keyword_dim:
            indices = torch.argmax(tensor, dim=self.dim)
        else:
            indices = torch.argmax(tensor, self.dim)
        first = indices.to(torch.int32)
        second = indices.to(torch.int32)
        return first, second.to(torch.float32)


class ArgmaxBoundary(torch.nn.Module):
    def __init__(self, dtype, return_raw, dim=1, keepdim=False):
        super().__init__()
        self.dtype = dtype
        self.return_raw = return_raw
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, tensor):
        indices = torch.argmax(tensor, dim=self.dim, keepdim=self.keepdim)
        converted = indices.to(self.dtype)
        return (indices, converted) if self.return_raw else converted


@pytest.mark.parametrize("dim", (0, 1, -1))
@pytest.mark.parametrize("keyword_dim", (False, True))
def test_rewrite_argmax_consumes_all_int32_casts(dim, keyword_dim):
    program = to_edge(
        torch.export.export(ArgmaxBranches(dim, keyword_dim), (torch.randn(2, 3),))
    ).exported_program()
    argmax = next(
        node
        for node in program.graph.nodes
        if node.target == exir_ops.edge.aten.argmax.default
    )
    assert len(argmax.users) == 2
    argmax.meta["rewrite_marker"] = "argmax"
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        result = RewriteArgmaxPass()(program.graph_module)
    assert result is not None and result.modified
    result.graph_module.graph.lint()
    nodes = list(result.graph_module.graph.nodes)
    lowered = [
        node for node in nodes if node.target == exir_ops.backend.tosa.ARGMAX.default
    ]
    assert len(lowered) == 1
    assert lowered[0].args[1] == dim % 2
    assert lowered[0].meta["val"].dtype == torch.int32
    assert lowered[0].meta["rewrite_marker"] == "argmax"
    assert all(node.target != exir_ops.edge.aten.argmax.default for node in nodes)
    casts = [
        node
        for node in nodes
        if node.target == exir_ops.edge.dim_order_ops._to_dim_order_copy.default
    ]
    assert len(casts) == 1
    assert casts[0].args[0] is lowered[0]
    assert casts[0].kwargs["dtype"] == torch.float32
    outputs = result.graph_module.graph.output_node().args[0]
    assert outputs[0] is lowered[0]
    assert outputs[1].meta["val"].dtype == torch.float32
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        repeated = RewriteArgmaxPass()(result.graph_module)
    assert repeated is not None and not repeated.modified


@pytest.mark.parametrize(
    "module",
    [
        ArgmaxBoundary(torch.int32, True),
        ArgmaxBoundary(torch.float32, False),
        ArgmaxBoundary(torch.int64, False),
        ArgmaxBoundary(torch.int32, False, dim=None),
        ArgmaxBoundary(torch.int32, False, keepdim=True),
    ],
)
def test_rewrite_argmax_preserves_unmatched_boundaries(module):
    program = to_edge(
        torch.export.export(module, (torch.randn(2, 3),))
    ).exported_program()
    original_code = program.graph_module.code
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        result = RewriteArgmaxPass()(program.graph_module)
    assert result is not None and not result.modified
    assert result.graph_module.code == original_code
    argmax = next(
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.argmax.default
    )
    assert argmax.meta["val"].dtype == torch.int64
    assert all(
        node.target != exir_ops.backend.tosa.ARGMAX.default
        for node in result.graph_module.graph.nodes
    )
