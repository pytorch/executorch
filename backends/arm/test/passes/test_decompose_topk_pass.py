# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm._passes import (
    ConvertInt64OutputOpsToInt32Pass,
    DecomposeTopKPass,
)
from executorch.backends.arm._passes.decompose_topk_pass import TOPK_OPS
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops


@pytest.mark.parametrize("k", [1, 4])
def test_decompose_topk_static_shapes_and_cumulative_masking(k):
    dtype = torch.float32

    class Model(torch.nn.Module):
        def forward(self, scores):
            values, indices = torch.topk(scores, k)
            return values, indices.int()

    ep = to_edge(
        torch.export.export(Model(), (torch.randn(2, 8, dtype=dtype),)),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    spec = TosaSpecification.create_from_string("TOSA-1.0+FP+INT")
    gm = ConvertInt64OutputOpsToInt32Pass(convert_cast_ops=False, tosa_spec=spec)(
        ep.graph_module
    ).graph_module
    with TosaLoweringContext(spec):
        result = DecomposeTopKPass(spec)(gm)
        repeated = DecomposeTopKPass(spec)(result.graph_module)
    assert result.modified and not repeated.modified
    nodes = list(result.graph_module.graph.nodes)
    assert not any(n.target in TOPK_OPS for n in nodes)
    argmaxes = [n for n in nodes if n.target == exir_ops.backend.tosa.ARGMAX.default]
    masks = [n for n in nodes if n.target == exir_ops.edge.aten.where.self]
    assert len(argmaxes) == k
    assert len(masks) == k - 1
    scores = next(n for n in nodes if n.op == "placeholder")
    for step, argmax in enumerate(argmaxes):
        assert argmax.meta["val"].dtype == torch.int32
        assert tuple(argmax.meta["val"].shape) == (2,)
        assert argmax.args[0] is (scores if step == 0 else masks[step - 1])
        if step < k - 1:
            assert masks[step].args[2] is argmax.args[0]
    gather = next(n for n in nodes if n.target == exir_ops.edge.aten.gather.default)
    assert gather.args[0] is scores
    assert gather.args[2].meta["val"].dtype == torch.int32
    assert tuple(gather.meta["val"].shape) == (2, k)
    assert gather.meta["val"].dtype == dtype
    if k > 1:
        sentinel = next(n for n in nodes if n.target == exir_ops.edge.aten.full.default)
        assert sentinel.args[1] == float("-inf")
        assert sentinel.kwargs["dtype"] == dtype
