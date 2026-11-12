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


@pytest.mark.parametrize("k", [1, 4, 8])
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


@pytest.mark.parametrize("k", [1, 4, 8])
def test_decompose_topk_generated_nodes_preserve_provenance(k: int) -> None:
    """Every replacement op must remain attributable to its TopK source.

    TopK has tuple-valued metadata, while its replacements are single tensors.
    The pass must retain source information without leaking TopK tuple shapes or
    its quantization parameters into the generated operations.

    """

    class Model(torch.nn.Module):
        def forward(self, scores: torch.Tensor):
            values, indices = torch.topk(scores, k)
            return values, indices.int()

    spec = TosaSpecification.create_from_string("TOSA-1.0+FP+INT")
    ep = to_edge(
        torch.export.export(Model(), (torch.randn(2, 8),)),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    converted = ConvertInt64OutputOpsToInt32Pass(
        convert_cast_ops=False, tosa_spec=spec
    )(ep.graph_module)
    assert converted is not None
    gm = converted.graph_module
    topk_nodes = [node for node in gm.graph.nodes if node.target in TOPK_OPS]
    assert len(topk_nodes) == 1
    topk = topk_nodes[0]

    # Distinguish the TopK provenance from its input producer's provenance.
    original_stack = {"topk_layer": ("topk_layer", torch.nn.Module)}
    original_source = [("topk_layer.forward", torch.topk)]
    topk.meta["nn_module_stack"] = original_stack
    topk.meta["source_fn_stack"] = original_source
    topk.meta["stack_trace"] = "originating TopK stack trace"
    topk.meta["debug_handle"] = 987654321
    topk.meta["input_qparams"] = {0: "do not copy"}
    topk.meta["output_qparams"] = {0: "do not copy"}

    with TosaLoweringContext(spec):
        result = DecomposeTopKPass(spec)(gm)
    assert result is not None
    assert result.modified

    all_nodes = list(result.graph_module.graph.nodes)
    assert not any(node.target in TOPK_OPS for node in all_nodes)

    # K=1 emits 3 nodes; K>1 emits 4*K+3, including masking.
    generated = [
        node
        for node in all_nodes
        if node.op == "call_function" and node.meta.get("debug_handle") == 987654321
    ]
    assert len(generated) == (3 if k == 1 else 4 * k + 3)
    assert (
        sum(node.target == exir_ops.backend.tosa.ARGMAX.default for node in generated)
        == k
    )
    assert (
        sum(node.target == exir_ops.edge.aten.gather.default for node in generated) == 1
    )
    if k > 1:
        assert (
            sum(node.target == exir_ops.edge.aten.where.self for node in generated)
            == k - 1
        )

    for node in generated:
        assert node.meta.get("nn_module_stack") == original_stack, node.name
        assert node.meta.get("source_fn_stack") == original_source, node.name
        assert "originating TopK stack trace" in node.meta.get("stack_trace", "")
        assert not node.meta.get("input_qparams"), node.name
        assert not node.meta.get("output_qparams"), node.name
        # ExportPass must recalculate the tensor-valued metadata: the original
        # TopK node returns a tuple, but every generated op returns a tensor.
        assert isinstance(node.meta.get("val"), torch.Tensor), node.name

    gather = next(
        node for node in generated if node.target == exir_ops.edge.aten.gather.default
    )
    assert tuple(gather.meta["val"].shape) == (2, k)
    assert gather.meta["val"].dtype == torch.float32
    assert tuple(gather.args[2].meta["val"].shape) == (2, k)
    assert gather.args[2].meta["val"].dtype == torch.int32
