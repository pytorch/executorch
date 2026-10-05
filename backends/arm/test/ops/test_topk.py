# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
from executorch.backends.arm._passes import ConvertInt64OutputOpsToInt32Pass
from executorch.backends.arm.test import common
from executorch.backends.arm.test.runner_utils import TosaReferenceModelDispatch
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    VgfPipeline,
)
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.backends.test.harness.stages import StageType
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.backend.operator_support import DontPartition
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.memory import alloc
from torch.fx.passes.operator_support import OperatorSupportBase

aten_op = "torch.ops.aten.topk.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_topk_default"
_CAST = exir_ops.edge.dim_order_ops._to_dim_order_copy.default
_CAST_OUT = torch.ops.dim_order_ops._to_dim_order_copy.out
_DELEGATE = torch.ops.higher_order.executorch_call_delegate
input_t = tuple[torch.Tensor]


class TopK(torch.nn.Module):
    def __init__(self, k, dim=-1, output="both", largest=True, sorted=True):
        super().__init__()
        self.k = k
        self.dim = dim
        self.output = output
        self.largest = largest
        self.sorted = sorted

    def forward(self, scores):
        values, indices = torch.topk(
            scores, self.k, self.dim, self.largest, self.sorted
        )
        if self.output == "values":
            return values
        if self.output == "indices":
            return indices
        if self.output == "int32":
            return values, indices.int()
        return values, indices


def _residual_ops(manager):
    return [
        node
        for node in manager.exported_program().graph.nodes
        if node.op == "call_function"
        and node.target not in (_DELEGATE, operator.getitem, alloc)
    ]


def _lower(module, inputs, spec="TOSA-1.0+FP+INT", checks=None, dynamic_shapes=None):
    return to_edge_transform_and_lower(
        torch.export.export(module, inputs, dynamic_shapes=dynamic_shapes),
        partitioner=[TOSAPartitioner(TosaCompileSpec(spec), additional_checks=checks)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )


def _execute(manager, inputs):
    with TosaReferenceModelDispatch():
        return manager.exported_program().module()(*inputs)


test_data = {
    f"{dtype}_k{k}_{shape}": (dtype, k, shape)
    for dtype in (torch.float16, torch.float32)
    for k in (1, 2, 3, 4)
    for shape in ("equal", "larger")
}


@common.parametrize("test_data", test_data)
def test_topk_tosa_FP(test_data):
    dtype, k, shape = test_data
    # E: Number of experts; K: Number of activated experts.
    # "equal" means E == K; "larger" means E > K.
    experts = k if shape == "equal" else 8
    scores = torch.arange(experts, dtype=dtype).unsqueeze(0)
    scores = torch.cat((scores.roll(2, 1), -scores.roll(3, 1)), dim=0)
    module = TopK(k, dim=1 if shape == "equal" else -1)
    pipeline = TosaPipelineFP[input_t](
        module,
        (scores,),
        aten_op,
        exir_op,
        tosa_extensions=[] if k == 1 else ["INT"],
        atol=0,
        rtol=0,
    )
    pipeline.count_tosa_ops({"ARGMAX": k, "SELECT": k - 1, "GATHER": 1})
    pipeline.run()


@common.parametrize("dtype", {"fp16": torch.float16, "fp32": torch.float32})
@common.parametrize("k", {"k1": 1, "k4": 4})
@common.SkipIfNoModelConverter
def test_topk_vgf_no_quant(dtype, k):
    scores = torch.tensor([[3.0, -1.0, 4.0, 0.0, 2.0, -2.0, 1.0, -3.0]], dtype=dtype)
    pipeline = VgfPipeline[input_t](
        TopK(k),
        (scores,),
        aten_op,
        exir_op,
        quantize=False,
        tosa_spec="TOSA-1.0+FP+INT",
        atol=0,
        rtol=0,
    )
    pipeline.run()


@common.parametrize("k", {"k1": 1, "k4": 4})
@common.parametrize("output", {key: key for key in ("values", "indices", "int32")})
def test_topk_output_interfaces_tosa_FP(k, output):
    scores = torch.tensor([[2.0, -1.0, 4.0, 0.0]], dtype=torch.float32)
    model = TopK(k, output=output)
    pipeline = TosaPipelineFP[input_t](
        model,
        (scores,),
        aten_op,
        exir_op,
        tosa_extensions=[] if k == 1 else ["INT"],
        atol=0,
        rtol=0,
    )
    pipeline.run()
    manager = pipeline.tester.get_artifact(StageType.TO_EDGE_TRANSFORM_AND_LOWER)
    residual = _residual_ops(manager)
    if output in ("values", "int32"):
        assert not residual
    else:
        assert len(residual) == 1
        assert residual[0].target in (_CAST, _CAST_OUT)
        assert residual[0].meta["val"].dtype == torch.int64


@common.parametrize("dtype", {"fp16": torch.float16, "fp32": torch.float32})
def test_topk_ties_and_extremes_tosa_FP(dtype):
    k = 4
    minimum, maximum = torch.finfo(dtype).min, torch.finfo(dtype).max
    scores = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [minimum] * 8,
            [maximum] * 8,
            [3.0, 2.0, 2.0, 2.0, 2.0, 1.0, -1.0, -2.0],
            [-4.0, -4.0, -2.0, -3.0, -2.0, -5.0, -1.0, -1.0],
            [minimum, maximum, 0.0, -0.0, 1.0, -1.0, minimum, maximum],
        ],
        dtype=dtype,
    )
    pipeline = TosaPipelineFP[input_t](
        TopK(k),
        (scores,),
        aten_op,
        exir_op,
        tosa_extensions=["INT"],
        run_on_tosa_ref_model=False,
    )
    pipeline.run()
    manager = pipeline.tester.get_artifact(StageType.TO_EDGE_TRANSFORM_AND_LOWER)
    values, indices = _execute(manager, (scores,))
    assert values.shape == indices.shape == (scores.shape[0], k)
    assert values.dtype == scores.dtype
    assert indices.dtype == torch.int64
    assert torch.all((indices >= 0) & (indices < scores.shape[1]))
    assert all(
        row.unique().numel() == k for row in indices
    ), "TopK must select K distinct indices per row"
    assert torch.all(values[:, :-1] >= values[:, 1:])
    torch.testing.assert_close(values, scores.gather(1, indices), atol=0, rtol=0)
    torch.testing.assert_close(values, scores.topk(k, dim=1).values, atol=0, rtol=0)
    expected_indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
    torch.testing.assert_close(indices, expected_indices, atol=0, rtol=0)


unsupported = {
    "rank3": (TopK(1), torch.randn(2, 3, 8), "TOSA-1.0+FP+INT"),
    "k5": (TopK(5), torch.randn(2, 8), "TOSA-1.0+FP+INT"),
    "missing_int": (TopK(2), torch.randn(2, 8), "TOSA-1.0+FP"),
    "missing_fp": (TopK(1), torch.randn(2, 8), "TOSA-1.0+INT"),
}


@common.parametrize("test_data", unsupported)
def test_topk_unsupported_tosa_FP(test_data):
    model, scores, spec = test_data
    manager = _lower(model, (scores,), spec)
    assert any(
        n.target == exir_ops.edge.aten.topk.default for n in _residual_ops(manager)
    )
    assert not any(
        n.target == _DELEGATE for n in manager.exported_program().graph.nodes
    )


@common.parametrize("axis", {"tokens": 0, "experts": 1})
def test_topk_dynamic_shapes_not_delegated_tosa_FP(axis):
    scores = torch.randn(3, 8)
    shapes = ({axis: torch.export.Dim("varying", min=3, max=16)},)
    manager = _lower(TopK(2), (scores,), dynamic_shapes=shapes)
    assert any(
        n.target == exir_ops.edge.aten.topk.default for n in _residual_ops(manager)
    )


class TopKGather(torch.nn.Module):
    def forward(self, scores, features):
        values, indices = torch.topk(scores, 3)
        return values, indices, torch.gather(features, 1, indices)


@common.parametrize("portable", {"delegated": False, "portable": True})
def test_topk_gather_boundary_tosa_FP(portable):
    scores = torch.tensor([[3.0, 1.0, 5.0, 0.0, 2.0, 4.0]])
    features = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]])
    checks = [DontPartition(exir_ops.edge.aten.gather.default)] if portable else None
    manager = _lower(TopKGather(), (scores, features), checks=checks)
    residual = _residual_ops(manager)
    gathers = [n for n in residual if n.target == exir_ops.edge.aten.gather.default]
    assert len(gathers) == int(portable)
    if portable:
        assert gathers[0].args[2].meta["val"].dtype == torch.int64
    assert not any(n.target == exir_ops.edge.aten.topk.default for n in residual)
    assert manager.to_executorch().buffer
    for current in (scores, scores.flip(1)):
        actual = _execute(manager, (current, features))
        torch.testing.assert_close(
            actual, TopKGather()(current, features), atol=0, rtol=0
        )


def test_topk_mixed_gather_consumers_tosa_FP():
    class Model(torch.nn.Module):
        def forward(self, scores, first, second):
            indices = torch.topk(scores, 2).indices
            return (
                torch.gather(first, 1, indices),
                torch.gather(second, 1, indices),
                indices,
            )

    class RejectSecondGather(OperatorSupportBase):
        def is_node_supported(self, submodules, node):
            return not (
                node.target == exir_ops.edge.aten.gather.default
                and node.args[0].name == "second"
            )

    scores = torch.tensor([[2.0, 4.0, 1.0, 3.0]])
    inputs = (scores, scores + 10, scores + 20)
    manager = _lower(Model(), inputs, checks=[RejectSecondGather()])
    residual = _residual_ops(manager)
    assert not any(n.target == exir_ops.edge.aten.topk.default for n in residual)
    gathers = [n for n in residual if n.target == exir_ops.edge.aten.gather.default]
    assert len(gathers) == 1
    assert gathers[0].args[0].name == "second"
    assert gathers[0].args[2].meta["val"].dtype == torch.int64
    assert manager.to_executorch().buffer
    torch.testing.assert_close(
        _execute(manager, inputs), Model()(*inputs), atol=0, rtol=0
    )


@common.parametrize(
    "target",
    {
        "topk": exir_ops.edge.aten.topk.default,
        "getitem": operator.getitem,
        "cast": _CAST,
    },
)
def test_topk_incomplete_index_chain_not_delegated_tosa_FP(target):
    scores = torch.tensor([[3.0, 1.0, 2.0, 0.0]])
    manager = _lower(TopK(2), (scores,), checks=[DontPartition(target)])
    assert any(
        n.target == exir_ops.edge.aten.topk.default for n in _residual_ops(manager)
    )
    assert not any(
        n.target == _DELEGATE for n in manager.exported_program().graph.nodes
    )
    torch.testing.assert_close(
        manager.exported_program().module()(scores), TopK(2)(scores)
    )


def test_topk_unsafe_index_arithmetic_tosa_FP():
    class Model(torch.nn.Module):
        def forward(self, scores):
            indices = torch.topk(scores, 1).indices
            return indices * indices

    scores = torch.zeros(1, 50001)
    scores[0, -1] = 1
    manager = _lower(Model(), (scores,), "TOSA-1.0+FP")
    actual = _execute(manager, (scores,))
    assert actual.dtype == torch.int64
    assert actual.item() == 2500000000
    assert manager.to_executorch().buffer


@common.parametrize("prepared", {"prepared": True, "unprepared": False})
def test_topk_legacy_partitioning_tosa_FP(prepared):
    scores = torch.tensor([[3.0, 1.0, 0.0, 2.0]])
    compile_spec = TosaCompileSpec("TOSA-1.0+FP+INT")
    manager = to_edge(
        torch.export.export(TopK(2), (scores,)),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    if prepared:
        manager = manager.transform(
            [
                ConvertInt64OutputOpsToInt32Pass(
                    convert_cast_ops=False, tosa_spec=compile_spec.tosa_spec
                )
            ]
        )
    manager = manager.to_backend(TOSAPartitioner(compile_spec))
    has_topk = any(
        n.target == exir_ops.edge.aten.topk.default for n in _residual_ops(manager)
    )
    assert has_topk != prepared
    if prepared:
        assert manager.to_executorch().buffer
        actual = _execute(manager, (scores,))
    else:
        actual = manager.exported_program().module()(scores)
    torch.testing.assert_close(actual, TopK(2)(scores), atol=0, rtol=0)
