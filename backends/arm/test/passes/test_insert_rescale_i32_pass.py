# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Tuple

import pytest
import torch
from executorch.backends.arm._passes import (
    FoldAndAnnotateQParamsPass,
    InsertRescaleInt32Pass,
)
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm._passes.replace_scalar_with_tensor_pass import (
    ReplaceScalarWithTensorByProfilePass,
)
from executorch.backends.arm._passes.tosa_lowering_utils import create_rescale_node
from executorch.backends.arm.common.annotation_meta import ArmAnnotationInfo
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


class MultipleOpsModel(torch.nn.Module):
    """A module containing ops that require INT32 inputs/outputs."""

    input_t = Tuple[torch.Tensor, torch.Tensor]

    def forward(self, x, y):
        a = x - y
        b = x * a
        c = torch.maximum(a, b)
        d = torch.abs(b)
        e = c + d
        f = e > a
        return f

    def get_inputs(self, dtype) -> input_t:
        if dtype == torch.float32:
            return (torch.rand(1, 3, 5, 6), torch.rand(1, 3, 5, 6))
        elif dtype == torch.int32:
            return (
                torch.randint(3, 5, (3,), dtype=torch.int32),
                torch.randint(3, 5, (3,), dtype=torch.int32),
            )
        else:
            raise ValueError("Not a valid input dtype for model")

    def get_num_expected_rescales(self):
        # "number of op nodes with i8 output" + "number of i8 node inputs"
        return 5 + 11


class SumModel(torch.nn.Module):
    input_t = Tuple[torch.Tensor]

    def forward(self, x):
        a = torch.sum(x, 2, keepdim=True)  # (1, 2, 1, 4)
        b = torch.sum(a, [1, 3], keepdim=True)  # (1, 1, 1, 1)
        c = torch.sum(b, [0, 2], keepdim=False)  # (1, 1)
        return c

    def get_inputs(self, dtype) -> input_t:
        if dtype == torch.float32:
            return (torch.rand(1, 2, 3, 4),)
        elif dtype == torch.int32:
            return (torch.randint(0, 10, (1, 2, 3, 4), dtype=torch.int32),)
        else:
            raise ValueError("Not a valid input dtype for model")

    def get_num_expected_rescales(self):
        # Two RESCALE nodes per SUM node
        return 6


def _test_model_with_f32_data(model):
    ops_not_before = {"executorch_exir_dialects_backend__ops_tosa_RESCALE_default"}
    ops_after = {
        "executorch_exir_dialects_backend__ops_tosa_RESCALE_default": model.get_num_expected_rescales(),
    }
    pipeline = PassPipeline[model.input_t](
        model,
        model.get_inputs(torch.float32),
        quantize=True,
        ops_not_before_pass=ops_not_before,
        ops_after_pass=ops_after,
        pass_list=[FoldAndAnnotateQParamsPass, InsertRescaleInt32Pass],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


def test_insert_rescale_int32_tosa_INT_sum():
    _test_model_with_f32_data(SumModel())


def test_insert_rescale_int32_tosa_INT_multiple_ops():
    _test_model_with_f32_data(MultipleOpsModel())


@pytest.mark.parametrize("dtype", (torch.int8, torch.int16))
@pytest.mark.parametrize(
    "unary_target",
    (None, exir_ops.edge.aten.tanh.default, exir_ops.edge.aten.log.default),
)
@pytest.mark.parametrize(
    "binary_target",
    (exir_ops.edge.aten.add.Tensor, exir_ops.edge.aten.mul.Tensor),
)
def test_insert_rescale_int32_zero_points_match_quantized_operands(
    dtype: torch.dtype,
    unary_target: Callable[..., object] | None,
    binary_target: Callable[..., object],
) -> None:
    graph = torch.fx.Graph()
    left = graph.placeholder("left")
    right = graph.placeholder("right")
    unary = graph.call_function(unary_target, (left,)) if unary_target else left
    binary = graph.call_function(binary_target, (unary, right))
    graph.output(binary)
    with FakeTensorMode():
        for node in (left, right, unary, binary):
            node.meta["val"] = torch.empty((2, 3), dtype=dtype)
    limits = torch.iinfo(dtype)
    qparams = QuantArgs(0.5, 0, limits.min, limits.max, dtype)
    if unary_target:
        unary.meta["input_qparams"] = {0: qparams}
        unary.meta["output_qparams"] = {0: qparams}
    binary.meta["input_qparams"] = {0: qparams, 1: qparams}
    binary.meta["output_qparams"] = {0: qparams}
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        result = InsertRescaleInt32Pass()(graph_module)

    assert result is not None
    result.graph_module.graph.lint()
    rescales = result.graph_module.graph.find_nodes(
        op="call_function", target=exir_ops.backend.tosa.RESCALE.default
    )
    assert len(rescales) == 3
    for rescale in rescales:
        assert rescale.args[3] == rescale.args[4] == 0
        assert isinstance(rescale.args[1], (tuple, list))
        assert isinstance(rescale.args[2], (tuple, list))
    output_rescale = rescales[-1]
    assert output_rescale.args[0].meta["val"].dtype == torch.int32
    assert output_rescale.meta["val"].dtype == dtype


@pytest.mark.parametrize(
    "binary_target",
    (exir_ops.edge.aten.add.Tensor, exir_ops.edge.aten.mul.Tensor),
)
@pytest.mark.parametrize("dtype", (torch.int8, torch.int16))
def test_insert_rescale_int32_uses_corrected_dtype_through_unannotated_view(
    binary_target, dtype
):
    graph = torch.fx.Graph()
    left = graph.placeholder("left")
    right = graph.placeholder("right")
    unary = graph.call_function(exir_ops.edge.aten.tanh.default, (left,))
    view = graph.call_function(exir_ops.edge.aten.view_copy.default, (unary, [2, 3]))
    binary = graph.call_function(binary_target, (view, right))
    graph.output(binary)
    with FakeTensorMode():
        for node in (left, right, binary):
            node.meta["val"] = torch.empty((2, 3), dtype=dtype)
        for node in (unary, view):
            node.meta["val"] = torch.empty((2, 3), dtype=torch.float32)
    limits = torch.iinfo(dtype)
    zero_point = 3 if dtype == torch.int8 else 0
    qparams = QuantArgs(0.5, zero_point, limits.min, limits.max, dtype)
    unary.meta.update(input_qparams={0: qparams}, output_qparams={0: qparams})
    binary.meta.update(
        input_qparams={0: qparams, 1: qparams}, output_qparams={0: qparams}
    )

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        scalar_result = ReplaceScalarWithTensorByProfilePass()(
            torch.fx.GraphModule({}, graph)
        )
        assert scalar_result is not None
        result = InsertRescaleInt32Pass()(scalar_result.graph_module)

    assert result is not None
    result.graph_module.graph.lint()
    input_rescales = [
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.RESCALE.default
        and node.kwargs["dtype"] == torch.int32
    ]
    assert len(input_rescales) == 2
    for rescale in input_rescales:
        assert "input_dtype" not in rescale.kwargs
        assert rescale.args[3] == zero_point


@pytest.mark.parametrize(
    "binary_target",
    (exir_ops.edge.aten.add.Tensor, exir_ops.edge.aten.mul.Tensor),
)
def test_insert_rescale_int32_preserves_partial_qdq(
    binary_target: Callable[..., object],
) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    x_q = graph.call_function(
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        (x, 0.5, 0, -128, 127, torch.int8),
    )
    x_dq = graph.call_function(
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        (x_q, 0.5, 0, -128, 127, torch.int8),
    )
    binary = graph.call_function(binary_target, (x_dq, y))
    binary.meta["custom"] = {
        ArmAnnotationInfo.CUSTOM_META_KEY: ArmAnnotationInfo(quantized=True)
    }
    output_q = graph.call_function(
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        (binary, 0.5, 0, -128, 127, torch.int8),
    )
    output_dq = graph.call_function(
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        (output_q, 0.5, 0, -128, 127, torch.int8),
    )
    graph.output(output_dq)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    fold_result = FoldAndAnnotateQParamsPass(preserve_partial_binary_tensor_qdq=True)(
        graph_module
    )
    assert fold_result is not None
    rescale_result = InsertRescaleInt32Pass()(fold_result.graph_module)
    assert rescale_result is not None

    assert binary.args == (x_dq, y)
    assert output_q in binary.users
    assert not rescale_result.graph_module.graph.find_nodes(
        op="call_function",
        target=exir_ops.backend.tosa.RESCALE.default,
        sort=False,
    )


@pytest.mark.parametrize("dtype", (torch.int8, torch.int16))
def test_rescale_builder_uses_input_metadata_dtype(dtype):
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    with FakeTensorMode():
        source.meta["val"] = torch.empty((2, 3), dtype=dtype)

    output = create_rescale_node(graph, (source, torch.int32, [1.0], 0, 0))
    graph.output(output)
    graph.lint()

    assert output.args[3] == 0
    assert not graph.find_nodes(
        op="call_function", target=exir_ops.backend.tosa.CONST.default
    )
    assert output.meta["val"].dtype == torch.int32
    assert source.meta["val"].dtype == dtype


def test_insert_rescale_int32_tosa_FP_dont_insert_rescales():
    module = MultipleOpsModel()
    input_t = Tuple[torch.Tensor, torch.Tensor]
    ops_not_before = {"executorch_exir_dialects_backend__ops_tosa_RESCALE_default"}
    # All inputs are already i32. Rescales should not be added.
    ops_not_after = {"executorch_exir_dialects_backend__ops_tosa_RESCALE_default"}
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(torch.int32),
        ops_not_before_pass=ops_not_before,
        ops_not_after_pass=ops_not_after,
        pass_list=[FoldAndAnnotateQParamsPass, InsertRescaleInt32Pass],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()
