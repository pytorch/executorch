# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import pytest
import torch
from executorch.backends.arm._passes.arm_pass_utils import (
    is_strictly_positive_tensor_node,
)
from executorch.backends.arm._passes.decompose_add_sub_alpha_pass import (
    DecomposeAddSubAlphaPass,
)
from executorch.backends.arm._passes.decompose_pow_tensor_tensor_pass import (
    DecomposePowTensorTensorPass,
)
from executorch.backends.arm._passes.scalars_to_attribute_pass import (
    ScalarsToAttributePass,
)
from executorch.backends.arm.quantizer.quantizer_support import (
    PowTensorTensorPositiveBaseCheck,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.backends.arm.tosa.specification import TosaSpecification


class Pow_TensorTensor(torch.nn.Module):
    aten_op = "torch.ops.aten.pow.Tensor_Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Tensor"

    input_t = Tuple[torch.Tensor | float, torch.Tensor | float]

    # The sign of the operands are important w.r.t. TOSA's spec of pow
    test_data = {
        "zero_base_pos_exp": lambda: (
            torch.zeros(1, 8, 3, 7),
            torch.abs(torch.randn((1, 8, 1, 7))) + 1e5,
        ),
        "pos_base": lambda: (
            torch.abs(torch.randn((3, 2, 4, 2))) + 1e5,
            torch.randn((1, 2, 4, 1)),
        ),
        "zero_base_zero_exp": lambda: (torch.zeros(2, 3), torch.zeros(2, 3)),
        "pos_base_zero_exp": lambda: (
            torch.abs(torch.randn((1, 7, 2, 3))) + 1e5,
            torch.zeros(1, 1, 2, 3),
        ),
        "neg_base_zero_exp": lambda: (
            -torch.abs(torch.randn((1, 2, 3, 4))) - 1e5,
            torch.zeros(1, 2, 3, 4),
        ),
        "base_has_lower_rank": lambda: (torch.ones(3, 4), torch.ones(1, 2, 3, 4)),
        "exp_has_lower_rank": lambda: (torch.ones(1, 2, 3, 4), torch.ones(3, 4)),
        "f16_tensors": lambda: (
            torch.HalfTensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]),
            torch.HalfTensor([[1.0, 2.0, 0.0]]),
        ),
    }

    test_data_bf16 = {
        "bf16_tensors": lambda: (
            torch.ones((2, 3), dtype=torch.bfloat16),
            torch.full((2, 3), 2, dtype=torch.bfloat16),
        ),
    }

    test_data_quant = {
        "positive_base_positive_exp": lambda: (
            torch.rand(2, 3, 4) * 1.5 + 0.25,
            torch.rand(2, 3, 4) * 1.5 + 0.25,
        ),
        "broadcast_exp": lambda: (
            torch.rand(1, 3, 4, 4) * 1.5 + 0.25,
            torch.rand(1, 3, 1, 1) * 1.5 + 0.25,
        ),
        "integer_valued_exp": lambda: (
            torch.rand(2, 4) * 1.5 + 0.25,
            torch.full((2, 4), 2.0),
        ),
        "broadcast_base": lambda: (
            torch.rand(2, 1) * 1.5 + 0.25,
            torch.rand(2, 4) * 1.5 + 0.25,
        ),
    }

    def forward(self, x: torch.Tensor | float, y: torch.Tensor | float):
        return torch.pow(x, y)


class Pow_TensorTensorPositiveBase(torch.nn.Module):
    """Tensor/Tensor pow with a graph-visible strictly-positive base."""

    input_t = Pow_TensorTensor.input_t

    def forward(self, x: torch.Tensor | float, y: torch.Tensor | float):
        # Encode the positivity guarantee in the graph using operators that
        # are supported by the U55 INT path:
        #
        #     abs(x) >= 0
        #     abs(x) + 0.25 > 0
        #
        # Unlike positive calibration inputs, this constrains all runtime
        # inputs reaching pow.
        return torch.pow(torch.abs(x) + 0.25, y)


class Pow_TensorScalar(torch.nn.Module):
    aten_op = "torch.ops.aten.pow.Tensor_Scalar"
    exir_op = "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar"

    input_t = Tuple[torch.Tensor]

    test_data = {
        # Test whole number exponents
        "exp_minus_three": lambda: (torch.randn((10, 5)).relu() + 0.1, -3.0),
        "exp_minus_one": lambda: (torch.randn((42,)).relu() + 0.1, -1.0),
        "exp_zero": lambda: (torch.randn((1, 2, 3, 7)).relu(), 0.0),
        "exp_one": lambda: (torch.randn((1, 4, 6, 2)).relu(), 1.0),
        "exp_two": lambda: (torch.randn((1, 2, 3, 6)), 2.0),
        # Test decimal exponent (base must be non-negative)
        "non_neg_base_exp_pos_decimal": lambda: (
            torch.abs(torch.randn((1, 2, 3, 6))),
            6.789,
        ),
        "neg_base_exp_pos_integer": lambda: (
            -torch.abs(torch.randn((1, 2, 3, 6))) - 10,
            3,
        ),
    }

    test_data_fp16 = {
        "exp_minus_three_fp16": lambda: (
            (torch.randn((10, 5), dtype=torch.float16).relu() + 0.1, -3.0)
        )
    }

    test_data_bf16 = {
        "exp_minus_three_bf16": lambda: (
            (torch.randn((10, 5), dtype=torch.bfloat16).relu() + 0.1, -3.0)
        )
    }

    def __init__(self, exp):
        super().__init__()
        self.exp = exp

    def forward(self, x: torch.Tensor):
        return torch.pow(x, self.exp)


x_fail = {
    "zero_base_zero_exp": "TOSA constraints: If x == 0 and y ⇐ 0, the result is undefined.",
    "neg_base_zero_exp": "TOSA constraints: If x == 0 and y ⇐ 0, the result is undefined.",
}


POW_TENSOR_TENSOR_INT_ATEN_OPS = [
    "torch.ops.aten.log.default",
    "torch.ops.aten.mul.Tensor",
    "torch.ops.aten.exp.default",
]

POW_TENSOR_TENSOR_INT_EXIR_OPS = [
    "executorch_exir_dialects_edge__ops_aten_log_default",
    "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
    "executorch_exir_dialects_edge__ops_aten_exp_default",
]


@common.parametrize(
    "test_data",
    Pow_TensorTensor.test_data | Pow_TensorTensor.test_data_bf16,
    x_fail,
    strict=False,
)
def test_pow_tensor_tensor_tosa_FP(test_data: Pow_TensorTensor.input_t):
    pipeline = TosaPipelineFP[Pow_TensorTensor.input_t](
        Pow_TensorTensor(),
        test_data(),
        Pow_TensorTensor.aten_op,
        Pow_TensorTensor.exir_op,
        tosa_extensions=["bf16"],
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    Pow_TensorTensor.test_data | Pow_TensorTensor.test_data_bf16,
    x_fail,
    strict=False,
)
@common.SkipIfNoModelConverter
def test_pow_tensor_tensor_vgf_no_quant(test_data: Pow_TensorTensor.input_t):
    pipeline = VgfPipeline[Pow_TensorTensor.input_t](
        Pow_TensorTensor(),
        test_data(),
        Pow_TensorTensor.aten_op,
        Pow_TensorTensor.exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    Pow_TensorScalar.test_data
    | Pow_TensorScalar.test_data_fp16
    | Pow_TensorScalar.test_data_bf16,
)
def test_pow_tensor_scalar_tosa_FP(test_data: Pow_TensorScalar.input_t):
    base, exp = test_data()
    pipeline = TosaPipelineFP[Pow_TensorScalar.input_t](
        Pow_TensorScalar(exp),
        (base,),
        Pow_TensorScalar.aten_op,
        Pow_TensorScalar.exir_op,
        tosa_extensions=["bf16"],
    )
    pipeline.run()


@common.parametrize("test_data", Pow_TensorScalar.test_data, strict=False)
def test_pow_tensor_scalar_tosa_INT(test_data: Pow_TensorScalar.input_t):
    base, exp = test_data()
    pipeline = TosaPipelineINT[Pow_TensorScalar.input_t](
        Pow_TensorScalar(exp),
        (base,),
        Pow_TensorScalar.aten_op,
        Pow_TensorScalar.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", Pow_TensorScalar.test_data)
@common.XfailIfNoCorstone300
def test_pow_tensor_scalar_u55_INT(test_data: Pow_TensorScalar.input_t):
    base, exp = test_data()
    pipeline = EthosU55PipelineINT[Pow_TensorScalar.input_t](
        Pow_TensorScalar(exp),
        (base,),
        Pow_TensorScalar.aten_op,
        Pow_TensorScalar.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", Pow_TensorScalar.test_data)
@common.XfailIfNoCorstone320
def test_pow_tensor_scalar_u85_INT(test_data: Pow_TensorScalar.input_t):
    base, exp = test_data()
    pipeline = EthosU85PipelineINT[Pow_TensorScalar.input_t](
        Pow_TensorScalar(exp),
        (base,),
        Pow_TensorScalar.aten_op,
        Pow_TensorScalar.exir_op,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    Pow_TensorScalar.test_data
    | Pow_TensorScalar.test_data_bf16
    | Pow_TensorScalar.test_data_fp16,
)
@common.SkipIfNoModelConverter
def test_pow_tensor_scalar_vgf_no_quant(test_data: Pow_TensorScalar.input_t):
    base, exp = test_data()
    pipeline = VgfPipeline[Pow_TensorScalar.input_t](
        Pow_TensorScalar(exp),
        (base,),
        Pow_TensorScalar.aten_op,
        Pow_TensorScalar.exir_op,
        quantize=False,
    )
    if base.dtype == torch.float16:
        pipeline.change_args("run_method_and_compare_outputs", atol=128.0, rtol=0.01)
    pipeline.run()


@common.parametrize(
    "test_data",
    Pow_TensorScalar.test_data,
)
@common.SkipIfNoModelConverter
def test_pow_tensor_scalar_vgf_quant(test_data: Pow_TensorScalar.input_t):
    base, exp = test_data()
    pipeline = VgfPipeline[Pow_TensorScalar.input_t](
        Pow_TensorScalar(exp),
        (base,),
        Pow_TensorScalar.aten_op,
        Pow_TensorScalar.exir_op,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", Pow_TensorTensor.test_data_quant)
@common.SkipIfNoModelConverter
def test_pow_tensor_tensor_vgf_quant(
    test_data: Pow_TensorTensor.input_t,
):
    pipeline = VgfPipeline[Pow_TensorTensor.input_t](
        Pow_TensorTensor(),
        test_data(),
        Pow_TensorTensor.aten_op,
        Pow_TensorTensor.exir_op,
        quantize=True,
    )

    pipeline.run_and_compare_to_initial_model(
        frobenius_threshold=0.3,
        cosine_threshold=0.9,
    )

    pipeline.run()


@common.parametrize("test_data", Pow_TensorTensor.test_data_quant)
def test_pow_tensor_tensor_tosa_INT(
    test_data: Pow_TensorTensor.input_t,
):
    pipeline = TosaPipelineINT[Pow_TensorTensor.input_t](
        Pow_TensorTensorPositiveBase(),
        test_data(),
        POW_TENSOR_TENSOR_INT_ATEN_OPS,
        POW_TENSOR_TENSOR_INT_EXIR_OPS,
        qtol=4,
    )
    pipeline.run()


@common.parametrize("test_data", Pow_TensorTensor.test_data_quant)
@common.XfailIfNoCorstone300
def test_pow_tensor_tensor_u55_INT(
    test_data: Pow_TensorTensor.input_t,
):
    pipeline = EthosU55PipelineINT[Pow_TensorTensor.input_t](
        Pow_TensorTensorPositiveBase(),
        test_data(),
        POW_TENSOR_TENSOR_INT_ATEN_OPS,
        POW_TENSOR_TENSOR_INT_EXIR_OPS,
        qtol=4,
    )
    pipeline.run()


@common.parametrize("test_data", Pow_TensorTensor.test_data_quant)
@common.XfailIfNoCorstone320
def test_pow_tensor_tensor_u85_INT(
    test_data: Pow_TensorTensor.input_t,
):
    pipeline = EthosU85PipelineINT[Pow_TensorTensor.input_t](
        Pow_TensorTensorPositiveBase(),
        test_data(),
        POW_TENSOR_TENSOR_INT_ATEN_OPS,
        POW_TENSOR_TENSOR_INT_EXIR_OPS,
        qtol=4,
    )
    pipeline.run()


def test_pow_tensor_tensor_int_rejects_base_lower_bound_quantized_to_zero():
    """A positive FP lower bound must stay positive after LOG input QDQ."""
    # abs(x) + 0.25 is structurally >= 0.25, but this calibration range makes
    # the LOG input INT8 scale large enough that 0.25 maps to the zero code.
    test_data = (
        torch.tensor([[0.0, 128.0]], dtype=torch.float32),
        torch.full((1, 2), -1.0, dtype=torch.float32),
    )

    pipeline = TosaPipelineINT[Pow_TensorTensor.input_t](
        Pow_TensorTensorPositiveBase(),
        test_data,
        POW_TENSOR_TENSOR_INT_ATEN_OPS,
        POW_TENSOR_TENSOR_INT_EXIR_OPS,
        qtol=4,
        run_on_tosa_ref_model=False,
    )

    pipeline.pop_stage("run_method_and_compare_outputs.original_model")

    with pytest.raises(Exception, match="InsertTableOpsPass") as exc_info:
        pipeline.run()

    cause = exc_info.value.__cause__
    assert isinstance(cause, RuntimeError)
    assert "Unsafe Tensor/Tensor POW INT decomposition" in str(cause)


def test_pow_tensor_tensor_int_requires_graph_proof_of_positive_base():
    """Calibration values must not be treated as a runtime positivity proof."""
    exponent = torch.full((2, 2), 2.0)

    unconstrained = torch.export.export(
        Pow_TensorTensor(),
        (
            torch.full((2, 2), -2.0),
            exponent,
        ),
    )
    constrained = torch.export.export(
        Pow_TensorTensorPositiveBase(),
        (
            torch.full((2, 2), -2.0),
            exponent,
        ),
    )

    unconstrained_pow = [
        node
        for node in unconstrained.graph_module.graph.nodes
        if node.target == torch.ops.aten.pow.Tensor_Tensor
    ]
    constrained_pow = [
        node
        for node in constrained.graph_module.graph.nodes
        if node.target == torch.ops.aten.pow.Tensor_Tensor
    ]

    assert len(unconstrained_pow) == 1
    assert len(constrained_pow) == 1

    # An ordinary runtime tensor can contain negative values. Even if all
    # calibration samples were positive, the optimization is not safe.
    assert not PowTensorTensorPositiveBaseCheck.check_pattern(unconstrained_pow)

    # abs(x) + 0.25 provides a structural proof that log(base) is safe.
    assert PowTensorTensorPositiveBaseCheck.check_pattern(constrained_pow)


def test_pow_tensor_tensor_decomposes_after_scalar_materialization():
    """Regression: materialized get_attr constants must remain provable."""
    exported = torch.export.export(
        Pow_TensorTensorPositiveBase(),
        (
            torch.randn(2, 2),
            torch.full((2, 2), 1.5),
        ),
    )

    graph_module = exported.graph_module
    scalar_result = ScalarsToAttributePass(tfa_pass=True)(graph_module)
    assert scalar_result is not None
    graph_module = scalar_result.graph_module

    pow_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.target == torch.ops.aten.pow.Tensor_Tensor
    ]
    assert len(pow_nodes) == 1
    assert isinstance(pow_nodes[0].args[0], torch.fx.Node)
    assert is_strictly_positive_tensor_node(pow_nodes[0].args[0])

    result = DecomposePowTensorTensorPass(
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        tfa_pass=True,
    )(graph_module)
    assert result is not None

    targets = {
        node.target
        for node in result.graph_module.graph.nodes
        if node.op == "call_function"
    }
    assert torch.ops.aten.pow.Tensor_Tensor not in targets
    assert torch.ops.aten.log.default in targets
    assert torch.ops.aten.mul.Tensor in targets
    assert torch.ops.aten.exp.default in targets


def test_pow_tensor_tensor_decomposes_after_add_alpha_canonicalization():
    """Non-default add alpha must not break the positive-base proof."""

    class PositiveBaseWithAlpha(torch.nn.Module):
        def forward(self, x, y):
            base = torch.add(torch.abs(x), 0.25, alpha=2.0)
            return torch.pow(base, y)

    exported = torch.export.export(
        PositiveBaseWithAlpha(),
        (
            torch.randn(2, 2),
            torch.full((2, 2), 1.5),
        ),
    )

    graph_module = exported.graph_module
    alpha_result = DecomposeAddSubAlphaPass(tfa_pass=True)(graph_module)
    assert alpha_result is not None
    graph_module = alpha_result.graph_module

    scalar_result = ScalarsToAttributePass(tfa_pass=True)(graph_module)
    assert scalar_result is not None
    graph_module = scalar_result.graph_module

    pow_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.target == torch.ops.aten.pow.Tensor_Tensor
    ]
    assert len(pow_nodes) == 1
    assert isinstance(pow_nodes[0].args[0], torch.fx.Node)
    assert is_strictly_positive_tensor_node(pow_nodes[0].args[0])

    result = DecomposePowTensorTensorPass(
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        tfa_pass=True,
    )(graph_module)
    assert result is not None

    targets = {
        node.target
        for node in result.graph_module.graph.nodes
        if node.op == "call_function"
    }
    assert torch.ops.aten.pow.Tensor_Tensor not in targets
    assert torch.ops.aten.log.default in targets
    assert torch.ops.aten.mul.Tensor in targets
    assert torch.ops.aten.exp.default in targets


def test_clamp_negative_max_does_not_prove_strictly_positive():
    """A positive clamp min is insufficient when max can force negativity."""

    class ClampModule(torch.nn.Module):
        def forward(self, x):
            return torch.clamp(x, min=0.25, max=-1.0)

    exported = torch.export.export(
        ClampModule(),
        (torch.zeros(2, dtype=torch.float32),),
    )
    clamp_nodes = [
        node
        for node in exported.graph_module.graph.nodes
        if node.target == torch.ops.aten.clamp.default
    ]
    assert len(clamp_nodes) == 1
    assert not is_strictly_positive_tensor_node(clamp_nodes[0])


def test_clamp_positive_bounds_prove_strictly_positive():
    """Two-sided clamp is safe when both effective bounds are positive."""

    class ClampModule(torch.nn.Module):
        def forward(self, x):
            return torch.clamp(x, min=0.25, max=1.0)

    exported = torch.export.export(
        ClampModule(),
        (torch.zeros(2, dtype=torch.float32),),
    )
    clamp_nodes = [
        node
        for node in exported.graph_module.graph.nodes
        if node.target == torch.ops.aten.clamp.default
    ]
    assert len(clamp_nodes) == 1
    assert is_strictly_positive_tensor_node(clamp_nodes[0])


def test_tiny_positive_scalar_does_not_prove_strictly_positive():
    """A Python scalar that underflows to zero in float32 is not a proof."""

    class TinyOffsetModule(torch.nn.Module):
        def forward(self, x):
            return torch.abs(x) + 1.0e-100

    exported = torch.export.export(
        TinyOffsetModule(),
        (torch.zeros(2, dtype=torch.float32),),
    )
    add_nodes = [
        node
        for node in exported.graph_module.graph.nodes
        if node.target == torch.ops.aten.add.Tensor
    ]
    assert len(add_nodes) == 1
    assert not is_strictly_positive_tensor_node(add_nodes[0])


def test_exp_does_not_prove_strictly_positive():
    """Floating-point exp may underflow to zero."""

    class ExpModule(torch.nn.Module):
        def forward(self, x):
            return torch.exp(x)

    exported = torch.export.export(
        ExpModule(),
        (torch.tensor([-1000.0], dtype=torch.float32),),
    )

    exp_nodes = [
        node
        for node in exported.graph_module.graph.nodes
        if node.target == torch.ops.aten.exp.default
    ]

    assert len(exp_nodes) == 1

    # exp(-1000) underflows to zero in float32, so graph structure alone
    # cannot establish the strict positivity required by the POW
    # decomposition.
    assert not is_strictly_positive_tensor_node(exp_nodes[0])
