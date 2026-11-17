# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Mish activation coverage for Arm TOSA, Ethos-U and VGF backends."""

# Review coverage: Mish inplace, INT8 reference and partitioning.
import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.backends.test.harness.stages import StageType


class _MishOpSupportModule(torch.nn.Module):
    """Mish model used to establish API coverage in generated support docs."""

    test_data = {
        "rank1_ones": lambda: torch.ones(5),
        "rank1_negative_ones": lambda: -torch.ones(5),
        "rank4_mixed": lambda: torch.linspace(-8.0, 8.0, 64).reshape(1, 2, 4, 8),
        "rank4_large_positive": lambda: (
            200.0 * torch.linspace(0.001, 1.0, 64).reshape(1, 2, 4, 8)
        ),
        "rank4_large_negative": lambda: (
            # Preserve a large negative range without losing the
            # significant Mish outputs to INT8 input quantization.
            -40.0
            * torch.linspace(0.001, 1.0, 64).reshape(1, 2, 4, 8)
        ),
        "rank4_large_mixed": lambda: (
            torch.linspace(-200.0, 200.0, 64).reshape(1, 2, 4, 8)
        ),
    }

    def forward(self, x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        # The comparison pipeline reuses the same input. Cloning also ensures
        # the test does not mutate the caller-owned tensor in the inplace case.
        return torch.nn.Mish(inplace=inplace)(x.clone())


# Names used by ArmTester's check_not after to_edge_transform_and_lower.
# In particular, no decomposed Mish arithmetic should remain on the CPU.
_FP_DECOMPOSED_EDGE_OPS = [
    "executorch_exir_dialects_edge__ops_aten_softplus_default",
    "executorch_exir_dialects_edge__ops_aten_exp_default",
    "executorch_exir_dialects_edge__ops_aten_log1p_default",
    "executorch_exir_dialects_edge__ops_aten_tanh_default",
    "executorch_exir_dialects_edge__ops_aten_where_self",
    "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
]
_MISH_EDGE_OPS = [
    "executorch_exir_dialects_edge__ops_aten_mish_default",
    "executorch_exir_dialects_edge__ops_aten_mish__default",
]


def _assert_full_delegation(pipeline) -> None:
    """Reject any computational fallback outside the delegate.

    The explicit input clone is a test harness detail. Quantize/dequantize
    boundary nodes and getitem are non-activation boundary operations.

    """
    edge = pipeline.tester.get_artifact(StageType.TO_EDGE_TRANSFORM_AND_LOWER)
    allowed = (
        "executorch_call_delegate",
        "getitem",
        "quantize_per",
        "dequantize_per",
        "aten.clone",
    )
    remaining = [
        str(node.target)
        for node in edge.exported_program().graph.nodes
        if node.op == "call_function"
        and not any(allowed_token in str(node.target) for allowed_token in allowed)
    ]
    assert not remaining, f"Mish left computational CPU operations: {remaining}"


def _enforce_full_delegation(pipeline) -> None:
    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        lambda: _assert_full_delegation(pipeline),
    )


def _frobenius_threshold(x: torch.Tensor) -> float:
    # Negative Mish outputs are close to zero, so their relative error is
    # disproportionately sensitive to small absolute INT8 differences.
    return 0.3 if bool(torch.all(x < 0)) else 0.15


@common.parametrize("test_data", _MishOpSupportModule.test_data)
def test_mish_tosa_FP(test_data):
    """Decomposed floating-point Mish must execute on the TOSA reference."""
    pipeline = TosaPipelineFP(
        _MishOpSupportModule().eval(),
        (test_data(), False),
        aten_op="torch.ops.aten.mish.default",
        exir_op=_FP_DECOMPOSED_EDGE_OPS,
    )
    _enforce_full_delegation(pipeline)
    pipeline.run()


@common.parametrize("test_data", _MishOpSupportModule.test_data)
def test_mish_tosa_FP_inplace(test_data):
    pipeline = TosaPipelineFP(
        _MishOpSupportModule().eval(),
        (test_data(), True),
        aten_op="torch.ops.aten.mish_.default",
        exir_op=_FP_DECOMPOSED_EDGE_OPS,
    )
    _enforce_full_delegation(pipeline)
    pipeline.run()


@common.parametrize("test_data", _MishOpSupportModule.test_data)
def test_mish_tosa_INT(test_data):
    """Check INT8 Mish numerics and require exactly one TOSA TABLE op."""
    x = test_data()
    pipeline = TosaPipelineINT(
        _MishOpSupportModule().eval(),
        (x, False),
        aten_op=[],
        exir_op=_MISH_EDGE_OPS,
        frobenius_threshold=_frobenius_threshold(x),
    )
    pipeline.count_tosa_ops({"TABLE": 1})
    _enforce_full_delegation(pipeline)
    pipeline.run()


@common.parametrize("test_data", _MishOpSupportModule.test_data)
def test_mish_tosa_INT_inplace(test_data):
    """In-place Mish must functionalize to an INT8 TABLE, without CPU ops."""
    x = test_data()
    pipeline = TosaPipelineINT(
        _MishOpSupportModule().eval(),
        (x, True),
        aten_op=[],
        exir_op=_MISH_EDGE_OPS,
        frobenius_threshold=_frobenius_threshold(x),
    )
    pipeline.count_tosa_ops({"TABLE": 1})
    _enforce_full_delegation(pipeline)
    pipeline.run()


@common.parametrize("test_data", _MishOpSupportModule.test_data)
@common.XfailIfNoCorstone300
def test_mish_u55_INT(test_data):
    """Execute quantized Mish on the U55 FVP, when available."""
    pipeline = EthosU55PipelineINT(
        _MishOpSupportModule().eval(),
        (test_data(), False),
        aten_ops="torch.ops.aten.mish.default",
        exir_ops=_MISH_EDGE_OPS,
    )
    _enforce_full_delegation(pipeline)
    pipeline.run()


@common.parametrize("test_data", _MishOpSupportModule.test_data)
@common.XfailIfNoCorstone300
def test_mish_u55_INT_inplace(test_data):
    pipeline = EthosU55PipelineINT(
        _MishOpSupportModule().eval(),
        (test_data(), True),
        aten_ops="torch.ops.aten.mish_.default",
        exir_ops=_MISH_EDGE_OPS,
    )
    _enforce_full_delegation(pipeline)
    pipeline.run()


@common.parametrize("test_data", _MishOpSupportModule.test_data)
@common.XfailIfNoCorstone320
def test_mish_u85_INT(test_data):
    """Quantized Mish must execute as one delegate on the U85 FVP."""
    pipeline = EthosU85PipelineINT(
        _MishOpSupportModule().eval(),
        (test_data(), False),
        aten_ops="torch.ops.aten.mish.default",
        exir_ops=_MISH_EDGE_OPS,
    )
    _enforce_full_delegation(pipeline)
    pipeline.run()


@common.parametrize("test_data", _MishOpSupportModule.test_data)
@common.XfailIfNoCorstone320
def test_mish_u85_INT_inplace(test_data):
    pipeline = EthosU85PipelineINT(
        _MishOpSupportModule().eval(),
        (test_data(), True),
        aten_ops="torch.ops.aten.mish_.default",
        exir_ops=_MISH_EDGE_OPS,
    )
    _enforce_full_delegation(pipeline)
    pipeline.run()


@common.parametrize("test_data", _MishOpSupportModule.test_data)
@common.SkipIfNoModelConverter
def test_mish_vgf_no_quant(test_data):
    """Validate FP Mish numerics and full VGF delegation."""
    pipeline = VgfPipeline(
        _MishOpSupportModule().eval(),
        (test_data(), False),
        aten_op="torch.ops.aten.mish.default",
        exir_op=[],
        quantize=False,
    )
    _enforce_full_delegation(pipeline)
    pipeline.run()


@common.parametrize("test_data", _MishOpSupportModule.test_data)
@common.SkipIfNoModelConverter
def test_mish_vgf_no_quant_inplace(test_data):
    pipeline = VgfPipeline(
        _MishOpSupportModule().eval(),
        (test_data(), True),
        aten_op="torch.ops.aten.mish_.default",
        exir_op=[],
        quantize=False,
    )
    _enforce_full_delegation(pipeline)
    pipeline.run()


@common.parametrize("test_data", _MishOpSupportModule.test_data)
@common.SkipIfNoModelConverter
def test_mish_vgf_quant(test_data):
    """Validate INT8 Mish numerics and full VGF delegation."""
    pipeline = VgfPipeline(
        _MishOpSupportModule().eval(),
        (test_data(), False),
        aten_op="torch.ops.aten.mish.default",
        exir_op=_MISH_EDGE_OPS,
        quantize=True,
    )
    _enforce_full_delegation(pipeline)
    pipeline.run()


@common.parametrize("test_data", _MishOpSupportModule.test_data)
@common.SkipIfNoModelConverter
def test_mish_vgf_quant_inplace(test_data):
    pipeline = VgfPipeline(
        _MishOpSupportModule().eval(),
        (test_data(), True),
        aten_op="torch.ops.aten.mish_.default",
        exir_op=_MISH_EDGE_OPS,
        quantize=True,
    )
    _enforce_full_delegation(pipeline)
    pipeline.run()
