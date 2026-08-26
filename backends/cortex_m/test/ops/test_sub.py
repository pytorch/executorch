# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from executorch.backends.arm.test.common import parametrize, xfail_type
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)


# Subtraction reuses quantized_add with the second operand's multiplier
# negated, so every lowered case here expects a quantized_add. Equal operands
# cannot see a dropped or misplaced negation, so the pairs are unlike
# everywhere except collapsed_output_scale, which is there for the overflow
# guard.
class CortexMTensorSub(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_sub_Tensor": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 1,
        "executorch_exir_dialects_edge__ops_aten_sub_Tensor": 0,
    }

    def forward(self, x, y):
        return x - y


class CortexMInplaceSub(CortexMTensorSub):
    def forward(self, x, y):
        return x.sub_(y)


class CortexMAlphaSub(CortexMTensorSub):
    """alpha scales the second operand, which is the one place it appears in the
    arithmetic, so it folds into that operand's multiplier."""

    def forward(self, x, y):
        return torch.sub(x, y, alpha=2)


class CortexMNegativeAlphaSub(CortexMTensorSub):
    def forward(self, x, y):
        return torch.sub(x, y, alpha=-2)


class CortexMFloatAlphaSub(torch.nn.Module):
    """A float alpha never reaches the lowering: FoldAndAnnotateQParamsPass
    re-traces with the dequantize nodes gone and aten refuses it on an int8
    sub, so the quantizer has to decline it and leave the op in fp32.

    The boundary quant/dequant pairs are pinned so that a quantizer which
    stopped annotating anything at all would not read as a successful decline.
    """

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_sub_Tensor": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 0,
        "executorch_exir_dialects_edge__ops_aten_sub_Tensor": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 3,
    }

    def forward(self, x, y):
        return torch.sub(x, y, alpha=0.5)


test_cases = {
    "tensor": McuTestCase(
        model=CortexMTensorSub(),
        example_inputs=(
            ramp_tensor(-10, 10, (4, 5)),
            ramp_tensor(-2, 8, (4, 5)),
        ),
    ),
    # Rank 4 in the layout the convolutions produce, which is where a rewrite
    # that only handles the small ranks would go unnoticed.
    "rank_4_channels_last": McuTestCase(
        model=CortexMTensorSub(),
        example_inputs=(
            ramp_tensor(-5, 5, (2, 8, 4, 4)).to(memory_format=torch.channels_last),
            ramp_tensor(-1, 9, (2, 8, 4, 4)).to(memory_format=torch.channels_last),
        ),
    ),
    "rank_5": McuTestCase(
        model=CortexMTensorSub(),
        example_inputs=(
            ramp_tensor(-5, 5, (2, 2, 2, 2, 2)),
            ramp_tensor(-3, 1, (2, 2, 2, 2, 2)),
        ),
    ),
    # One operand an order of magnitude wider than the other, so the two
    # multipliers differ and swapping them changes the result.
    "mismatched_scales": McuTestCase(
        model=CortexMTensorSub(),
        example_inputs=(
            ramp_tensor(-100, 100, (32,)),
            ramp_tensor(-3, 7, (32,)),
        ),
    ),
    # Cancelling operands: the output scale collapses to the observer's
    # eps floor and the kernel's requantization runs one shift below the point
    # at which its int32 left shift would wrap. Sits just inside what the
    # overflow guard permits.
    "collapsed_output_scale": McuTestCase(
        model=CortexMTensorSub(),
        example_inputs=(
            ramp_tensor(-100, 100, (32,)),
            ramp_tensor(-100, 100, (32,)),
        ),
    ),
    "inplace": McuTestCase(
        model=CortexMInplaceSub(),
        example_inputs=(
            ramp_tensor(-10, 10, (4, 5)),
            ramp_tensor(-2, 8, (4, 5)),
        ),
    ),
    "alpha_int": McuTestCase(
        model=CortexMAlphaSub(),
        example_inputs=(
            ramp_tensor(-10, 10, (4, 5)),
            ramp_tensor(-2, 8, (4, 5)),
        ),
    ),
    # A negative alpha turns the subtraction back into an addition, so the
    # folded coefficient has to carry the sign rather than the op.
    "alpha_negative": McuTestCase(
        model=CortexMNegativeAlphaSub(),
        example_inputs=(
            ramp_tensor(-7, 13, (4, 5)),
            ramp_tensor(-1, 5, (4, 5)),
        ),
    ),
    "alpha_float": McuTestCase(
        model=CortexMFloatAlphaSub(),
        example_inputs=(
            ramp_tensor(-10, 10, (4, 5)),
            ramp_tensor(-2, 8, (4, 5)),
        ),
    ),
}

xfails: dict[str, xfail_type] = {}


def test_cancelling_operands_are_refused(cortex_m_target):
    """Operands that cancel leave an output scale the kernel cannot reach.

    `collapsed_output_scale` above is the same shape one shift lower, where the
    kernel still holds; widening the operands is what pushes it over.
    """
    tester = CortexMTester(
        CortexMTensorSub(),
        (ramp_tensor(-150, 150, (32,)), ramp_tensor(-150, 150, (32,))),
        target_config=cortex_m_target,
    )
    tester.quantize().export().to_edge()
    with pytest.raises(Exception) as raised:
        tester.run_passes()
    # The pass manager wraps whatever a pass raises, so check the cause.
    cause = raised.value
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert "the int32 kernel cannot hold" in str(cause), cause


@parametrize("test_case", test_cases, xfails=xfails)
def test_dialect_sub(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms, test_case.model.ops_after_transforms
    )


@parametrize("test_case", test_cases, xfails=xfails)
def test_implementation_sub(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_implementation()
