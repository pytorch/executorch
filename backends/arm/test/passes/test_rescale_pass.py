# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineINT,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode

input_t = Tuple[torch.Tensor, torch.Tensor]  # Input x


def test_rescale_op():
    sample_inputs = [
        # (data, out_dtype, scale, in_zp, out_zp)
        (
            torch.randint(low=0, high=100, size=(4, 4, 4), dtype=torch.int8),
            torch.int32,
            0.2,
            2,
            0,
        ),
        (
            torch.randint(low=0, high=100, size=(4, 4, 4), dtype=torch.int32),
            torch.int8,
            0.2,
            0,
            -128,
        ),
        (
            torch.randint(low=0, high=100, size=(4, 4, 4), dtype=torch.int8),
            torch.int8,
            0.8,
            10,
            127,
        ),
    ]

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+INT")
    ), FakeTensorMode() as mode:
        for sample_input in sample_inputs:
            exir_ops.backend.tosa.RESCALE.default(
                *tuple(
                    [
                        mode.from_tensor(i) if isinstance(i, torch.Tensor) else i
                        for i in sample_input
                    ]
                )
            )


def test_nonzero_zp_for_int32():

    sample_inputs = [
        (
            torch.randint(low=0, high=100, size=(4, 4, 4), dtype=torch.int8),
            torch.int32,
            0.2,
            2,  # Should be 0, expect error
            1,
        ),
        (
            torch.randint(low=0, high=100, size=(4, 4, 4), dtype=torch.int32),
            torch.int8,
            0.2,
            1,
            1,  # Should be 0, expect error
        ),
    ]

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+INT")
    ), FakeTensorMode() as mode:
        for sample_input in sample_inputs:
            with pytest.raises(
                ValueError, match="TOSA requires (output|input)_zp to be zero"
            ):
                exir_ops.backend.tosa.RESCALE.default(
                    *tuple(
                        [
                            mode.from_tensor(i) if isinstance(i, torch.Tensor) else i
                            for i in sample_input
                        ]
                    )
                )


def test_zp_outside_range():

    sample_inputs = [
        (
            torch.randint(low=0, high=100, size=(4, 4, 4), dtype=torch.int8),
            torch.int32,
            0.2,
            128,  # Should be <128, expect error
            0,
        ),
        (
            torch.randint(low=0, high=100, size=(4, 4, 4), dtype=torch.int32),
            torch.int8,
            0.2,
            0,
            -129,  # Should be >-129m expect error
        ),
    ]
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+INT")
    ), FakeTensorMode() as mode:
        for sample_input in sample_inputs:
            with pytest.raises(
                Exception, match="(in_zp|out_zp)=-?[0-9]* outside valid range"
            ):
                exir_ops.backend.tosa.RESCALE.default(
                    *tuple(
                        [
                            mode.from_tensor(i) if isinstance(i, torch.Tensor) else i
                            for i in sample_input
                        ]
                    )
                )


class RescaleNetwork(torch.nn.Module):
    test_data = {
        "rand": (torch.rand(5), torch.rand(5)),
        "randn": (torch.randn(5, 2), torch.randn(5, 1)),
        "ones": (torch.ones(1, 10, 4, 6), torch.ones(1, 10, 4, 6)),
        "randn_ones": (torch.randn(1, 1, 4, 4), torch.ones(1, 1, 4, 1)),
        "randn_large": (10000 * torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 1)),
    }

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        a = y.exp()
        g = (a + 5).log()
        c = a + x
        d = c - g
        e = c * d
        f = e.sigmoid()

        return f


@common.parametrize("test_data", RescaleNetwork.test_data)
def test_quantized_rescale_tosa_bi(test_data: tuple[torch.Tensor, torch.Tensor]):
    """Tests a model with many ops that requires rescales. As more ops are quantized to int32 and
    need the InsertRescalesPass, make sure that they play nicely together."""
    module = RescaleNetwork()
    pipeline = TosaPipelineINT(
        module=module,
        test_data=test_data,
        aten_op=[],
        exir_op=[],
    )
    if not conftest.is_option_enabled("tosa_ref_model"):
        pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


u55_xfails = {
    "ones": "MLBEDSW-11032: ILLEGAL_OFM_BASE error: Base addresses must be aligned to brick depth on u55.",
    "randn_ones": "MLBEDSW-11032: ILLEGAL_OFM_BASE error: Base addresses must be aligned to brick depth on u55.",
    "randn_large": "MLBEDSW-11032: ILLEGAL_OFM_BASE error: Base addresses must be aligned to brick depth on u55.",
}


@common.parametrize("test_data", RescaleNetwork.test_data, xfails=u55_xfails)
@common.XfailIfNoCorstone300
def test_quantized_rescale_u55(test_data: tuple[torch.Tensor, torch.Tensor]):
    """Tests a model with many ops that requires rescales. As more ops are quantized to int32 and
    need the InsertRescalesPass, make sure that they play nicely together."""
    module = RescaleNetwork()
    pipeline = EthosU55PipelineINT(
        module=module,
        test_data=test_data,
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", RescaleNetwork.test_data)
@common.XfailIfNoCorstone320
def test_quantized_rescale_u85(test_data: tuple[torch.Tensor, torch.Tensor]):
    """Tests a model with many ops that requires rescales. As more ops are quantized to int32 and
    need the InsertRescalesPass, make sure that they play nicely together."""
    module = RescaleNetwork()
    pipeline = EthosU85PipelineINT(
        module=module,
        test_data=test_data,
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()
