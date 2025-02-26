# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import pytest

import torch
import torch.library
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized
from torch.testing._internal import optests


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
    for sample_input in sample_inputs[1:2]:
        torch.library.opcheck(torch.ops.tosa._rescale, sample_input)


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
    for sample_input in sample_inputs:
        with pytest.raises(optests.generate_tests.OpCheckError):
            torch.library.opcheck(torch.ops.tosa._rescale, sample_input)


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
    for sample_input in sample_inputs:
        with pytest.raises(optests.generate_tests.OpCheckError):
            torch.library.opcheck(torch.ops.tosa._rescale, sample_input)


class RescaleNetwork(torch.nn.Module):
    test_parameters = [
        (torch.rand(5), torch.rand(5)),
        (torch.randn(5, 2), torch.randn(5, 1)),
        (torch.ones(1, 10, 4, 6), torch.ones(1, 10, 4, 6)),
        (torch.randn(1, 1, 4, 4), torch.ones(1, 1, 4, 1)),
        (10000 * torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 1)),
    ]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        a = y.exp()
        g = (a + 5).log()
        c = a + x
        d = c - g
        e = c * d
        f = e.sigmoid()

        return f


def _test_rescale_pipeline(
    module: torch.nn.Module, test_data: tuple[torch.Tensor, torch.Tensor]
):
    """Tests a model with many ops that requires rescales. As more ops are quantized to int32 and
    need the InsertRescalesPass, make sure that they play nicely together."""
    tester = (
        ArmTester(
            module,
            example_inputs=test_data,
            compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
        )
        .quantize()
        .export()
        .to_edge_transform_and_lower()
        .to_executorch()
    )
    if conftest.is_option_enabled("tosa_ref_model"):
        tester.run_method_and_compare_outputs(test_data)


def _test_rescale_pipeline_ethosu(
    module: torch.nn.Module, compile_spec, test_data: tuple[torch.Tensor, torch.Tensor]
):
    tester = (
        ArmTester(
            module,
            example_inputs=test_data,
            compile_spec=compile_spec,
        )
        .quantize()
        .export()
        .to_edge_transform_and_lower()
        .to_executorch()
        .serialize()
    )
    if conftest.is_option_enabled("corstone_fvp"):
        tester.run_method_and_compare_outputs(inputs=test_data)


class TestRescales(unittest.TestCase):

    @parameterized.expand(RescaleNetwork.test_parameters)
    @pytest.mark.tosa_ref_model
    def test_quantized_rescale(self, x, y):
        _test_rescale_pipeline(RescaleNetwork(), (x, y))

    @parameterized.expand(RescaleNetwork.test_parameters)
    @pytest.mark.corstone_fvp
    def test_quantized_rescale_U55(self, x, y):
        _test_rescale_pipeline_ethosu(
            RescaleNetwork(), common.get_u55_compile_spec(), (x, y)
        )

    @parameterized.expand(RescaleNetwork.test_parameters)
    @pytest.mark.corstone_fvp
    def test_quantized_rescale_U85(self, x, y):
        _test_rescale_pipeline_ethosu(
            RescaleNetwork(), common.get_u85_compile_spec(), (x, y)
        )
