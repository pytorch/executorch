# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

import torch
from executorch.backends.arm.test import common, conftest

from executorch.backends.arm.test.tester.arm_tester import ArmTester

from torchaudio.models import Conformer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestConformer(unittest.TestCase):
    """Tests Torchaudio Conformer"""

    # Adjust nbr below as we increase op support. Note: most of the delegates
    # calls are directly consecutive to each other in the .pte. The reason
    # for that is some assert ops are removed by passes in the
    # .to_executorch step, i.e. after Arm partitioner.
    ops_after_partitioner = {
        "executorch_exir_dialects_edge__ops_aten_arange_start_step": 1,
        "executorch_exir_dialects_edge__ops_aten_full_like_default": 4,
        "executorch_exir_dialects_edge__ops_aten_max_default": 1,
        "executorch_exir_dialects_edge__ops_aten_mul_Scalar": 4,
        "executorch_exir_dialects_edge__ops_aten_eq_Scalar": 2,
        "executorch_exir_dialects_edge__ops_aten_where_self": 4,
        "executorch_exir_dialects_edge__ops_aten_logical_not_default": 4,
        "executorch_exir_dialects_edge__ops_aten_any_dim": 2,
        "torch.ops.aten._assert_scalar.default": 10,
        "torch.ops.aten._local_scalar_dense.default": 1,
        "torch.ops.aten.scalar_tensor.default": 2,
        "torch.ops.higher_order.executorch_call_delegate": 5,
    }

    dim = 16
    lengths = torch.randint(1, 100, (10,), dtype=torch.int32)
    input_data = torch.rand(10, int(lengths.max()), dim)
    conformer = Conformer(
        input_dim=dim,
        num_heads=4,
        ffn_dim=64,
        num_layers=2,
        depthwise_conv_kernel_size=31,
    )
    conformer = conformer.eval()

    def test_conformer_tosa_MI(self):
        (
            ArmTester(
                self.conformer,
                example_inputs=(self.input_data, self.lengths),
                compile_spec=common.get_tosa_compile_spec(tosa_spec="TOSA-0.80+MI"),
            )
            .export()
            .to_edge_transform_and_lower()
            .dump_operator_distribution()
            .check_count(self.ops_after_partitioner)
            .to_executorch()
            # TODO(MLETORCH-632): Fix numerical errors
            .run_method_and_compare_outputs(
                inputs=(self.input_data, self.lengths), rtol=1, atol=5
            )
        )

    @unittest.expectedFailure  # TODO(MLETORCH-635)
    def test_conformer_tosa_BI(self):
        (
            ArmTester(
                self.conformer,
                example_inputs=(self.input_data, self.lengths),
                compile_spec=common.get_tosa_compile_spec(tosa_spec="TOSA-0.80+BI"),
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(
                qtol=1, rtol=1, atol=5, inputs=(self.input_data, self.lengths)
            )
        )

    @unittest.expectedFailure  # TODO(MLETORCH-635)
    def test_conformer_u55_BI(self):
        tester = (
            ArmTester(
                self.conformer,
                example_inputs=(self.input_data, self.lengths),
                compile_spec=common.get_u55_compile_spec(),
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(
                atol=1.0, qtol=1, inputs=(self.input_data, self.lengths)
            )

    @unittest.expectedFailure  # TODO(MLETORCH-635)
    def test_conformer_u85_BI(self):
        tester = (
            ArmTester(
                self.conformer,
                example_inputs=(self.input_data, self.lengths),
                compile_spec=common.get_u85_compile_spec(),
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(
                atol=1.0, qtol=1, inputs=(self.input_data, self.lengths)
            )
