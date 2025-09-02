# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from executorch.backends.arm._passes import InsertCastForOpsWithInt64InputPass

from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.stable_diffusion.stable_diffusion_module_test_configs import (
    T5_encoder_config,
)
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from transformers import T5EncoderModel


class TestT5EncoderModel(unittest.TestCase):
    """
    Test class of T5EncoderModel.
    T5EncoderModel is one of the text_encoder used by Stable Diffusion 3.5 Medium
    """

    # Adjust nbr below as we increase op support. Note: most of the delegates
    # calls are directly consecutive to each other in the .pte. The reason
    # for that is some assert ops are removed by passes in the
    # .to_executorch step, i.e. after Arm partitioner.
    ops_after_partitioner = {
        "executorch_exir_dialects_edge__ops_aten__to_copy_default": 2,
        "executorch_exir_dialects_edge__ops_aten_abs_default": 1,
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 3,
        "executorch_exir_dialects_edge__ops_aten_arange_start_step": 2,
        "executorch_exir_dialects_edge__ops_aten_full_like_default": 1,
        "executorch_exir_dialects_edge__ops_aten_gt_Scalar": 1,
        "executorch_exir_dialects_edge__ops_aten_lt_Scalar": 1,
        "executorch_exir_dialects_edge__ops_aten_minimum_default": 1,
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
        "executorch_exir_dialects_edge__ops_aten_sub_Tensor": 1,
        "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 2,
        "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
        "executorch_exir_dialects_edge__ops_aten_where_self": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 3,
        "torch.ops.higher_order.executorch_call_delegate": 2,
    }

    def _prepare_inputs(
        self,
        batch_size=12,
        seq_length=7,
        vocab_size=1000,
    ):
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, seq_length),
            dtype=torch.long,
        )
        return (input_ids,)

    def prepare_model_and_inputs(self):
        t5_encoder_config = T5_encoder_config

        t5_encoder_model = T5EncoderModel(t5_encoder_config)
        t5_encoder_model.eval()
        t5_encoder_model_inputs = self._prepare_inputs()

        return t5_encoder_model, t5_encoder_model_inputs

    def test_T5EncoderModel_tosa_MI(self):
        t5_encoder_model, t5_encoder_model_inputs = self.prepare_model_and_inputs()
        with torch.no_grad():
            (
                ArmTester(
                    t5_encoder_model,
                    example_inputs=t5_encoder_model_inputs,
                    compile_spec=common.get_tosa_compile_spec(tosa_spec="TOSA-1.0+FP"),
                    transform_passes=[InsertCastForOpsWithInt64InputPass()],
                )
                .export()
                .to_edge_transform_and_lower()
                .dump_operator_distribution()
                .check_count(self.ops_after_partitioner)
                .to_executorch()
                .run_method_and_compare_outputs(
                    inputs=t5_encoder_model_inputs,
                )
            )

    def test_T5EncoderModel_tosa_INT(self):
        t5_encoder_model, t5_encoder_model_inputs = self.prepare_model_and_inputs()
        with torch.no_grad():
            (
                ArmTester(
                    t5_encoder_model,
                    example_inputs=t5_encoder_model_inputs,
                    compile_spec=common.get_tosa_compile_spec(tosa_spec="TOSA-1.0+INT"),
                )
                .quantize()
                .export()
                .to_edge_transform_and_lower()
                .dump_operator_distribution()
                .to_executorch()
                .run_method_and_compare_outputs(
                    inputs=t5_encoder_model_inputs,
                )
            )
