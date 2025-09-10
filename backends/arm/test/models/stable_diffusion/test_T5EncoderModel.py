# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from executorch.backends.arm._passes import (
    ConvertInt64ConstOpsToInt32Pass,
    ConvertInt64OutputOpsToInt32Pass,
    InsertInt32CastsAfterInt64PlaceholdersPass,
)

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

    # Adjust nbr below as we increase op support.
    ops_after_partitioner_FP = {
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 2,
        "torch.ops.higher_order.executorch_call_delegate": 2,
    }

    ops_after_partitioner_INT = {
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 3,
        "torch.ops.higher_order.executorch_call_delegate": 3,
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

    def test_T5EncoderModel_tosa_FP(self):
        t5_encoder_model, t5_encoder_model_inputs = self.prepare_model_and_inputs()
        with torch.no_grad():
            (
                ArmTester(
                    t5_encoder_model,
                    example_inputs=t5_encoder_model_inputs,
                    compile_spec=common.get_tosa_compile_spec(tosa_spec="TOSA-1.0+FP"),
                    transform_passes=[
                        ConvertInt64ConstOpsToInt32Pass(),
                        ConvertInt64OutputOpsToInt32Pass(),
                        InsertInt32CastsAfterInt64PlaceholdersPass(),
                    ],
                )
                .export()
                .to_edge_transform_and_lower()
                .dump_operator_distribution()
                .check_count(self.ops_after_partitioner_FP)
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
                .check_count(self.ops_after_partitioner_INT)
                .to_executorch()
                .run_method_and_compare_outputs(
                    inputs=t5_encoder_model_inputs,
                )
            )
