# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import pytest
import torch
from executorch.backends.arm._passes import (
    ConvertInt64ConstOpsToInt32Pass,
    ConvertInt64OutputOpsToInt32Pass,
    InsertCastForOpsWithInt64InputPass,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.stable_diffusion.stable_diffusion_module_test_configs import (
    CLIP_text_encoder_config,
)
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from transformers import CLIPTextModelWithProjection


class TestCLIPTextModelWithProjection(unittest.TestCase):
    """
    Test class of CLIPTextModelWithProjection.
    CLIPTextModelWithProjection is one of the text_encoder used by Stable Diffusion 3.5 Medium
    """

    # Adjust nbr below as we increase op support. Note: most of the delegates
    # calls are directly consecutive to each other in the .pte. The reason
    # for that is some assert ops are removed by passes in the
    # .to_executorch step, i.e. after Arm partitioner.
    ops_after_partitioner = {
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 3,
        "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
        "executorch_exir_dialects_edge__ops_aten_argmax_default": 1,
        "torch.ops.higher_order.executorch_call_delegate": 1,
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
        clip_text_encoder_config = CLIP_text_encoder_config

        text_encoder_model = CLIPTextModelWithProjection(clip_text_encoder_config)
        text_encoder_model.eval()
        text_encoder_model_inputs = self._prepare_inputs()

        return text_encoder_model, text_encoder_model_inputs

    def test_CLIPTextModelWithProjection_tosa_FP(self):
        text_encoder_model, text_encoder_model_inputs = self.prepare_model_and_inputs()
        with torch.no_grad():
            (
                ArmTester(
                    text_encoder_model,
                    example_inputs=text_encoder_model_inputs,
                    compile_spec=common.get_tosa_compile_spec(tosa_spec="TOSA-1.0+FP"),
                    transform_passes=[
                        InsertCastForOpsWithInt64InputPass(),
                        ConvertInt64ConstOpsToInt32Pass(),
                        ConvertInt64OutputOpsToInt32Pass(),
                    ],
                )
                .export()
                .to_edge_transform_and_lower()
                .dump_operator_distribution()
                .check_count(self.ops_after_partitioner)
                .to_executorch()
                .run_method_and_compare_outputs(
                    inputs=text_encoder_model_inputs,
                )
            )

    @pytest.mark.xfail(raises=AssertionError, reason="Output difference.")
    def test_CLIPTextModelWithProjection_tosa_INT(self):
        text_encoder_model, text_encoder_model_inputs = self.prepare_model_and_inputs()
        with torch.no_grad():
            (
                ArmTester(
                    text_encoder_model,
                    example_inputs=text_encoder_model_inputs,
                    compile_spec=common.get_tosa_compile_spec(tosa_spec="TOSA-1.0+INT"),
                )
                .quantize()
                .export()
                .to_edge_transform_and_lower()
                .dump_operator_distribution()
                .to_executorch()
                .run_method_and_compare_outputs(
                    inputs=text_encoder_model_inputs,
                )
            )
