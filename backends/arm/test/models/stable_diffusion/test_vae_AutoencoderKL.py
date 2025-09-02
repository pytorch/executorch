# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.utils.testing_utils import floats_tensor

from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.stable_diffusion.stable_diffusion_module_test_configs import (
    AutoencoderKL_config,
)
from executorch.backends.arm.test.tester.arm_tester import ArmTester


class TestAutoencoderKL(unittest.TestCase):
    """
    Test class of AutoencoderKL.
    AutoencoderKL is the encoder/decoder used by Stable Diffusion 3.5 Medium
    """

    def _prepare_inputs(self, batch_size=4, num_channels=3, sizes=(32, 32)):
        image = floats_tensor((batch_size, num_channels) + sizes)
        return (image,)

    def prepare_model_and_inputs(self):

        class AutoencoderWrapper(AutoencoderKL):
            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs).sample

        vae_config = AutoencoderKL_config

        auto_encoder_model = AutoencoderWrapper(**vae_config)

        auto_encoder_model_inputs = self._prepare_inputs()

        return auto_encoder_model, auto_encoder_model_inputs

    def test_AutoencoderKL_tosa_FP(self):
        auto_encoder_model, auto_encoder_model_inputs = self.prepare_model_and_inputs()
        with torch.no_grad():
            (
                ArmTester(
                    auto_encoder_model,
                    example_inputs=auto_encoder_model_inputs,
                    compile_spec=common.get_tosa_compile_spec(tosa_spec="TOSA-1.0+FP"),
                )
                .export()
                .to_edge_transform_and_lower()
                .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                .to_executorch()
                .run_method_and_compare_outputs(
                    inputs=auto_encoder_model_inputs,
                )
            )

    def test_AutoencoderKL_tosa_INT(self):
        auto_encoder_model, auto_encoder_model_inputs = self.prepare_model_and_inputs()
        with torch.no_grad():
            (
                ArmTester(
                    auto_encoder_model,
                    example_inputs=auto_encoder_model_inputs,
                    compile_spec=common.get_tosa_compile_spec(tosa_spec="TOSA-1.0+INT"),
                )
                .quantize()
                .export()
                .to_edge_transform_and_lower()
                .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                .to_executorch()
                .run_method_and_compare_outputs(
                    inputs=auto_encoder_model_inputs,
                    atol=1.0,  # TODO: MLETORCH-990 Reduce tolerance of vae(AutoencoderKL) with INT
                )
            )
