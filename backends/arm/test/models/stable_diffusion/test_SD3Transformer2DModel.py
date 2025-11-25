# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from diffusers.models.transformers import SD3Transformer2DModel

from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.stable_diffusion.stable_diffusion_module_test_configs import (
    SD3Transformer2DModel_init_dict,
)
from executorch.backends.arm.test.tester.arm_tester import ArmTester


class TestSD3Transformer2DModel(unittest.TestCase):
    """
    Test class of AutoenSD3Transformer2DModelcoderKL.
    SD3Transformer2DModel is the transformer model used by Stable Diffusion 3.5 Medium
    """

    # Adjust nbr below as we increase op support.
    ops_after_partitioner_FP = {
        "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1,
        "torch.ops.higher_order.executorch_call_delegate": 1,
    }

    ops_after_partitioner_INT = {
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 2,
        "torch.ops.higher_order.executorch_call_delegate": 2,
    }

    def _prepare_inputs(
        self,
        batch_size=2,
        num_channels=4,
        height=32,
        width=32,
        embedding_dim=32,
        sequence_length=154,
        max_timestep=1000,
    ):
        hidden_states = torch.randn(
            (
                batch_size,
                num_channels,
                height,
                width,
            )
        )
        encoder_hidden_states = torch.randn(
            (
                batch_size,
                sequence_length,
                embedding_dim,
            )
        )
        pooled_prompt_embeds = torch.randn(
            (
                batch_size,
                embedding_dim * 2,
            )
        )
        timestep = torch.randint(low=0, high=max_timestep, size=(batch_size,))

        input_dict = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_prompt_embeds,
            "timestep": timestep,
        }

        return tuple(input_dict.values())

    def prepare_model_and_inputs(self):

        class SD3Transformer2DModelWrapper(SD3Transformer2DModel):
            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs).sample

        init_dict = SD3Transformer2DModel_init_dict

        sd35_transformer2D_model = SD3Transformer2DModelWrapper(**init_dict)
        sd35_transformer2D_model_inputs = self._prepare_inputs()

        return sd35_transformer2D_model, sd35_transformer2D_model_inputs

    def test_SD3Transformer2DModel_tosa_FP(self):
        sd35_transformer2D_model, sd35_transformer2D_model_inputs = (
            self.prepare_model_and_inputs()
        )
        with torch.no_grad():
            (
                ArmTester(
                    sd35_transformer2D_model,
                    example_inputs=sd35_transformer2D_model_inputs,
                    compile_spec=common.get_tosa_compile_spec(tosa_spec="TOSA-1.0+FP"),
                )
                .export()
                .to_edge_transform_and_lower()
                .check_count(self.ops_after_partitioner_FP)
                .to_executorch()
                .run_method_and_compare_outputs(
                    inputs=sd35_transformer2D_model_inputs,
                    rtol=1.0,  # TODO: MLETORCH-875: Reduce tolerance of SD3Transformer2DModel with FP and INT
                    atol=4.0,
                )
            )

    def test_SD3Transformer2DModel_tosa_INT(self):
        sd35_transformer2D_model, sd35_transformer2D_model_inputs = (
            self.prepare_model_and_inputs()
        )
        with torch.no_grad():
            (
                ArmTester(
                    sd35_transformer2D_model,
                    example_inputs=sd35_transformer2D_model_inputs,
                    compile_spec=common.get_tosa_compile_spec(tosa_spec="TOSA-1.0+INT"),
                )
                .quantize()
                .export()
                .to_edge_transform_and_lower()
                .check_count(self.ops_after_partitioner_INT)
                .to_executorch()
                .run_method_and_compare_outputs(
                    inputs=sd35_transformer2D_model_inputs,
                    qtol=1.0,  # TODO: MLETORCH-875: Reduce tolerance of SD3Transformer2DModel with FP and INT
                    rtol=1.0,
                    atol=4.0,
                )
            )
