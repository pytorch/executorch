# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from diffusers.models.transformers import SD3Transformer2DModel

from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.stable_diffusion.stable_diffusion_module_test_configs import (
    SD3Transformer2DModel_init_dict,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t4 = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class TestSD3Transformer2DModel:
    """
    Test class of AutoenSD3Transformer2DModelcoderKL.
    SD3Transformer2DModel is the transformer model used by Stable Diffusion 3.5 Medium
    """

    # Adjust nbr below as we increase op support.
    ops_after_partitioner_FP = {
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 1,
        "executorch_exir_dialects_edge__ops_aten_view_copy_default": 2,
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1,
        "torch.ops.higher_order.executorch_call_delegate": 1,
    }

    ops_after_partitioner_INT = {
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        "executorch_exir_dialects_edge__ops_aten_view_copy_default": 2,
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


def test_SD3Transformer2DModel_tosa_FP():
    sd35_transformer2D_model, sd35_transformer2D_model_inputs = (
        TestSD3Transformer2DModel().prepare_model_and_inputs()
    )
    with torch.no_grad():
        pipeline = TosaPipelineFP[input_t4](
            sd35_transformer2D_model,
            sd35_transformer2D_model_inputs,
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            rtol=1.0,  # TODO: MLETORCH-875: Reduce tolerance of SD3Transformer2DModel with FP and INT
            atol=4.0,
        )
        pipeline.change_args(
            "check_count.exir", TestSD3Transformer2DModel.ops_after_partitioner_FP
        )
        pipeline.run()


def test_SD3Transformer2DModel_tosa_INT():
    sd35_transformer2D_model, sd35_transformer2D_model_inputs = (
        TestSD3Transformer2DModel().prepare_model_and_inputs()
    )
    with torch.no_grad():
        pipeline = TosaPipelineINT[input_t4](
            sd35_transformer2D_model,
            sd35_transformer2D_model_inputs,
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            qtol=1.0,  # TODO: MLETORCH-875: Reduce tolerance of SD3Transformer2DModel with FP and INT
            rtol=1.0,
            atol=4.0,
        )
        pipeline.change_args(
            "check_count.exir", TestSD3Transformer2DModel.ops_after_partitioner_INT
        )
        pipeline.run()


@common.SkipIfNoModelConverter
def test_SD3Transformer2DModel_vgf_FP():
    sd35_transformer2D_model, sd35_transformer2D_model_inputs = (
        TestSD3Transformer2DModel().prepare_model_and_inputs()
    )
    with torch.no_grad():
        pipeline = VgfPipeline[input_t4](
            sd35_transformer2D_model,
            sd35_transformer2D_model_inputs,
            aten_op=[],
            exir_op=[],
            tosa_version="TOSA-1.0+FP",
            use_to_edge_transform_and_lower=True,
            rtol=1.0,  # TODO: MLETORCH-875: Reduce tolerance of SD3Transformer2DModel with FP and INT
            atol=4.0,
        )
        pipeline.change_args(
            "check_count.exir", TestSD3Transformer2DModel.ops_after_partitioner_FP
        )
        pipeline.run()


@common.SkipIfNoModelConverter
def test_SD3Transformer2DModel_vgf_INT():
    sd35_transformer2D_model, sd35_transformer2D_model_inputs = (
        TestSD3Transformer2DModel().prepare_model_and_inputs()
    )
    with torch.no_grad():
        pipeline = VgfPipeline[input_t4](
            sd35_transformer2D_model,
            sd35_transformer2D_model_inputs,
            aten_op=[],
            exir_op=[],
            tosa_version="TOSA-1.0+INT",
            use_to_edge_transform_and_lower=True,
            qtol=1.0,  # TODO: MLETORCH-875: Reduce tolerance of SD3Transformer2DModel with FP and INT
            rtol=1.0,
            atol=4.0,
        )
        pipeline.change_args(
            "check_count.exir", TestSD3Transformer2DModel.ops_after_partitioner_INT
        )
        pipeline.run()
