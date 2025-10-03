# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.utils.testing_utils import floats_tensor

from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.stable_diffusion.stable_diffusion_module_test_configs import (
    AutoencoderKL_config,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t = Tuple[torch.Tensor]


class TestAutoencoderKL:
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


def test_AutoencoderKL_tosa_FP():
    auto_encoder_model, auto_encoder_model_inputs = (
        TestAutoencoderKL().prepare_model_and_inputs()
    )
    with torch.no_grad():
        pipeline = TosaPipelineFP[input_t](
            auto_encoder_model,
            auto_encoder_model_inputs,
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
        )
        pipeline.run()


def test_AutoencoderKL_tosa_INT():
    auto_encoder_model, auto_encoder_model_inputs = (
        TestAutoencoderKL().prepare_model_and_inputs()
    )
    with torch.no_grad():
        pipeline = TosaPipelineINT[input_t](
            auto_encoder_model,
            auto_encoder_model_inputs,
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            atol=1.0,  # TODO: MLETORCH-990 Reduce tolerance of vae(AutoencoderKL) with INT
        )
        pipeline.run()


@common.SkipIfNoModelConverter
def test_AutoencoderKL_vgf_FP():
    auto_encoder_model, auto_encoder_model_inputs = (
        TestAutoencoderKL().prepare_model_and_inputs()
    )
    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            auto_encoder_model,
            auto_encoder_model_inputs,
            aten_op=[],
            exir_op=[],
            tosa_version="TOSA-1.0+FP",
            use_to_edge_transform_and_lower=True,
        )
        pipeline.run()


@common.SkipIfNoModelConverter
def test_AutoencoderKL_vgf_INT():
    auto_encoder_model, auto_encoder_model_inputs = (
        TestAutoencoderKL().prepare_model_and_inputs()
    )
    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            auto_encoder_model,
            auto_encoder_model_inputs,
            aten_op=[],
            exir_op=[],
            tosa_version="TOSA-1.0+INT",
            use_to_edge_transform_and_lower=True,
            atol=1.0,  # TODO: MLETORCH-990 Reduce tolerance of vae(AutoencoderKL) with INT
        )
        pipeline.run()
