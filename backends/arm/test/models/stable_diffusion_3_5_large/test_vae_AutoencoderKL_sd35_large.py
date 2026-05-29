# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.stable_diffusion_3_5_large.test_configs_sd35_large import (
    get_tiny_sd35_large_vae_config,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.examples.models.stable_diffusion_3_5_large.model import (
    SD3VAEDecoderWrapper,
)

input_t = Tuple[torch.Tensor]


class TestAutoencoderKL:
    """Test helper for SD3.5 Large AutoencoderKL config."""

    ops_after_partitioner_FP = {
        "torch.ops.higher_order.executorch_call_delegate": 1,
    }

    ops_after_partitioner_INT = {
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 1,
        "torch.ops.higher_order.executorch_call_delegate": 1,
    }

    ops_after_partitioner_vgf_quantize = ops_after_partitioner_FP
    ops_after_partitioner_vgf_no_quantize = ops_after_partitioner_FP

    def create_config(self):
        """Create a tiny SD3.5 Large-like AutoencoderKL config for tests."""
        return get_tiny_sd35_large_vae_config()

    def create_dummy_inputs(
        self,
        batch_size: int = 1,
        latent_channels: int = 16,
        latent_size: int = 4,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor]:
        """Create dummy inputs for the SD3 VAE decoder tests."""
        # SD3.5 Large uses VAE decoder latent channels=16 and latent size=128.
        # Keep this unit-test default spatial size smaller for TOSA runtime.
        return (
            torch.randn(
                batch_size,
                latent_channels,
                latent_size,
                latent_size,
                dtype=dtype,
            ),
        )

    def create_model(self) -> SD3VAEDecoderWrapper:
        """Instantiate wrapped AutoencoderKL decoder for tests."""
        diffusers_autoencoders = pytest.importorskip("diffusers.models.autoencoders")
        AutoencoderKL = diffusers_autoencoders.AutoencoderKL
        return SD3VAEDecoderWrapper(AutoencoderKL(**self.create_config())).eval()


def test_vae_tosa_FP():
    """Run the AutoencoderKL TOSA FP test."""
    test_helper = TestAutoencoderKL()

    with torch.no_grad():
        pipeline = TosaPipelineFP[input_t](
            test_helper.create_model(),
            test_helper.create_dummy_inputs(),
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
        )
        pipeline.change_args(
            "check_count.exir", TestAutoencoderKL.ops_after_partitioner_FP
        )
        pipeline.run()


def test_vae_tosa_INT():
    """Run the AutoencoderKL TOSA INT test."""
    test_helper = TestAutoencoderKL()

    with torch.no_grad():
        pipeline = TosaPipelineINT[input_t](
            test_helper.create_model(),
            test_helper.create_dummy_inputs(),
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            atol=9e-2,
            frobenius_threshold=None,
            cosine_threshold=None,
        )
        pipeline.change_args(
            "check_count.exir", TestAutoencoderKL.ops_after_partitioner_INT
        )
        pipeline.run()


@common.SkipIfNoModelConverter
def test_vae_vgf_no_quant():
    """Run the AutoencoderKL VGF no-quant test."""
    test_helper = TestAutoencoderKL()

    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            test_helper.create_model(),
            test_helper.create_dummy_inputs(),
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            quantize=False,
        )
        pipeline.change_args(
            "check_count.exir",
            TestAutoencoderKL.ops_after_partitioner_vgf_no_quantize,
        )
        pipeline.run()


@common.SkipIfNoModelConverter
def test_vae_vgf_quant():
    """Run the AutoencoderKL VGF quant test."""
    test_helper = TestAutoencoderKL()

    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            test_helper.create_model(),
            test_helper.create_dummy_inputs(),
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            quantize=True,
        )
        pipeline.change_args(
            "check_count.exir",
            TestAutoencoderKL.ops_after_partitioner_vgf_quantize,
        )
        pipeline.run()
