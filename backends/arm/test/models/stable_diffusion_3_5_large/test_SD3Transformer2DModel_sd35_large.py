# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.stable_diffusion_3_5_large.test_configs_sd35_large import (
    get_tiny_sd35_large_transformer_config,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.examples.models.stable_diffusion_3_5_large.model import (
    SD3TransformerWrapper,
)

input_t4 = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class TestSD3Transformer2DModel:
    """Test helper for SD3.5 Large SD3Transformer2DModel config."""

    ops_after_partitioner_FP = {
        "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1,
        "torch.ops.higher_order.executorch_call_delegate": 1,
    }

    ops_after_partitioner_INT = {
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
        "torch.ops.higher_order.executorch_call_delegate": 1,
    }

    ops_after_partitioner_vgf_quantize = {
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1,
        "torch.ops.higher_order.executorch_call_delegate": 1,
    }
    ops_after_partitioner_vgf_no_quantize = ops_after_partitioner_FP

    def create_config(self):
        """Create a tiny SD3.5 Large-like MMDiT config for tests."""
        return get_tiny_sd35_large_transformer_config()

    def create_dummy_inputs(
        self,
        batch_size: int = 2,
        latent_channels: int = 4,
        latent_size: int = 32,
        seq_length: int = 77,
        joint_attention_dim: int = 16,
        pooled_projection_dim: int = 32,
        max_timestep: int = 1000,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create dummy inputs for the SD3Transformer2DModel tests."""
        # SD3.5 Large uses latent channels=16, latent size=128, T5 seq length=256,
        # joint_attention_dim=4096, and pooled_projection_dim=2048. Keep this
        # unit-test default smaller for TOSA runtime and VGF memory limits.
        return (
            torch.randn(
                batch_size,
                latent_channels,
                latent_size,
                latent_size,
                dtype=dtype,
            ),
            torch.randint(low=0, high=max_timestep, size=(batch_size,)),
            torch.randn(
                batch_size,
                seq_length,
                joint_attention_dim,
                dtype=dtype,
            ),
            torch.randn(batch_size, pooled_projection_dim, dtype=dtype),
        )

    def create_model(self) -> SD3TransformerWrapper:
        """Instantiate wrapped SD3Transformer2DModel for tests."""
        SD3Transformer2DModel = pytest.importorskip(
            "diffusers.models.transformers"
        ).SD3Transformer2DModel
        return SD3TransformerWrapper(
            SD3Transformer2DModel(**self.create_config())
        ).eval()


def test_sd3_transformer_tosa_FP():
    """Run the SD3Transformer2DModel TOSA FP test."""
    test_helper = TestSD3Transformer2DModel()

    with torch.no_grad():
        pipeline = TosaPipelineFP[input_t4](
            test_helper.create_model(),
            test_helper.create_dummy_inputs(),
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
        )
        pipeline.change_args(
            "check_count.exir", TestSD3Transformer2DModel.ops_after_partitioner_FP
        )
        pipeline.run()


def test_sd3_transformer_tosa_INT():
    """Run the SD3Transformer2DModel TOSA INT test."""
    test_helper = TestSD3Transformer2DModel()

    with torch.no_grad():
        pipeline = TosaPipelineINT[input_t4](
            test_helper.create_model(),
            test_helper.create_dummy_inputs(),
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            frobenius_threshold=None,
            cosine_threshold=None,
        )
        pipeline.change_args(
            "check_count.exir", TestSD3Transformer2DModel.ops_after_partitioner_INT
        )
        pipeline.run()


@common.SkipIfNoModelConverter
def test_sd3_transformer_vgf_no_quant():
    """Run the SD3Transformer2DModel VGF no-quant test."""
    test_helper = TestSD3Transformer2DModel()

    with torch.no_grad():
        pipeline = VgfPipeline[input_t4](
            test_helper.create_model(),
            test_helper.create_dummy_inputs(),
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            quantize=False,
        )
        pipeline.change_args(
            "check_count.exir",
            TestSD3Transformer2DModel.ops_after_partitioner_vgf_no_quantize,
        )
        pipeline.run()


@common.SkipIfNoModelConverter
def test_sd3_transformer_vgf_quant():
    """Run the SD3Transformer2DModel VGF quant test."""
    test_helper = TestSD3Transformer2DModel()

    with torch.no_grad():
        pipeline = VgfPipeline[input_t4](
            test_helper.create_model(),
            test_helper.create_dummy_inputs(),
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            quantize=True,
        )
        pipeline.change_args(
            "check_count.exir",
            TestSD3Transformer2DModel.ops_after_partitioner_vgf_quantize,
        )
        pipeline.run()
