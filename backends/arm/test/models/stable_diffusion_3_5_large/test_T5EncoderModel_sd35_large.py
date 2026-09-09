# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.stable_diffusion_3_5_large.test_configs_sd35_large import (
    get_int64_to_int32_passes,
    get_tiny_sd35_large_t5_config,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.examples.models.stable_diffusion_3_5_large.model import (
    SD3T5TextEncoderWrapper,
)
from transformers import T5EncoderModel

input_t = Tuple[torch.Tensor]


class TestT5EncoderModel:
    """Test helper for SD3.5 Large T5EncoderModel config."""

    ops_after_partitioner_FP = {
        "executorch_exir_dialects_edge__ops_aten_clamp_Tensor": 4,
        "executorch_exir_dialects_edge__ops_aten_isinf_default": 4,
        "executorch_exir_dialects_edge__ops_aten_where_self": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 2,
        "torch.ops.higher_order.executorch_call_delegate": 10,
    }

    ops_after_partitioner_INT = {
        "executorch_exir_dialects_edge__ops_aten_isinf_default": 4,
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 5,
        "executorch_exir_dialects_edge__ops_aten_where_self": 5,
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 21,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 30,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 24,
        "aten.scalar_tensor.default": 9,
        "torch.ops.higher_order.executorch_call_delegate": 24,
    }

    ops_after_partitioner_vgf_quantize = {
        "executorch_exir_dialects_edge__ops_aten_clamp_Tensor": 4,
        "executorch_exir_dialects_edge__ops_aten_isinf_default": 4,
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1,
        "torch.ops.higher_order.executorch_call_delegate": 9,
    }

    ops_after_partitioner_vgf_no_quantize = ops_after_partitioner_vgf_quantize

    def create_dummy_inputs(
        self,
        config,
        batch_size: int = 1,
        seq_length: int = 2,
        dtype: torch.dtype = torch.long,
    ) -> tuple[torch.Tensor]:
        """Create dummy inputs for the T5EncoderModel tests."""
        # SD3.5 Large uses (batch_size, seq_length) = (1, 256) for T5.
        # Keep this unit-test default smaller for TOSA runtime.
        return (
            torch.randint(
                low=0,
                high=config.vocab_size,
                size=(batch_size, seq_length),
                dtype=dtype,
            ),
        )

    def create_config(self):
        """Create a tiny SD3.5 Large-like T5 config for tests."""
        return get_tiny_sd35_large_t5_config()

    def create_model(self, config) -> SD3T5TextEncoderWrapper:
        """Instantiate wrapped T5EncoderModel for tests."""
        return SD3T5TextEncoderWrapper(
            T5EncoderModel(config).to(dtype=config.dtype)  # type: ignore[call-arg]
        ).eval()


def test_t5_encoder_tosa_FP():
    """Run the T5EncoderModel TOSA FP test."""
    test_helper = TestT5EncoderModel()
    config = test_helper.create_config()

    with torch.no_grad():
        pipeline = TosaPipelineFP[input_t](
            test_helper.create_model(config),
            test_helper.create_dummy_inputs(config),
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            atol=2e-2,
            transform_passes=get_int64_to_int32_passes(),
        )
        pipeline.change_args(
            "check_count.exir", TestT5EncoderModel.ops_after_partitioner_FP
        )
        pipeline.run()


def test_t5_encoder_tosa_INT():
    """Run the T5EncoderModel TOSA INT test."""
    test_helper = TestT5EncoderModel()
    config = test_helper.create_config()

    with torch.no_grad():
        pipeline = TosaPipelineINT[input_t](
            test_helper.create_model(config),
            test_helper.create_dummy_inputs(config),
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            atol=3e-2,
            frobenius_threshold=None,
            cosine_threshold=None,
        )
        pipeline.change_args(
            "check_count.exir", TestT5EncoderModel.ops_after_partitioner_INT
        )
        pipeline.run()


@common.SkipIfNoModelConverter
def test_t5_encoder_vgf_no_quant():
    """Run the T5EncoderModel VGF no-quant test."""
    test_helper = TestT5EncoderModel()
    config = test_helper.create_config()

    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            test_helper.create_model(config),
            test_helper.create_dummy_inputs(config),
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            atol=8e-3,
            transform_passes=get_int64_to_int32_passes(),
            quantize=False,
        )
        pipeline.change_args(
            "check_count.exir",
            TestT5EncoderModel.ops_after_partitioner_vgf_no_quantize,
        )
        pipeline.run()


@common.SkipIfNoModelConverter
def test_t5_encoder_vgf_quant():
    """Run the T5EncoderModel VGF quant test."""
    test_helper = TestT5EncoderModel()
    config = test_helper.create_config()

    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            test_helper.create_model(config),
            test_helper.create_dummy_inputs(config),
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            atol=1.2e-2,
            quantize=True,
        )
        pipeline.change_args(
            "check_count.exir",
            TestT5EncoderModel.ops_after_partitioner_vgf_quantize,
        )
        pipeline.run()
