# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch
from executorch.backends.arm._passes import (
    ConvertInt64ConstOpsToInt32Pass,
    ConvertInt64OutputOpsToInt32Pass,
    InsertInt32CastsAfterInt64PlaceholdersPass,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.stable_diffusion_3_5_large.test_configs_sd35_large import (
    get_tiny_sd35_large_text_encoder_2_config,
    get_tiny_sd35_large_text_encoder_config,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.examples.models.stable_diffusion_3_5_large.model import (
    SD3CLIPTextEncoderWrapper,
)
from transformers import CLIPTextModelWithProjection

input_t = Tuple[torch.Tensor]

_INT64_TO_INT32_PASSES = [
    ConvertInt64ConstOpsToInt32Pass(),
    ConvertInt64OutputOpsToInt32Pass(),
    InsertInt32CastsAfterInt64PlaceholdersPass(),
]


class TestCLIPTextModelWithProjection:
    """Test helper for SD3.5 Large CLIPTextModelWithProjection configs."""

    ops_after_partitioner_FP = {
        "executorch_exir_dialects_edge__ops_aten_argmax_default": 1,
        "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 2,
        "torch.ops.higher_order.executorch_call_delegate": 2,
    }

    ops_after_partitioner_vgf_no_quantize = {
        "executorch_exir_dialects_edge__ops_aten_argmax_default": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 2,
        "torch.ops.higher_order.executorch_call_delegate": 2,
    }
    ops_after_partitioner_vgf_quantize = {
        "executorch_exir_dialects_edge__ops_aten_argmax_default": 1,
        "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 2,
        "torch.ops.higher_order.executorch_call_delegate": 1,
    }

    def create_dummy_inputs(
        self,
        config,
        batch_size: int = 1,
        seq_length: int = 2,
        dtype: torch.dtype = torch.long,
    ) -> tuple[torch.Tensor]:
        """Create dummy inputs for the CLIPTextModelWithProjection tests."""
        # SD3.5 Large uses (batch_size, seq_length) = (1, 77) for both CLIP-L
        # and CLIP-bigG. Keep this unit-test default smaller for TOSA runtime.
        return (
            torch.randint(
                low=0,
                high=config.vocab_size,
                size=(batch_size, seq_length),
                dtype=dtype,
            ),
        )

    def create_model(
        self,
        config,
    ) -> SD3CLIPTextEncoderWrapper:
        """Instantiate wrapped CLIPTextModelWithProjection for tests."""
        return SD3CLIPTextEncoderWrapper(
            CLIPTextModelWithProjection(config).to(dtype=config.dtype)  # type: ignore[call-arg]
        ).eval()

    @staticmethod
    def ops_after_partitioner_INT(config) -> dict[str, int]:
        if config.num_hidden_layers == 2:
            return {
                "executorch_exir_dialects_edge__ops_aten_add_Tensor": 2,
                "executorch_exir_dialects_edge__ops_aten_argmax_default": 1,
                "executorch_exir_dialects_edge__ops_aten_where_self": 2,
                "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 12,
                "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 18,
                "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 12,
                "torch.ops.higher_order.executorch_call_delegate": 7,
            }

        raise ValueError(
            f"Unexpected CLIP config: hidden_act={config.hidden_act}, "
            f"num_hidden_layers={config.num_hidden_layers}"
        )


@pytest.mark.parametrize(
    ("config_factory", "atol"),
    (
        (get_tiny_sd35_large_text_encoder_config, 1e-2),  # FP atol
        (get_tiny_sd35_large_text_encoder_2_config, 1.5e-2),  # FP atol
    ),
    ids=("text_encoder", "text_encoder_2"),
)
def test_clip_text_model_with_projection_tosa_FP(config_factory, atol):
    """Run the CLIPTextModelWithProjection TOSA FP test for a given config."""
    test_helper = TestCLIPTextModelWithProjection()
    config = config_factory()

    with torch.no_grad():
        pipeline = TosaPipelineFP[input_t](
            test_helper.create_model(config),
            test_helper.create_dummy_inputs(config),
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            atol=atol,
            transform_passes=_INT64_TO_INT32_PASSES,
        )
        pipeline.change_args(
            "check_count.exir", TestCLIPTextModelWithProjection.ops_after_partitioner_FP
        )
        pipeline.run()


@pytest.mark.parametrize(
    ("config_factory", "atol"),
    (
        (get_tiny_sd35_large_text_encoder_config, 5.5e-2),  # INT atol
        (get_tiny_sd35_large_text_encoder_2_config, 6e-2),  # INT atol
    ),
    ids=("text_encoder", "text_encoder_2"),
)
def test_clip_text_model_with_projection_tosa_INT(config_factory, atol):
    """Run the CLIPTextModelWithProjection TOSA INT test for a given config."""
    test_helper = TestCLIPTextModelWithProjection()
    config = config_factory()

    with torch.no_grad():
        pipeline = TosaPipelineINT[input_t](
            test_helper.create_model(config),
            test_helper.create_dummy_inputs(config),
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            atol=atol,
            frobenius_threshold=None,
            cosine_threshold=None,
        )
        pipeline.change_args(
            "check_count.exir",
            TestCLIPTextModelWithProjection.ops_after_partitioner_INT(config),
        )
        pipeline.run()


@common.SkipIfNoModelConverter
@pytest.mark.parametrize(
    ("config_factory",),
    (
        (get_tiny_sd35_large_text_encoder_config,),
        (get_tiny_sd35_large_text_encoder_2_config,),
    ),
    ids=("text_encoder", "text_encoder_2"),
)
def test_clip_text_model_with_projection_vgf_no_quant(config_factory):
    """Run the CLIPTextModelWithProjection VGF no-quant test."""
    test_helper = TestCLIPTextModelWithProjection()
    config = config_factory()

    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            test_helper.create_model(config),
            test_helper.create_dummy_inputs(config),
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            atol=5e-3,
            transform_passes=_INT64_TO_INT32_PASSES,
            quantize=False,
        )
        pipeline.change_args(
            "check_count.exir",
            TestCLIPTextModelWithProjection.ops_after_partitioner_vgf_no_quantize,
        )
        pipeline.run()


@common.SkipIfNoModelConverter
@pytest.mark.parametrize(
    ("config_factory", "atol"),
    (
        (get_tiny_sd35_large_text_encoder_config, 5.5e-2),
        (get_tiny_sd35_large_text_encoder_2_config, 6e-2),
    ),
    ids=("text_encoder", "text_encoder_2"),
)
def test_clip_text_model_with_projection_vgf_quant(config_factory, atol):
    """Run the CLIPTextModelWithProjection VGF quant test."""
    test_helper = TestCLIPTextModelWithProjection()
    config = config_factory()

    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            test_helper.create_model(config),
            test_helper.create_dummy_inputs(config),
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            atol=atol,
            quantize=True,
        )
        pipeline.change_args(
            "check_count.exir",
            TestCLIPTextModelWithProjection.ops_after_partitioner_vgf_quantize,
        )
        pipeline.run()
