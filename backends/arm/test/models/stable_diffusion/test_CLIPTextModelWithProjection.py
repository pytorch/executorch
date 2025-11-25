# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from executorch.backends.arm._passes import (
    ConvertInt64ConstOpsToInt32Pass,
    ConvertInt64OutputOpsToInt32Pass,
    InsertInt32CastsAfterInt64PlaceholdersPass,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.stable_diffusion.stable_diffusion_module_test_configs import (
    CLIP_text_encoder_config,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from transformers import CLIPTextModelWithProjection

input_t = Tuple[torch.Tensor]


class TestCLIPTextModelWithProjection:
    """
    Test class of CLIPTextModelWithProjection.
    CLIPTextModelWithProjection is one of the text_encoder used by Stable Diffusion 3.5 Medium
    """

    # Adjust nbr below as we increase op support.
    ops_after_partitioner_FP = {
        "executorch_exir_dialects_edge__ops_aten_argmax_default": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 2,
        "torch.ops.higher_order.executorch_call_delegate": 2,
    }

    ops_after_partitioner_INT = {
        "executorch_exir_dialects_edge__ops_aten_argmax_default": 1,
        "executorch_exir_dialects_edge__ops_aten_index_select_default": 1,
        "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 1,
        "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
        "executorch_exir_dialects_edge__ops_aten_where_self": 1,
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 2,
        "torch.ops.aten.scalar_tensor.default": 1,
        "torch.ops.higher_order.executorch_call_delegate": 2,
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


def test_CLIPTextModelWithProjection_tosa_FP():
    text_encoder_model, text_encoder_model_inputs = (
        TestCLIPTextModelWithProjection().prepare_model_and_inputs()
    )
    with torch.no_grad():
        pipeline = TosaPipelineFP[input_t](
            text_encoder_model,
            text_encoder_model_inputs,
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            transform_passes=[
                ConvertInt64ConstOpsToInt32Pass(),
                ConvertInt64OutputOpsToInt32Pass(),
                InsertInt32CastsAfterInt64PlaceholdersPass(),
            ],
        )
        pipeline.change_args(
            "check_count.exir", TestCLIPTextModelWithProjection.ops_after_partitioner_FP
        )
        pipeline.run()


def test_CLIPTextModelWithProjection_tosa_INT():
    text_encoder_model, text_encoder_model_inputs = (
        TestCLIPTextModelWithProjection().prepare_model_and_inputs()
    )
    with torch.no_grad():
        pipeline = TosaPipelineINT[input_t](
            text_encoder_model,
            text_encoder_model_inputs,
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            atol=0.8,
        )
        pipeline.change_args(
            "check_count.exir",
            TestCLIPTextModelWithProjection.ops_after_partitioner_INT,
        )
        pipeline.run()


@common.SkipIfNoModelConverter
def test_CLIPTextModelWithProjection_vgf_FP():
    text_encoder_model, text_encoder_model_inputs = (
        TestCLIPTextModelWithProjection().prepare_model_and_inputs()
    )
    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            text_encoder_model,
            text_encoder_model_inputs,
            aten_op=[],
            exir_op=[],
            tosa_version="TOSA-1.0+FP",
            use_to_edge_transform_and_lower=True,
            atol=4,  # TODO: Investiage numerical issue: MAX Diff ~50%
            transform_passes=[
                ConvertInt64ConstOpsToInt32Pass(),
                ConvertInt64OutputOpsToInt32Pass(),
                InsertInt32CastsAfterInt64PlaceholdersPass(),
            ],
        )
        pipeline.change_args(
            "check_count.exir", TestCLIPTextModelWithProjection.ops_after_partitioner_FP
        )
        pipeline.run()


@common.SkipIfNoModelConverter
def test_CLIPTextModelWithProjection_vgf_INT():
    text_encoder_model, text_encoder_model_inputs = (
        TestCLIPTextModelWithProjection().prepare_model_and_inputs()
    )
    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            text_encoder_model,
            text_encoder_model_inputs,
            aten_op=[],
            exir_op=[],
            tosa_version="TOSA-1.0+INT",
            use_to_edge_transform_and_lower=True,
            atol=0.8,
        )
        pipeline.change_args(
            "check_count.exir",
            TestCLIPTextModelWithProjection.ops_after_partitioner_INT,
        )
        pipeline.run()
