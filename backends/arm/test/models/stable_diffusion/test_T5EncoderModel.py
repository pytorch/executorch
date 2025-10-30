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
    T5_encoder_config,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from transformers import T5EncoderModel

input_t = Tuple[torch.Tensor]


class TestT5EncoderModel:
    """
    Test class of T5EncoderModel.
    T5EncoderModel is one of the text_encoder used by Stable Diffusion 3.5 Medium
    """

    # Adjust nbr below as we increase op support.
    ops_after_partitioner_FP = {
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 2,
        "torch.ops.higher_order.executorch_call_delegate": 2,
    }

    ops_after_partitioner_INT = {
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 3,
        "torch.ops.higher_order.executorch_call_delegate": 3,
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
        t5_encoder_config = T5_encoder_config

        t5_encoder_model = T5EncoderModel(t5_encoder_config)
        t5_encoder_model.eval()
        t5_encoder_model_inputs = self._prepare_inputs()

        return t5_encoder_model, t5_encoder_model_inputs


def test_T5EncoderModel_tosa_FP():
    t5_encoder_model, t5_encoder_model_inputs = (
        TestT5EncoderModel().prepare_model_and_inputs()
    )
    with torch.no_grad():
        pipeline = TosaPipelineFP[input_t](
            t5_encoder_model,
            t5_encoder_model_inputs,
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
            "check_count.exir", TestT5EncoderModel.ops_after_partitioner_FP
        )
        pipeline.run()


def test_T5EncoderModel_tosa_INT():
    t5_encoder_model, t5_encoder_model_inputs = (
        TestT5EncoderModel().prepare_model_and_inputs()
    )
    with torch.no_grad():
        pipeline = TosaPipelineINT[input_t](
            t5_encoder_model,
            t5_encoder_model_inputs,
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
        )
        pipeline.change_args(
            "check_count.exir", TestT5EncoderModel.ops_after_partitioner_INT
        )
        pipeline.run()


@common.SkipIfNoModelConverter
def test_T5EncoderModel_vgf_FP():
    t5_encoder_model, t5_encoder_model_inputs = (
        TestT5EncoderModel().prepare_model_and_inputs()
    )
    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            t5_encoder_model,
            t5_encoder_model_inputs,
            aten_op=[],
            exir_op=[],
            tosa_version="TOSA-1.0+FP",
            use_to_edge_transform_and_lower=True,
            transform_passes=[
                ConvertInt64ConstOpsToInt32Pass(),
                ConvertInt64OutputOpsToInt32Pass(),
                InsertInt32CastsAfterInt64PlaceholdersPass(),
            ],
        )
        pipeline.change_args(
            "check_count.exir", TestT5EncoderModel.ops_after_partitioner_FP
        )
        pipeline.run()


@common.SkipIfNoModelConverter
def test_T5EncoderModel_vgf_INT():
    t5_encoder_model, t5_encoder_model_inputs = (
        TestT5EncoderModel().prepare_model_and_inputs()
    )
    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            t5_encoder_model,
            t5_encoder_model_inputs,
            aten_op=[],
            exir_op=[],
            tosa_version="TOSA-1.0+INT",
            use_to_edge_transform_and_lower=True,
        )
        pipeline.change_args(
            "check_count.exir", TestT5EncoderModel.ops_after_partitioner_INT
        )
        pipeline.run()
