# Copyright 2025 Arm Limited and/or its affiliates.
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
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

from transformers import AutoTokenizer, T5ForConditionalGeneration

input_t3 = Tuple[
    torch.LongTensor, torch.LongTensor, torch.LongTensor
]  # (input_ids, attention_mask, decoder_input_ids)


class TestT5ForConditionalGeneration:
    # Adjust nbr below as we increase op support.
    ops_after_partitioner_FP = {
        "executorch_exir_dialects_edge__ops_aten_where_self": 2,
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 5,
        "torch.ops.higher_order.executorch_call_delegate": 2,
    }

    ops_after_partitioner_INT = {
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 3,
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 10,
        "torch.ops.higher_order.executorch_call_delegate": 3,
    }

    ops_after_partitioner_vgf_no_quantize = {
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 4,
        "torch.ops.higher_order.executorch_call_delegate": 2,
    }

    ops_after_partitioner_vgf_quantize = ops_after_partitioner_vgf_no_quantize

    def _prepare_inputs(
        self,
        prompt: str,
    ):
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids  # (1, src_len)
        attention_mask = enc.attention_mask  # (1, src_len)
        # T5 uses <pad> as BOS / decoder start
        bos_id = tokenizer.pad_token_id
        decoder_input_ids = torch.tensor([[bos_id]], dtype=torch.long)  # (1, 1)
        return input_ids, attention_mask, decoder_input_ids

    def prepare_model_and_inputs(self, prompt):
        class T5ForConditionalGenerationWrapper(T5ForConditionalGeneration):
            def forward(self, input_ids, attention_mask, decoder_input_ids):
                out = super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    use_cache=False,  # simpler, export-friendly
                    return_dict=True,
                )
                return out.logits  # Tensor: (B, tgt_len=1, vocab)

        model = T5ForConditionalGenerationWrapper.from_pretrained("google-t5/t5-small")
        model.config.use_cache = False
        inputs = self._prepare_inputs(prompt)

        return model, inputs


@pytest.mark.slow
def test_T5ForConditionalGeneration_tosa_FP():
    prompt = "summarize: studies have shown that owning a dog is good for you"
    model, inputs = TestT5ForConditionalGeneration().prepare_model_and_inputs(prompt)
    with torch.no_grad():
        pipeline = TosaPipelineFP[input_t3](
            model,
            inputs,
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
            "check_count.exir", TestT5ForConditionalGeneration.ops_after_partitioner_FP
        )
        pipeline.run()


@pytest.mark.slow
def test_T5ForConditionalGeneration_tosa_INT():
    prompt = "summarize: studies have shown that owning a dog is good for you"
    model, inputs = TestT5ForConditionalGeneration().prepare_model_and_inputs(prompt)
    with torch.no_grad():
        pipeline = TosaPipelineINT[input_t3](
            model,
            inputs,
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            atol=20,  # TODO: MLETORCH-1703: Reduce the tolerance of quantized T5ForConditionalGeneration
        )
        pipeline.change_args(
            "check_count.exir",
            TestT5ForConditionalGeneration.ops_after_partitioner_INT,
        )
        pipeline.run()


@pytest.mark.slow
@common.SkipIfNoModelConverter
def test_T5ForConditionalGeneration_vgf_no_quant():
    prompt = "summarize: studies have shown that owning a dog is good for you"
    model, inputs = TestT5ForConditionalGeneration().prepare_model_and_inputs(prompt)
    with torch.no_grad():
        pipeline = VgfPipeline[input_t3](
            model,
            inputs,
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            transform_passes=[
                ConvertInt64ConstOpsToInt32Pass(),
                ConvertInt64OutputOpsToInt32Pass(),
                InsertInt32CastsAfterInt64PlaceholdersPass(),
            ],
            quantize=False,
        )
        pipeline.change_args(
            "check_count.exir",
            TestT5ForConditionalGeneration.ops_after_partitioner_vgf_no_quantize,
        )
        pipeline.run()


@pytest.mark.slow
@common.SkipIfNoModelConverter
def test_T5ForConditionalGeneration_vgf_quant():
    prompt = "summarize: studies have shown that owning a dog is good for you"
    model, inputs = TestT5ForConditionalGeneration().prepare_model_and_inputs(prompt)
    with torch.no_grad():
        pipeline = VgfPipeline[input_t3](
            model,
            inputs,
            aten_op=[],
            exir_op=[],
            use_to_edge_transform_and_lower=True,
            atol=20,  # TODO: MLETORCH-1703: Reduce the tolerance of quantized T5ForConditionalGeneration
            quantize=True,
        )
        pipeline.change_args(
            "check_count.exir",
            TestT5ForConditionalGeneration.ops_after_partitioner_vgf_quantize,
        )
        pipeline.run()
