# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Tuple

import pytest
import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.DeepSeek_R1_Distill_Qwen.deepseek_r1_distill_qwen_test_config import (
    get_deepseek_r1_distill_qwen_1_5b_checkpoint_config,
)
from executorch.backends.arm.test.models.model_test_utils import to_bfloat16
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    VgfPipeline,
)

from transformers.models.qwen2.modeling_qwen2 import (  # noqa: E402
    Qwen2ForCausalLM,
    Qwen2Model,
)

input_t = Tuple[torch.Tensor, ...]
config_factory_t = Callable[[], object]


def _make_deepseek_r1_distill_qwen_1_5b_model_config():
    config = get_deepseek_r1_distill_qwen_1_5b_checkpoint_config()
    config._attn_implementation = "sdpa"
    config.use_cache = False
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    return config


def _make_deepseek_r1_distill_qwen_e2e_test_config():
    config = _make_deepseek_r1_distill_qwen_1_5b_model_config()

    config.vocab_size = 1024
    config.bos_token_id = 1
    config.eos_token_id = 2
    config.hidden_size = 128
    config.intermediate_size = 384
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.max_position_embeddings = 1024
    config.max_window_layers = 2
    config.layer_types = ["full_attention"] * config.num_hidden_layers

    return config


def _make_position_ids(
    batch_size: int, seq_length: int, device: torch.device
) -> torch.Tensor:
    return torch.arange(seq_length, device=device).unsqueeze(0).repeat(batch_size, 1)


def _make_model_inputs(config, batch_size: int = 1, seq_length: int = 8) -> input_t:
    inputs_embeds = torch.randn(batch_size, seq_length, config.hidden_size)
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    position_ids = _make_position_ids(batch_size, seq_length, inputs_embeds.device)
    return inputs_embeds, attention_mask, position_ids


class DeepSeekR1DistillQwenModelTestModule(torch.nn.Module):
    @classmethod
    def prepare_model_and_inputs(
        cls, config_factory=_make_deepseek_r1_distill_qwen_e2e_test_config
    ):
        raise NotImplementedError


class BaseModelWrapper(DeepSeekR1DistillQwenModelTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen2Model(config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        return outputs.last_hidden_state

    @classmethod
    def prepare_model_and_inputs(
        cls, config_factory=_make_deepseek_r1_distill_qwen_e2e_test_config
    ):
        torch.manual_seed(0)
        config = config_factory()
        model = cls(config).eval()
        return model, _make_model_inputs(config)


class CausalLMWrapper(DeepSeekR1DistillQwenModelTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen2ForCausalLM(config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        return outputs.logits

    @classmethod
    def prepare_model_and_inputs(
        cls, config_factory=_make_deepseek_r1_distill_qwen_e2e_test_config
    ):
        torch.manual_seed(0)
        config = config_factory()
        model = cls(config).eval()
        return model, _make_model_inputs(config)


@dataclass(frozen=True)
class DeepSeekR1DistillQwenModelTestCase:
    model_cls: type[DeepSeekR1DistillQwenModelTestModule]
    config_factory: config_factory_t = _make_deepseek_r1_distill_qwen_e2e_test_config
    atol: float = 1e-3
    rtol: float = 1e-3
    qtol: int = 1
    tosa_spec: str | None = None


REDUCED_TOSA_FP_TEST_CASES: dict[str, DeepSeekR1DistillQwenModelTestCase] = {
    "base_model": DeepSeekR1DistillQwenModelTestCase(model_cls=BaseModelWrapper),
    "causal_lm": DeepSeekR1DistillQwenModelTestCase(
        model_cls=CausalLMWrapper,
        atol=3e-2,
        rtol=1e-2,
    ),
}

REDUCED_TOSA_BF16_TEST_CASES: dict[str, DeepSeekR1DistillQwenModelTestCase] = {
    "base_model": DeepSeekR1DistillQwenModelTestCase(
        model_cls=BaseModelWrapper,
        atol=0.1,
        rtol=0.1,
    ),
    "causal_lm": DeepSeekR1DistillQwenModelTestCase(
        model_cls=CausalLMWrapper,
        atol=0.1,
        rtol=0.1,
    ),
}

REDUCED_VGF_NO_QUANT_TEST_CASES: dict[str, DeepSeekR1DistillQwenModelTestCase] = {
    "base_model": DeepSeekR1DistillQwenModelTestCase(model_cls=BaseModelWrapper),
}

REDUCED_VGF_NO_QUANT_BF16_TEST_CASES: dict[str, DeepSeekR1DistillQwenModelTestCase] = {
    "base_model": DeepSeekR1DistillQwenModelTestCase(
        model_cls=BaseModelWrapper,
        atol=0.1,
        rtol=0.1,
        tosa_spec="TOSA-1.0+FP+bf16",
    ),
}

CHECKPOINT_TOSA_BF16_XLARGE_TEST_CASES: dict[
    str, DeepSeekR1DistillQwenModelTestCase
] = {
    "base_model": DeepSeekR1DistillQwenModelTestCase(
        model_cls=BaseModelWrapper,
        config_factory=_make_deepseek_r1_distill_qwen_1_5b_model_config,
        atol=0.1,
        rtol=0.1,
    ),
}

CHECKPOINT_VGF_NO_QUANT_BF16_XLARGE_TEST_CASES: dict[
    str, DeepSeekR1DistillQwenModelTestCase
] = {
    "base_model": DeepSeekR1DistillQwenModelTestCase(
        model_cls=BaseModelWrapper,
        config_factory=_make_deepseek_r1_distill_qwen_1_5b_model_config,
        atol=0.1,
        rtol=0.1,
        tosa_spec="TOSA-1.0+FP+bf16",
    ),
}


def _run_tosa_fp_model_test(
    test_case: DeepSeekR1DistillQwenModelTestCase,
    *,
    bf16: bool = False,
):
    model, inputs = test_case.model_cls.prepare_model_and_inputs(
        test_case.config_factory
    )
    if bf16:
        model, inputs = to_bfloat16(model, inputs)
    with torch.no_grad():
        if bf16:
            pipeline = TosaPipelineFP[input_t](
                model,
                inputs,
                aten_op=[],
                exir_op=[],
                tosa_extensions=["bf16"],
                atol=test_case.atol,
                rtol=test_case.rtol,
                qtol=test_case.qtol,
            )
        else:
            pipeline = TosaPipelineFP[input_t](
                model,
                inputs,
                aten_op=[],
                exir_op=[],
                atol=test_case.atol,
                rtol=test_case.rtol,
                qtol=test_case.qtol,
            )
        pipeline.run()


def _run_vgf_no_quant_model_test(
    test_case: DeepSeekR1DistillQwenModelTestCase,
    *,
    bf16: bool = False,
):
    model, inputs = test_case.model_cls.prepare_model_and_inputs(
        test_case.config_factory
    )
    if bf16:
        model, inputs = to_bfloat16(model, inputs)
    with torch.no_grad():
        if test_case.tosa_spec is not None:
            pipeline = VgfPipeline[input_t](
                model,
                inputs,
                aten_op=[],
                exir_op=[],
                quantize=False,
                tosa_spec=test_case.tosa_spec,
                atol=test_case.atol,
                rtol=test_case.rtol,
                qtol=test_case.qtol,
            )
        else:
            pipeline = VgfPipeline[input_t](
                model,
                inputs,
                aten_op=[],
                exir_op=[],
                quantize=False,
                atol=test_case.atol,
                rtol=test_case.rtol,
                qtol=test_case.qtol,
            )
        pipeline.run()


@pytest.mark.slow
@common.parametrize("test_case", REDUCED_TOSA_FP_TEST_CASES)
def test_deepseek_r1_distill_qwen_full_models_tosa_FP(
    test_case: DeepSeekR1DistillQwenModelTestCase,
):
    _run_tosa_fp_model_test(test_case)


@pytest.mark.slow
@common.parametrize("test_case", REDUCED_TOSA_BF16_TEST_CASES)
def test_deepseek_r1_distill_qwen_full_models_tosa_FP_bf16(
    test_case: DeepSeekR1DistillQwenModelTestCase,
):
    _run_tosa_fp_model_test(test_case, bf16=True)


@pytest.mark.slow
@common.SkipIfNoModelConverter
@common.parametrize("test_case", REDUCED_VGF_NO_QUANT_TEST_CASES)
def test_deepseek_r1_distill_qwen_full_models_vgf_no_quant(
    test_case: DeepSeekR1DistillQwenModelTestCase,
):
    _run_vgf_no_quant_model_test(test_case)


@pytest.mark.slow
@common.SkipIfNoModelConverter
@common.parametrize("test_case", REDUCED_VGF_NO_QUANT_BF16_TEST_CASES)
def test_deepseek_r1_distill_qwen_full_models_vgf_no_quant_bf16(
    test_case: DeepSeekR1DistillQwenModelTestCase,
):
    _run_vgf_no_quant_model_test(test_case, bf16=True)


@pytest.mark.slow
@pytest.mark.xlarge
@common.parametrize("test_case", CHECKPOINT_TOSA_BF16_XLARGE_TEST_CASES)
def test_deepseek_r1_distill_qwen_1_5b_full_models_tosa_FP_bf16(
    test_case: DeepSeekR1DistillQwenModelTestCase,
):
    _run_tosa_fp_model_test(test_case, bf16=True)


@pytest.mark.slow
@pytest.mark.xlarge
@common.SkipIfNoModelConverter
@common.parametrize("test_case", CHECKPOINT_VGF_NO_QUANT_BF16_XLARGE_TEST_CASES)
def test_deepseek_r1_distill_qwen_1_5b_full_models_vgf_no_quant_bf16(
    test_case: DeepSeekR1DistillQwenModelTestCase,
):
    _run_vgf_no_quant_model_test(test_case, bf16=True)
