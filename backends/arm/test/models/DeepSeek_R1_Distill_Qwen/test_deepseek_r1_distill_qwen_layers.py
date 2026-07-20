# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import pytest
import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.DeepSeek_R1_Distill_Qwen.deepseek_r1_distill_qwen_test_config import (
    get_deepseek_r1_distill_qwen_1_5b_checkpoint_config,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    VgfPipeline,
)

pytest.importorskip("transformers.models.qwen2")

from transformers.models.qwen2.modeling_qwen2 import (  # noqa: E402
    apply_rotary_pos_emb,
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    repeat_kv,
)

input_t = Tuple[torch.Tensor, ...]


def _make_deepseek_r1_distill_qwen_1_5b_layer_config():
    config = get_deepseek_r1_distill_qwen_1_5b_checkpoint_config()
    config._attn_implementation = "sdpa"
    return config


def _make_position_ids(
    batch_size: int, seq_length: int, device: torch.device
) -> torch.Tensor:
    return torch.arange(seq_length, device=device).unsqueeze(0).repeat(batch_size, 1)


def _make_causal_mask(
    batch_size: int, seq_length: int, device: torch.device
) -> torch.Tensor:
    mask = torch.full(
        (seq_length, seq_length), torch.finfo(torch.float32).min, device=device
    )
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)


def _make_rope_embeddings(
    config,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary = Qwen2RotaryEmbedding(config)
    return rotary(hidden_states, position_ids)


class DeepSeekR1DistillQwenTestModule(torch.nn.Module):
    @classmethod
    def prepare_model_and_inputs(cls):
        raise NotImplementedError


def _to_bfloat16(
    model: torch.nn.Module, inputs: input_t
) -> tuple[torch.nn.Module, input_t]:
    return model.to(torch.bfloat16), tuple(
        (
            x.to(torch.bfloat16)
            if isinstance(x, torch.Tensor) and x.is_floating_point()
            else x
        )
        for x in inputs
    )


class RotaryEmbeddingModel(DeepSeekR1DistillQwenTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.rotary = Qwen2RotaryEmbedding(config)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        cos, sin = self.rotary(hidden_states, position_ids)
        return cos + sin

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_deepseek_r1_distill_qwen_1_5b_layer_config()
        model = cls(config).eval()
        hidden_states = torch.randn(2, 8, config.hidden_size)
        position_ids = _make_position_ids(2, 8, hidden_states.device)
        return model, (hidden_states, position_ids)


class RotaryApplyModel(DeepSeekR1DistillQwenTestModule):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)
        return q_embed.mean(dim=1) + k_embed.mean(dim=1)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_deepseek_r1_distill_qwen_1_5b_layer_config()
        model = cls().eval()
        hidden_states = torch.randn(2, 8, config.hidden_size)
        position_ids = _make_position_ids(2, 8, hidden_states.device)
        cos, sin = _make_rope_embeddings(config, hidden_states, position_ids)
        head_dim = config.hidden_size // config.num_attention_heads
        q = torch.randn(2, config.num_attention_heads, 8, head_dim)
        k = torch.randn(2, config.num_key_value_heads, 8, head_dim)
        return model, (q, k, cos, sin)


class RepeatKVModel(DeepSeekR1DistillQwenTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.n_rep = config.num_attention_heads // config.num_key_value_heads

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return repeat_kv(hidden_states, self.n_rep)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_deepseek_r1_distill_qwen_1_5b_layer_config()
        model = cls(config).eval()
        head_dim = config.hidden_size // config.num_attention_heads
        hidden_states = torch.randn(2, config.num_key_value_heads, 8, head_dim)
        return model, (hidden_states,)


class AttentionModel(DeepSeekR1DistillQwenTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.attn = Qwen2Attention(config, layer_idx=0)
        self.rotary = Qwen2RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        cos, sin = self.rotary(hidden_states, position_ids)
        attn_output, _ = self.attn(
            hidden_states=hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=attention_mask,
        )
        return attn_output

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_deepseek_r1_distill_qwen_1_5b_layer_config()
        model = cls(config).eval()
        hidden_states = torch.randn(2, 8, config.hidden_size)
        attention_mask = _make_causal_mask(2, 8, hidden_states.device)
        position_ids = _make_position_ids(2, 8, hidden_states.device)
        return model, (hidden_states, attention_mask, position_ids)


class RMSNormModel(DeepSeekR1DistillQwenTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_deepseek_r1_distill_qwen_1_5b_layer_config()
        model = cls(config).eval()
        hidden_states = torch.randn(2, 8, config.hidden_size)
        return model, (hidden_states,)


class MLPModel(DeepSeekR1DistillQwenTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.mlp = Qwen2MLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_deepseek_r1_distill_qwen_1_5b_layer_config()
        model = cls(config).eval()
        hidden_states = torch.randn(2, 8, config.hidden_size)
        return model, (hidden_states,)


class DecoderLayerModel(DeepSeekR1DistillQwenTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.layer = Qwen2DecoderLayer(config, layer_idx=0)
        self.rotary = Qwen2RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        cos, sin = self.rotary(hidden_states, position_ids)
        return self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=(cos, sin),
        )

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_deepseek_r1_distill_qwen_1_5b_layer_config()
        model = cls(config).eval()
        hidden_states = torch.randn(2, 8, config.hidden_size)
        attention_mask = _make_causal_mask(2, 8, hidden_states.device)
        position_ids = _make_position_ids(2, 8, hidden_states.device)
        return model, (hidden_states, attention_mask, position_ids)


class FinalNormModel(DeepSeekR1DistillQwenTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_deepseek_r1_distill_qwen_1_5b_layer_config()
        model = cls(config).eval()
        hidden_states = torch.randn(2, 8, config.hidden_size)
        return model, (hidden_states,)


@dataclass(frozen=True)
class DeepSeekR1DistillQwenTestCase:
    model_cls: type[DeepSeekR1DistillQwenTestModule]
    atol: float = 1e-3
    rtol: float = 1e-3
    qtol: int = 1
    transform_passes: tuple = field(default_factory=tuple)


TOSA_FP_TEST_CASES: dict[str, DeepSeekR1DistillQwenTestCase] = {
    "rotary_embedding": DeepSeekR1DistillQwenTestCase(model_cls=RotaryEmbeddingModel),
    "rotary_apply": DeepSeekR1DistillQwenTestCase(model_cls=RotaryApplyModel),
    "repeat_kv": DeepSeekR1DistillQwenTestCase(model_cls=RepeatKVModel),
    "attention": DeepSeekR1DistillQwenTestCase(model_cls=AttentionModel),
    "rms_norm": DeepSeekR1DistillQwenTestCase(model_cls=RMSNormModel),
    "mlp": DeepSeekR1DistillQwenTestCase(model_cls=MLPModel),
    "decoder_layer": DeepSeekR1DistillQwenTestCase(model_cls=DecoderLayerModel),
    "final_norm": DeepSeekR1DistillQwenTestCase(model_cls=FinalNormModel),
}

TOSA_BF16_TEST_CASES: dict[str, DeepSeekR1DistillQwenTestCase] = {
    "rotary_embedding": DeepSeekR1DistillQwenTestCase(
        model_cls=RotaryEmbeddingModel,
        atol=1e-2,
        rtol=1e-2,
    ),
    "rotary_apply": DeepSeekR1DistillQwenTestCase(
        model_cls=RotaryApplyModel,
        atol=1e-2,
        rtol=1e-2,
    ),
    "repeat_kv": DeepSeekR1DistillQwenTestCase(
        model_cls=RepeatKVModel,
        atol=1e-2,
        rtol=1e-2,
    ),
    "attention": DeepSeekR1DistillQwenTestCase(
        model_cls=AttentionModel,
        atol=1e-2,
        rtol=1e-2,
    ),
    "rms_norm": DeepSeekR1DistillQwenTestCase(
        model_cls=RMSNormModel,
        atol=1e-2,
        rtol=1e-2,
    ),
    "mlp": DeepSeekR1DistillQwenTestCase(
        model_cls=MLPModel,
        atol=1e-2,
        rtol=1e-2,
    ),
    "decoder_layer": DeepSeekR1DistillQwenTestCase(
        model_cls=DecoderLayerModel,
        atol=1e-2,
        rtol=1e-2,
    ),
    "final_norm": DeepSeekR1DistillQwenTestCase(
        model_cls=FinalNormModel,
        atol=1e-2,
        rtol=1e-2,
    ),
}

VGF_NO_QUANT_TEST_CASES: dict[str, DeepSeekR1DistillQwenTestCase] = {
    "rotary_embedding": DeepSeekR1DistillQwenTestCase(model_cls=RotaryEmbeddingModel),
    "rotary_apply": DeepSeekR1DistillQwenTestCase(model_cls=RotaryApplyModel),
    "repeat_kv": DeepSeekR1DistillQwenTestCase(model_cls=RepeatKVModel),
    "attention": DeepSeekR1DistillQwenTestCase(model_cls=AttentionModel),
    "rms_norm": DeepSeekR1DistillQwenTestCase(model_cls=RMSNormModel),
    "mlp": DeepSeekR1DistillQwenTestCase(model_cls=MLPModel),
    "decoder_layer": DeepSeekR1DistillQwenTestCase(model_cls=DecoderLayerModel),
    "final_norm": DeepSeekR1DistillQwenTestCase(model_cls=FinalNormModel),
}

VGF_NO_QUANT_BF16_TEST_CASES: dict[str, DeepSeekR1DistillQwenTestCase] = (
    TOSA_BF16_TEST_CASES
)


@common.parametrize(
    "test_case",
    TOSA_FP_TEST_CASES,
)
def test_deepseek_r1_distill_qwen_tosa_FP(
    test_case: DeepSeekR1DistillQwenTestCase,
):
    model, inputs = test_case.model_cls.prepare_model_and_inputs()
    with torch.no_grad():
        pipeline = TosaPipelineFP[input_t](
            model,
            inputs,
            aten_op=[],
            exir_op=[],
            transform_passes=list(test_case.transform_passes),
        )
        pipeline.run()


@common.parametrize(
    "test_case",
    TOSA_BF16_TEST_CASES,
)
def test_deepseek_r1_distill_qwen_tosa_FP_bf16(
    test_case: DeepSeekR1DistillQwenTestCase,
):
    model, inputs = test_case.model_cls.prepare_model_and_inputs()
    model, inputs = _to_bfloat16(model, inputs)
    with torch.no_grad():
        pipeline = TosaPipelineFP[input_t](
            model,
            inputs,
            aten_op=[],
            exir_op=[],
            transform_passes=list(test_case.transform_passes),
            tosa_extensions=["bf16"],
            atol=test_case.atol,
            rtol=test_case.rtol,
        )
        pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize(
    "test_case",
    VGF_NO_QUANT_TEST_CASES,
)
def test_deepseek_r1_distill_qwen_vgf_no_quant(
    test_case: DeepSeekR1DistillQwenTestCase,
):
    model, inputs = test_case.model_cls.prepare_model_and_inputs()
    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            model,
            inputs,
            aten_op=[],
            exir_op=[],
            quantize=False,
            atol=test_case.atol,
            rtol=test_case.rtol,
            qtol=test_case.qtol,
            transform_passes=list(test_case.transform_passes),
        )
        pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize(
    "test_case",
    VGF_NO_QUANT_BF16_TEST_CASES,
)
def test_deepseek_r1_distill_qwen_vgf_no_quant_bf16(
    test_case: DeepSeekR1DistillQwenTestCase,
):
    model, inputs = test_case.model_cls.prepare_model_and_inputs()
    model, inputs = _to_bfloat16(model, inputs)
    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            model,
            inputs,
            aten_op=[],
            exir_op=[],
            quantize=False,
            atol=test_case.atol,
            rtol=test_case.rtol,
            qtol=test_case.qtol,
            transform_passes=list(test_case.transform_passes),
        )
        pipeline.run()
