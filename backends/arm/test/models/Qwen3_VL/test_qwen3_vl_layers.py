# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.Qwen3_VL.qwen3_vl_test_config import (
    get_qwen3_vl_2b_instruct_checkpoint_config,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    VgfPipeline,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionMLP,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionRotaryEmbedding,
)

input_t = Tuple[torch.Tensor | int, ...]


def _make_qwen3_vl_2b_instruct_layer_config():
    config = get_qwen3_vl_2b_instruct_checkpoint_config()
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    return config


def _make_text_position_ids(
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


def _make_image_grid_thw(device: torch.device) -> torch.Tensor:
    return torch.tensor([[1, 4, 4]], dtype=torch.long, device=device)


def _make_pixel_values(config, device: torch.device) -> torch.Tensor:
    grid_thw = _make_image_grid_thw(device)
    patch_volume = (
        config.vision_config.in_channels
        * config.vision_config.temporal_patch_size
        * config.vision_config.patch_size
        * config.vision_config.patch_size
    )
    num_patches = int(torch.prod(grid_thw[0]).item())
    return torch.randn(num_patches, patch_volume, device=device)


def _make_vision_position_embeddings(
    config, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    grid_thw = _make_image_grid_thw(device)
    num_patches = int(torch.prod(grid_thw[0]).item())
    head_dim = config.vision_config.hidden_size // config.vision_config.num_heads
    return (
        torch.randn(num_patches, head_dim, device=device),
        torch.randn(num_patches, head_dim, device=device),
    )


def _make_vision_cu_seqlens(device: torch.device) -> torch.Tensor:
    grid_thw = _make_image_grid_thw(device)
    num_patches = int(torch.prod(grid_thw[0]).item())
    return torch.tensor([0, num_patches], dtype=torch.int32, device=device)


class Qwen3VLTestModule(torch.nn.Module):
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


class Qwen3VLVisionMLPModel(Qwen3VLTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.mlp = Qwen3VLVisionMLP(config.vision_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_qwen3_vl_2b_instruct_layer_config()
        model = cls(config).eval()
        hidden_states = torch.randn(16, config.vision_config.hidden_size)
        return model, (hidden_states,)


class VisionPatchEmbedModel(Qwen3VLTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_embed = Qwen3VLVisionPatchEmbed(config.vision_config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(pixel_values)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_qwen3_vl_2b_instruct_layer_config()
        model = cls(config).eval()
        return model, (_make_pixel_values(config, torch.device("cpu")),)


class VisionRotaryEmbeddingModel(Qwen3VLTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        head_dim = config.vision_config.hidden_size // config.vision_config.num_heads
        self.rotary = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

    def forward(self, max_hw: int) -> torch.Tensor:
        return self.rotary(max_hw)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_qwen3_vl_2b_instruct_layer_config()
        grid_thw = _make_image_grid_thw(torch.device("cpu"))
        max_hw = int(grid_thw[:, 1:].max().item())
        model = cls(config).eval()
        return model, (max_hw,)


class VisionRotaryApplyModel(Qwen3VLTestModule):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        q_embed, k_embed = apply_rotary_pos_emb_vision(q, k, cos, sin)
        return q_embed + k_embed

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_qwen3_vl_2b_instruct_layer_config()
        model = cls().eval()
        cos, sin = _make_vision_position_embeddings(config, torch.device("cpu"))
        head_dim = config.vision_config.hidden_size // config.vision_config.num_heads
        q = torch.randn(cos.shape[0], config.vision_config.num_heads, head_dim)
        k = torch.randn(cos.shape[0], config.vision_config.num_heads, head_dim)
        return model, (q, k, cos, sin)


class VisionAttentionModel(Qwen3VLTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.attn = Qwen3VLVisionAttention(config.vision_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=(cos, sin),
        )

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_qwen3_vl_2b_instruct_layer_config()
        model = cls(config).eval()
        hidden_states = model.attn.qkv.weight.new_empty(
            16, config.vision_config.hidden_size
        ).normal_()
        cos, sin = _make_vision_position_embeddings(config, hidden_states.device)
        cu_seqlens = _make_vision_cu_seqlens(hidden_states.device)
        return model, (hidden_states, cu_seqlens, cos, sin)


class VisionBlockModel(Qwen3VLTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.block = Qwen3VLVisionBlock(config.vision_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=(cos, sin),
        )

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_qwen3_vl_2b_instruct_layer_config()
        model = cls(config).eval()
        hidden_states = torch.randn(16, config.vision_config.hidden_size)
        cos, sin = _make_vision_position_embeddings(config, hidden_states.device)
        cu_seqlens = _make_vision_cu_seqlens(hidden_states.device)
        return model, (hidden_states, cu_seqlens, cos, sin)


class VisionPatchMergerModel(Qwen3VLTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.merger = Qwen3VLVisionPatchMerger(config.vision_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.merger(hidden_states)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_qwen3_vl_2b_instruct_layer_config()
        model = cls(config).eval()
        hidden_states = torch.randn(
            config.vision_config.spatial_merge_size**2,
            config.vision_config.hidden_size,
        )
        return model, (hidden_states,)


class TextRMSNormModel(Qwen3VLTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm = Qwen3VLTextRMSNorm(
            config.text_config.hidden_size, eps=config.text_config.rms_norm_eps
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_qwen3_vl_2b_instruct_layer_config()
        model = cls(config).eval()
        hidden_states = torch.randn(2, 8, config.text_config.hidden_size)
        return model, (hidden_states,)


class TextRotaryEmbeddingModel(Qwen3VLTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.rotary = Qwen3VLTextRotaryEmbedding(config.text_config)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        cos, sin = self.rotary(hidden_states, position_ids)
        return cos + sin

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_qwen3_vl_2b_instruct_layer_config()
        model = cls(config).eval()
        hidden_states = torch.randn(2, 8, config.text_config.hidden_size)
        position_ids = _make_text_position_ids(2, 8, hidden_states.device)
        return model, (hidden_states, position_ids)


class TextRotaryApplyModel(Qwen3VLTestModule):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)
        return q_embed.mean(dim=1) + k_embed.mean(dim=1)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_qwen3_vl_2b_instruct_layer_config()
        model = cls().eval()
        hidden_states = torch.randn(2, 8, config.text_config.hidden_size)
        position_ids = _make_text_position_ids(2, 8, hidden_states.device)
        cos, sin = Qwen3VLTextRotaryEmbedding(config.text_config)(
            hidden_states, position_ids
        )
        q = torch.randn(
            2,
            config.text_config.num_attention_heads,
            8,
            config.text_config.head_dim,
        )
        k = torch.randn(
            2,
            config.text_config.num_key_value_heads,
            8,
            config.text_config.head_dim,
        )
        return model, (q, k, cos, sin)


class TextAttentionModel(Qwen3VLTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.attn = Qwen3VLTextAttention(config.text_config, layer_idx=0)
        self.rotary = Qwen3VLTextRotaryEmbedding(config.text_config)

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
        config = _make_qwen3_vl_2b_instruct_layer_config()
        model = cls(config).eval()
        hidden_states = torch.randn(2, 8, config.text_config.hidden_size)
        attention_mask = _make_causal_mask(2, 8, hidden_states.device)
        position_ids = _make_text_position_ids(2, 8, hidden_states.device)
        return model, (hidden_states, attention_mask, position_ids)


class QKNormModel(Qwen3VLTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.attn = Qwen3VLTextAttention(config.text_config, layer_idx=0)

    def forward(self, q_states: torch.Tensor, k_states: torch.Tensor) -> torch.Tensor:
        q_states = self.attn.q_norm(q_states)
        k_states = self.attn.k_norm(k_states)
        return q_states.mean(dim=(-1, -2)) + k_states.mean(dim=(-1, -2))

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_qwen3_vl_2b_instruct_layer_config()
        model = cls(config).eval()
        q_states = torch.randn(
            2,
            8,
            config.text_config.num_attention_heads,
            config.text_config.head_dim,
        )
        k_states = torch.randn(
            2,
            8,
            config.text_config.num_key_value_heads,
            config.text_config.head_dim,
        )
        return model, (q_states, k_states)


class TextMLPModel(Qwen3VLTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.mlp = Qwen3VLTextMLP(config.text_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_qwen3_vl_2b_instruct_layer_config()
        model = cls(config).eval()
        hidden_states = torch.randn(2, 8, config.text_config.hidden_size)
        return model, (hidden_states,)


class TextDecoderLayerModel(Qwen3VLTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.layer = Qwen3VLTextDecoderLayer(config.text_config, layer_idx=0)
        self.rotary = Qwen3VLTextRotaryEmbedding(config.text_config)

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
            position_ids=position_ids,
            position_embeddings=(cos, sin),
        )

    @classmethod
    def prepare_model_and_inputs(cls):
        config = _make_qwen3_vl_2b_instruct_layer_config()
        model = cls(config).eval()
        hidden_states = torch.randn(2, 8, config.text_config.hidden_size)
        attention_mask = _make_causal_mask(2, 8, hidden_states.device)
        position_ids = _make_text_position_ids(2, 8, hidden_states.device)
        return model, (hidden_states, attention_mask, position_ids)


@dataclass(frozen=True)
class Qwen3VLTestCase:
    model_cls: type[Qwen3VLTestModule]
    transform_passes: tuple = field(default_factory=tuple)


TOSA_FP_TEST_CASES: dict[str, Qwen3VLTestCase] = {
    "vision_mlp": Qwen3VLTestCase(model_cls=Qwen3VLVisionMLPModel),
    "vision_patch_embed": Qwen3VLTestCase(model_cls=VisionPatchEmbedModel),
    "vision_rotary_embedding": Qwen3VLTestCase(model_cls=VisionRotaryEmbeddingModel),
    "vision_rotary_apply": Qwen3VLTestCase(model_cls=VisionRotaryApplyModel),
    "vision_attention": Qwen3VLTestCase(model_cls=VisionAttentionModel),
    "vision_block": Qwen3VLTestCase(model_cls=VisionBlockModel),
    "vision_patch_merger": Qwen3VLTestCase(model_cls=VisionPatchMergerModel),
    "text_rms_norm": Qwen3VLTestCase(model_cls=TextRMSNormModel),
    "text_rotary_embedding": Qwen3VLTestCase(model_cls=TextRotaryEmbeddingModel),
    "text_rotary_apply": Qwen3VLTestCase(model_cls=TextRotaryApplyModel),
    "text_attention": Qwen3VLTestCase(model_cls=TextAttentionModel),
    "qk_norm": Qwen3VLTestCase(model_cls=QKNormModel),
    "text_mlp": Qwen3VLTestCase(model_cls=TextMLPModel),
    "text_decoder_layer": Qwen3VLTestCase(model_cls=TextDecoderLayerModel),
}

VGF_NO_QUANT_TEST_CASES: dict[str, Qwen3VLTestCase] = TOSA_FP_TEST_CASES

TOSA_BF16_TEST_CASES: dict[str, Qwen3VLTestCase] = {
    "vision_mlp": TOSA_FP_TEST_CASES["vision_mlp"],
    "vision_patch_embed": TOSA_FP_TEST_CASES["vision_patch_embed"],
    "vision_rotary_embedding": TOSA_FP_TEST_CASES["vision_rotary_embedding"],
    "vision_rotary_apply": TOSA_FP_TEST_CASES["vision_rotary_apply"],
    "vision_attention": TOSA_FP_TEST_CASES["vision_attention"],
    "vision_block": TOSA_FP_TEST_CASES["vision_block"],
    "vision_patch_merger": TOSA_FP_TEST_CASES["vision_patch_merger"],
    "text_rms_norm": TOSA_FP_TEST_CASES["text_rms_norm"],
    "qk_norm": TOSA_FP_TEST_CASES["qk_norm"],
}


@common.parametrize(
    "test_case",
    TOSA_FP_TEST_CASES,
)
def test_qwen3_vl_tosa_FP(test_case: Qwen3VLTestCase):
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
def test_qwen3_vl_tosa_FP_bf16(test_case: Qwen3VLTestCase):
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
            atol=1e-2,
            rtol=1e-2,
        )
        pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize(
    "test_case",
    VGF_NO_QUANT_TEST_CASES,
)
def test_qwen3_vl_vgf_no_quant(test_case: Qwen3VLTestCase):
    model, inputs = test_case.model_cls.prepare_model_and_inputs()
    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            model,
            inputs,
            aten_op=[],
            exir_op=[],
            quantize=False,
            transform_passes=list(test_case.transform_passes),
        )
        pipeline.run()
