# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pytest
import torch
import torch.nn.functional as F
from executorch.backends.arm.test import common
from executorch.backends.arm.test.models.Qwen3_VL.qwen3_vl_test_config import (
    get_qwen3_vl_2b_instruct_checkpoint_config,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    VgfPipeline,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)

input_t = Tuple[torch.Tensor, ...]


def _make_qwen3_vl_2b_instruct_layer_config():
    config = get_qwen3_vl_2b_instruct_checkpoint_config()
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    return config


def _make_qwen3_vl_e2e_test_config():
    config = _make_qwen3_vl_2b_instruct_layer_config()

    config.text_config.vocab_size = 1024
    config.text_config.bos_token_id = 1
    config.text_config.eos_token_id = 2
    config.text_config.hidden_size = 128
    config.text_config.intermediate_size = 384
    config.text_config.num_hidden_layers = 2
    config.text_config.num_attention_heads = 4
    config.text_config.num_key_value_heads = 2
    config.text_config.head_dim = 32
    config.text_config.max_position_embeddings = 1024
    config.text_config.rope_parameters["mrope_section"] = [4, 4, 4]
    config.text_config.rope_scaling["mrope_section"] = [4, 4, 4]

    config.vision_config.deepstack_visual_indexes = [0]
    config.vision_config.depth = 2
    config.vision_config.hidden_size = 128
    config.vision_config.intermediate_size = 512
    config.vision_config.num_heads = 4
    config.vision_config.num_position_embeddings = 16
    config.vision_config.out_hidden_size = 128

    return config


def _make_text_position_ids(
    batch_size: int, seq_length: int, device: torch.device
) -> torch.Tensor:
    return torch.arange(seq_length, device=device).unsqueeze(0).repeat(batch_size, 1)


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


class Qwen3VLModelTestModule(torch.nn.Module):
    @classmethod
    def prepare_model_and_inputs(cls):
        raise NotImplementedError


def _to_bfloat16_model_and_floating_inputs(
    model: torch.nn.Module, inputs: input_t
) -> tuple[torch.nn.Module, input_t]:
    """Convert model and floating inputs for BF16 backend coverage."""

    return model.to(torch.bfloat16), tuple(
        (
            x.to(torch.bfloat16)
            if isinstance(x, torch.Tensor) and x.is_floating_point()
            else x
        )
        for x in inputs
    )


class TextModelWrapper(Qwen3VLModelTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen3VLTextModel(config.text_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return outputs.last_hidden_state

    @classmethod
    def prepare_model_and_inputs(cls):
        torch.manual_seed(0)
        config = _make_qwen3_vl_e2e_test_config()
        model = cls(config).eval()
        input_ids = torch.randint(0, 128, (2, 8), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        position_ids = _make_text_position_ids(2, 8, input_ids.device)
        return model, (input_ids, attention_mask, position_ids)


class LowerableVisionModelWrapper(Qwen3VLModelTestModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.visual = Qwen3VLVisionModel(config.vision_config)

        with torch.no_grad():
            grid_thw = _make_image_grid_thw(self.visual.pos_embed.weight.device)
            pos_embeds = self.visual.fast_pos_embed_interpolate(grid_thw)

            rotary_pos_emb = self.visual.rot_pos_emb(grid_thw)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        self.register_buffer("pos_embeds", pos_embeds)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("cu_seqlens", cu_seqlens)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.visual.patch_embed(pixel_values)
        hidden_states = hidden_states + self.pos_embeds

        position_embeddings = (self.cos, self.sin)
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.visual.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.visual.deepstack_visual_indexes:
                deepstack_feature = self.visual.deepstack_merger_list[
                    self.visual.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.visual.merger(hidden_states)

        # Keep deepstack feature extraction in the exported graph without
        # changing the model output.
        deepstack_residual = hidden_states.new_zeros(())
        for deepstack_feature in deepstack_feature_lists:
            deepstack_residual = deepstack_residual + deepstack_feature.sum() * 0

        return hidden_states + deepstack_residual

    @classmethod
    def prepare_model_and_inputs(cls):
        torch.manual_seed(0)
        config = _make_qwen3_vl_e2e_test_config()
        model = cls(config).eval()
        pixel_values = _make_pixel_values(config, torch.device("cpu"))
        return model, (pixel_values,)


@dataclass(frozen=True)
class Qwen3VLModelTestCase:
    model_cls: type[Qwen3VLModelTestModule]
    run_on_vulkan_runtime: bool = True
    atol: float = 1e-3
    rtol: float = 1e-3


TOSA_FP_TEST_CASES: dict[str, Qwen3VLModelTestCase] = {
    "vision_model": Qwen3VLModelTestCase(
        model_cls=LowerableVisionModelWrapper,
    ),
    "text_model": Qwen3VLModelTestCase(
        model_cls=TextModelWrapper,
        atol=3e-2,
        rtol=1e-2,
    ),
}

VGF_NO_QUANT_TEST_CASES: dict[str, Qwen3VLModelTestCase] = {
    "vision_model": Qwen3VLModelTestCase(
        model_cls=LowerableVisionModelWrapper,
        run_on_vulkan_runtime=False,
    ),
    "text_model": Qwen3VLModelTestCase(
        model_cls=TextModelWrapper,
        run_on_vulkan_runtime=False,
    ),
}


@pytest.mark.slow
@common.parametrize("test_case", TOSA_FP_TEST_CASES)
def test_qwen3_vl_full_models_tosa_FP(test_case: Qwen3VLModelTestCase):
    model, inputs = test_case.model_cls.prepare_model_and_inputs()
    with torch.no_grad():
        pipeline = TosaPipelineFP[input_t](
            model,
            inputs,
            aten_op=[],
            exir_op=[],
            atol=test_case.atol,
            rtol=test_case.rtol,
        )
        pipeline.run()


@pytest.mark.slow
@common.parametrize("test_case", TOSA_FP_TEST_CASES)
def test_qwen3_vl_full_models_tosa_FP_bf16(test_case: Qwen3VLModelTestCase):
    model, inputs = test_case.model_cls.prepare_model_and_inputs()
    model, inputs = _to_bfloat16_model_and_floating_inputs(model, inputs)
    # Slightly higher atol for TOSA BF16 on aarch64 (MLETORCH-2048: numeric mismatch)
    atol = (
        0.4
        if common.is_aarch64_host()
        and test_case.model_cls is LowerableVisionModelWrapper
        else 0.1
    )
    with torch.no_grad():
        pipeline = TosaPipelineFP[input_t](
            model,
            inputs,
            aten_op=[],
            exir_op=[],
            tosa_extensions=["bf16"],
            atol=atol,
            rtol=0.1,
        )
        pipeline.run()


@pytest.mark.slow
@common.SkipIfNoModelConverter
@common.parametrize("test_case", VGF_NO_QUANT_TEST_CASES)
def test_qwen3_vl_full_models_vgf_no_quant(test_case: Qwen3VLModelTestCase):
    model, inputs = test_case.model_cls.prepare_model_and_inputs()
    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            model,
            inputs,
            aten_op=[],
            exir_op=[],
            quantize=False,
            run_on_vulkan_runtime=test_case.run_on_vulkan_runtime,
        )
        pipeline.run()


@pytest.mark.slow
@common.SkipIfNoModelConverter
@common.parametrize("test_case", VGF_NO_QUANT_TEST_CASES)
def test_qwen3_vl_full_models_vgf_no_quant_bf16(test_case: Qwen3VLModelTestCase):
    model, inputs = test_case.model_cls.prepare_model_and_inputs()
    model, inputs = _to_bfloat16_model_and_floating_inputs(model, inputs)
    with torch.no_grad():
        pipeline = VgfPipeline[input_t](
            model,
            inputs,
            aten_op=[],
            exir_op=[],
            quantize=False,
            run_on_vulkan_runtime=test_case.run_on_vulkan_runtime,
            tosa_spec="TOSA-1.0+FP+bf16",
        )
        pipeline.run()
