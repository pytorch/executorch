# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ExecuTorch-friendly LFM2.5-VL model. Mirrors examples/models/llava/model.py."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from executorch.examples.models.lfm2_5_vl.convert_weights import lfm2_5_vl_to_meta
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    replace_kv_cache_with_custom_kv_cache,
)
from executorch.examples.models.llama.source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
)
from executorch.examples.models.model_base import EagerModelBase
from torch.export import Dim
from transformers import AutoModelForImageTextToText, AutoProcessor

MAX_SEQ_LEN = 2048
IMAGE_SIZE = 512
PATCH_SIZE = 16
FIXED_H, FIXED_W = 32, 32

_DEFAULT_PARAMS = Path(__file__).parent / "config" / "lfm2_5_vl_1_6b_config.json"


class Lfm2p5Vl(torch.nn.Module):
    def __init__(self, hf_model: AutoModelForImageTextToText, params: ModelArgs) -> None:
        super().__init__()
        self.model_ = hf_model
        self.text_model_args = params
        self.text_model = construct_transformer(params)

        if params.use_sdpa_with_kv_cache_op:
            self.text_model = replace_kv_cache_with_custom_kv_cache(self.text_model)
            self.text_model = replace_sdpa_with_custom_op(self.text_model)

        self.text_model.load_state_dict(
            state_dict=self._translate_weights(), strict=False, assign=True
        )
        self._patch_positional_embeddings()

    def _patch_positional_embeddings(self) -> None:
        embeddings = self.model_.model.vision_tower.vision_model.embeddings
        orig = embeddings.position_embedding.weight.data
        sqrt_n = int(math.sqrt(orig.shape[0]))

        grid = orig.reshape(sqrt_n, sqrt_n, -1).permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(
            grid, size=(FIXED_H, FIXED_W), mode="bilinear", align_corners=False, antialias=True
        )
        pe = resized.squeeze(0).permute(1, 2, 0).reshape(FIXED_H * FIXED_W, -1).contiguous()
        embeddings.register_buffer("_precomputed_pe", pe, persistent=False)
        embeddings.resize_positional_embeddings = lambda *_args, **_kw: embeddings._precomputed_pe

    def _translate_weights(self) -> dict[str, torch.Tensor]:
        raw: dict[str, torch.Tensor] = {}
        for k, v in self.model_.model.language_model.state_dict().items():
            raw[f"model.language_model.{k}"] = v
        for k, v in self.model_.lm_head.state_dict().items():
            raw[f"model.language_model.lm_head.{k}"] = v
        return lfm2_5_vl_to_meta(raw)

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model_.model.language_model.get_input_embeddings()(tokens)

    def image_embedding(self, nchw_pixels: torch.Tensor) -> torch.Tensor:
        """[B, 3, 512, 512] float32 pixels in [0, 255] -> [B, 256, D]."""
        x = (nchw_pixels / 255.0 - 0.5) / 0.5

        x = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        x = x.permute(0, 2, 3, 4, 5, 1).reshape(1, FIXED_H * FIXED_W, PATCH_SIZE * PATCH_SIZE * 3)

        out = self.model_.model.vision_tower(
            pixel_values=x,
            pixel_attention_mask=None,
            spatial_shapes=torch.tensor([[FIXED_H, FIXED_W]], dtype=torch.int64, device=x.device),
            return_dict=True,
        )
        feats = out.last_hidden_state.reshape(-1, FIXED_H, FIXED_W, out.last_hidden_state.shape[-1])
        projected = self.model_.model.multi_modal_projector(feats)
        return projected.reshape(1, -1, projected.shape[-1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_embedding(images)


class Lfm2p5VlModel(EagerModelBase):
    def __init__(
        self,
        *,
        use_sdpa_with_kv_cache_op: bool = True,
        use_kv_cache: bool = True,
        max_seq_len: int = MAX_SEQ_LEN,
        max_context_len: int = MAX_SEQ_LEN,
        model_dir: str = "LiquidAI/LFM2.5-VL-1.6B",
        params_path: str | None = None,
    ) -> None:
        self.use_sdpa_with_kv_cache_op = use_sdpa_with_kv_cache_op
        self.max_context_len = max_context_len
        self.max_seq_len = max_seq_len
        self.model_dir = model_dir

        resolved = Path(params_path) if params_path else _DEFAULT_PARAMS
        params = json.loads(resolved.read_text())

        self.text_model_args = ModelArgs(
            max_batch_size=1,
            max_seq_len=max_seq_len,
            max_context_len=max_context_len,
            use_kv_cache=use_kv_cache,
            use_sdpa_with_kv_cache_op=use_sdpa_with_kv_cache_op,
            enable_dynamic_shape=False,
            **params,
        )

        self.hf_model = AutoModelForImageTextToText.from_pretrained(
            model_dir, device_map="cpu", torch_dtype=torch.float32
        )
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.tokenizer = self.processor.tokenizer

    def get_eager_model(self) -> torch.nn.Module:
        return Lfm2p5Vl(self.hf_model, self.text_model_args).to(dtype=torch.float32)

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.randint(0, 256, (1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32),)

    def get_dynamic_shapes(self) -> None:
        return None
