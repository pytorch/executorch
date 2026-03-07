# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# An ExecuTorch-friendly implementation of LFM2.5-VL-1.6B.
# Mirrors examples/models/llava/model.py in structure.

import json
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
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
IMAGE_SIZE = 512  # 512x512 -> 32x32 patches -> 256 tokens after projector
PATCH_SIZE = 16
FIXED_H, FIXED_W = 32, 32

# Path to the bundled config for LFM2.5-VL-1.6B, used when no --params is given.
_DEFAULT_PARAMS = os.path.join(
    os.path.dirname(__file__), "config", "lfm2_5_vl_1_6b_config.json"
)


class Lfm2p5Vl(torch.nn.Module):
    def __init__(
        self,
        hf_model: AutoModelForImageTextToText,
        params: ModelArgs,
    ):
        super().__init__()
        self.model_ = hf_model
        self.text_model_args = params
        self.text_model = construct_transformer(self.text_model_args)

        # Source transforms must happen before load_state_dict so buffers exist
        if params.use_sdpa_with_kv_cache_op:
            self.text_model = replace_kv_cache_with_custom_kv_cache(self.text_model)
            self.text_model = replace_sdpa_with_custom_op(self.text_model)

        # Load translated weights from HF model (strict=False: KV/conv buffers
        # are runtime state initialised as zeros by construct_transformer)
        self.text_model.load_state_dict(
            state_dict=self._translate_weights(),
            strict=False,
            assign=True,
        )

        # Pre-compute bilinear PE for 32x32 grid and patch the resize method.
        # This avoids dynamic F.interpolate calls that are not exportable.
        self._patch_positional_embeddings()

    def _patch_positional_embeddings(self) -> None:
        orig = self.model_.model.vision_tower.vision_model.embeddings.position_embedding.weight.data
        num_positions, dim = orig.shape
        sqrt_num = int(math.sqrt(num_positions))
        grid = orig.reshape(sqrt_num, sqrt_num, dim)
        resized = F.interpolate(
            grid.permute(2, 0, 1).unsqueeze(0),
            size=(FIXED_H, FIXED_W),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        # .contiguous() prevents dim_order mismatches in portable aten::add.out kernel
        precomputed_pe = (
            resized.squeeze(0).permute(1, 2, 0).reshape(FIXED_H * FIXED_W, dim).contiguous()
        )

        def _patched_resize(positional_embeddings, height=None, width=None, max_length=None):
            return precomputed_pe

        self.model_.model.vision_tower.vision_model.embeddings.resize_positional_embeddings = (
            _patched_resize
        )

    def _translate_weights(self):
        """Translate HF LFM2-VL state dict keys to ET construct_transformer format.

        Handles all layer types (conv + full_attention) and the critical
        in_proj [3*dim, dim] -> three [dim, dim] split for conv layers.
        """
        from executorch.examples.models.lfm2_5_vl.convert_weights import (
            lfm2_5_vl_to_meta,
        )

        # Build a flat state dict with the "model.language_model." prefix that
        # convert_weights expects (same layout as the HF safetensors file).
        raw = {}
        for k, v in self.model_.model.language_model.state_dict().items():
            raw[f"model.language_model.{k}"] = v
        # lm_head is a separate top-level module in the VL wrapper
        for k, v in self.model_.lm_head.state_dict().items():
            raw[f"model.language_model.lm_head.{k}"] = v

        return lfm2_5_vl_to_meta(raw)

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model_.model.language_model.get_input_embeddings()(tokens)

    def encode_images(self, nchw_pixels: torch.Tensor) -> torch.Tensor:
        """Accept [1, 3, 512, 512] NCHW float32 pixels in [0, 255].

        Bakes in normalize + patch extraction so the C++ runner only needs
        to resize to 512x512 and pass raw pixel values -- same as LLaVA runner.
        """
        # Normalize: (x/255 - 0.5) / 0.5
        x = nchw_pixels / 255.0
        x = (x - 0.5) / 0.5

        # Extract 16x16 patches in HW-major order -> [1, 1024, PATCH_SIZE*PATCH_SIZE*3]
        x = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        x = x.permute(0, 2, 3, 4, 5, 1).reshape(
            1, FIXED_H * FIXED_W, PATCH_SIZE * PATCH_SIZE * 3
        )

        FULL_MASK = torch.ones(1, FIXED_H * FIXED_W, dtype=torch.int32)
        out = self.model_.model.vision_tower(
            pixel_values=x,
            pixel_attention_mask=FULL_MASK,
            spatial_shapes=torch.tensor([[FIXED_H, FIXED_W]], dtype=torch.int64),
            return_dict=True,
        )
        feats = out.last_hidden_state  # [1, 1024, vision_hidden_size]
        feats = feats.reshape(feats.shape[0], FIXED_H, FIXED_W, -1)
        projected = self.model_.model.multi_modal_projector(feats)  # [1, 16, 16, 2048]
        return projected.reshape(1, -1, projected.shape[-1])  # [1, 256, 2048]

    def image_embedding(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_images(images)

    def prefill_embedding(
        self,
        prompt_before_image: torch.Tensor,
        images: torch.Tensor,
        prompt_after_image: torch.Tensor,
    ) -> torch.Tensor:
        image_embeds = self.image_embedding(images)
        embeds_before = self.embed_tokens(prompt_before_image)
        embeds_after = self.embed_tokens(prompt_after_image)
        return torch.cat((embeds_before, image_embeds, embeds_after), dim=1)

    def step(
        self, token: torch.Tensor, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Single decode step. Input: one token. Output: logits."""
        token_embeds = self.embed_tokens(token).unsqueeze(0)
        return self.text_model.forward(None, {"input_pos": input_pos}, token_embeds)

    def prefill(
        self,
        prompt_before_image: torch.Tensor,
        images: torch.Tensor,
        prompt_after_image: torch.Tensor,
    ) -> Tuple[int, torch.Tensor]:
        embeds = self.prefill_embedding(prompt_before_image, images, prompt_after_image)
        return embeds.shape[1], self.text_model.forward(
            None, {"input_pos": torch.tensor([0])}, embeds
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_embedding(images)


class Lfm2p5VlModel(EagerModelBase):
    def __init__(
        self,
        use_sdpa_with_kv_cache_op: bool = True,
        max_seq_len: int = MAX_SEQ_LEN,
        max_context_len: int = MAX_SEQ_LEN,
        # HF auto-load path (model ID or local dir containing the full VL checkpoint)
        model_dir: str = "LiquidAI/LFM2-VL-1.6B",
        # Path to params JSON (architecture config). Defaults to bundled
        # config/lfm2_5_vl_1_6b_config.json if not provided.
        params_path: Optional[str] = None,
    ):
        self.use_sdpa_with_kv_cache_op = use_sdpa_with_kv_cache_op
        self.max_context_len = max_context_len
        self.max_seq_len = max_seq_len
        self.model_dir = model_dir

        # Load architecture config from JSON (mirrors LLaMA model.py pattern)
        resolved_params = params_path or _DEFAULT_PARAMS
        with open(resolved_params, "r") as f:
            params = json.loads(f.read())

        self.text_model_args = ModelArgs(
            max_batch_size=1,
            max_seq_len=max_seq_len,
            max_context_len=max_context_len,
            use_kv_cache=True,
            use_sdpa_with_kv_cache_op=use_sdpa_with_kv_cache_op,
            # CRITICAL: False avoids .item() in rope.get_freqs which crashes
            # FakeTensor export via to_edge_transform_and_lower
            enable_dynamic_shape=False,
            **params,
        )

        self.hf_model = AutoModelForImageTextToText.from_pretrained(
            model_dir, device_map="cpu", torch_dtype=torch.float32
        )
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.tokenizer = self.processor.tokenizer

        self.conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        # Lazy-initialized inputs
        self.input = None
        self.example_image = None

    def get_eager_model(self) -> torch.nn.Module:
        model = Lfm2p5Vl(self.hf_model, self.text_model_args)
        model.to(dtype=torch.float32)
        return model

    def get_example_inputs(self):
        """Returns a [1, 3, 512, 512] pixel tensor for the vision encoder."""
        if self.example_image is not None:
            return self.example_image
        self.example_image = (
            torch.randint(0, 256, (1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32),
        )
        return self.example_image

    def get_inputs_for_prefill(self):
        """Returns (prompt_before_image, pixel_values, prompt_after_image)."""
        if self.input is not None:
            return self.input

        text = self.processor.apply_chat_template(
            self.conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)],
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        index = torch.where(input_ids == self.hf_model.config.image_token_index)[1]
        prompt_before = input_ids[:, : index[0]]
        prompt_after = input_ids[:, index[-1] + 1 :]
        self.input = (prompt_before, *self.get_example_inputs(), prompt_after)
        return self.input

    def get_dynamic_shapes(self):
        return self._get_image_dynamic_shapes()

    def _get_image_dynamic_shapes(self):
        # Vision encoder has fixed 512x512 input -- no dynamic shapes needed
        return None

    def _get_prompt_dynamic_shapes(self):
        dim = Dim("token_dim", min=1, max=self.max_seq_len)
        return ({1: dim}, {1: dim})
