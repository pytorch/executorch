# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# An ExecuTorch friendly implementation of Llava-1.5.abs

import math

import re

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
import torch
import torchvision
from executorch.examples.models.llama2.llama_transformer import ModelArgs, Transformer
from executorch.examples.models.llama2.source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
)
from executorch.examples.models.model_base import EagerModelBase
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)

from llava.conversation import conv_templates

from llava.mm_utils import get_model_name_from_path, tokenizer_image_token

from llava.model.builder import load_pretrained_model

from llava.model.llava_arch import LlavaMetaForCausalLM

from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from PIL import Image

from torch import nn
from torch.export import Dim

from transformers import LlamaForCausalLM


@dataclass
class PreprocessConfig:
    crop_size: dict
    image_mean: list[float]
    image_std: list[float]
    rescale_factor: float


class Llava(torch.nn.Module):
    def __init__(
        self,
        llava_model: LlavaMetaForCausalLM,
        image_processor: CLIPVisionTower,
        config: PreprocessConfig,
    ):
        super().__init__()
        self.config = config
        self.model_ = llava_model
        self.text_model_args = ModelArgs(
            use_kv_cache=True,
            vocab_size=self.model_.config.vocab_size,
            hidden_dim=self.model_.config.intermediate_size,
            max_batch_size=1,  # doesn't work with default batch size 32
            ffn_dim_multiplier=1,  # TODO: a hack to make rotary embedding happy
            enable_dynamic_shape=True,  # allow parallel prefill
            use_sdpa_with_kv_cache_op=True,
            use_hf_rope=True,
        )
        self.embed_tokens = nn.Embedding(
            self.model_.config.vocab_size,
            self.model_.config.hidden_size,
            self.model_.config.pad_token_id,
        )
        self.text_model = Transformer(self.text_model_args)
        # use custom op for SDPA
        self.text_model = replace_sdpa_with_custom_op(self.text_model)
        # load state dict
        self.text_model.load_state_dict(
            state_dict=self._translate_state_dict_for_text_model(),
            strict=False,
            assign=True,
        )
        self.embed_tokens.load_state_dict(
            state_dict=self.get_model().embed_tokens.state_dict(),
            strict=True,
            assign=True,
        )
        self.image_processor = image_processor
        self.vision_tower = self.get_model().vision_tower
        self.mm_projector = self.get_model().mm_projector

    def _translate_state_dict_for_text_model(self) -> Dict[str, Any]:
        state_dict = self.model_.state_dict()
        key_map = {
            # fmt: off
            r"model.layers.([0-9]+).self_attn.q_proj.": r"layers.\1.attention.wq.",
            r"model.layers.([0-9]+).self_attn.k_proj.": r"layers.\1.attention.wk.",
            r"model.layers.([0-9]+).self_attn.v_proj.": r"layers.\1.attention.wv.",
            r"model.layers.([0-9]+).self_attn.o_proj.": r"layers.\1.attention.wo.",
            r"model.layers.([0-9]+).input_layernorm.": r"layers.\1.attention_norm.",
            r"model.layers.([0-9]+).mlp.gate_proj.": r"layers.\1.feed_forward.w1.",
            r"model.layers.([0-9]+).mlp.down_proj.": r"layers.\1.feed_forward.w2.",
            r"model.layers.([0-9]+).mlp.up_proj.": r"layers.\1.feed_forward.w3.",
            r"model.layers.([0-9]+).post_attention_layernorm.": r"layers.\1.ffn_norm.",
            r"model.norm.": r"norm.",
            # r"model.embed_tokens.": r"tok_embeddings.", # load separately
            r"lm_head.": r"output.",
            # fmt: on
        }

        new_state_dict = {}

        def get_new_key(old_key: str) -> str:
            for old_pattern, replacement in key_map.items():
                if (new_key := re.sub(old_pattern, replacement, old_key)) != old_key:
                    return new_key

            return old_key

        # Convert module keys from hf transformer to Llama transformer.
        for old_key in state_dict.keys():
            new_key = get_new_key(old_key)

            new_state_dict[new_key] = state_dict[old_key]

        return new_state_dict

    def get_model(self):
        return self.model_.get_model()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(dtype=self.get_model().dtype)
        image_features = self.vision_tower(images)
        image_features = self.mm_projector(image_features)
        return image_features

    def image_preprocess(self, img: torch.Tensor) -> torch.Tensor:
        w = max(img.shape[1], img.shape[2])
        # pad the image with median rgb value, to make a square
        v_padding = (w - img.shape[1]) / 2
        h_padding = (w - img.shape[2]) / 2
        l_pad = int(math.ceil(h_padding))
        t_pad = int(math.ceil(v_padding))
        r_pad = int(math.floor(h_padding))
        b_pad = int(math.floor(v_padding))
        resized = torchvision.transforms.v2.functional.pad(
            img,
            padding=(l_pad, t_pad, r_pad, b_pad),
            fill=tuple(int(x * 255) for x in self.image_processor.image_mean),
        )
        # TODO: implement _upsample_bicubic_aa.out in portable kernel library.
        # here padded shape should be max(h, w) x max(h, w)
        # skipping resize for now due to missing _upsample_bicubic_aa kernel in portable
        # resized = resize(
        #     padded,
        #     size=[
        #         self.image_processor.crop_size["height"],
        #         self.image_processor.crop_size["width"],
        #     ],
        #     interpolation="bicubic",
        # )
        # torch._check(resized.size(1) == self.config.crop_size["height"])
        # torch._check(resized.size(2) == self.config.crop_size["width"])
        # print(resized.shape)
        # cropped = torchvision.transforms.v2.functional.center_crop(img, output_size=[w, w])
        # print(cropped.shape)
        scaled = resized * self.config.rescale_factor
        # print(scaled)
        normed = torchvision.transforms.v2.functional.normalize(
            scaled, self.config.image_mean, self.config.image_std
        )
        # print(normed)
        return normed.unsqueeze(0)

    def forward(
        self, token: torch.Tensor, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Input is embeddings from prompt and image (after image encoder). Return logits."""
        token_embeds = self.embed_tokens(token).unsqueeze(0)
        return self.text_model.forward(None, input_pos, token_embeds)

    def image_embedding(self, images: torch.Tensor) -> torch.Tensor:
        preprocessed_img = self.image_preprocess(images)
        return self.encode_images(preprocessed_img)

    def prefill_embedding(
        self,
        prompt_before_image: torch.Tensor,
        images: torch.Tensor,
        prompt_after_image: torch.Tensor,
    ) -> torch.Tensor:
        image_embeds = self.image_embedding(images)
        embeds_before_img = self.embed_tokens(prompt_before_image)
        embeds_after_img = self.embed_tokens(prompt_after_image)
        result = torch.cat((embeds_before_img, image_embeds, embeds_after_img), dim=1)
        return result

    def prefill(
        self,
        prompt_before_image: torch.Tensor,
        images: torch.Tensor,
        prompt_after_image: torch.Tensor,
    ) -> torch.Tensor:
        """Avoiding the torch.where() call to find <image> placeholder and insert image embedding. Taking 3 inputs instead."""
        embeds = self.prefill_embedding(prompt_before_image, images, prompt_after_image)
        return self.text_model.forward(None, torch.tensor([0]), embeds)

    def prefill_ref(
        self,
        prompt_before_image: torch.Tensor,
        images: torch.Tensor,
        prompt_after_image: torch.Tensor,
    ) -> torch.Tensor:
        """Avoiding the torch.where() call to find <image> placeholder and insert image embedding. Taking 3 inputs instead."""
        embeds = self.prefill_embedding(prompt_before_image, images, prompt_after_image)
        return LlamaForCausalLM.forward(
            self.model_,
            inputs_embeds=embeds,
            return_dict=False,
            use_cache=False,
            output_hidden_states=False,
        )


def get_prompt(query: str, mm_use_im_start_end: bool, model_name: str) -> str:
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    def get_conv_mode(model_name: str) -> str:
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        return conv_mode

    conv = conv_templates[get_conv_mode(model_name)].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


class LlavaModel(EagerModelBase):
    def __init__(self):
        self.model_path = "liuhaotian/llava-v1.5-7b"
        self.tokenizer, self.model, self.image_processor, context_len = (
            load_pretrained_model(
                model_path=self.model_path,
                model_base=None,
                model_name=get_model_name_from_path(self.model_path),
                device_map="cpu",
                device="cpu",
            )
        )
        self.config = PreprocessConfig(
            self.image_processor.crop_size,
            self.image_processor.image_mean,
            self.image_processor.image_std,
            self.image_processor.rescale_factor,
        )
        self.image = Image.open(
            requests.get(
                "https://llava-vl.github.io/static/images/view.jpg", stream=True
            ).raw
        )
        self.args = type(
            "Args",
            (),
            {
                "model_path": self.model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(self.model_path),
                "query": "What are the things I should be cautious about when I visit here?",
                "conv_mode": None,
                "sep": ",",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512,
            },
        )()

    def get_eager_model(self):
        model = Llava(self.model, self.image_processor, self.config)
        return model

    def get_example_inputs(self):

        model_name = get_model_name_from_path(self.model_path)
        prompt = get_prompt(self.args.query, False, model_name)
        self.input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cpu()
        )
        index = torch.where(self.input_ids == IMAGE_TOKEN_INDEX)[1]
        prompt_before_image = self.input_ids[:, :index]
        # print(prompt_before_image.shape)
        prompt_after_image = self.input_ids[:, index + 1 :]
        # print(prompt_after_image.shape)
        imagr = torchvision.transforms.functional.pil_to_tensor(self.image)
        ratio = (
            max(imagr.shape[1], imagr.shape[2])
            / self.image_processor.crop_size["height"]
        )
        output_size = (int(imagr.shape[1] / ratio), int(imagr.shape[2] / ratio))
        resized = torchvision.transforms.Resize(size=output_size)(imagr)
        self.inputs = (prompt_before_image, resized, prompt_after_image)
        return self.inputs

    def get_dynamic_shapes(self):
        height = Dim("height", min=8, max=336)
        # height = Dim("height", min=8, max=4091)
        token_dim_1 = Dim("token_dim_1", min=2, max=3518)
        token_dim_2 = Dim("token_dim_2", min=2, max=3518)
        width = Dim("width", min=28, max=336)
        # width = Dim("width", min=9, max=4092)
        dynamic_shapes = [{1: token_dim_1}, {1: height, 2: width}, {1: token_dim_2}]
        return dynamic_shapes
