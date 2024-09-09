# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# An ExecuTorch friendly implementation of Llava-1.5.

import re

from typing import Any, Dict, Optional, Tuple

import requests
import torch
from executorch.examples.models.llama2.llama_transformer import ModelArgs, Transformer

from executorch.examples.models.llama2.source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
)
from executorch.examples.models.llava.image_util import prepare_image
from executorch.examples.models.model_base import EagerModelBase
from PIL import Image

from torch.export import Dim
from torchvision.transforms.v2 import functional as F

from transformers import (
    AutoProcessor,
    CLIPImageProcessor,
    LlamaForCausalLM,
    LlavaForConditionalGeneration,
)


class Llava(torch.nn.Module):
    def __init__(
        self,
        llava_model: LlavaForConditionalGeneration,
        image_processor: CLIPImageProcessor,
        use_sdpa_with_kv_cache_op: bool = True,
        max_seq_len: int = 768,
    ):
        super().__init__()
        self.use_sdpa_with_kv_cache_op = use_sdpa_with_kv_cache_op
        self.model_ = llava_model
        self.image_processor = image_processor
        self.vision_feature_layer = self.model_.config.vision_feature_layer
        self.vision_feature_select_strategy = (
            self.model_.config.vision_feature_select_strategy
        )
        self.text_model_args = ModelArgs(
            use_kv_cache=True,
            vocab_size=self.model_.config.text_config.vocab_size,
            hidden_dim=self.model_.config.text_config.intermediate_size,
            max_batch_size=1,  # doesn't work with default batch size 32
            ffn_dim_multiplier=1,  # TODO: a hack to make rotary embedding happy
            enable_dynamic_shape=True,  # allow parallel prefill
            use_sdpa_with_kv_cache_op=use_sdpa_with_kv_cache_op,  # use sdpa_with_kv_cache op
            use_hf_rope=True,
            max_seq_len=max_seq_len,
        )
        self.text_model = Transformer(self.text_model_args)
        # use custom op for SDPA.
        if use_sdpa_with_kv_cache_op:
            self.text_model = replace_sdpa_with_custom_op(self.text_model)
        # load state dict
        self.text_model.load_state_dict(
            state_dict=self._translate_state_dict_for_text_model(),
            strict=False,
            assign=True,
        )

    def _translate_state_dict_for_text_model(self) -> Dict[str, Any]:
        state_dict = self.model_.language_model.state_dict()
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

    def _feature_select(self, image_outputs):
        selected_image_feature = image_outputs.hidden_states[self.vision_feature_layer]

        if self.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                f"Unexpected select feature: {self.vision_feature_select_strategy}"
            )
        return selected_image_feature

    def get_model(self):
        return self.model_.get_model()

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model_.language_model.model.embed_tokens(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(dtype=self.model_.dtype)
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.model_.vision_tower(
                    image.to(
                        device=self.model_.device, dtype=self.model_.dtype
                    ).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self._feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.model_.vision_tower(
                images.to(device=self.model_.device, dtype=self.model_.dtype),
                output_hidden_states=True,
            )
            image_features = self._feature_select(image_forward_outs).to(images.dtype)
        image_features = self.model_.multi_modal_projector(image_features)
        return image_features

    def image_preprocess(self, img: torch.Tensor) -> torch.Tensor:
        target_h = self.image_processor.crop_size["height"]
        target_w = self.image_processor.crop_size["width"]
        # pad the image with median rgb value, to make a square
        l_pad = (target_w - img.shape[2]) // 2
        t_pad = (target_h - img.shape[1]) // 2
        # ceil division
        r_pad = -((target_w - img.shape[2]) // -2)
        b_pad = -((target_h - img.shape[1]) // -2)

        torch._check(l_pad >= 0)
        torch._check(t_pad >= 0)
        torch._check(r_pad >= 0)
        torch._check(b_pad >= 0)

        # This is different from the original implementation, due to export limitations.
        resized = torch.nn.functional.pad(
            img,
            (l_pad, r_pad, t_pad, b_pad),
        )
        # originally:
        # resized = F.pad(
        #     img,
        #     padding=(l_pad, t_pad, r_pad, b_pad),
        #     fill=tuple(int(x * 255) for x in self.image_mean),
        # )

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
        # cropped = F.center_crop(img, output_size=[w, w])
        # print(cropped.shape)
        scaled = resized * self.image_processor.rescale_factor
        # print(scaled)
        normed = F.normalize(
            scaled, self.image_processor.image_mean, self.image_processor.image_std
        )
        # print(normed)
        return normed.unsqueeze(0)

    def step(
        self, token: torch.Tensor, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Input is one token. Return logits for next token."""
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

    # prefill using the in house text_model of llama transformer
    def prefill(
        self,
        prompt_before_image: torch.Tensor,
        images: torch.Tensor,
        prompt_after_image: torch.Tensor,
    ) -> Tuple[int, torch.Tensor]:
        """Avoiding the torch.where() call to find <image> placeholder and insert image embedding. Taking 3 inputs instead."""
        embeds = self.prefill_embedding(prompt_before_image, images, prompt_after_image)
        # returns the prefilled token length too, because the text model generates one logits in each forward call.
        return embeds.shape[1], self.text_model.forward(None, torch.tensor([0]), embeds)

    # reference prefill using the text model in HF
    def prefill_ref(
        self,
        prompt_before_image: torch.Tensor,
        images: torch.Tensor,
        prompt_after_image: torch.Tensor,
    ) -> torch.Tensor:
        """Avoiding the torch.where() call to find <image> placeholder and insert image embedding. Taking 3 inputs instead."""
        embeds = self.prefill_embedding(prompt_before_image, images, prompt_after_image)
        return LlamaForCausalLM.forward(
            self.model_.language_model,
            inputs_embeds=embeds,
            return_dict=False,
            use_cache=False,
            output_hidden_states=False,
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return self.image_embedding(images)


class LlavaModel(EagerModelBase):
    def __init__(self, use_sdpa_with_kv_cache_op=True, max_seq_len=768):
        self.use_sdpa_with_kv_cache_op = use_sdpa_with_kv_cache_op
        self.max_seq_len = max_seq_len
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        self.model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            device_map="cpu",
        )
        self.image = Image.open(
            requests.get(
                "https://llava-vl.github.io/static/images/view.jpg", stream=True
            ).raw
        )
        self.prompt = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>
What are the things I should be cautious about when I visit here? ASSISTANT:"""
        self.model_name = "llava-1.5-7b-hf"
        # set input to None and initialize them lazily
        self.input = None
        self.resized_image = None

    def get_eager_model(self):
        model = Llava(
            self.model,
            self.image_processor,
            self.use_sdpa_with_kv_cache_op,
            self.max_seq_len,
        )
        model.to(dtype=torch.float32)
        return model

    def get_example_inputs(self):
        """Returns a resized image as input to model.forward()."""
        if self.resized_image:
            return self.resized_image
        resized = prepare_image(
            self.image,
            self.image_processor.crop_size["height"],
            self.image_processor.crop_size["width"],
        )
        self.resized_image = (resized,)
        return self.resized_image

    def get_inputs_for_prefill(self):
        """Returns prompts as well as image."""
        if self.input:
            return self.input
        self.input_ids = self.tokenizer.encode(self.prompt, return_tensors="pt").cpu()
        index = torch.where(self.input_ids == self.model.config.image_token_index)[1]
        self.prompt_before_image = self.input_ids[:, :index]
        # print(prompt_before_image.shape)
        self.prompt_after_image = self.input_ids[:, index + 1 :]
        # print(prompt_after_image.shape)
        self.input = (
            self.prompt_before_image,
            *self.get_example_inputs(),
            self.prompt_after_image,
        )
        return self.input

    def get_dynamic_shapes(self):
        return self._get_image_dynamic_shapes()

    def _get_image_dynamic_shapes(self):
        # only support even number of height and width for now
        _height = Dim(
            "_height", min=1, max=self.image_processor.crop_size["height"] // 2
        )
        _width = Dim("_width", min=1, max=self.image_processor.crop_size["width"] // 2)
        height = 2 * _height
        width = 2 * _width
        dynamic_shapes = [{1: height, 2: width}]
        return dynamic_shapes

    def _get_prompt_dynamic_shapes(self):
        dim = torch.export.Dim("token_dim", min=2, max=self.max_seq_len)
        text_model_dynamic_shapes = ({0: 1}, {1: dim})
        return text_model_dynamic_shapes
