# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# An ExecuTorch friendly implementation of Llava-1.5.

import math

import re
from dataclasses import dataclass, field

from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import torch
import torchvision
from executorch.examples.models.llama2.llama_transformer import ModelArgs, Transformer

from executorch.examples.models.llama2.source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
)
from executorch.examples.models.model_base import EagerModelBase
from PIL import Image

from torch import nn
from torch.export import Dim

from torchtune.models.clip import clip_vision_encoder
from torchvision.transforms.v2 import functional as F

from transformers import (
    AutoProcessor,
    CLIPImageProcessor,
    LlamaForCausalLM,
    LlavaForConditionalGeneration,
)


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input):
        return input * torch.sigmoid(1.702 * input)


@dataclass
class VisionArgs:
    tile_size: int = 336
    patch_size: int = 14
    embed_dim: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    out_indices: List[int] = field(default_factory=list)
    output_cls_projection: bool = False
    max_num_tiles: int = 1
    in_channels: int = 3
    intermediate_act: nn.Module = QuickGELUActivation()

    def __post_init__(self):
        if not self.out_indices:
            self.out_indices = [self.num_layers - 1]


@dataclass
class ProjectorArgs:
    in_channels: int = 1024
    out_channels: int = 4096
    activation: nn.Module = nn.GELU()


@dataclass
class LlavaArgs:
    vision_args: VisionArgs = VisionArgs()
    text_args: ModelArgs = ModelArgs()
    projector_args: ProjectorArgs = ProjectorArgs()
    vision_feature_select_strategy: str = "default"
    pad_token_id: int = 32001
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32


@dataclass
class PreprocessorArgs:
    image_mean: List[float] = field(
        default_factory=lambda: [0.48145466, 0.4578275, 0.40821073]
    )
    image_std: List[float] = field(
        default_factory=lambda: [0.26862954, 0.26130258, 0.27577711]
    )
    rescale_factor: float = 0.00392156862745098


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, args: ProjectorArgs):
        super().__init__()

        self.linear_1 = nn.Linear(args.in_channels, args.out_channels, bias=True)
        self.act = args.activation
        self.linear_2 = nn.Linear(args.out_channels, args.out_channels, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Llava(torch.nn.Module):
    def __init__(
        self,
        llava_model: LlavaForConditionalGeneration,
        preprocessor_args: PreprocessorArgs,
        llava_args: LlavaArgs,
    ):
        super().__init__()
        self.model_ = llava_model
        self.preprocessor_args = preprocessor_args
        self.llava_args = llava_args

        self.use_sdpa_with_kv_cache_op = (
            self.llava_args.text_args.use_sdpa_with_kv_cache_op
        )

        self.vision_feature_select_strategy = (
            self.llava_args.vision_feature_select_strategy
        )
        self.vision_tower = clip_vision_encoder(**self.llava_args.vision_args.__dict__)
        self.mm_projector = LlavaMultiModalProjector(llava_args.projector_args)

        self.embed_tokens = nn.Embedding(
            self.llava_args.text_args.vocab_size,
            self.llava_args.text_args.dim,  # this may not right
            self.llava_args.pad_token_id,
        )

        self.text_model = Transformer(self.llava_args.text_args)
        # use custom op for SDPA.
        if self.use_sdpa_with_kv_cache_op:
            self.text_model = replace_sdpa_with_custom_op(self.text_model)
        # load state dict
        self.text_model.load_state_dict(
            state_dict=self._translate_state_dict_for_text_model(),
            strict=False,
            assign=True,
        )
        self.embed_tokens.load_state_dict(
            state_dict=self.model_.language_model.model.embed_tokens.state_dict(),
            strict=True,
            assign=True,
        )
        self.vision_tower.load_state_dict(
            state_dict=self._translate_state_dict_for_vision_model(),
            strict=True,
            assign=True,
        )

        self.mm_projector.load_state_dict(
            state_dict=self.model_.multi_modal_projector.state_dict(),
            strict=True,
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

    def _translate_state_dict_for_vision_model(self) -> Dict[str, Any]:
        state_dict = self.model_.vision_tower.state_dict()

        model2_state_dict = {}

        # Define the mapping from old names to new names
        model1_prefix = "vision_model."
        name_mapping = {
            f"{model1_prefix}embeddings.class_embedding": "cls_token_embedding.cls_embedding",
            f"{model1_prefix}embeddings.position_embedding.weight": "token_pos_embedding.positional_embedding",
            f"{model1_prefix}embeddings.patch_embedding.weight": "conv.weight",
            f"{model1_prefix}pre_layrnorm.weight": "ln_pre.weight",
            f"{model1_prefix}pre_layrnorm.bias": "ln_pre.bias",
            f"{model1_prefix}post_layernorm.weight": "ln_post.weight",
            f"{model1_prefix}post_layernorm.bias": "ln_post.bias",
        }

        # Use regular expressions to define the mapping for each layer
        patterns = [
            (
                rf"{model1_prefix}encoder\.layers\.([0-9]+)\.self_attn\.(k|q|v)_proj\.(weight|bias)",
                lambda match: f"transformer_layers.{match.group(1)}.self_attn.in_proj_{match.group(3)}",
            ),
            (
                rf"{model1_prefix}encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.(weight|bias)",
                lambda match: f"transformer_layers.{match.group(1)}.self_attn.out_proj.{match.group(2)}",
            ),
            (
                rf"{model1_prefix}encoder\.layers\.([0-9]+)\.mlp\.fc(1|2)\.(weight|bias)",
                lambda match: f"transformer_layers.{match.group(1)}.linear{match.group(2)}.{match.group(3)}",
            ),
            (
                rf"{model1_prefix}encoder\.layers\.([0-9]+)\.layer_norm(1|2)\.(weight|bias)",
                lambda match: f"transformer_layers.{match.group(1)}.norm{match.group(2)}.{match.group(3)}",
            ),
        ]

        # Apply the patterns to update the name mapping
        for pattern, replacement in patterns:
            for key in list(state_dict.keys()):
                if re.match(pattern, key):
                    new_key = re.sub(pattern, replacement, key)
                    name_mapping[key] = new_key

        # Process the combined self-attention weights and biases
        temp_state_dict = {}
        for k, v in state_dict.items():
            if k in name_mapping:
                new_k = name_mapping[k]
                if "in_proj_weight" in new_k or "in_proj_bias" in new_k:
                    if new_k not in temp_state_dict:
                        temp_state_dict[new_k] = {"q": None, "k": None, "v": None}
                    if "q_proj" in k:
                        temp_state_dict[new_k]["q"] = v
                    elif "k_proj" in k:
                        temp_state_dict[new_k]["k"] = v
                    elif "v_proj" in k:
                        temp_state_dict[new_k]["v"] = v
                else:
                    temp_state_dict[new_k] = v

        # Final processing of the combined self-attention weights and biases
        for k, v in temp_state_dict.items():
            if isinstance(v, dict):
                model2_state_dict[k] = torch.cat([v["q"], v["k"], v["v"]], dim=0)
            else:
                model2_state_dict[k] = v

        return model2_state_dict

    def _feature_select(self, image_outputs):
        selected_image_feature = image_outputs[1][0].view(
            *image_outputs[1][0].shape[2:]
        )

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

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(dtype=self.model_.dtype)
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(
                        device=self.model_.device, dtype=self.model_.dtype
                    ).unsqueeze(0),
                )
                image_feature = self._feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.model_.device, dtype=self.model_.dtype),
            )
            image_features = self._feature_select(image_forward_outs).to(images.dtype)
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
        resized = F.pad(
            img,
            padding=(l_pad, t_pad, r_pad, b_pad),
            fill=tuple(int(x * 255) for x in self.preprocessor_args.image_mean),
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
        # cropped = F.center_crop(img, output_size=[w, w])
        # print(cropped.shape)
        scaled = resized * self.preprocessor_args.rescale_factor
        # print(scaled)
        normed = F.normalize(
            scaled, self.preprocessor_args.image_mean, self.preprocessor_args.image_std
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
        preprocessed_img = preprocessed_img.view(1, 1, *preprocessed_img.shape)
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
    ) -> (int, torch.Tensor):
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
    def __init__(self, use_sdpa_with_kv_cache_op=True):
        self.use_sdpa_with_kv_cache_op = use_sdpa_with_kv_cache_op
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

    def get_preprocessor_args(self):
        return PreprocessorArgs(
            image_mean=self.image_processor.image_mean,
            image_std=self.image_processor.image_std,
            rescale_factor=self.image_processor.rescale_factor,
        )

    def get_llava_args(self):
        return LlavaArgs(
            text_args=ModelArgs(
                use_kv_cache=True,
                vocab_size=self.model.config.text_config.vocab_size,
                hidden_dim=self.model.config.text_config.intermediate_size,
                max_batch_size=1,  # doesn't work with default batch size 32
                ffn_dim_multiplier=1,  # TODO: a hack to make rotary embedding happy
                enable_dynamic_shape=True,  # allow parallel prefill
                use_sdpa_with_kv_cache_op=self.use_sdpa_with_kv_cache_op,  # use sdpa_with_kv_cache op
                use_hf_rope=True,
            ),
            vision_feature_select_strategy="default",
            pad_token_id=self.model.config.pad_token_id,
            device=self.model.device,
            dtype=self.model.dtype,
        )

    def get_eager_model(self):
        model = Llava(
            self.model,
            self.get_preprocessor_args(),
            self.get_llava_args(),
        )
        model.to(dtype=torch.float32)
        return model

    def get_example_inputs(self):
        """Returns a resized image as input to model.forward()."""
        if self.resized_image:
            return self.resized_image
        imagr = torchvision.transforms.functional.pil_to_tensor(self.image)
        ratio = (
            max(imagr.shape[1], imagr.shape[2])
            / self.image_processor.crop_size["height"]
        )
        output_size = (int(imagr.shape[1] / ratio), int(imagr.shape[2] / ratio))
        self.resized_image = (torchvision.transforms.Resize(size=output_size)(imagr),)
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
        height = Dim("height", min=8, max=336)
        width = Dim("width", min=28, max=336)
        dynamic_shapes = [{1: height, 2: width}]
        return dynamic_shapes

    def _get_prompt_dynamic_shapes(self):
        dim = torch.export.Dim("token_dim", min=2, max=2048)
        text_model_dynamic_shapes = ({0: 1}, {1: dim})
        return text_model_dynamic_shapes
