# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from examples.models.model_base import EagerModelBase
from llava.eval.run_llava import load_images, process_images
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token

from llava.model.builder import load_pretrained_model
from torch import nn

import torchvision
from torchvision.transforms import v2

from dataclasses import dataclass
from torch.export import Dim
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

from executorch.examples.models.llama2.llama_transformer import TransformerBlock, ModelArgs

@dataclass
class PreprocessConfig:
    crop_size: dict
    image_mean: list[float]
    image_std: list[float]
    rescale_factor: float


class TextModelTransformer(nn.Module):
    """Mostly copied from examples/models/llama2/llama_transformer but taking a embedding"""
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.use_kv_cache = params.use_kv_cache

        freqs_cos, freqs_sin = precompute_freqs_cis(
            params.dim // params.n_heads,
            (
                params.max_seq_len  # Normal llama2.
                if params.ffn_dim_multiplier is None
                else params.max_seq_len * 2  # Sharded checkpoint.
            ),
            params.rope_freq_base,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        h: torch.Tensor,
        input_pos: Optional[
            torch.Tensor
        ] = None,  # Scalar tensor indicating size of window of the caches
    ) -> torch.Tensor:
        _bsz = h.shape[0]
        seqlen = h.shape[1]
        # h = self.tok_embeddings(tokens)

        if self.use_kv_cache:
            assert (
                input_pos is not None
            ), "input_pos must be provided when use_kv_cache is True"

            # when KV cache is used, seqlen is most likely 1. We want to slice from the start_pos.
            freqs_cos = self.freqs_cos[input_pos]
            freqs_sin = self.freqs_sin[input_pos]
        else:
            assert input_pos is None, "input_pos is unused when use_kv_cache is False"
            freqs_cos = self.freqs_cos[:seqlen]
            freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(
                h,
                freqs_cos,
                freqs_sin,
                input_pos,
            )

        h = self.norm(h)

        logits = self.output(h)
        return logits
    
class Llava(torch.nn.Module):
    def __init__(self, llava_model: LlavaMetaForCausalLM, config: PreprocessConfig):
        super().__init__()
        self.config = config
        self.model_ = llava_model
    
    def get_model(self):
        return self.model_.get_model()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(dtype=torch.float16)
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
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
        padded = torchvision.transforms.v2.functional.pad(img, padding=(l_pad, t_pad, r_pad, b_pad), fill=tuple(int(x*255) for x in image_processor.image_mean))
        # here padded shape should be max(h, w) x max(h, w)
        resized = resize(padded, size=[image_processor.crop_size['height'], image_processor.crop_size['width']], interpolation="bicubic")
        torch._check(resized.size(1) == self.config.crop_size['height'])
        torch._check(resized.size(2) == self.config.crop_size['width'])
        # print(resized.shape)
        # cropped = torchvision.transforms.v2.functional.center_crop(img, output_size=[w, w])
        # print(cropped.shape)
        scaled = resized * self.config.rescale_factor
        # print(scaled)
        normed = torchvision.transforms.v2.functional.normalize(scaled, self.config.image_mean, self.config.image_std)
        # print(normed)
        return normed.unsqueeze(0)
    
    def prepare_inputs_labels_for_multimodal_one_image(self, prompt_before_image: torch.Tensor, images: torch.Tensor, prompt_after_image: torch.Tensor) -> torch.Tensor:
        assert isinstance(input_ids, torch.Tensor), f"Expecting input_ids to be a tensor, got {input_ids}"
        assert input_ids.shape[0] == 1, f"Expecting input_ids to be of shape [1, num_tokens], got {input_ids.shape}"
        prompt_before_image = prompt_before_image.squeeze(0)
        prompt_after_image = prompt_after_image.squeeze(0)

        # preprocessed_img = self.image_preprocess(imagt)
        # preprocessed_img = torch.unsqueeze(preprocessed_img, dim=0)

        embeds_before_img = self.get_model().embed_tokens(prompt_before_image).unsqueeze(0)
        embeds_after_img = self.get_model().embed_tokens(prompt_after_image).unsqueeze(0)

        image_embeds = self.encode_images(images)
        # new_input_embeds = torch.cat(input_embeds, image_embeds)
        result = torch.cat((embeds_before_img, image_embeds, embeds_after_img), dim=1)
        return result
    
    def text_model_forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """Input is embeddings from prompt and image (after image encoder). Return logits."""


    def forward(self, prompt_before_image: torch.Tensor, images: torch.Tensor, prompt_after_image: torch.Tensor) -> torch.Tensor:
        preprocessed_img = self.image_preprocess(images)
        embeds = self.prepare_inputs_labels_for_multimodal_one_image(prompt_before_image, preprocessed_img, prompt_after_image)
        return LlamaForCausalLM.forward(self.model_, inputs_embeds=embeds, return_dict=False, use_cache=False, output_hidden_states=False)


class LlavaModel(EagerModelBase):
    def __init__(self):
        model_path = "liuhaotian/llava-v1.5-7b"
        self.tokenizer_, self.model_, self.image_processor_, context_len = (
            load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=get_model_name_from_path(model_path),
            )
        )
        self.config_ = PreprocessConfig(image_processor.crop_size, image_processor.image_mean, image_processor.image_std,
                image_processor.rescale_factor)

    def get_eager_model(self):
        model = Llava(self.model_, self.config_)
        return model

    def get_example_inputs(self):
        prompt = ""
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer_, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cpu()
        )
        index = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]
        prompt_before_image = input_ids[:, :index]
        # print(prompt_before_image.shape)
        prompt_after_image = input_ids[:, index+1:]
        # print(prompt_after_image.shape)
        inputs = (prompt_before_image, imagr, prompt_after_image)
        return inputs

    def get_dynamic_shapes(self):
        length = Dim('length', min=8, max=4091)
        token_dim_1 = Dim('token_dim_1', min=2, max=3518)
        token_dim_2 = Dim('token_dim_2', min=2, max=3518)
        width = Dim('width', min=9, max=4092)
        dynamic_shapes = [{1: token_dim_1}, {1: length, 2: width}, {1: token_dim_2}]
        return dynamic_shapes

