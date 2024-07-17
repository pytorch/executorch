# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import os
import re

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
import torch
from pathlib import Path
import torchvision

from executorch.examples.models.model_base import EagerModelBase
from executorch.examples.models.llama2.source_transformation.sdpa import replace_sdpa_with_custom_op
from executorch.examples.models.llama2.llama_transformer import (
    FeedForward,
    KVCache,
    ModelArgs,
    RMSNorm,
    SDPA,
    TransformerBlock,
)
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)

from llava.conversation import conv_templates, SeparatorStyle
from llava.eval.run_llava import eval_model, load_images, process_images
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token

from llava.model.builder import load_pretrained_model

from llava.model.llava_arch import LlavaMetaForCausalLM

# from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge

from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from PIL import Image

from torch import nn
from torch.export import Dim
from torch.nn import functional as F
from torchvision.transforms import v2
from torchvision.transforms._functional_tensor import resize

from transformers import LlamaForCausalLM


@dataclass
class PreprocessConfig:
    crop_size: dict
    image_mean: list[float]
    image_std: list[float]
    rescale_factor: float


def precompute_freqs_cis(dim: int, end: int, theta: float):
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, device="cpu", dtype=torch.int64).float() / dim)
    )
    t = torch.arange(end, device=freqs.device, dtype=torch.int64).type_as(
        freqs
    )  # pyre-ignore
    freqs = torch.outer(t, freqs).float()  # pyre-ignore
    emb = torch.cat((freqs, freqs), dim=-1)
    freqs_cos = torch.cos(emb)
    freqs_sin = torch.sin(emb)
    return freqs_cos, freqs_sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.use_kv_cache = args.use_kv_cache
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        # args.dim = 4096, args.n_heads = 32, self.head_dim = 4096 / 32 = 125
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.layer_id = layer_id

        causal_mask = torch.tril(
            torch.ones(
                self.max_seq_len,
                self.max_seq_len,
                dtype=torch.bool,
                device="cpu",
            )
        )
        self.register_buffer("mask", causal_mask, persistent=False)

        if self.use_kv_cache:
            self.kv_cache = KVCache(
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
                not args.use_sdpa_with_kv_cache_op,  # if we are using the custom op dont transpose the cache. Expect untransposed q k v
                args.enable_dynamic_shape,
            )
            self.SDPA = SDPA(
                kv_cache=self.kv_cache,
                dim=self.dim,
                head_dim=self.head_dim,
                n_rep=self.n_rep,
                max_seq_len=self.max_seq_len,
                enable_dynamic_shape=args.enable_dynamic_shape,
            )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        # We need view_copy elimination
        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        q, k = apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)

        assert input_pos is not None
        # print(f"in attention: q: {torch.isnan(q).any()}, {torch.isnan(k).any()}, {torch.isnan(v).any()}")
        output = self.SDPA(input_pos, q, k, v, bsz, seqlen, self.mask)
        # print(f"in attention: out {output[0, 0, :3]}")
        return self.wo(output)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.use_kv_cache = args.use_kv_cache
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, layer_id)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin, input_pos=None):  # x: 1xN
        h = self.attention.forward(
            self.attention_norm(x), freqs_cos, freqs_sin, input_pos
        )
        # print(f"after attention: {h[0, 0, :3]}, is_nan: {torch.isnan(h).any()}")
        h = x + h
        out = h + self.feed_forward(self.ffn_norm(h))
        # print(f"after ffn: {out[0, 0, :3]}, is_nan: {torch.isnan(out).any()}")
        return out


class TextModelTransformer(nn.Module):
    """Mostly copied from examples/models/llama2/llama_transformer but taking a embedding"""

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.use_kv_cache = params.use_kv_cache
        self.max_seq_len = params.max_seq_len

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

        assert (
            input_pos is not None
        ), "input_pos must be provided when use_kv_cache is True"

        # when KV cache is used, seqlen is most likely 1. We want to slice from the start_pos.
        input_pos_item = input_pos[-1].item()
        torch._check_is_size(input_pos_item)
        # Setting this to max_seq_len but the resulting
        # asserts from export are ignore anyway, so the particular
        # value doesn't matter.
        # Also in future when we want to support infinite generation
        # input_pos can take any value until eos is encountered.
        torch._check(input_pos_item < self.max_seq_len)
        freqs_cos = self.freqs_cos.narrow(0, input_pos_item, seqlen)
        freqs_sin = self.freqs_sin.narrow(0, input_pos_item, seqlen)
    

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
            # rope_theta=self.model_.config.rope_theta,
            ffn_dim_multiplier=1,  # a hack to make rotary embedding happy
            enable_dynamic_shape=True,
            use_sdpa_with_kv_cache_op=True,
        )
        self.embed_tokens = nn.Embedding(self.model_.config.vocab_size, self.model_.config.hidden_size, self.model_.config.pad_token_id) 
        self.text_model = TextModelTransformer(self.text_model_args)
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
        return self.text_model.forward(token_embeds, input_pos)

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
        embeds_before_img = (
            self.embed_tokens(prompt_before_image)
        )
        embeds_after_img = (
            self.embed_tokens(prompt_after_image)
        )
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
        return self.text_model.forward(embeds, torch.tensor([0]))

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
        self.image_path = Path(__file__).with_name("view.jpg")
        self.args = type(
            "Args",
            (),
            {
                "model_path": self.model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(self.model_path),
                "query": "What are the things I should be cautious about when I visit here?",
                "conv_mode": None,
                "image_file": "./view.jpg",
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
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cpu()
        )
        index = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]
        prompt_before_image = input_ids[:, :index]
        # print(prompt_before_image.shape)
        prompt_after_image = input_ids[:, index + 1 :]
        # print(prompt_after_image.shape)
        imagr = torchvision.io.read_image(self.image_path)
        ratio = max(imagr.shape[1], imagr.shape[2]) / self.image_processor.crop_size["height"]
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
