# READ:
# ~/src/llava_diff to run in CPU
# pip install -I torch for newer torch version for fp16

import torch

import os

from torch import nn

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.eval.run_llava import eval_model, load_images, process_images
# from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge

from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.model.llava_arch import LlavaMetaForCausalLM

from transformers import LlamaForCausalLM

from dataclasses import dataclass
import math
from executorch.examples.models.llama2.llama_transformer import TransformerBlock, ModelArgs, RMSNorm, precompute_freqs_cis
from typing import Optional, Dict, Any
import re
import requests
from PIL import Image


import torchvision
from torchvision.transforms._functional_tensor import resize
from torchvision.transforms import v2

# model_path = "liuhaotian/llava-v1.6-vicuna-7b"
# only this one works
os.environ['HF_TOKEN'] = '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'
model_path = "liuhaotian/llava-v1.5-7b"


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

        if self.use_kv_cache:
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
        else:
            assert input_pos is None, "input_pos is unused when use_kv_cache is False"
            freqs_cos = self.freqs_cos[:seqlen]
            freqs_sin = self.freqs_sin[:seqlen]

        for i, layer in enumerate(self.layers):
            h = layer(
                h,
                freqs_cos,
                freqs_sin,
                input_pos,
            )
            if i < 3:
                torch.save(h, f"/Users/larryliu/Desktop/layer_{i}.pt")

        h = self.norm(h)

        logits = self.output(h)
        return logits
    
class Llava(torch.nn.Module):
    def __init__(self, llava_model: LlavaMetaForCausalLM, image_processor: CLIPVisionTower,  config: PreprocessConfig):
        super().__init__()
        self.config = config
        self.model_ = llava_model
        self.text_model_args = ModelArgs(
            use_kv_cache=True, 
            vocab_size=self.model_.config.vocab_size, 
            hidden_dim=self.model_.config.intermediate_size,
            max_batch_size=1, # doesn't work with default batch size 32
        )
        self.text_model = TextModelTransformer(self.text_model_args)
        # load state dict
        self.text_model.load_state_dict(
            state_dict=self._translate_state_dict_for_text_model(),
            strict=False,
            assign=True,
        )
        self.image_processor = image_processor

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
            # r"model.embed_tokens.": r"tok_embeddings.", # not needed
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
        padded = torchvision.transforms.v2.functional.pad(img, padding=(l_pad, t_pad, r_pad, b_pad), fill=tuple(int(x*255) for x in self.image_processor.image_mean))
        # here padded shape should be max(h, w) x max(h, w)
        resized = resize(padded, size=[self.image_processor.crop_size['height'], self.image_processor.crop_size['width']], interpolation="bicubic")
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
        assert isinstance(prompt_before_image, torch.Tensor), f"Expecting prompt_before_image to be a tensor, got {prompt_before_image}"
        assert prompt_before_image.shape[0] == 1, f"Expecting prompt_before_image to be of shape [1, num_tokens], got {prompt_before_image.shape}"
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
    
    def forward(self, token: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Input is embeddings from prompt and image (after image encoder). Return logits."""
        token_embeds = self.get_model().embed_tokens(token).unsqueeze(0)
        return self.text_model.forward(token_embeds, input_pos)

    def prefill(self, prompt_before_image: torch.Tensor, images: torch.Tensor, prompt_after_image: torch.Tensor) -> torch.Tensor:
        """Avoiding the torch.where() call to find <image> placeholder and insert image embedding. Taking 3 inputs instead."""
        preprocessed_img = self.image_preprocess(images)
        embeds = self.prepare_inputs_labels_for_multimodal_one_image(prompt_before_image, preprocessed_img, prompt_after_image)
        return self.text_model.forward(embeds, torch.tensor([0]))

    def prefill_ref(self, prompt_before_image: torch.Tensor, images: torch.Tensor, prompt_after_image: torch.Tensor) -> torch.Tensor:
        """Avoiding the torch.where() call to find <image> placeholder and insert image embedding. Taking 3 inputs instead."""
        preprocessed_img = self.image_preprocess(images)
        embeds = self.prepare_inputs_labels_for_multimodal_one_image(prompt_before_image, preprocessed_img, prompt_after_image)
        return LlamaForCausalLM.forward(self.model_, inputs_embeds=embeds, return_dict=False, use_cache=False, output_hidden_states=False)
    
def download_image() -> str:
    image = Image.open(requests.get('https://llava-vl.github.io/static/images/view.jpg', stream=True).raw)
    temp_file = "./view.jpg"
    image.save(temp_file)
    return temp_file

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

def get_image_tensor(args, image_processor, model) -> torch.Tensor:
    image_files = args.image_file.split(args.sep)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    return image_sizes, images_tensor

def main():
    temp_file = download_image()
    image_files = [temp_file]  # IMG_3997

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": "What are the things I should be cautious about when I visit here?",
        "conv_mode": None,
        "image_file": image_files[0],
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device_map="cpu", device="cpu"
    )

    prompt = get_prompt(args.query, model.config.mm_use_im_start_end, model_name)

    # # uncomment this line for end to end eager mode run
    # eval_model(args)
    imagr = torchvision.io.read_image(image_files[0])
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cpu()
    )

    pre_config = PreprocessConfig(image_processor.crop_size, image_processor.image_mean, image_processor.image_std,
                              image_processor.rescale_factor)
    llava = Llava(model, image_processor, pre_config)
    index = torch.where(input_ids==IMAGE_TOKEN_INDEX)[1]
    prompt_before_image = input_ids[:, :index]
    prompt_after_image = input_ids[:, index+1:]
    llava = llava.to(torch.float32) # overflow error with fp16

    prefill_logits_ref = llava.prefill_ref(prompt_before_image, imagr, prompt_after_image)[0]
    prefill_logits = llava.prefill(prompt_before_image, imagr, prompt_after_image)
    # context_len = prefill_logits.shape[0]
    # print(prefill_logits)
    # # first token
    # new_tokens = [torch.argmax(prefill_logits[-1, :]).item()]

    # for i in range(args.max_new_tokens):
    #     print(i, tokenizer.decode(new_tokens[i]))
    #     logits = llava.forward(new_tokens[i], context_len + i)
    #     new_tokens.append(torch.argmax(logits[-1, :]))


if __name__ == "__main__":
    main()