# READ:
# ~/src/llava_diff to run in CPU
# pip install -I torch for newer torch version for fp16

import math

import os
import re

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
import torch

import torchvision
from executorch.examples.models.llama2.llama_transformer import (
    FeedForward,
    KVCache,
    ModelArgs,
    RMSNorm,
    SDPA,
)
from model import LlavaModel
from PIL import Image

from torch import nn
from torch.nn import functional as F
from torchvision.transforms import v2
from torchvision.transforms._functional_tensor import resize

from transformers import LlamaForCausalLM

# model_path = "liuhaotian/llava-v1.6-vicuna-7b"
# only this one works
os.environ["HF_TOKEN"] = "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"


def download_image() -> str:
    image = Image.open(
        requests.get(
            "https://llava-vl.github.io/static/images/view.jpg", stream=True
        ).raw
    )
    temp_file = "./view.jpg"
    image.save(temp_file)
    return temp_file


def main():
    temp_file = download_image()
    image_files = [temp_file]  # IMG_3997

    llava_model = LlavaModel()
    llava = llava_model.get_eager_model()

    llava = llava.to(torch.float32)  # overflow error with fp16
    inputs = llava_model.get_example_inputs()

    prefill_logits = llava.prefill(*inputs)
    # prefill_logits_ref = llava.prefill_ref(prompt_before_image, imagr, prompt_after_image)[0]
    # prefill_logits = llava.prefill(prompt_before_image, imagr, prompt_after_image)
    context_len = prefill_logits.shape[1]
    print(prefill_logits)
    # first token
    new_tokens = [torch.argmax(prefill_logits[..., -1, :]).item()]
    # print(tokenizer.decode(new_tokens))
    for i in range(llava_model.args.max_new_tokens):
        print(i, llava_model.tokenizer.decode(new_tokens[i]))
        logits = llava.forward(
            torch.tensor([new_tokens[i]]), torch.tensor([context_len + i])
        )
        new_tokens.append(torch.argmax(logits[-1, :]))


if __name__ == "__main__":
    main()
