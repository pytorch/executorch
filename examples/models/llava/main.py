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


def main():

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
