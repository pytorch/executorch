# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

import torch
from executorch.examples.models.llava.image_util import prepare_image
from executorch.examples.models.llava.model import LlavaModel
from executorch.extension.pybindings.portable_lib import _load_for_executorch
from PIL import Image

# Custom ops has to be loaded after portable_lib.
from executorch.extension.llm.custom_ops import sdpa_with_kv_cache  # noqa # usort: skip
from executorch.kernels import quantized  # noqa # usort: skip

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


def main():
    args = sys.argv[1:]
    if len(args) == 0:
        print(
            "Usage: python test_pte.py <model_path> <image_path?>. If no image, will use default image."
        )
        sys.exit(1)

    llava_module = _load_for_executorch(args[0])

    llava_model = LlavaModel()

    prompt_before_image, resized, prompt_after_image = (
        llava_model.get_inputs_for_prefill()
    )
    if len(args) == 2:
        image_path = args[1]
        image = Image.open(image_path)
        resized = prepare_image(image, target_h=336, target_w=336)

    start_pos = 0
    # pte prefill prompt before img
    pte_embeds_before_img = llava_module.run_method(
        "token_embedding", (prompt_before_image,)
    )[0]
    pte_prefill_before_img = llava_module.run_method(
        "text_model",
        (torch.tensor([start_pos], dtype=torch.int64), pte_embeds_before_img),
    )[0]
    print(pte_prefill_before_img)

    start_pos += prompt_before_image.shape[1]

    # pte prefill image
    logging.warning("Image encoder started")
    pte_embeds_img = llava_module.run_method("image_encoder", (resized,))[0]
    logging.warning("Image encoder finished")
    logging.warning("Image token prefill started")
    pte_prefill_img = llava_module.run_method(
        "text_model",
        (
            torch.tensor([start_pos], dtype=torch.int64),
            pte_embeds_img,
        ),
    )[0]
    logging.warning("Image token prefill finished")
    print(pte_prefill_img)

    start_pos += pte_embeds_img.shape[1]

    # pte prefill prompt after img
    logging.warning("Text token prefill started")
    pte_embeds_after_img = llava_module.run_method(
        "token_embedding", (prompt_after_image,)
    )[0]
    pte_prefill_after_img = llava_module.run_method(
        "text_model",
        (torch.tensor([start_pos], dtype=torch.int64), pte_embeds_after_img),
    )[0]
    logging.warning("Text token prefill finished")
    print(pte_prefill_after_img)

    # being tested, using llama_transformer
    new_tokens = [torch.argmax(pte_prefill_after_img[..., -1, :]).item()]
    for i in range(4):
        print(i, llava_model.tokenizer.decode(new_tokens[i]))
        token_embeds = llava_module.run_method(
            "token_embedding", (torch.tensor([[new_tokens[i]]], dtype=torch.int64),)
        )[0]
        logits = llava_module.run_method(
            "text_model",
            (torch.tensor([start_pos + i], dtype=torch.int64), token_embeds),
        )[0]
        new_tokens.append(torch.argmax(logits[..., -1, :]).item())

    outputs = llava_model.tokenizer.batch_decode(
        torch.tensor([new_tokens]), skip_special_tokens=True
    )[0].strip()
    print(outputs)


if __name__ == "__main__":
    main()
