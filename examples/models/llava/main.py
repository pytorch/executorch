# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from model import LlavaModel


def main():

    llava_model = LlavaModel()
    llava = llava_model.get_eager_model()

    prompt_before_image, resized, prompt_after_image = llava_model.get_example_inputs()
    logging.info(f"Prompt: {llava_model.prompt}")
    preprocessed = llava.image_preprocess(resized)
    with torch.inference_mode():
        output_ids = llava_model.model.generate(
            llava_model.input_ids,
            images=preprocessed,
            image_sizes=[preprocessed.size],
            do_sample=False,
            num_beams=1,
            max_new_tokens=10,
            use_cache=True,
        )

    outputs = llava_model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
        0
    ].strip()
    logging.info(f"Reference output: {outputs}")

    # comparing with llava result
    # prefill_logits = llava.prefill(prompt_before_image, resized, prompt_after_image)
    # prefill_logits_ref = llava.prefill_ref(*inputs)[0]
    # print(f"Prefill logits all close? {torch.allclose(prefill_logits, prefill_logits_ref, atol=1e-3)}")

    # prefill_logits = llava.prefill(*inputs)
    # context_len = prefill_logits.shape[1]
    # print(prefill_logits)
    # # first token
    # new_tokens = [torch.argmax(prefill_logits[..., -1, :]).item()]
    # # print(tokenizer.decode(new_tokens))
    # for i in range(llava_model.args.max_new_tokens):
    #     print(i, llava_model.tokenizer.decode(new_tokens[i]))
    #     logits = llava.forward(
    #         torch.tensor([new_tokens[i]]), torch.tensor([context_len + i])
    #     )
    #     new_tokens.append(torch.argmax(logits[-1, :]))
    prefill_logits = llava.prefill(prompt_before_image, resized, prompt_after_image)
    context_len = prefill_logits.shape[1]
    logging.info(prefill_logits)
    new_tokens = [torch.argmax(prefill_logits[..., -1, :]).item()]
    i = 0
    logging.info(i, llava_model.tokenizer.decode(new_tokens[i]))
    logits = llava.step(torch.tensor([new_tokens[i]]), torch.tensor([context_len + i]))
    logging.info(logits)


if __name__ == "__main__":
    main()
