import sys

import torch

from executorch.examples.models.llama2.custom_ops import sdpa_with_kv_cache  # noqa
from executorch.examples.models.llava.model import LlavaModel
from executorch.extension.pybindings.portable_lib import _load_for_executorch


def main():
    args = sys.argv[1:]
    llava_module = _load_for_executorch(args[0])

    llava_model = LlavaModel()

    prompt_before_image, resized, prompt_after_image = (
        llava_model.get_inputs_for_prefill()
    )

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

    start_pos += pte_prefill_before_img.shape[1]

    # pte prefill image
    pte_embeds_img = llava_module.run_method("image_encoder", (resized,))[0]
    pte_prefill_img = llava_module.run_method(
        "text_model",
        (
            torch.tensor([start_pos], dtype=torch.int64),
            pte_embeds_img,
        ),
    )[0]
    print(pte_prefill_img)

    start_pos += pte_prefill_img.shape[1]

    # pte prefill prompt after img
    pte_embeds_after_img = llava_module.run_method(
        "token_embedding", (prompt_after_image,)
    )[0]
    pte_prefill_after_img = llava_module.run_method(
        "text_model",
        (torch.tensor([start_pos], dtype=torch.int64), pte_embeds_after_img),
    )[0]
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
