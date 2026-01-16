# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code=import-not-found

import torch

try:
    import nncf  # type: ignore[import-untyped]
    from pytorch_tokenizers import get_tokenizer  # type: ignore[import-untyped]
except ImportError:
    raise ImportError("Please install nncf via backends/openvino/requirements.txt")


def get_calibration_data(
    module: torch.fx.GraphModule, tokenizer, prompts: str, max_len: int
):
    # TODO: change criteria & support batch inputs if necessary
    pos = torch.tensor(0, dtype=torch.int64)
    token_list = tokenizer.encode(prompts, bos=True, eos=False)

    with torch.no_grad():
        while token_list[-1] != tokenizer.eos_id and pos < max_len:
            logits = module(
                torch.full((1, 1), token_list[pos]),
                {"input_pos": torch.tensor((pos,))},
            )
            pos += 1
            if pos >= len(token_list):
                token_list.append(torch.argmax(logits[:], dim=-1).item())
    token_list = [
        (
            pos,
            token,
        )
        for pos, token in enumerate(token_list)
    ]
    return token_list


def transform_fn(token_pos_map: tuple[int, str]):
    # tokenized_text = tokenizer.encode(prompts, bos=False, eos=False)
    inputs = ()
    inputs = (
        torch.tensor(token_pos_map[1]).unsqueeze(0).unsqueeze(0),
        {"input_pos": torch.tensor([token_pos_map[0]])},
    )

    return inputs


def apply_nncf_data_aware_compression(
    builder_exported, quantizers, awq: bool, scale_estimation: bool
):
    tokenizer = get_tokenizer(builder_exported.tokenizer_path)

    builder_exported.calibration_data = get_calibration_data(
        builder_exported.pre_autograd_graph_module,
        tokenizer,
        builder_exported.calibration_data,
        builder_exported.max_seq_len,
    )

    builder_exported.pre_autograd_graph_module = (
        nncf.experimental.torch.fx.compress_pt2e(
            builder_exported.pre_autograd_graph_module,
            quantizer=quantizers[0],
            dataset=nncf.Dataset(
                builder_exported.calibration_data,
                transform_func=transform_fn,
            ),
            awq=awq,
            scale_estimation=scale_estimation,
        )
    )
    return builder_exported