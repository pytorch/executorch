# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse

import torch

from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import to_edge
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

from transformers import Phi3ForCausalLM

from .phi_3_mini import Phi3Mini


def export(args) -> None:
    torch.manual_seed(0)

    if args.context_length == "4k":
        model_name = "microsoft/Phi-3-mini-4k-instruct"
    elif args.context_length == "128k":
        model_name = "microsoft/Phi-3-mini-128k-instruct"
    else:
        raise Exception(
            f"Invalid context length {args.context_length}. Should be either 4k or 128k"
        )

    with torch.no_grad():
        model = Phi3Mini(
            # pyre-ignore: Undefined attribute [16]: Module `transformers` has no attribute `Phi3ForCausalLM`
            model=Phi3ForCausalLM.from_pretrained(model_name),
            max_batch_size=1,
            max_seq_len=args.seq_len,
        )
        example_inputs = (
            torch.tensor(
                [[1048, 263, 931, 746]], dtype=torch.long, requires_grad=False
            ),
        )
        dynamic_shapes = {
            "input_ids": {
                1: torch.export.Dim("sequence_length", min=1, max=args.seq_len)
            }
        }

        xnnpack_quant_config = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True
        )
        xnnpack_quantizer = XNNPACKQuantizer()
        xnnpack_quantizer.set_global(xnnpack_quant_config)

        model = capture_pre_autograd_graph(
            model, example_inputs, dynamic_shapes=dynamic_shapes
        )
        model = prepare_pt2e(model, xnnpack_quantizer)  # pyre-fixme[6]
        model(*example_inputs)
        model = convert_pt2e(model)
        DuplicateDynamicQuantChainPass()(model)
        # TODO(lunwenh): update it to use export once
        # https://github.com/pytorch/pytorch/issues/128394 is resolved.
        model = torch.export._trace._export(
            model,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
            pre_dispatch=False,
        )

    edge_config = get_xnnpack_edge_compile_config()
    edge_manager = to_edge(model, compile_config=edge_config)
    edge_manager = edge_manager.to_backend(XnnpackPartitioner())
    et_program = edge_manager.to_executorch()

    with open(args.output_name, "wb") as file:
        file.write(et_program.buffer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--context_length",
        type=str,
        default="4k",
        choices=["4k", "128k"],
        help="Phi-3-mini provides two context length variants: 4k and 128k",
    )
    parser.add_argument(
        "-s",
        "--seq_len",
        type=int,
        default=128,
        help="Maximum number of tokens including prompt to generate",
    )
    parser.add_argument(
        "-o",
        "--output_name",
        default="phi-3-mini.pte",
        help="Override the output filename of the saved pte model file.",
    )
    export(parser.parse_args())


if __name__ == "__main__":
    main()
