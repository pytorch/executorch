# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import torch
from executorch.extension.llm.export.builder import DType, LLMEdgeManager

from executorch.extension.llm.export.partitioner_lib import get_xnnpack_partitioner
from executorch.extension.llm.export.quantizer_lib import (
    DynamicQuantLinearOptions,
    get_pt2e_quantizers,
    PT2EQuantOptions,
)

from transformers import Phi3ForCausalLM

from .phi_3_mini import Phi3Mini


def main(args) -> None:
    torch.manual_seed(42)

    if args.context_length == "4k":
        model_name = "microsoft/Phi-3-mini-4k-instruct"
    elif args.context_length == "128k":
        model_name = "microsoft/Phi-3-mini-128k-instruct"
    else:
        raise Exception(
            f"Invalid context length {args.context_length}. Should be either 4k or 128k"
        )

    (
        LLMEdgeManager(
            model=Phi3Mini(
                # pyre-ignore: Undefined attribute [16]: Module `transformers` has no attribute `Phi3ForCausalLM`
                model=Phi3ForCausalLM.from_pretrained(model_name),
                max_batch_size=1,
                max_seq_len=args.seq_len,
            ),
            modelname="phi-3-mini",
            max_seq_len=args.seq_len,
            dtype=DType.fp32,
            use_kv_cache=True,
            example_inputs=(
                torch.tensor(
                    [[1048, 263, 931, 746]], dtype=torch.long, requires_grad=False
                ),
            ),
            enable_dynamic_shape=True,
            dynamic_shapes={
                "input_ids": {
                    1: torch.export.Dim("sequence_length", min=1, max=args.seq_len)
                }
            },
            verbose=True,
        )
        .set_output_dir(".")
        .capture_pre_autograd_graph()
        .pt2e_quantize(
            get_pt2e_quantizers(PT2EQuantOptions(None, DynamicQuantLinearOptions()))
        )
        .export_to_edge(strict=False)
        .to_backend([get_xnnpack_partitioner()])
        .to_executorch()
        .save_to_pte(f"{args.output_name}.pte")
    )


if __name__ == "__main__":
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
    main(parser.parse_args())
