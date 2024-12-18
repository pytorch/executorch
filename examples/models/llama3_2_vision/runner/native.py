# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from typing import Optional

import torch

from executorch.examples.models.llama.export_llama_lib import (
    EXECUTORCH_DEFINED_MODELS,
    TORCHTUNE_DEFINED_MODELS,
)
from executorch.examples.models.llama3_2_vision.runner.generation import (
    TorchTuneLlamaRunner,
)

from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)

# Load custom ops and quantized ops.
from executorch.extension.pybindings import portable_lib  # noqa # usort: skip

# Note: import this after portable_lib
from executorch.extension.llm.custom_ops import custom_ops  # noqa # usort: skip
from executorch.kernels import quantized  # noqa


class NativeLlamaRunner(TorchTuneLlamaRunner):
    """
    Runs llama via ExecuTorch with provided pte file.
    """

    def __init__(self, args):
        with open(args.params, "r") as f:
            params = json.loads(f.read())
        super().__init__(
            tokenizer_path=args.tokenizer,
            max_seq_len=args.max_len,
            max_batch_size=1,
            use_kv_cache=args.kv_cache,
            vocab_size=params["vocab_size"],
        )
        # Save the loaded model bytes to prevent data from going out of
        # scope after the `with` and getting cleaned up by Python's
        # garbage collector.
        self.model_bytes = None
        with open(args.pte, "rb") as f:
            self.model_bytes = f.read()
            # Need to use _load_for_executorch_from_buffer instead of
            # _load_for_executorch because the latter uses MmapDataLoader,
            # which doesn't have load_into() implemented, which is needed
            # for loading initialized mutable buffers.
            self.model = _load_for_executorch_from_buffer(self.model_bytes)
        self.use_kv_cache = args.kv_cache

    def forward(
        self,
        tokens: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        return (
            self.model.forward((tokens, input_pos, mask))
            if self.use_kv_cache
            else self.model.forward((tokens,))
        )[0]


def build_args_parser() -> argparse.ArgumentParser:
    # TODO: merge these with build_args_parser from export_llama_lib.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="llama3",
        choices=EXECUTORCH_DEFINED_MODELS + TORCHTUNE_DEFINED_MODELS,
    )

    parser.add_argument(
        "-f",
        "--pte",
        type=str,
        default=None,
        help="path to exported executorch .pte file",
    )

    parser.add_argument(
        "-p", "--params", type=str, default=None, help="model params file"
    )

    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "-kv",
        "--kv_cache",
        action="store_true",
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="Maximum length of the generated response sequence.",
    )

    return parser


def main() -> None:
    parser = build_args_parser()
    args = parser.parse_args()
    runner = NativeLlamaRunner(args)
    generated_tokens = runner.text_completion(
        prompt=args.prompt,
        temperature=args.temperature,
    )
    print(f"Response: {generated_tokens}")


if __name__ == "__main__":
    main()  # pragma: no cover
