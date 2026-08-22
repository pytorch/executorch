# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from executorch.extension.gguf_util.converter import convert_to_pte
from executorch.extension.gguf_util.load_gguf import load_file


def save_pte_program(_, pte_file) -> None:
    # TODO (mnachin): Save the PTE program to a file
    print(f"Saving PTE program to {pte_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gguf_file",
        type=str,
        help="The GGUF file to load.",
    )
    parser.add_argument(
        "--pte_file",
        type=str,
        help="The path to save the PTE file.",
    )
    args = parser.parse_args()

    # Step 1: Load the GGUF file
    gguf_model_args, gguf_weights = load_file(args.gguf_file)

    # Step 2: Convert the GGUF model to PTE
    # Currently, underneath the hood, it is first converting the GGUF model
    # to a PyTorch model (nn.Module), then exporting to ET.
    #
    # NOTE: In the future, it may makes sense to refactor out the conversion from GGUF to nn.Module
    # into its own package that can be shared between ExecuTorch and PyTorch core. I can
    # imagine that there will be a need to do load GGUF file directly into PyTorch core, and
    # use torch.compile/AOTInductor to accelerate on server, without ever touching ExecuTorch.
    #
    # TODO(mnachin): Add a knob to delegate to various backends.
    pte_program = convert_to_pte(gguf_model_args, gguf_weights)

    # Step 3: Save the PTE program so that
    # it can be used by the ExecuTorch runtime
    save_pte_program(pte_program, args.pte_file)


if __name__ == "__main__":
    main()
