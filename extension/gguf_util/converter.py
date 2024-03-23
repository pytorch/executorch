# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.extension.gguf_util.load_gguf import GGUFModelArgs, GGUFWeights


def convert_to_pte(model_args: GGUFModelArgs, weights: GGUFWeights) -> None:
    """Convert a GGUF model into a PTE file, an ExecuTorch program.

    Args:
        model_args: The arguments for the GGUF model.
        weights: The weights of the GGUF model.
    """

    # Switch statement based on the architecture enum.
    # Each enum has its own converter function.
    if model_args.arch == "llama":
        from executorch.extension.gguf_util.converters.llama_converter import (
            convert_to_pte as llama_convert_to_pte,
        )

        return llama_convert_to_pte(model_args, weights)
    else:
        raise NotImplementedError("Unsupported architecture.")
