# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO(mnachin): Move this file to torchao

from typing import Any, Mapping

from executorch.extension.gguf_util.load_gguf import GGUFModelArgs, GGUFWeights


def convert_gguf_to_checkpoint(
    gguf_model_args: GGUFModelArgs,
    gguf_weights: GGUFWeights,
) -> Mapping[str, Any]:
    """
    Convert a GGUF model to a checkpoint/state_dict

    Args:
        gguf_model_args: The GGUF model args.
        gguf_weights: The GGUF weights.
    """
    # Switch statement based on the architecture enum.
    # Each enum has its own converter function.
    if gguf_model_args.arch == "llama":
        from executorch.extension.gguf_util.gguf_converters.llama_converter import (
            convert_to_state_dict as llama_convert_to_state_dict,
        )

        return llama_convert_to_state_dict(gguf_weights)
    else:
        raise NotImplementedError("Unsupported architecture.")
