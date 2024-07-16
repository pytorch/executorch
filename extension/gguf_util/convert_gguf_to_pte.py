# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.extension.gguf_util.convert_gguf_to_checkpoint import (
    convert_gguf_to_checkpoint,
)
from executorch.extension.gguf_util.load_gguf import GGUFModelArgs, GGUFWeights
from executorch.extension.gguf_util.quantize import quantize


def convert_to_pte(
    gguf_model_args: GGUFModelArgs,
    gguf_weights: GGUFWeights,
) -> None:
    """Convert a GGUF model into a PTE file, an ExecuTorch program.

    Args:
        gguf_model_args: The arguments for the GGUF model.
        gguf_weights: The weights of the GGUF model.
    """

    # Step 1: Create the corresponding checkpoint state dict
    state_dict = convert_gguf_to_checkpoint(gguf_model_args, gguf_weights)

    # Switch statement based on the architecture enum.
    # Each enum has its own converter function.
    if gguf_model_args.arch == "llama":
        from executorch.extension.gguf_util.pte_converters.llama_pte_converter import (
            convert_to_pte as llama_convert_to_pte,
            _create_pt_model as create_llama_pt_model,
        )
                
        # Step 2: Create the PyTorch nn.Module.
        pt_model = create_llama_pt_model(gguf_model_args)
        
        # Step 3: Quantize
        pt_model = quantize(pt_model, gguf_model_args)

        # Step 4: Convert the PyTorch nn.Module to a PTE program.
        return llama_convert_to_pte(pt_model, state_dict, gguf_model_args)

    else:
        raise NotImplementedError("Unsupported architecture.")
