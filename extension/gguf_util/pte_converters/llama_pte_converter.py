from typing import Any, Mapping

import torch
from executorch.extension.gguf_util.load_gguf import GGUFModelArgs

from torch import nn


def _load_weights_into_nn(
    pt_model: nn.Module,
    state_dict: Mapping[str, Any],
    gguf_model_args: GGUFModelArgs,
):

    # We need to fake initialize the mask, to match with the llama_transformer.py
    for id in range(gguf_model_args.block_count):
        mask_name = f"layers.{id}.attention.mask"
        mask = torch.full(
            (1, 1, pt_model.params.max_seq_len, pt_model.params.max_seq_len),
            float("-inf"),
        )
        mask = torch.triu(mask, diagonal=1)
        state_dict[mask_name] = mask

    pt_model.load_state_dict(state_dict)
    return


def _create_pte_program(pt_model: nn.Module) -> bytes:
    # TODO (mnachin): Export
    return


def convert_to_pte(
    pt_model: nn.Module,
    state_dict: Mapping[str, Any],
    gguf_model_args: GGUFModelArgs,
) -> bytes:
    """Convert a GGUF model into an ExecuTorch program.

    Args:
        model_args: The arguments for the GGUF model.
        weights: The weights of the GGUF model.
    """

    print("Load the weights into the PyTorch model")
    _load_weights_into_nn(pt_model, state_dict)

    # Step 3: Export to ExecuTorch
    print("Exporting to ExecuTorch.")
    pte_program = _create_pte_program(pt_model)
    return pte_program
