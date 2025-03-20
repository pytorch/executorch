import argparse
from typing import Dict

import torch

from torchtune.models.convert_weights import get_mapped_key

from torchtune.training import FullModelHFCheckpointer

# Standard _FROM_META weight mapping of Meta weights to TorchTune + additional bias weight mappings.
_SMOLLM_FROM_META = {
    "tok_embeddings.weight": "tok_embeddings.weight",
    "norm.weight": "norm.scale",
    "output.weight": "output.weight",
    "layers.{}.attention.wk.weight": "layers.{}.attn.k_proj.weight",
    "layers.{}.attention.wq.weight": "layers.{}.attn.q_proj.weight",
    "layers.{}.attention.wv.weight": "layers.{}.attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "layers.{}.attn.output_proj.weight",
    "layers.{}.attention_norm.weight": "layers.{}.sa_norm.scale",
    "layers.{}.ffn_norm.weight": "layers.{}.mlp_norm.scale",
    "layers.{}.feed_forward.w1.weight": "layers.{}.mlp.w1.weight",
    "layers.{}.feed_forward.w2.weight": "layers.{}.mlp.w2.weight",
    "layers.{}.feed_forward.w3.weight": "layers.{}.mlp.w3.weight",
}


def smollm_tune_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from torchtune's format to Meta's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in torchtune's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _SMOLLM_FROM_META.items()}
    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value

    return converted_state_dict


def main():
    parser = argparse.ArgumentParser(
        description="Convert SmolLM weights to Meta format."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to directory containing checkpoint files",
    )
    parser.add_argument("output", type=str, help="Path to the output checkpoint")

    args = parser.parse_args()

    # Don't necessarily need to use TorchTune checkpointer, can just aggregate checkpoint files by ourselves.
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=args.input_dir,
        checkpoint_files=["model.safetensors"],
        output_dir=".",
        model_type="LLAMA",
    )

    print("Loading checkpoint...")
    sd = checkpointer.load_checkpoint()

    print("Converting checkpoint...")
    sd = smollm_tune_to_meta(sd["model"])

    torch.save(sd, args.output)
    print(f"Checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
