import argparse
import os
from typing import Dict

import torch
from safetensors.torch import load_file
from torchtune.models.convert_weights import get_mapped_key

# Standard _FROM_META weight mapping of Meta weights to TorchTune + additional bias weight mappings.
_GLM_FROM_META = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
    "layers.{}.attention.wk.weight": "model.layers.{}.self_attn.k_proj.weight",
    "layers.{}.attention.wq.weight": "model.layers.{}.self_attn.q_proj.weight",
    "layers.{}.attention.wv.weight": "model.layers.{}.self_attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",
    "layers.{}.attention_norm.weight": "model.layers.{}.input_layernorm.weight",
    "layers.{}.ffn_norm.weight": "model.layers.{}.post_attention_layernorm.weight",
    "layers.{}.feed_forward.gate_up_proj.weight": "model.layers.{}.mlp.gate_up_proj.weight",
    "layers.{}.feed_forward.down_proj.weight": "model.layers.{}.mlp.down_proj.weight",
}


def glm_tune_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
    inverted_mapping_dict = {v: k for k, v in _GLM_FROM_META.items()}

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value

    if "lm_head.weight" not in state_dict:
        converted_state_dict["output.weight"] = converted_state_dict[
            "tok_embeddings.weight"
        ]

    return converted_state_dict


def convert_weights(input_dir: str, output_file: str) -> None:
    pt_path = os.path.join(input_dir, "model.safetensors")
    print("Loading checkpoint from file...")
    sd = load_file(pt_path)

    print("Converting checkpoint...")
    sd = glm_tune_to_meta(sd)

    print("Saving checkpoint...")
    torch.save(sd, output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Convert GLM weights to Meta format.")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to directory containing checkpoint files",
    )
    parser.add_argument("output", type=str, help="Path to the output checkpoint")

    args = parser.parse_args()
    convert_weights(args.input_dir, args.output)


if __name__ == "__main__":
    main()
