import argparse
from typing import Dict

import torch

from executorch.examples.models.smollm3.convert_weights import load_checkpoint
from torchtune.models.convert_weights import get_mapped_key

# Standard _FROM_META weight mapping of Meta weights to TorchTune + additional bias weight mappings.
_INTERNVL_TO_META = {
    "language_model.model.embed_tokens.weight": "tok_embeddings.weight",
    "language_model.model.norm.weight": "norm.weight",
    "language_model.model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "language_model.model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
    "language_model.model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "language_model.model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
    "language_model.model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "language_model.model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
    "language_model.model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "language_model.model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "language_model.model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "language_model.model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "language_model.model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "language_model.model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
}


def internvl_tune_to_meta(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from torchtune's format to Meta's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in torchtune's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    converted_text_model_state_dict = {}
    for key, value in state_dict.items():
        try:
            new_key = get_mapped_key(key, _INTERNVL_TO_META)
            converted_text_model_state_dict[new_key] = value
        except:
            # only preserve parameters of text decoder
            pass
    converted_text_model_state_dict["output.weight"] = state_dict[
        "language_model.lm_head.weight"
    ]
    return converted_text_model_state_dict


def convert_weights(input_dir: str, output_file: str) -> None:
    print("Loading checkpoint...")
    sd = load_checkpoint(input_dir)
    print("Converting checkpoint...")
    converted_sd = internvl_tune_to_meta(sd)
    print("Saving checkpoint...")
    torch.save(converted_sd, output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert InternVL3 weights to Meta format."
    )
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
