import argparse
from typing import Dict

import torch

from torchtune.models.convert_weights import get_mapped_key
from executorch.examples.models.checkpoint import load_checkpoint_from_pytorch_model

from torchtune.training import FullModelHFCheckpointer

_HF_PHI_4_FROM_META = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "layers.{}.attention.wq.weight": "model.layers.{}.self_attn.q_proj.weight",
    "layers.{}.attention.wk.weight": "model.layers.{}.self_attn.k_proj.weight",
    "layers.{}.attention.wv.weight": "model.layers.{}.self_attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",
    "layers.{}.attention_norm.weight": "model.layers.{}.input_layernorm.weight",
    "layers.{}.ffn_norm.weight": "model.layers.{}.post_attention_layernorm.weight",
    "layers.{}.feed_forward.w1.weight": "model.layers.{}.mlp.gate_proj.weight",
    "layers.{}.feed_forward.w3.weight": "model.layers.{}.mlp.up_proj.weight",
    "layers.{}.feed_forward.w2.weight": "model.layers.{}.mlp.down_proj.weight",
    "output.weight": "lm_head.weight",
}


def phi_4_hf_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from hf's format to Meta's format.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in hf's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _HF_PHI_4_FROM_META.items()}

    for key, value in state_dict.items():
        if key.endswith("mlp.gate_up_proj.weight"):
            # Split the gate_up_proj into gate_proj and up_proj
            hidden_dim = value.shape[0] // 2
            assert 2 * hidden_dim == value.shape[0]
            gate = value[0:hidden_dim, :]
            up = value[hidden_dim:, :]
            for new_key, new_value in [("gate_proj", gate), ("up_proj", up)]:
                new_key = key.replace("gate_up_proj", new_key)
                new_key = get_mapped_key(new_key, inverted_mapping_dict)
                converted_state_dict[new_key] = new_value
        elif key.endswith("self_attn.qkv_proj.weight"):
            # Split the qkv_proj into q_proj, k_proj, and v_proj
            q_dim = value.shape[1]
            kv_dim = (value.shape[0] - q_dim) // 2
            assert 2 * kv_dim + q_dim == value.shape[0]
            q = value[0:q_dim, :]
            k = value[q_dim : (q_dim + kv_dim), :]
            v = value[(q_dim + kv_dim) :, :]
            for new_key, new_value in [("q_proj", q), ("k_proj", k), ("v_proj", v)]:
                new_key = key.replace("qkv_proj", new_key)
                new_key = get_mapped_key(new_key, inverted_mapping_dict)
                converted_state_dict[new_key] = new_value
        else:
            new_key = get_mapped_key(key, inverted_mapping_dict)
            converted_state_dict[new_key] = value
    return converted_state_dict


# Standard _FROM_META weight mapping of Meta weights to TorchTune.
_PHI_4_FROM_META = {
    "tok_embeddings.weight": "tok_embeddings.weight",
    "norm.weight": "norm.scale",
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


def phi_4_tune_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
    inverted_mapping_dict = {v: k for k, v in _PHI_4_FROM_META.items()}

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value

    # Input and output embeddings are tied.
    converted_state_dict["output.weight"] = converted_state_dict[
        "tok_embeddings.weight"
    ]
    return converted_state_dict


def phi_4_tune_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
    inverted_mapping_dict = {v: k for k, v in _PHI_4_FROM_META.items()}

    # Single checkpoint
    model_path = os.path.join(input_dir, "pytorch_model.bin")
    if os.path.exists(model_path):
        state_dict = torch.load(
            model_path, weights_only=True, map_location=torch.device("cpu")
        )
        return state_dict

    # Input and output embeddings are tied.
    converted_state_dict["output.weight"] = converted_state_dict[
        "tok_embeddings.weight"
    ]
    return converted_state_dict


def convert_weights(input_dir_or_checkpoint: str, output_file: str) -> None:
    try:
        sd = load_checkpoint_from_pytorch_model(input_dir_or_checkpoint)
        print("Converting checkpoint...")
        sd = phi_4_hf_to_meta(sd)
    except FileNotFoundError:
        checkpointer = FullModelHFCheckpointer(
            checkpoint_dir=input_dir_or_checkpoint,
            checkpoint_files=[
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
            output_dir=".",
            model_type="PHI4",
        )
        print("Loading checkpoint from directory...")
        sd = checkpointer.load_checkpoint()
        sd = sd["model"]
        print("Converting checkpoint...")
        sd = phi_4_tune_to_meta(sd)

    print("Saving checkpoint...")
    torch.save(sd, output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Phi-4-mini weights to Meta format."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to directory containing checkpoint files, or path to a single checkpoint file.",
    )
    parser.add_argument("output", type=str, help="Path to the output checkpoint")

    args = parser.parse_args()
    convert_weights(args.input_dir, args.output)


if __name__ == "__main__":
    main()
