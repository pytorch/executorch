import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from executorch.examples.models.checkpoint import get_mapped_key

from safetensors.torch import load_file

_UNSLOTH_TO_META = {
    "base_model.model.model.layers.{}.mlp.down_proj.lora_A.weight": "layers.{}.feed_forward.w2.lora_a.weight",
    "base_model.model.model.layers.{}.mlp.down_proj.lora_B.weight": "layers.{}.feed_forward.w2.lora_b.weight",
    "base_model.model.model.layers.{}.mlp.gate_proj.lora_A.weight": "layers.{}.feed_forward.w1.lora_a.weight",
    "base_model.model.model.layers.{}.mlp.gate_proj.lora_B.weight": "layers.{}.feed_forward.w1.lora_b.weight",
    "base_model.model.model.layers.{}.mlp.up_proj.lora_A.weight": "layers.{}.feed_forward.w3.lora_a.weight",
    "base_model.model.model.layers.{}.mlp.up_proj.lora_B.weight": "layers.{}.feed_forward.w3.lora_b.weight",
    "base_model.model.model.layers.{}.self_attn.k_proj.lora_A.weight": "layers.{}.attention.wk.lora_a.weight",
    "base_model.model.model.layers.{}.self_attn.k_proj.lora_B.weight": "layers.{}.attention.wk.lora_b.weight",
    "base_model.model.model.layers.{}.self_attn.o_proj.lora_A.weight": "layers.{}.attention.wo.lora_a.weight",
    "base_model.model.model.layers.{}.self_attn.o_proj.lora_B.weight": "layers.{}.attention.wo.lora_b.weight",
    "base_model.model.model.layers.{}.self_attn.q_proj.lora_A.weight": "layers.{}.attention.wq.lora_a.weight",
    "base_model.model.model.layers.{}.self_attn.q_proj.lora_B.weight": "layers.{}.attention.wq.lora_b.weight",
    "base_model.model.model.layers.{}.self_attn.v_proj.lora_A.weight": "layers.{}.attention.wv.lora_a.weight",
    "base_model.model.model.layers.{}.self_attn.v_proj.lora_B.weight": "layers.{}.attention.wv.lora_b.weight",
}


def unsloth_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from unsloth format to Meta's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in unsloth format.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    converted_state_dict = {}

    for key, value in state_dict.items():
        try:
            new_key = get_mapped_key(key, _UNSLOTH_TO_META)
        except Exception as e:
            raise ValueError(f"Key {key} not found in mapping") from e

        converted_state_dict[new_key] = value
    return converted_state_dict


def load_and_convert_unsloth_to_meta(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a checkpoint file and convert it to Meta's format.

    Args:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    state_dict = load_file(checkpoint_path)
    return unsloth_to_meta(state_dict)


def convert_hf_to_meta(
    input_dir: str, output_file: str, params_output: str | None = None
) -> None:
    """Convert HuggingFace Llama weights to Meta format, including RoPE ordering."""
    from executorch.examples.models.qwen3.convert_weights import (
        load_checkpoint_from_safetensors,
    )
    from torchtune.models.convert_weights import hf_to_tune, tune_to_meta

    config = json.loads((Path(input_dir) / "config.json").read_text())
    state_dict = load_checkpoint_from_safetensors(input_dir)
    state_dict = tune_to_meta(
        hf_to_tune(
            state_dict,
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            dim=config["hidden_size"],
            head_dim=config.get("head_dim"),
        )
    )
    if config.get("tie_word_embeddings"):
        state_dict["output.weight"] = state_dict["tok_embeddings.weight"]
    torch.save(state_dict, output_file)
    if params_output:
        rope_scaling = config.get("rope_scaling")
        params = {
            "dim": config["hidden_size"],
            "hidden_dim": config["intermediate_size"],
            "n_heads": config["num_attention_heads"],
            "n_kv_heads": config["num_key_value_heads"],
            "n_layers": config["num_hidden_layers"],
            "norm_eps": config["rms_norm_eps"],
            "rope_theta": config["rope_theta"],
            "use_scaled_rope": bool(rope_scaling),
            "rope_scale_factor": rope_scaling["factor"] if rope_scaling else 8,
            "vocab_size": config["vocab_size"],
        }
        Path(params_output).write_text(json.dumps(params, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Llama checkpoints to Meta format."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    hf_parser = subparsers.add_parser(
        "hf-to-meta", help="Convert a HuggingFace snapshot."
    )
    hf_parser.add_argument("input_dir")
    hf_parser.add_argument("output")
    hf_parser.add_argument("--params-output")
    for command, help_text in (
        ("unsloth-to-meta", "Convert a PyTorch Unsloth state dict."),
        (
            "load-and-convert-unsloth-to-meta",
            "Load and convert an Unsloth Safetensors checkpoint.",
        ),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("input")
        command_parser.add_argument("output")
    args = parser.parse_args()
    if args.command == "hf-to-meta":
        convert_hf_to_meta(args.input_dir, args.output, args.params_output)
    elif args.command == "unsloth-to-meta":
        state_dict = torch.load(args.input, map_location="cpu", weights_only=True)
        torch.save(unsloth_to_meta(state_dict), args.output)
    else:
        torch.save(load_and_convert_unsloth_to_meta(args.input), args.output)
