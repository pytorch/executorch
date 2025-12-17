import argparse
import os
from typing import Dict

import torch

from torchtune.models.convert_weights import get_mapped_key

# Standard _FROM_META weight mapping of Meta weights to TorchTune + additional bias weight mappings.
_HF__CODEGEN_2_FROM_META = {
    "tok_embeddings.weight": "transformer.wte.weight",
    "layers.{}.attention_norm.weight": "transformer.h.{}.ln_1.weight",
    "layers.{}.attention_norm.bias": "transformer.h.{}.ln_1.bias",
    "layers.{}.attention.wq.weight": "transformer.h.{}.attn.q_proj.weight",
    "layers.{}.attention.wk.weight": "transformer.h.{}.attn.k_proj.weight",
    "layers.{}.attention.wv.weight": "transformer.h.{}.attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "transformer.h.{}.attn.out_proj.weight",
    "layers.{}.feed_forward.fc_in.weight": "transformer.h.{}.mlp.fc_in.weight",
    "layers.{}.feed_forward.fc_in.bias": "transformer.h.{}.mlp.fc_in.bias",
    "layers.{}.feed_forward.fc_out.weight": "transformer.h.{}.mlp.fc_out.weight",
    "layers.{}.feed_forward.fc_out.bias": "transformer.h.{}.mlp.fc_out.bias",
    "norm.weight": "transformer.ln_f.weight",
    "norm.bias": "transformer.ln_f.bias",
    "output.weight": "lm_head.weight",
    "output.bias": "lm_head.bias",
}


def codegen_hf_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    converted_state_dict = {}
    keys_to_remove = []
    for key in state_dict:
        if ".attn.causal_mask" in key:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        state_dict.pop(key)
    inverted_mapping_dict = {v: k for k, v in _HF__CODEGEN_2_FROM_META.items()}
    for key, value in state_dict.items():
        if key.endswith("attn.qkv_proj.weight"):
            mp_num = 8  # This number is from modeling_codegen.py
            dim, dim_kv = value.shape
            block = dim // mp_num
            split_size = block // 3

            qkv_blocks = value.reshape(mp_num, block, dim_kv)
            q_blocks = qkv_blocks[:, 0:split_size, :]
            v_blocks = qkv_blocks[:, split_size : 2 * split_size, :]
            k_blocks = qkv_blocks[:, 2 * split_size : 3 * split_size, :]

            q = q_blocks.reshape(-1, dim_kv)
            v = v_blocks.reshape(-1, dim_kv)
            k = k_blocks.reshape(-1, dim_kv)

            for new_key, new_value in [("q_proj", q), ("k_proj", k), ("v_proj", v)]:
                new_key = key.replace("qkv_proj", new_key)
                new_key = get_mapped_key(new_key, inverted_mapping_dict)
                converted_state_dict[new_key] = new_value
        else:
            mapped_key = get_mapped_key(key, inverted_mapping_dict)
            converted_state_dict[mapped_key] = value

    return converted_state_dict


def convert_weights(input_dir_or_checkpoint: str, output_file: str) -> None:
    pt_path = os.path.join(input_dir_or_checkpoint, "pytorch_model.bin")
    print("Loading checkpoint from file...")
    sd = torch.load(pt_path, map_location="cpu", weights_only=True)
    print("Converting checkpoint...")
    sd = codegen_hf_to_meta(sd)

    print("Saving checkpoint...")
    torch.save(sd, output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Codegen weights to Meta format."
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
