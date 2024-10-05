# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
from typing import Any, Dict, Tuple

import torch

from executorch.examples.models.model_base import EagerModelBase
from torchtune.models.llama3_2_vision._convert_weights import llama3_vision_meta_to_tune
from torchtune.models.llama3_2_vision._component_builders import llama3_2_vision_decoder
from executorch.examples.models.checkpoint import (
    get_default_model_resource_dir,
    get_checkpoint_dtype,
)


def to_decoder_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and formats the decoder-related weights from the checkpoint. The checkpoint contains
    weight names prefixed with "encoder"/"decoder", such as "encoder.layer.etc" or "decoder.norm.scale".
    To load the text decoder on its own, the "decoder" prefix needs to be removed.
    """
    return {".".join(weight.split(".")[1:]): value for weight, value in checkpoint.items() if weight.startswith("decoder")}

class Llama3_2Decoder(EagerModelBase):
    """
    Just the text decoder portions of the Llama3.2 multimodal model.
    """

    def __init__(self, **kwargs):
        # Set member vars from kwargs.
        self.max_seq_len = kwargs.get("max_seq_len", 8192)
        self.encoder_max_seq_len = kwargs.get("encoder_max_seq_len", int(4 * (448 / 14) ** 2 + 1))
        self.generate_full_logits = kwargs.get("generate_full_logits", False)
        self.enable_dynamic_shape = kwargs.get("enable_dynamic_shape", False)
        self.output_prune_map_path = kwargs.get("output_prune_map_path", None)
        # TODO: enable kv cache with TransformerDecoder's setup_cache().
        self.use_kv_cache = kwargs.get("use_kv_cache", False)
        self.use_sdpa_with_kv_cache = kwargs.get("use_sdpa_with_kv_cache", False)
        self.verbose = kwargs.get("verbose", False)
        self.args = kwargs.get("args", None)


        ckpt_dir = get_default_model_resource_dir()
        # Single checkpoint file.
        checkpoint_path = kwargs.get("checkpoint", ckpt_dir / "demo_rand_params.pth")
        # Sharded checkpoint.
        checkpoint_dir = kwargs.get("checkpoint_dir", None)
        params_path = kwargs.get("params", ckpt_dir / "demo_config.json")

        # Load checkpoint and params.
        device = "cpu"
        if checkpoint_dir is not None:
            raise NotImplementedError("Sharded checkpoint not yet supported for Llama3_2Decoder.")
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device, mmap=True)
        checkpoint = llama3_vision_meta_to_tune(checkpoint)
        checkpoint = to_decoder_checkpoint(checkpoint)
        with open(params_path, "r") as f:
            params = json.loads(f.read())

        # Find dtype from checkpoint. (skip for now)
        self.dtype = get_checkpoint_dtype(checkpoint)

        # Load model.
        # Cannot use "with torch.device("meta"):" because it causes some exceptions during export,
        # i.e. the model isn't fully initialized or something.
        self.model_ = llama3_2_vision_decoder(
            vocab_size=params["vocab_size"],
            num_layers=params["n_layers"],
            fusion_interval=params["fusion_interval"],
            num_special_tokens=params["n_special_tokens"],
            num_heads=params["n_heads"],
            num_kv_heads=params["n_kv_heads"],
            embed_dim=params["dim"],
            max_seq_len=self.max_seq_len,
            encoder_max_seq_len=self.encoder_max_seq_len,
            rope_base=params["rope_theta"],
            intermediate_dim=params["intermediate_dim"],
        )
        # Save params for future use.
        for param_name, param_val in params.items():
            setattr(self.model_, param_name, param_val)

        # Quantize. (skip for now)

        # Load checkpoint.
        missing, unexpected = self.model_.load_state_dict(
            checkpoint,
            strict=False,
            assign=True,
        )
        if kwargs.get("verbose", False):
            print("============= missing keys ================")
            print(missing)
            print("============= /missing ================")
            print("============= unexpected keys ================")
            print(unexpected)
            print("============= /unexpected ================")

        # Prune the output layer if output_prune_map is provided.
        output_prune_map = None
        if self.output_prune_map_path is not None:
            from executorch.examples.models.llama2.source_transformation.prune_output import prune_output_vocab

            with open(self.output_prune_map_path, "r") as f:
                output_prune_map = json.load(f)
            # Change keys from string to int (json only supports string keys)
            output_prune_map = {int(k): v for (k, v) in output_prune_map.items()}

            self.model_ = prune_output_vocab(self.model_, output_prune_map)

    def get_eager_model(self) -> torch.nn.Module:
        if self.dtype:
            return self.model_.to(self.dtype)
        else:
            return self.model_.to(torch.float16)

    def get_example_inputs(self) -> Tuple[Tuple, Dict]:
        return (
            (torch.ones(1, 64, dtype=torch.long),), # positional inputs
            {
                # "mask": None,
                # "encoder_input": None,
                # "encoder_mask": None,
                # "input_pos": torch.ones(64, dtype=torch.long),
            } # kwarg inputs
        )

    def get_dynamic_shapes(self):
        dim = torch.export.Dim("token_dim", min=1,max=self.max_seq_len)
        dynamic_shapes = {
            "tokens": {0: 1, 1: dim},
            # "encoder_input": {0:1, 1:dim_enc, 2:4096},
            # "encoder_mask": {0:1, 1:dim, 2:dim_enc},
            # "mask": None,
            # "input_pos" : {0: dim},
        }
        return dynamic_shapes
        
