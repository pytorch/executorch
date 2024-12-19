# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
import os
from typing import Any, Dict

import torch
from executorch.examples.models.checkpoint import (
    get_checkpoint_dtype,
    get_default_model_resource_dir,
)

from executorch.examples.models.model_base import EagerModelBase
from executorch.extension.llm.modules.attention import replace_mha_with_inference_mha
from torchtune.models.llama3_2_vision._component_builders import llama3_2_vision_decoder
from torchtune.models.llama3_2_vision._convert_weights import llama3_vision_meta_to_tune


def to_decoder_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and formats the decoder-related weights from the checkpoint. The checkpoint contains
    weight names prefixed with "encoder"/"decoder", such as "encoder.layer.etc" or "decoder.norm.scale".
    To load the text decoder on its own, the "decoder" prefix needs to be removed.
    """
    return {
        ".".join(weight.split(".")[1:]): value
        for weight, value in checkpoint.items()
        if weight.startswith("decoder")
    }


class Llama3_2Decoder(EagerModelBase):
    """
    Just the text decoder portions of the Llama3.2 multimodal model.
    """

    def __init__(self, **kwargs):
        # Set member vars from kwargs.
        self.max_seq_len = kwargs.get(
            "max_seq_len", 8192
        )  # Trained to be a lot larger, but this value is kept small because of static kv cache at the moment.
        self.encoder_max_seq_len = kwargs.get(
            "encoder_max_seq_len", int(4 * (448 / 14) ** 2 + 1)
        )  # Same as above.
        self.generate_full_logits = kwargs.get("generate_full_logits", False)
        self.enable_dynamic_shape = kwargs.get("enable_dynamic_shape", False)
        self.output_prune_map_path = kwargs.get("output_prune_map_path", None)
        self.use_kv_cache = kwargs.get("use_kv_cache", False)
        self.verbose = kwargs.get("verbose", False)
        self.args = kwargs.get("args", None)
        self.dtype = kwargs.get("dtype", torch.float16)
        self.use_checkpoint = False

        ckpt_dir = get_default_model_resource_dir(__file__)
        # Single checkpoint file.
        checkpoint_path = kwargs.get("checkpoint", ckpt_dir / "demo_rand_params.pth")
        if os.path.isfile(checkpoint_path):
            self.use_checkpoint = True

        # Sharded checkpoint.
        checkpoint_dir = kwargs.get("checkpoint_dir", None)
        params_path = kwargs.get("params", ckpt_dir / "demo_config.json")

        self.causal_mask = torch.tril(
            torch.ones(
                size=(self.max_seq_len, self.max_seq_len),
                dtype=torch.bool,
            )
        )
        self.input_pos = torch.arange(self.max_seq_len, dtype=torch.int64)

        # Load checkpoint and params.
        device = "cpu"
        if checkpoint_dir is not None:
            raise NotImplementedError(
                "Sharded checkpoint not yet supported for Llama3_2Decoder."
            )
        elif self.use_checkpoint:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False, mmap=True
            )
            checkpoint = llama3_vision_meta_to_tune(checkpoint)
            checkpoint = to_decoder_checkpoint(checkpoint)
            self.dtype = get_checkpoint_dtype(checkpoint)

        with open(params_path, "r") as f:
            params = json.loads(f.read())

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
        self.model_.requires_grad_(False)

        # Source transformation for MultiHeadAttention
        self.model_ = replace_mha_with_inference_mha(self.model_)
        # Save params for future use.
        for param_name, param_val in params.items():
            setattr(self.model_, param_name, param_val)

        # Quantize. (skip for now)

        if self.use_checkpoint:
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
            from executorch.examples.models.llama2.source_transformation.prune_output import (
                prune_output_vocab,
            )

            with open(self.output_prune_map_path, "r") as f:
                output_prune_map = json.load(f)
            # Change keys from string to int (json only supports string keys)
            output_prune_map = {int(k): v for (k, v) in output_prune_map.items()}

            self.model_ = prune_output_vocab(self.model_, output_prune_map)

        if self.use_kv_cache:
            print("Setting up KV cache on the model...")
            self.model_.setup_caches(
                batch_size=1,
                dtype=self.dtype,
                encoder_max_seq_len=self.encoder_max_seq_len,
                decoder_max_seq_len=self.max_seq_len,
            )
        # number of tokens for example input
        self.n_tokens = 34
        self.model_.to(self.dtype)

    def get_eager_model(self) -> torch.nn.Module:
        return self.model_

    def get_example_inputs(self):
        return (torch.ones(1, self.n_tokens, dtype=torch.int64),)

    def get_example_kwarg_inputs(self):
        # For export we must use the prefill versions of the
        # causal mask and input_pos.

        # Make input_pos and mask contiguous in memory.
        input_pos = self.input_pos[None, : self.n_tokens]
        mask = self.causal_mask[None, : self.n_tokens]
        contiguous_input_pos = torch.empty_like(
            input_pos, memory_format=torch.contiguous_format
        )
        contiguous_input_pos.data.copy_(input_pos.data)
        contiguous_mask = torch.empty_like(mask, memory_format=torch.contiguous_format)
        contiguous_mask.data.copy_(mask.data)

        # Hardcoding # of tiles to be 2. image tokens per tile is 1601.
        if self.use_kv_cache:
            return {
                "input_pos": contiguous_input_pos,
                "mask": contiguous_mask,
                "encoder_input": torch.randn(
                    1, self.encoder_max_seq_len, self.model_.dim, dtype=self.dtype
                ),
                "encoder_mask": torch.ones(
                    [1, self.n_tokens, self.encoder_max_seq_len], dtype=torch.bool
                ),
            }
        else:
            return None

    def get_dynamic_shapes(self):
        batch_size = 1
        dim_seq_len = torch.export.Dim("token_dim", min=1, max=self.max_seq_len)
        # Hardcoding # of tiles to be 2. image tokens per tile is 1601.
        if self.use_kv_cache:
            dynamic_shapes = {
                "tokens": {0: batch_size, 1: dim_seq_len},
                "encoder_input": None,
                "encoder_mask": {0: 1, 1: dim_seq_len, 2: None},
                "mask": {0: batch_size, 1: dim_seq_len, 2: None},
                "input_pos": {0: batch_size, 1: dim_seq_len},
            }
        else:
            dynamic_shapes = {
                "tokens": {0: batch_size, 1: dim_seq_len},
            }
        return dynamic_shapes
