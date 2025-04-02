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

from executorch.examples.models.checkpoint import get_checkpoint_dtype
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import Rope, RotaryEmbedding
from executorch.examples.models.model_base import EagerModelBase
from executorch.extension.llm.modules.attention import (
    replace_mha_with_inference_mha,
    replace_rope_with_inference_rope,
)

from torchtune.models import convert_weights

from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE

from torchtune.models.llama3_2._component_builders import lora_llama3_2


class Llama3_2_Lora(EagerModelBase):
    def __init__(self, **kwargs):
        # Set member vars from kwargs.
        self.max_seq_len = kwargs.get(
            "max_seq_len", 8192
        )  # Trained to be a lot larger, but this value is kept small because of static kv cache at the moment.
        # self.encoder_max_seq_len = kwargs.get(
        #     "encoder_max_seq_len", int(4 * (448 / 14) ** 2 + 1)
        # )  # Same as above.
        self.generate_full_logits = kwargs.get("generate_full_logits", False)
        self.enable_dynamic_shape = kwargs.get("enable_dynamic_shape", True)
        self.output_prune_map_path = kwargs.get("output_prune_map_path", None)
        self.use_kv_cache = kwargs.get("use_kv_cache", False)
        self.verbose = kwargs.get("verbose", False)
        self.args = kwargs.get("args", None)
        self.dtype = kwargs.get("dtype", torch.float16)
        self.use_checkpoint = False
        self.max_context_len = kwargs.get("max_context_len", 8192)

        # Single checkpoint file.
        checkpoint_path = kwargs.get("checkpoint")

        if os.path.isfile(checkpoint_path):
            self.use_checkpoint = True

        params_path = kwargs.get("params")
        adapter_path = kwargs.get("adapter")

        # self.input_pos = torch.arange(self.max_seq_len, dtype=torch.int64)
        # Load checkpoint and params.
        device = "cpu"
        if self.use_checkpoint:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False, mmap=True
            )
            checkpoint = convert_weights.meta_to_tune(checkpoint)
            self.dtype = get_checkpoint_dtype(checkpoint)

            adapter = torch.load(
                adapter_path, map_location="cpu", mmap=True, weights_only=False
            )

            checkpoint.update(adapter)

        with open(params_path, "r") as f:
            params = json.loads(f.read())

        # Load model.
        # Cannot use "with torch.device("meta"):" because it causes some exceptions during export,
        # i.e. the model isn't fully initialized or something.
        self.model_ = lora_llama3_2(
            lora_attn_modules=[
                "q_proj",
            ],
            apply_lora_to_mlp=False,
            apply_lora_to_output=False,
            # llama3_2 args
            vocab_size=params["vocab_size"],
            num_layers=params["n_layers"],
            num_heads=params["n_heads"],
            num_kv_heads=params["n_kv_heads"],
            embed_dim=params["dim"],
            max_seq_len=self.max_seq_len,  # 131072
            # intermediate_dim=params["intermediate_dim"], # 8192, calc is 4096
            # LoRA args. TODO take in the adapter config.
            lora_rank=8,
            lora_alpha=16,
        )
        self.model_.requires_grad_(False)
        for param_name, param_val in params.items():
            setattr(self.model_, param_name, param_val)

        setattr(self.model_, "enable_dynamic_shape", self.enable_dynamic_shape)
        # Source transformation for MultiHeadAttention
        self.model_ = replace_mha_with_inference_mha(self.model_)

        model_args: ModelArgs = ModelArgs(
            max_seq_len=self.max_seq_len,
            max_context_len=self.max_context_len,
            use_kv_cache=self.use_kv_cache,
            generate_full_logits=self.generate_full_logits,
            enable_dynamic_shape=self.enable_dynamic_shape,
            **params,
        )
        # Source transformation for RoPE
        # self.model_ = replace_rope_with_inference_rope(self.model_, model_args)

        setattr(self.model_, "checkpoint_dtype", self.dtype)
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

        self.model_.to(self.dtype)
        eg = torch.tensor([[2, 3, 4]], dtype=torch.int64)
        ip = torch.tensor([[0, 1, 2]], dtype=torch.long)
        # self.model_.forward(eg, input_pos=ip)
        # breakpoint()  # 2, OK.
        self.model_.forward(eg, input_pos=ip)

    def get_eager_model(self) -> torch.nn.Module:
        return self.model_

    def get_example_inputs(self):
        return (torch.tensor([[2, 3, 4]], dtype=torch.int64),)
        # return (
        #     torch.tensor([[2, 3, 4]], dtype=torch.long),
        #     {"input_pos": torch.tensor([0], dtype=torch.long)},
        # )
        # return (torch.ones(1, self.n_tokens, dtype=torch.int64),)

    # eg=torch.tensor([[2, 3, 4]], dtype=torch.int64)
    # ip=torch.tensor([[0, 1, 2]], dtype=torch.long)
    def get_example_kwarg_inputs(self):
        return {"input_pos": torch.tensor([[0, 1, 2]], dtype=torch.long)}

    def get_dynamic_shapes(self):
        dim = torch.export.Dim("token_dim", min=1, max=self.max_seq_len - 1)
        return ({1: dim}, {1: dim})
