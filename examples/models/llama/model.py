# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
import os
from typing import Dict, Tuple

import torch
from executorch.examples.models.checkpoint import (
    get_checkpoint_dtype,
    get_default_model_resource_dir,
)

from executorch.examples.models.llama.llama_transformer import ModelArgs, Transformer

try:
    from .fairseq2 import convert_to_llama_checkpoint

except ImportError:

    def convert_to_llama_checkpoint(**kwargs):
        raise NotImplementedError(
            "Please install fairseq2 with `pip install fairseq2`."
        )


from ..model_base import EagerModelBase


class Llama2Model(EagerModelBase):
    def __init__(self, **kwargs):
        resource_dir = get_default_model_resource_dir(__file__)

        # Use single checkpoint file.
        checkpoint_path = kwargs.get(
            "checkpoint", resource_dir / "demo_rand_params.pth"
        )
        params_path = kwargs.get("params", resource_dir / "demo_config.json")

        # Check if checkpoint_dir was provided for a sharded checkpoint.
        checkpoint_dir = kwargs.get("checkpoint_dir", None)

        self.use_kv_cache = kwargs.get("use_kv_cache", False)
        self.use_sdpa_with_kv_cache_op = kwargs.get("use_sdpa_with_kv_cache", False)
        self.generate_full_logits = kwargs.get("generate_full_logits", False)
        self.enable_dynamic_shape = kwargs.get("enable_dynamic_shape", False)
        self.input_prune_map_path = kwargs.get("input_prune_map_path", None)
        self.output_prune_map_path = kwargs.get("output_prune_map_path", None)
        self.max_seq_len = kwargs.get("max_seq_len", 128)
        self.args = kwargs.get("args", None)

        # The example is using a dummy small model with random weights for demo purpose only.
        # Follow the instruction in https://github.com/facebookresearch/llama to download the model.
        device = "cpu"
        # flake8: noqa: TOR102
        cps = []
        # Load sharded checkpoint.
        if checkpoint_dir is not None:
            # Load multiple checkpoint; ignore the single path.
            checkpoint_path = None
            for i in range(4):
                cp_name = f"consolidated.{i}.pth"
                print(f"Loading {cp_name}")
                cps.append(
                    torch.load(
                        os.path.join(checkpoint_dir, cp_name),
                        map_location=device,
                        mmap=True,
                    )
                )
            checkpoint = {}
            for key in cps[0].keys():
                if not torch.allclose(cps[0][key], cps[1][key]):
                    values = (cps[0][key], cps[1][key], cps[2][key], cps[3][key])
                    if "wo" in key or "w2" in key:
                        # Concat on dim=1 for "wo" and "w2".
                        checkpoint[key] = torch.cat(values, dim=1)
                    else:
                        # Concat on dim=0 for everything else.
                        checkpoint[key] = torch.cat(values, dim=0)
                else:
                    # Do not duplicate layers shared between each checkpoint.
                    checkpoint[key] = cps[0][key]
        # Load single checkpoint.
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device, mmap=True)

        # If given checkpoint is fairseq, convert to llama checkpoint.
        fairseq2_checkpoint = kwargs.get("fairseq2", False)
        if fairseq2_checkpoint:
            print("Using fairseq2 checkpoint")
            checkpoint = convert_to_llama_checkpoint(checkpoint=checkpoint)
        if "model" in checkpoint:
            # NB: some checkpoint contains a "model" field, which is the actual weights dict
            checkpoint = checkpoint["model"]

        # Check if user gave a fairseq2 checkpoint unknowingly without specifying --fairseq2.
        if (not fairseq2_checkpoint) and checkpoint.get(
            "final_proj.weight", None
        ) is not None:
            raise ValueError(
                """
************************************************************
This looks like a Fairseq2 checkpoint (based on the presence
of `final_proj.weight`.

You can import Fairseq2 checkpoints using the --fairseq2
option, but --fairseq2 was not specified.  Please verify
the checkpoint format to avoid generating faulty models.
************************************************************
"""
            )

        # Get checkpoint dtype.
        self.dtype = get_checkpoint_dtype(checkpoint)

        with open(params_path, "r") as f:
            params = json.loads(f.read())
        output_prune_map = None
        if self.output_prune_map_path is not None:
            with open(self.output_prune_map_path, "r") as f:
                output_prune_map = json.load(f)
            # Change keys from string to int (json only supports string keys).
            output_prune_map = {int(k): v for (k, v) in output_prune_map.items()}
        input_prune_map = None
        if self.input_prune_map_path is not None:
            with open(self.input_prune_map_path, "r") as f:
                input_prune_map = json.load(f)
            # Change keys from string to int (json only supports string keys).
            input_prune_map = {int(k): v for (k, v) in input_prune_map.items()}

        model_args: ModelArgs = ModelArgs(
            max_seq_len=self.max_seq_len,
            max_batch_size=1,
            use_kv_cache=self.use_kv_cache,
            use_sdpa_with_kv_cache_op=self.use_sdpa_with_kv_cache_op,
            generate_full_logits=self.generate_full_logits,
            input_prune_map=input_prune_map,
            output_prune_map=output_prune_map,
            enable_dynamic_shape=self.enable_dynamic_shape,
            **params,
        )

        if model_args.use_scaled_rope:
            # Older models don't have use_scaled_rope configuration
            assert self.args.model not in ["llama2", "stories110m"]

            # Llama3_2 and newer models in ExecuTorch repo should set larger scale factor
            if self.args.model not in ["llama3", "llama3_1"]:
                model_args.rope_scale_factor = 32

        if kwargs.get("verbose", False):
            print("============= weights ================")
            print("{key} : {weights.numel()} : {weights.size()}")
            for key, weights in checkpoint.items():
                print(f"{key} : {weights.numel()} : {weights.size()}")
            print("============= /weights ================")

        # Within the device="meta" context, tensors that are created do not carry data.
        # They possess all other metadata a tensor carries such as size, stride, requires_grad.
        with torch.device("meta"):
            self.model_ = Transformer(model_args)

        if "int8" in str(checkpoint_path):
            print("Using int8 weight-only quantization!")
            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.examples.models.source_transformation.quantize`
            from ..source_transformation.quantize import WeightOnlyInt8QuantHandler

            simple_quantizer = WeightOnlyInt8QuantHandler(self.model_)
            self.model_ = simple_quantizer.convert_for_runtime()
        elif "8da4w" in str(checkpoint_path):
            print("Using int4 weight and int8 dynamic activation quantization!")
            from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

            self.model_ = Int8DynActInt4WeightQuantizer()._convert_for_runtime(
                self.model_
            )
        elif hasattr(self.args, "use_spin_quant") and self.args.use_spin_quant:
            print("Using SPIN quantization.")
            self._transform_for_pre_quantization(checkpoint, model_args)

            from .source_transformation.pre_quantization import (
                sanitize_checkpoint_from_pre_quantization,
            )

            sanitize_checkpoint_from_pre_quantization(checkpoint)
        elif hasattr(self.args, "use_qat") and self.args.use_qat:
            print("Using QAT quantization.")
            self._transform_for_pre_quantization(checkpoint, model_args)
            if hasattr(self.args, "use_lora") and self.args.use_lora:
                assert model_args.lora_args["rank"] == self.args.use_lora
                from .source_transformation.lora import (
                    transform_linear_for_lora_after_quantization,
                )

                self.model_ = transform_linear_for_lora_after_quantization(
                    self.model_,
                    checkpoint,
                    self.args.use_lora,
                )

            from .source_transformation.pre_quantization import (
                sanitize_checkpoint_from_pre_quantization,
            )

            sanitize_checkpoint_from_pre_quantization(checkpoint)

        if hasattr(self.args, "use_attention_sink") and self.args.use_attention_sink:
            from .source_transformation.attention_sink import enable_attention_sink

            attention_sink_params = self.args.use_attention_sink.split(",")
            assert len(attention_sink_params) == 3
            sink_size = int(attention_sink_params[0])
            window_size = int(attention_sink_params[1])
            eviction_batch_size = int(attention_sink_params[2])

            assert self.args.max_seq_length == sink_size + window_size

            self.model_ = enable_attention_sink(
                module=self.model_,
                params=model_args,
                sink_size=sink_size,
                window_size=window_size,
                eviction_batch_size=eviction_batch_size,
            )

        # assign=True: load params/buffers by assignment instead of performing an in-place copy.
        # Because we are using device="meta", tensors do not have memory associated with them
        # and an in-place copy is a no-op. Use assign=True in load_state_dict for this scenario.
        missing, unexpected = self.model_.load_state_dict(
            checkpoint,
            strict=False,
            assign=True,
        )  # self.model_ = Transformer(gptconf)
        if kwargs.get("verbose", False):
            print("============= missing keys ================")
            print(missing)
            print("============= /missing ================")
            print("============= unexpected keys ================")
            print(unexpected)
            print("============= /unexpected ================")

        # Prune the input layer if input_prune_map is provided
        if input_prune_map is not None:
            from .source_transformation.prune_vocab import prune_input_vocab

            self.model_ = prune_input_vocab(self.model_, input_prune_map)

        # Prune the output layer if output_prune_map is provided
        if output_prune_map is not None:
            from .source_transformation.prune_vocab import prune_output_vocab

            self.model_ = prune_output_vocab(self.model_, output_prune_map)

    def get_eager_model(self) -> torch.nn.Module:
        if self.dtype:
            # convert to the type of the provided checkpoint
            # input and output are torch.long, so signature unchanged
            return self.model_.to(self.dtype)
        else:
            # int8 quantization code has some bf16,
            # switch all to FP32
            return self.model_.to(torch.float32)

    def get_example_inputs(self):
        if self.use_kv_cache:
            return self.get_example_inputs_kvcache_sdpa()
        else:
            return (
                torch.tensor(
                    [[1, 2, 3]], dtype=torch.long
                ),  # tokens, with kv cache our input token length is always just 1 token.
            )

    # assumption is the custom op doesnt support dynamic shape right now. It might but its untested so lets first get static shape working
    def get_example_inputs_kvcache_sdpa(self):
        if self.enable_dynamic_shape:
            return (
                torch.tensor([[2, 3, 4]], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
            )
        else:
            return (
                torch.tensor(
                    [[1]], dtype=torch.long
                ),  # tokens, with kv cache our input token length is always just 1 token.
                torch.tensor(
                    [0], dtype=torch.long
                ),  # start_pos, what token of output are we on.
            )

    def _transform_for_pre_quantization(self, checkpoint, model_args):
        assert hasattr(self.args, "preq_mode"), "preq_mode must be specified"
        assert self.args.preq_mode in [
            "8da4w",
            "8da4w_output_8da8w",
        ], f"Quantization mode {self.args.preq_mode} is not compatible with SpinQuant."
        assert hasattr(
            self.args, "preq_group_size"
        ), "preq_group_size must be specified"
        assert hasattr(self.args, "dtype_override"), "dtype_override must be specified"
        from .source_transformation.pre_quantization import (
            transform_linear_for_pre_quantization,
        )

        assert self.args.preq_group_size == model_args.quantization_args["group_size"]

        mapping = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }

        # Transform the output layer first if needed.
        if self.args.preq_mode == "8da4w_output_8da8w":
            from .source_transformation.pre_quantization import (
                transform_output_linear_for_pre_quantization,
            )

            self.model_ = transform_output_linear_for_pre_quantization(
                module=self.model_,
                checkpoint=checkpoint,
                dtype=mapping[self.args.dtype_override],
            )

        self.model_ = transform_linear_for_pre_quantization(
            self.model_,
            checkpoint,
            self.args.preq_group_size,
            mapping[self.args.dtype_override],
        )

        embedding_bit_width, embedding_group_size = None, None
        if hasattr(self.args, "preq_embedding_quantize"):
            embedding_bit_width, embedding_group_size = (
                self.args.preq_embedding_quantize.split(",")
            )
            from .source_transformation.pre_quantization import (
                transform_embedding_for_pre_quantization,
            )

            if (
                embedding_group_size == "none"
                or embedding_group_size == "None"
                or embedding_group_size == "0"
            ):
                embedding_group_size = None
            else:
                embedding_group_size = int(embedding_group_size)

            self.model_ = transform_embedding_for_pre_quantization(
                self.model_,
                checkpoint,
                mapping[self.args.dtype_override],
                int(embedding_bit_width),
                embedding_group_size,
            )
