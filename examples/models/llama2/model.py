# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import torch

from executorch.examples.models.llama2.llama_transformer import ModelArgs, Transformer

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
        import pkg_resources

        # default path to the resource file
        # It currently supports 3 ways of specifying the checkpoint location:
        # 1. Using default path locates in examples/models/llama2/params
        # 2. Passing in the checkpoint path and params via kwargs
        # 3. Using the path from pkg_resources, only works with buck2
        try:
            # The 3rd way, if we can import this path, we are running with buck2, all resources can be accessed with pkg_resources.resource_filename
            # pyre-ignore
            from executorch.examples.models.llama2 import params

            ckpt_dir = Path(
                pkg_resources.resource_filename(
                    "executorch.examples.models.llama2", "params"
                )
            )
        except:
            # The 1st way
            ckpt_dir = Path(__file__).absolute().parent / "params"

        checkpoint_path = (
            kwargs["checkpoint"]
            if "checkpoint" in kwargs
            else ckpt_dir / "demo_rand_params.pth"
        )

        params_path = (
            kwargs["params"] if "params" in kwargs else ckpt_dir / "demo_config.json"
        )

        self.use_kv_cache = (
            kwargs["use_kv_cache"] if "use_kv_cache" in kwargs else False
        )
        self.use_sdpa_with_kv_cache_op = (
            kwargs["use_sdpa_with_kv_cache"]
            if "use_sdpa_with_kv_cache" in kwargs
            else False
        )

        self.max_seq_len = kwargs["max_seq_len"] if "max_seq_len" in kwargs else 128
        # The example is using a dummy small model with random weights for demo purpose only.
        # Follow the instruction in https://github.com/facebookresearch/llama to download the model
        device = "cpu"
        # flake8: noqa: TOR102
        checkpoint = torch.load(checkpoint_path, map_location=device, mmap=True)
        fairseq2_checkpoint = kwargs.get("fairseq2", False)
        if fairseq2_checkpoint:
            print("Using fairseq2 checkpoint")
            checkpoint = convert_to_llama_checkpoint(checkpoint=checkpoint)
        if "model" in checkpoint:
            # NB: some checkpoint contains a "model" field, which is the actual weights dict
            checkpoint = checkpoint["model"]

        if (not fairseq2_checkpoint) and checkpoint.get(
            "final_proj.weight", None
        ) is not None:
            print(
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

        # get checkpoint dtype
        self.dtype = None
        if len(checkpoint) > 0:
            first = checkpoint[next(iter(checkpoint))]
            self.dtype = first.dtype
            mismatched_dtypes = [
                (key, value.dtype)
                for key, value in checkpoint.items()
                if value.dtype != self.dtype
            ]
            if len(mismatched_dtypes) > 0:
                print(
                    f"Mixed dtype model. Dtype of {first.key}: {first.dtype}. Mismatches in the checkpoint: {mismatched_dtypes}"
                )
        with open(params_path, "r") as f:
            params = json.loads(f.read())
        max_seq_len = self.max_seq_len
        max_batch_size = 1
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            use_kv_cache=self.use_kv_cache,
            use_sdpa_with_kv_cache_op=self.use_sdpa_with_kv_cache_op,
            **params,
        )
        if kwargs.get("fairseq2", False):
            print("Using fairseq2 checkpoint")
            checkpoint = convert_to_llama_checkpoint(checkpoint=checkpoint)
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
            from .quantize import WeightOnlyInt8QuantHandler

            simple_quantizer = WeightOnlyInt8QuantHandler(self.model_)
            self.model_ = simple_quantizer.convert_for_runtime()
        elif "8da4w" in str(checkpoint_path):
            print("Using int4 weight and int8 dynamic activation quantization!")
            from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

            self.model_ = Int8DynActInt4WeightQuantizer()._convert_for_runtime(
                self.model_
            )

        # assign=True: load params/buffers by assignment instead of performing an in-place copy.
        # Because we are using device="meta", tensors do not have memory associated with them
        # and an in-place copy is a no-op. Use assign=True in load_state_dict for this scenario.
        self.model_.load_state_dict(
            checkpoint,
            strict=False,
            assign=True,
        )  # self.model_ = Transformer(gptconf)

    def get_eager_model(self):
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
        return (
            torch.tensor(
                [[1]], dtype=torch.long
            ),  # tokens, with kv cache our input token length is always just 1 token.
            torch.tensor(
                [0], dtype=torch.long
            ),  # start_pos, what token of output are we on.)
        )
