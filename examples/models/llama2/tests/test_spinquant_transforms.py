# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.examples.models.llama2.llama_transformer import ModelArgs, Transformer
from executorch.examples.models.llama2.source_transformation.spin_quant import (
    sanitize_checkpoint_from_spinquant,
    transform_for_spinquant,
)
from torchao.quantization.utils import group_quantize_tensor_symmetric


class SpinQuantTests(unittest.TestCase):
    def test_transforms_for_spinquant(self):

        # Step 1: Create llama class with dummy weights
        params = {
            "dim": 768,
            "multiple_of": 32,
            "n_heads": 12,
            "n_layers": 12,
            "norm_eps": 1e-05,
            "vocab_size": 32000,
        }

        model_args = ModelArgs(
            max_seq_len=2048,
            max_batch_size=1,
            use_kv_cache=False,
            use_sdpa_with_kv_cache_op=False,
            generate_full_logits=False,
            enable_dynamic_shape=True,
            **params,
        )

        model = Transformer(model_args)
        checkpoint = model.state_dict()

        # Step 2:
        # Do group-wise quantization and amend the checkpoints with
        # int8 weight and fp32 scales
        group_size = 32
        n_bit = 4
        scales_precision = torch.float32
        for fqn, mod in model.named_modules():
            # Quantize everything except the last layer
            if isinstance(mod, torch.nn.Linear) and ("output" not in fqn):
                weight = mod.weight.data
                (
                    weight_int8,
                    scales,
                    zeros,
                ) = group_quantize_tensor_symmetric(
                    weight.to(torch.float32), n_bit, group_size, scales_precision
                )
                checkpoint[f"{fqn}.weight"] = weight_int8.to("cpu")
                checkpoint[f"{fqn}.scale"] = scales.to("cpu")

        # Step 3:
        # Transform the model so that it is compatible with the new checkpoint
        transform_for_spinquant(
            model,
            checkpoint,
            32,
            "8da4w",
            torch.float32,
        )
        sanitize_checkpoint_from_spinquant(
            checkpoint,
            -1,
        )

        model.load_state_dict(
            checkpoint,
            strict=False,
            assign=True,
        )

        new_checkpoint = model.state_dict()

        for k, v in checkpoint.items():
            # The new_checkpoint contains zeros so
            # have to iterate over the keys.
            self.assertTrue(torch.allclose(new_checkpoint[k], v))
