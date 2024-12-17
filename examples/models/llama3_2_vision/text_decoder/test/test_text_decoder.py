# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Export and ExecuTorch tests for CLIP vision encoder are covered by test_models.sh.
# Only test AOTI in this file
import json
import os
import tempfile
import unittest

import torch

from executorch.examples.models.llama3_2_vision.text_decoder.model import (
    Llama3_2Decoder,
)
from torch.testing import assert_close

params = {
    "dim": 2048,
    "ffn_dim_multiplier": 1.3,
    "fusion_interval": 2,
    "intermediate_dim": 14336,
    "multiple_of": 1024,
    "n_heads": 32,
    "n_kv_heads": 8,
    "n_layers": 2,
    "n_special_tokens": 8,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "use_scaled_rope": True,
    "vision_chunk_size": 560,
    "vision_max_num_chunks": 4,
    "vocab_size": 21008,
    "vision_num_cross_attention_layers": 1,
}


class TextDecoderTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def _set_requires_grad_false(self, model: torch.nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = False
        for child in model.children():
            self._set_requires_grad_false(child)

    def test_llama3_2_text_decoder_aoti(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w") as param_file:
            json.dump(params, param_file, indent=2)
            param_file.flush()
            model = Llama3_2Decoder(
                encoder_max_seq_len=6404,
                generate_full_logits=True,
                enable_dynamic_shape=True,
                use_kv_cache=True,
                params=param_file.name,
                dtype=torch.float32,
            )
        encoder = model.get_eager_model().eval()
        self._set_requires_grad_false(encoder)

        # AOTI
        with torch.no_grad(), torch.inference_mode():
            ep = torch.export.export(
                encoder,
                model.get_example_inputs(),
                kwargs=model.get_example_kwarg_inputs(),
                dynamic_shapes=model.get_dynamic_shapes(),
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = torch._inductor.aoti_compile_and_package(
                ep,
                package_path=os.path.join(tmpdir, "text_decoder.pt2"),
            )
            encoder_aoti = torch._inductor.aoti_load_package(path)

            y = encoder_aoti(
                *model.get_example_inputs(), **model.get_example_kwarg_inputs()
            )

        eager_res = encoder.forward(
            *model.get_example_inputs(), **model.get_example_kwarg_inputs()
        )
        assert_close(y, eager_res, rtol=1e-4, atol=1e-4)
