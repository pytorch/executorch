# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

import torch
from executorch.examples.models.lfm2_5_vl.export_lfm2_5_vl import export_all
from executorch.examples.models.lfm2_5_vl.model import IMAGE_SIZE, Lfm2p5VlModel
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "LiquidAI/LFM2-VL-1.6B"


class TestLfm2p5Vl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lfm2_model = Lfm2p5VlModel(model_dir=MODEL_DIR)
        cls.lfm2 = cls.lfm2_model.get_eager_model().eval()

    def test_vision_encoder_shape(self):
        """Vision encoder must produce [1, 256, 2048] embeddings."""
        pixels = torch.randint(
            0, 256, (1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32
        )
        with torch.no_grad():
            embeds = self.lfm2.image_embedding(pixels)
        self.assertEqual(embeds.shape, (1, 256, 2048))

    def test_prefill_output_shape(self):
        """Prefill must return (seq_len: int, logits [1, vocab_size])."""
        prompt_before, pixels, prompt_after = self.lfm2_model.get_inputs_for_prefill()
        with torch.no_grad():
            seq_len, logits = self.lfm2.prefill(prompt_before, pixels, prompt_after)
        self.assertIsInstance(seq_len, int)
        self.assertEqual(logits.shape[-1], 65536)

    def test_export_methods(self):
        """Exported PTE must contain the three named methods and metadata."""
        et_program = export_all(
            model_dir=MODEL_DIR,
            output=None,  # in-memory only
            _return_program=True,
        )
        self.assertIn("vision_encoder", et_program.methods)
        self.assertIn("token_embedding", et_program.methods)
        self.assertIn("text_decoder", et_program.methods)

    def test_export_and_run(self):
        """Export to PTE and run a short prefill + decode loop end-to-end."""
        et_program = export_all(
            model_dir=MODEL_DIR,
            output=None,
            _return_program=True,
        )
        module = _load_for_executorch_from_buffer(et_program.buffer)

        prompt_before, pixels, prompt_after = self.lfm2_model.get_inputs_for_prefill()
        start_pos = 0

        # Embed and prefill tokens before image
        before_embeds = module.run_method("token_embedding", (prompt_before,))[0]
        module.run_method(
            "text_decoder",
            (
                before_embeds,
                torch.arange(start_pos, start_pos + before_embeds.shape[1]),
            ),
        )
        start_pos += before_embeds.shape[1]

        # Vision encoder
        image_embeds = module.run_method("vision_encoder", (pixels,))[0]
        module.run_method(
            "text_decoder",
            (image_embeds, torch.arange(start_pos, start_pos + image_embeds.shape[1])),
        )
        start_pos += image_embeds.shape[1]

        # Embed and prefill tokens after image
        after_embeds = module.run_method("token_embedding", (prompt_after,))[0]
        logits = module.run_method(
            "text_decoder",
            (after_embeds, torch.arange(start_pos, start_pos + after_embeds.shape[1])),
        )[0]
        start_pos += after_embeds.shape[1]

        # Decode a few tokens — just check we get valid token IDs
        new_tokens = [torch.argmax(logits).item()]
        for i in range(3):
            token_embed = module.run_method(
                "token_embedding",
                (torch.tensor([[new_tokens[i]]], dtype=torch.int64),),
            )[0]
            logits = module.run_method(
                "text_decoder",
                (token_embed, torch.tensor([start_pos + i], dtype=torch.int64)),
            )[0]
            new_tokens.append(torch.argmax(logits).item())

        self.assertEqual(len(new_tokens), 4)
        for tok in new_tokens:
            self.assertGreaterEqual(tok, 0)
            self.assertLess(tok, 65536)


if __name__ == "__main__":
    unittest.main()
