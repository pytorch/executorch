# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

import torch

from executorch.examples.models.llava.model import LlavaModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestLlava(unittest.TestCase):
    def setUp(self):
        self.llava_model = LlavaModel()
        self.llava = self.llava_model.get_eager_model()
        self.prompt_before_image, self.resized, self.prompt_after_image = (
            self.llava_model.get_inputs_for_prefill()
        )

    def test_prefill_logits(self):
        prefill_logits = self.llava.prefill(
            self.prompt_before_image, self.resized, self.prompt_after_image
        )
        prefill_logits_ref = self.llava.prefill_ref(
            self.prompt_before_image, self.resized, self.prompt_after_image
        )[0]
        self.assertTrue(torch.allclose(prefill_logits, prefill_logits_ref, atol=3e-2))

    def test_generated_output(self):
        # source of truth, using HF llava
        preprocessed = self.llava.image_preprocess(self.resized)
        with torch.inference_mode():
            output_ids = self.llava_model.generate(
                self.llava_model.input_ids,
                images=preprocessed,
                image_sizes=[preprocessed.size],
                do_sample=False,
                num_beams=1,
                max_new_tokens=5,
                use_cache=True,
            )

        ref_outputs = self.llava_model.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

        # being tested, using llama_transformer
        prefill_logits = self.llava.prefill(
            self.prompt_before_image, self.resized, self.prompt_after_image
        )
        context_len = prefill_logits.shape[1]
        new_tokens = [torch.argmax(prefill_logits[..., -1, :]).item()]
        for i in range(4):
            logits = self.llava.step(
                torch.tensor([new_tokens[i]]), torch.tensor([context_len + i])
            )
            new_tokens.append(torch.argmax(logits[-1, :]).item())

        outputs = self.llava_model.tokenizer.batch_decode(
            torch.tensor([new_tokens]), skip_special_tokens=True
        )[0].strip()
        self.assertEqual(outputs, ref_outputs)
