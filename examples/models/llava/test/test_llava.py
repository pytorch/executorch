# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

import torch
from executorch.examples.models.llava.export_llava import export_all

from executorch.examples.models.llava.model import LlavaModel
from executorch.extension.llm.custom_ops import sdpa_with_kv_cache  # noqa
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)

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
            output_ids = self.llava_model.model.generate(
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

    def test_llava_export(self):
        # export llava and make sure e2e works
        llava_model = LlavaModel(use_sdpa_with_kv_cache_op=True)

        prompt_before_image, resized, prompt_after_image = (
            llava_model.get_inputs_for_prefill()
        )
        executorch_program = export_all(llava_model)
        llava_module = _load_for_executorch_from_buffer(executorch_program.buffer)

        start_pos = 0
        # pte prefill prompt before img
        pte_embeds_before_img = llava_module.run_method(
            "token_embedding", (prompt_before_image,)
        )[0]
        pte_prefill_before_img = llava_module.run_method(
            "text_model",
            (torch.tensor([start_pos], dtype=torch.int64), pte_embeds_before_img),
        )[0]

        start_pos += pte_prefill_before_img.shape[1]

        # pte prefill image
        pte_embeds_img = llava_module.run_method("image_encoder", (resized,))[0]
        pte_prefill_img = llava_module.run_method(
            "text_model",
            (
                torch.tensor([start_pos], dtype=torch.int64),
                pte_embeds_img,
            ),
        )[0]

        start_pos += pte_prefill_img.shape[1]

        # pte prefill prompt after img
        pte_embeds_after_img = llava_module.run_method(
            "token_embedding", (prompt_after_image,)
        )[0]
        pte_prefill_after_img = llava_module.run_method(
            "text_model",
            (torch.tensor([start_pos], dtype=torch.int64), pte_embeds_after_img),
        )[0]

        # being tested, using llama_transformer
        new_tokens = [torch.argmax(pte_prefill_after_img[..., -1, :]).item()]
        self.assertEquals(new_tokens[0], 1932)  # When
        for i in range(4):
            print(i, llava_model.tokenizer.decode(new_tokens[i]))
            token_embeds = llava_module.run_method(
                "token_embedding", (torch.tensor([[new_tokens[i]]], dtype=torch.int64),)
            )[0]
            logits = llava_module.run_method(
                "text_model",
                (torch.tensor([start_pos + i], dtype=torch.int64), token_embeds),
            )[0]
            new_tokens.append(torch.argmax(logits[..., -1, :]).item())

        outputs = llava_model.tokenizer.batch_decode(
            torch.tensor([new_tokens]), skip_special_tokens=True
        )[0].strip()
        print(outputs)
        self.assertEqual(new_tokens.size(), 5)
