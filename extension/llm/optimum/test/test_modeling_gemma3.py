# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.extension.llm.optimum.image_text_to_text import (
    load_image_text_to_text_model,
)
from executorch.extension.llm.optimum.modeling import (
    ExecuTorchModelForImageTextToTextCausalLM,
)
from executorch.extension.llm.optimum.xnnpack import export_to_executorch_with_xnnpack
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def test_gemma3_image_text_to_text_generation_with_custom_sdpa_kv_cache_8da4w_8we(
        self,
    ):

        model_id = "google/gemma-3-4b-it"

        module = load_image_text_to_text_model(
            model_id,
            use_custom_sdpa=True,
            use_custom_kv_cache=True,
            qlinear=True,
            qembedding=True,
        )
        model = export_to_executorch_with_xnnpack(module)
        et_model = ExecuTorchModelForImageTextToTextCausalLM(
            model, PretrainedConfig.from_pretrained(model_id)
        )
        # Generate
        image_url = "https://llava-vl.github.io/static/images/view.jpg"
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_url},
                    {
                        "type": "text",
                        "text": "What are the things I should be cautious about when I visit here?",
                    },
                ],
            },
        ]
        processor = AutoProcessor.from_pretrained(model_id)
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        output = et_model.generate(
            AutoTokenizer.from_pretrained(model_id),
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=50,
        )
        self.assertEqual(
            output,
            """Okay, let's analyze the image and discuss potential cautions for visiting this location.

Based on the picture, we're looking at a serene lake scene with mountains in the background, a wooden pier, and a generally calm appearance. However""",
        )
