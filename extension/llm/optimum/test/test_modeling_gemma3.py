# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from optimum.exporters.executorch.tasks.image_text_to_text import (
    load_image_text_to_text_model,
)
from optimum.executorch import ExecuTorchModelForCausalLM
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
        executorch_program = export_to_executorch_with_xnnpack(module)
        
        # Verify the program was created successfully
        self.assertIsNotNone(executorch_program)
        
        # Note: For actual usage, use optimum-executorch API:
        # model = ExecuTorchModelForCausalLM.from_pretrained(
        #     model_id, task="image-text-to-text", recipe="xnnpack",
        #     use_custom_sdpa=True, use_custom_kv_cache=True
        # )
        # This test demonstrates ExecuTorch-specific XNNPACK optimizations
