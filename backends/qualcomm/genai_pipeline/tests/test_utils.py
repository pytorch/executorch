# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext

# Shared test constants
TEST_MODEL_NAME = "test_model"
TEST_SOC_MODEL = "SM8750"
TEST_PROMPT = ["test"]
TEST_ARTIFACT_DIR = "/tmp/test_artifacts"
TEST_PTE_PATH = Path("/tmp/test.pte")


def make_test_context(**kwargs) -> PipelineContext:
    defaults = {
        "model_name": TEST_MODEL_NAME,
        "soc_model": TEST_SOC_MODEL,
        "prompt": TEST_PROMPT,
        "artifact_dir": TEST_ARTIFACT_DIR,
    }
    defaults.update(kwargs)
    return PipelineContext(**defaults)
