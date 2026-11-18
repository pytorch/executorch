# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
)

# Shared test constants
TEST_MODEL_NAME = "test_model"
TEST_SOC_MODEL = "SM8750"
TEST_PROMPT = ["test"]
TEST_ARTIFACT_DIR = "/tmp/test_artifacts"
TEST_PTE_PATH = Path("/tmp/test.pte")

# Real enum values rather than mocks: both import without a device or the QNN
# SDK, and a mock named after an enum member does not actually carry that
# member's identity, so it cannot catch a wrong value being passed through.
TEST_BACKEND_TYPE = QnnExecuTorchBackendType.kHtpBackend
TEST_SOC_CHIPSET = QcomChipset.SM8750

# The compiled-artifact map a single-graph text-only model produces, in the
# graph-keyed shape the compilation stage emits and inference consumes.
TEST_ARTIFACT_PATHS = {ARTIFACT_TEXT_DECODER: TEST_PTE_PATH}


def make_test_context(**kwargs) -> PipelineContext:
    defaults = {
        "model_name": TEST_MODEL_NAME,
        "soc_model": TEST_SOC_MODEL,
        "prompt": TEST_PROMPT,
        "artifact_dir": TEST_ARTIFACT_DIR,
    }
    defaults.update(kwargs)
    return PipelineContext(**defaults)
