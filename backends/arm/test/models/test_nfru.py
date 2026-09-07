# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pytest
import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineINT,
    VgfPipeline,
)
from huggingface_hub import hf_hub_download
from ng_model_gym.usecases.nfru.model.nfru_v1_nn import (  # type: ignore[import-not-found,import-untyped]
    NFRUAutoEncoder,
)


_RELEASE_REFS = (
    os.environ.get("GITHUB_REF", ""),
    os.environ.get("GITHUB_REF_NAME", ""),
    os.environ.get("GITHUB_BASE_REF", ""),
)
_IS_FROZEN_RELEASE = any(
    ref.removeprefix("refs/heads/").startswith("release/") for ref in _RELEASE_REFS
)
pytestmark = pytest.mark.skipif(
    _IS_FROZEN_RELEASE,
    reason="NFRU tests depend on resources fetched from main.",
)

_NFRU_QAT_INPUTS = (torch.ones((1, 16, 64, 64)),)


def nfru() -> NFRUAutoEncoder:
    weights = hf_hub_download(  # nosec B615
        repo_id="Arm/neural-frame-rate-upscaling",
        filename="nfru_v1_fp32.pt",
        revision="main",
        cache_dir=os.environ.get("RUNNER_TEMP"),
    )
    checkpoint = torch.load(
        weights,
        map_location=torch.device("cpu"),
        weights_only=True,
    )["model_state_dict"]
    prefix = "network.auto_encoder."
    assert all(key.startswith(prefix) for key in checkpoint)

    model = NFRUAutoEncoder()
    model.load_state_dict(
        {key.removeprefix(prefix): value for key, value in checkpoint.items()},
        strict=True,
    )
    return model


def test_nfru_qat_tosa_INT() -> None:
    pipeline = TosaPipelineINT(
        nfru(),
        _NFRU_QAT_INPUTS,
        aten_op=[],
        exir_op=[],
        per_channel_quantization=False,
        use_to_edge_transform_and_lower=True,
        is_qat=True,
        frobenius_threshold=None,
        cosine_threshold=None,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_nfru_qat_vgf_INT() -> None:
    pipeline = VgfPipeline(
        nfru(),
        _NFRU_QAT_INPUTS,
        aten_op=[],
        exir_op=[],
        run_on_vulkan_runtime=True,
        quantize=True,
        per_channel_quantization=False,
        use_to_edge_transform_and_lower=True,
        is_qat=True,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
