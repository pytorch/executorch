# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import errno
import os
from typing import Any

import pytest
import torch


REAL_AND_RANDOM_DATA = {
    "real_data": True,
    "random_data": False,
}

PTQ_AND_QAT_DATA = {
    "ptq": False,
    "qat": True,
}


def download_model_weights(*, repo_id: str, filename: str, revision: str) -> str:
    """Reuse the HF cache, retrying read-only CI caches in writable storage."""
    from huggingface_hub import hf_hub_download  # pyrefly: ignore [missing-import]

    try:
        return hf_hub_download(  # nosec B615
            repo_id=repo_id,
            filename=filename,
            revision=revision,
        )
    except OSError as error:
        runner_temp = os.environ.get("RUNNER_TEMP")
        if error.errno != errno.EROFS or not runner_temp:
            raise

    return hf_hub_download(  # nosec B615
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=os.path.join(runner_temp, "hf_cache", "hub"),
    )


def skip_if_frozen_release(model_name: str):
    refs = (
        os.environ.get("GITHUB_REF", ""),
        os.environ.get("GITHUB_REF_NAME", ""),
        os.environ.get("GITHUB_BASE_REF", ""),
    )
    is_frozen_release = any(
        ref.removeprefix("refs/heads/").startswith("release/") for ref in refs
    )
    return pytest.mark.skipif(
        is_frozen_release,
        reason=f"{model_name} tests depend on resources fetched from main.",
    )


def to_bfloat16(
    model: torch.nn.Module, inputs: tuple[Any, ...]
) -> tuple[torch.nn.Module, tuple[Any, ...]]:
    return model.to(torch.bfloat16), tuple(
        (
            x.to(torch.bfloat16)
            if isinstance(x, torch.Tensor) and x.is_floating_point()
            else x
        )
        for x in inputs
    )
