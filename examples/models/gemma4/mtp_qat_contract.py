# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import hashlib
from typing import Any

import torch


OFFICIAL_QAT_NUM_CENTROIDS: int = 2048
OFFICIAL_QAT_TOKENS_PER_CENTROID: int = 128
OFFICIAL_QAT_CENTROID_TOP_K: int = 32
OFFICIAL_QAT_SELECTED_TOKEN_COUNT: int = (
    OFFICIAL_QAT_CENTROID_TOP_K * OFFICIAL_QAT_TOKENS_PER_CENTROID
)


def validate_qat_token_ordering(ordering: torch.Tensor) -> dict[str, Any]:
    if ordering.numel() != (
        OFFICIAL_QAT_NUM_CENTROIDS * OFFICIAL_QAT_TOKENS_PER_CENTROID
    ):
        raise ValueError("QAT token ordering must contain 262144 entries")
    if ordering.dtype not in (torch.int32, torch.int64):
        raise ValueError("QAT token ordering must use an integer dtype")

    raw_shape = list(ordering.shape)
    canonical = ordering.detach().to(dtype=torch.int64, device="cpu").reshape(-1)
    sorted_values = torch.sort(canonical).values
    expected = torch.arange(canonical.numel(), dtype=torch.int64)
    if not torch.equal(sorted_values, expected):
        raise ValueError("QAT token ordering must be an exact permutation")

    digest = hashlib.sha256(canonical.numpy().tobytes()).hexdigest()
    return {
        "max": int(canonical.max().item()),
        "min": int(canonical.min().item()),
        "numel": canonical.numel(),
        "permutationExact": True,
        "rawShape": raw_shape,
        "sha256": digest,
        "shape": [OFFICIAL_QAT_NUM_CENTROIDS, OFFICIAL_QAT_TOKENS_PER_CENTROID],
        "uniqueCount": canonical.numel(),
    }
