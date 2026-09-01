# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers for selecting CUDA export paths from the requested target."""

import os
import re

import torch


def cuda_targets_are_sm90_or_newer() -> bool:
    """Return whether every requested NVIDIA CUDA target is SM90 or newer.

    Explicit ``TORCH_CUDA_ARCH_LIST`` targets take precedence over the local
    export device. This keeps per-architecture AOT exports deterministic while
    retaining local-device detection for the usual native-export workflow.
    """
    if torch.version.hip is not None:
        return False

    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch_list:
        target_majors = []
        for target in re.split(r"[;,\s]+", arch_list):
            target = target.strip().lower().removeprefix("sm_").removeprefix("compute_")
            target = target.removesuffix("+ptx").removesuffix("a")
            if not target:
                continue
            match = re.fullmatch(r"(\d+)(?:\.(\d+))?", target)
            if match is None:
                return False
            major = int(match.group(1))
            if match.group(2) is None and major >= 10:
                major //= 10
            target_majors.append(major)
        return bool(target_majors) and min(target_majors) >= 9

    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
