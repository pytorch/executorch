# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from executorch_native._native import MethodInfo, Model, Session, TensorInfo

__all__ = ["MethodInfo", "Model", "Session", "TensorInfo", "cmake_prefix_path"]


def cmake_prefix_path() -> str:
    """Return the installed Native SDK prefix for CMake consumers."""
    return str(Path(__file__).resolve().parent)
