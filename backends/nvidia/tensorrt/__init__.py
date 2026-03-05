# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT backend for ExecuTorch.

This module provides TensorRT delegate support for accelerating
PyTorch models on NVIDIA GPUs.
"""

import os
import platform
import sys

# On Jetson (aarch64 with JetPack), make system and user-installed packages
# available inside venvs. This is needed for TensorRT (JetPack system package)
# and optionally for packages like onnx installed via pip at system level.
if platform.machine() == "aarch64" and os.path.exists("/etc/nv_tegra_release"):
    _py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for _pkg_dir in [
        f"/usr/lib/{_py_ver}/dist-packages",
        os.path.expanduser(f"~/.local/lib/{_py_ver}/site-packages"),
    ]:
        if os.path.isdir(_pkg_dir) and _pkg_dir not in sys.path:
            sys.path.append(_pkg_dir)

from executorch.backends.nvidia.tensorrt.backend import TensorRTBackend
from executorch.backends.nvidia.tensorrt.partitioner import TensorRTPartitioner

__all__ = [
    "TensorRTBackend",
    "TensorRTPartitioner",
]
