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

# On Jetson (aarch64 with JetPack), use system-installed TensorRT instead of pip.
if platform.machine() == "aarch64" and os.path.exists("/etc/nv_tegra_release"):
    _py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    _system_dist_packages = f"/usr/lib/{_py_ver}/dist-packages"
    if os.path.isdir(_system_dist_packages) and _system_dist_packages not in sys.path:
        sys.path.append(_system_dist_packages)

from executorch.backends.nvidia.tensorrt.backend import TensorRTBackend
from executorch.backends.nvidia.tensorrt.partitioner import TensorRTPartitioner

__all__ = [
    "TensorRTBackend",
    "TensorRTPartitioner",
]
