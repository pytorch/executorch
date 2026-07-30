# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for locating ExecuTorch's packaged assets.

``cmake_prefix_path`` points at the directory that contains the installed
ExecuTorch CMake package config, so a C++ project can discover it with:

    cmake -DCMAKE_PREFIX_PATH="$(python -c 'import executorch.utils as u; print(u.cmake_prefix_path)')"
"""

import os as _os

# Mirror torch.utils.cmake_prefix_path: <package_root>/share/cmake. This file
# lives at <package_root>/utils/__init__.py, so go up one level.
cmake_prefix_path = _os.path.join(
    _os.path.dirname(_os.path.dirname(__file__)), "share", "cmake"
)

__all__ = ["cmake_prefix_path"]
