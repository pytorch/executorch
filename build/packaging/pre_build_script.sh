#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euxo pipefail

# This script is run before building ExecuTorch binaries

# Manually install build requirements because `python setup.py bdist_wheel` does
# not install them. TODO(dbort): Switch to using `python -m build --wheel`,
# which does install them. Though we'd need to disable build isolation to be
# able to see the installed torch package.
readonly BUILD_DEPS=(
  # This list must match the build-system.requires list from pyproject.toml.
  "cmake"
  "pip>=23"
  "pyyaml"
  "setuptools>=63"
  "tomli"
  "wheel"
  "zstd"
)
pip install --progress-bar off "${BUILD_DEPS[@]}"
