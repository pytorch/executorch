#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Before doing anything, cd to the real path of the directory containing this script
# so we avoid repeated symlink segments in downstream CMake paths.
cd -- "$( realpath "$( dirname -- "${BASH_SOURCE[0]}" )" )" &> /dev/null || /bin/true
./run_python_script.sh ./install_executorch.py "$@"

# Install git hooks if inside a git repo
if git rev-parse --git-dir > /dev/null 2>&1; then
    git config core.hooksPath .githooks
fi
