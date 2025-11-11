#!/bin/bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

git_dir=$(git rev-parse --git-dir)
ln $git_dir/../backends/arm/scripts/pre-push $git_dir/hooks
ln $git_dir/../backends/arm/scripts/pre-commit $git_dir/hooks