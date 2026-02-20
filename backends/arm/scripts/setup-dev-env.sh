#!/bin/bash
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

git_dir=$(git rev-parse --git-dir)

mkdir -p "$git_dir/hooks"
(
  cd "$git_dir/hooks" || exit 1
  if [ -e "pre-push" ] || [ -L "pre-push" ]; then
      echo "Hook '$git_dir/hooks/pre-push' already exists. Please remove it before running $(basename "$0")."
  else
      ln -s "../../backends/arm/scripts/pre-push" "pre-push"
  fi
  if [ -e "pre-commit" ] || [ -L "pre-commit" ]; then
      echo "Hook '$git_dir/hooks/pre-commit' already exists. Please remove it before running $(basename "$0")."
  else
    ln -s "../../backends/arm/scripts/pre-commit" pre-commit
  fi
)