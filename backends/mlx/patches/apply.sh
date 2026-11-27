#!/bin/bash
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Idempotently apply local patches to the MLX submodule before its build.
#
# Usage: apply.sh <mlx_source_dir> <patch> [<patch> ...]
#
# For each patch: skip it if it is already applied (reverse-check succeeds),
# otherwise apply it. A patch that neither reverse-applies nor forward-applies
# (e.g. context drift after an MLX bump) fails loudly via `set -e`.
set -euo pipefail

mlx_dir="$1"
shift

for patch in "$@"; do
  if git -C "$mlx_dir" apply --reverse --check "$patch" 2>/dev/null; then
    echo "MLX patch already applied, skipping: $patch"
  else
    echo "Applying MLX patch: $patch"
    git -C "$mlx_dir" apply --verbose "$patch"
  fi
done
