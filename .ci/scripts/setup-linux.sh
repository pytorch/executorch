#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

read -r BUILD_TOOL BUILD_MODE EDITABLE < <(parse_args "$@")
echo "Build tool: $BUILD_TOOL, Mode: $BUILD_MODE"

# As Linux job is running inside a Docker container, all of its dependencies
# have already been installed, so we use PyTorch build from source here instead
# of nightly. This allows CI to test against latest commits from PyTorch
# The image or PyTorch CI may intentionally provide a different source pin.
torch_pin=$(cd / && "${PYTHON_EXECUTABLE:-python}" -c 'import torch; print(torch.version.git_version)')
if [[ "${EDITABLE:-false}" == "true" ]]; then
  install_executorch --use-pt-pinned-commit --editable
else
  install_executorch --use-pt-pinned-commit
fi
# Dependency installation must preserve the pinned PyTorch build.
TORCH_VERSION="${torch_pin}" bash .ci/scripts/install_pinned_pytorch.sh --verify-only
build_executorch_runner "${BUILD_TOOL}" "${BUILD_MODE}"

if [[ "${GITHUB_BASE_REF:-}" == *main* || "${GITHUB_BASE_REF:-}" == *gh* ]]; then
  do_not_use_nightly_on_ci
fi
