#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
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
if [[ "${EDITABLE:-false}" == "true" ]]; then
  install_executorch --use-pt-pinned-commit --editable
else
  install_executorch --use-pt-pinned-commit
fi
build_executorch_runner "${BUILD_TOOL}" "${BUILD_MODE}"

if [[ "${GITHUB_BASE_REF:-}" == *main* || "${GITHUB_BASE_REF:-}" == *gh* ]]; then
  do_not_use_nightly_on_ci
fi
