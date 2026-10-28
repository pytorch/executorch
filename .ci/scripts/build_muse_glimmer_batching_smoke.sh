#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run after the normal batching build; reuse its backend and package settings.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="${1:-${ROOT}/cmake-out/examples/models/muse-glimmer-batching}"
cmake -S "${ROOT}/examples/models/muse-glimmer" -B "$BUILD_DIR" \
  -DMUSE_GLIMMER_BUILD_BATCHING=ON -DMUSE_GLIMMER_BUILD_CI_SMOKE=ON
cmake --build "$BUILD_DIR" --target muse_glimmer_batching_smoke --parallel
