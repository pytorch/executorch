#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

AAR_URL="https://ossci-android.s3.us-west-1.amazonaws.com/executorch/release/executorch-241002/executorch.aar"
AAR_SHASUM_URL="https://ossci-android.s3.us-west-1.amazonaws.com/executorch/release/executorch-241002/executorch.aar.sha256sums"
LIBS_PATH="$(dirname "$0")/app/libs"

mkdir -p "$LIBS_PATH"

pushd "$LIBS_PATH"
curl -O "${AAR_SHASUM_URL}"
shasum --check --status executorch.aar.sha256sums || curl "${AAR_URL}" -o executorch.aar
popd
