#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

AAR_URL="https://ossci-android.s3.us-west-1.amazonaws.com/executorch/release/0.3/executorch-llama.aar"
AAR_SHASUM="f06cc1606e5e05f00fd0ae721f5d37d56124fd28"

LIBS_PATH="$(dirname "$0")/app/libs"
AAR_PATH="${LIBS_PATH}/executorch-llama.aar"

mkdir -p "$LIBS_PATH"

if [[ ! -f "${AAR_PATH}" || "${AAR_SHASUM}" != $(shasum "${AAR_PATH}" | awk '{print $1}') ]]; then
  curl "${AAR_URL}" -o "${AAR_PATH}"
fi
