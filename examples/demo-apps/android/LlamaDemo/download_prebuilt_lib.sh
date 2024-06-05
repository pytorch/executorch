#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

AAR_URL="https://ossci-android.s3.us-west-1.amazonaws.com/executorch/release/0.2.1/executorch-llama.aar"
AAR_SHASUM="2973b1c41aa2c2775482d7cc7c803d0f6ca282c1"

LIBS_PATH="$(dirname "$0")/app/libs"
AAR_PATH="${LIBS_PATH}/executorch-llama.aar"

mkdir -p "$LIBS_PATH"

if [[ ! -f "${AAR_PATH}" || "${AAR_SHASUM}" != $(shasum "${AAR_PATH}" | awk '{print $1}') ]]; then
  curl "${AAR_URL}" -o "${AAR_PATH}"
fi
