#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

AAR_URL="https://gha-artifacts.s3.amazonaws.com/pytorch/executorch/9357260259/artifact/executorch.aar"
AAR_SHASUM="2081b318fefe105e5f92249350c4551a1f3826ec"

LIBS_PATH="$(dirname "$0")/app/libs"
AAR_PATH="${LIBS_PATH}/executorch-llama.aar"

mkdir -p "$LIBS_PATH"

if [[ ! -f "${AAR_PATH}" || "${AAR_SHASUM}" != $(shasum "${AAR_PATH}" | awk '{print $1}') ]]; then
  curl "${AAR_URL}" -o "${AAR_PATH}"
fi
