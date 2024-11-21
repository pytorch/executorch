#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

VERSION="0.4.0.20241120"
FRAMEWORKS=(
  "backend_coreml"
  "backend_mps"
  "backend_xnnpack"
  "executorch"
  "kernels_custom"
  "kernels_optimized"
  "kernels_portable"
  "kernels_quantized"
)

cd "$(dirname "$0")" || exit

for FRAMEWORK in "${FRAMEWORKS[@]}"; do
  rm -f "${FRAMEWORK}-${VERSION}.zip"
  rm -rf "${FRAMEWORK}.xcframework"
  curl -sSLO "https://ossci-ios.s3.amazonaws.com/executorch/${FRAMEWORK}-${VERSION}.zip" && \
  unzip -q "${FRAMEWORK}-${VERSION}.zip" && \
  rm "${FRAMEWORK}-${VERSION}.zip"
done
