#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

frameworks=(
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

for framework in "${frameworks[@]}"; do
  rm -f "${framework}-latest.zip"
  rm -rf "${framework}.xcframework"
  curl -sSLO "https://ossci-ios.s3.amazonaws.com/executorch/${framework}-latest.zip" && \
  unzip -q "${framework}-latest.zip" && \
  rm "${framework}-latest.zip"
done
