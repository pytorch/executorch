#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi
which "${PYTHON_EXECUTABLE}"

BASEDIR=$(dirname "$(realpath $0)")

prepare_xor() {
  pushd "${BASEDIR}/../../training/"
  python3 -m examples.XOR.export_model  --outdir "${BASEDIR}/src/androidTest/resources/"
  mv "${BASEDIR}/src/androidTest/resources/xor.pte" "${BASEDIR}/src/androidTest/resources/xor_full.pte"
  python3 -m examples.XOR.export_model  --outdir "${BASEDIR}/src/androidTest/resources/" --external
  popd
}

prepare_tinyllama() {
  local S3_BASE="https://ossci-android.s3.amazonaws.com/executorch/stories/snapshot-20260114"
  curl -C - -Ls "${S3_BASE}/stories110M.pte" --output "${BASEDIR}/src/androidTest/resources/stories.pte"
  curl -C - -Ls "${S3_BASE}/tokenizer.model" --output "${BASEDIR}/src/androidTest/resources/tokenizer.bin"
}

prepare_golden() {
  local url="https://gha-artifacts.s3.amazonaws.com/pytorch/executorch/test-backend-artifacts/golden-artifacts-xnnpack/golden_artifacts_26022500.zip"
  curl -sL -o /tmp/golden.zip "$url"
  unzip -o /tmp/golden.zip -d /tmp/golden/
  for model in mobilenet_v2 vit_b_16; do
    cp "/tmp/golden/xnnpack/${model}.pte" "${BASEDIR}/src/androidTest/resources/"
    cp /tmp/golden/xnnpack/${model}_input*.bin "${BASEDIR}/src/androidTest/resources/"
    cp /tmp/golden/xnnpack/${model}_expected_output*.bin "${BASEDIR}/src/androidTest/resources/" 2>/dev/null || echo "Warning: no expected_output files for ${model}"
  done
}

prepare_xor
prepare_tinyllama
prepare_golden
