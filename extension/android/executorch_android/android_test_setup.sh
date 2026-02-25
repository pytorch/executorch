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
RESOURCES_DIR="${BASEDIR}/src/androidTest/resources"
PUSH_ARTIFACTS_DIR="${BASEDIR}/build/push-artifacts"
mkdir -p "${RESOURCES_DIR}" "${PUSH_ARTIFACTS_DIR}"

prepare_xor() {
  pushd "${BASEDIR}/../../training/"
  python3 -m examples.XOR.export_model  --outdir "${RESOURCES_DIR}/"
  mv "${RESOURCES_DIR}/xor.pte" "${RESOURCES_DIR}/xor_full.pte"
  python3 -m examples.XOR.export_model  --outdir "${RESOURCES_DIR}/" --external
  popd
}

prepare_tinyllama() {
  local S3_BASE="https://ossci-android.s3.amazonaws.com/executorch/stories/snapshot-20260114"
  # stories.pte is ~29MB; push via adb instead of bundling in APK
  curl -C - -Ls "${S3_BASE}/stories110M.pte" --output "${PUSH_ARTIFACTS_DIR}/stories.pte"
  curl -C - -Ls "${S3_BASE}/tokenizer.model" --output "${RESOURCES_DIR}/tokenizer.bin"
}

prepare_golden() {
  local url="https://gha-artifacts.s3.amazonaws.com/pytorch/executorch/test-backend-artifacts/golden-artifacts-xnnpack/golden_artifacts_26022500.zip"
  curl -sL -o /tmp/golden.zip "$url"
  unzip -o /tmp/golden.zip -d /tmp/golden/
  for model in mobilenet_v2 vit_b_16; do
    cp /tmp/golden/xnnpack/${model}_input*.bin "${RESOURCES_DIR}/"
    cp /tmp/golden/xnnpack/${model}_expected_output*.bin "${RESOURCES_DIR}/" 2>/dev/null || echo "Warning: no expected_output files for ${model}"
  done
  # mobilenet_v2.pte (~13MB) stays in APK resources
  cp "/tmp/golden/xnnpack/mobilenet_v2.pte" "${RESOURCES_DIR}/"
  # vit_b_16.pte (~330MB) is too large for APK; push via adb
  cp "/tmp/golden/xnnpack/vit_b_16.pte" "${PUSH_ARTIFACTS_DIR}/"
}

prepare_xor
prepare_tinyllama
prepare_golden
