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
mkdir -p "${RESOURCES_DIR}"

prepare_xor() {
  pushd "${BASEDIR}/../../training/"
  python3 -m examples.XOR.export_model  --outdir "${RESOURCES_DIR}/"
  mv "${RESOURCES_DIR}/xor.pte" "${RESOURCES_DIR}/xor_full.pte"
  python3 -m examples.XOR.export_model  --outdir "${RESOURCES_DIR}/" --external
  popd
}

prepare_xor

# Large test assets (LLM weights, golden XNNPACK outputs) are no longer bundled
# in the APK. In CI they are downloaded and adb-pushed by scripts/run_android_emulator.sh.
# For local runs, you must push them yourself before running connectedAndroidTest.
PUSH_DIR="/data/local/tmp/executorch"
REQUIRED_ASSETS=(stories.pte tokenizer.bin mobilenet_v2.pte mobilenet_v2_expected_output.bin vit_b_16.pte vit_b_16_expected_output.bin)
MISSING=()
for asset in "${REQUIRED_ASSETS[@]}"; do
  if ! adb shell "[ -f ${PUSH_DIR}/${asset} ]" 2>/dev/null; then
    MISSING+=("$asset")
  fi
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo ""
  echo "WARNING: The following test assets are missing from ${PUSH_DIR} on the device:"
  printf "  - %s\n" "${MISSING[@]}"
  echo ""
  echo "LlmModuleInstrumentationTest and ModuleE2ETest will fail without them."
  echo "See scripts/run_android_emulator.sh for download URLs and adb push commands."
  echo ""
fi
