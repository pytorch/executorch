#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

ARTIFACTS_DIR_NAME="$1"
APP_PATH="extension/benchmark/apple/Benchmark/Benchmark"
DERIVED_DATA_PATH="$(pwd)/cmake-out/ios-benchmark-derived-data"
BUILD_DIR="${DERIVED_DATA_PATH}/Build/Products"
MODE="Release"
PLATFORM="iphoneos"

xcodebuild build-for-testing \
  -project "${APP_PATH}.xcodeproj" \
  -scheme Benchmark \
  -configuration "${MODE}" \
  -derivedDataPath "${DERIVED_DATA_PATH}" \
  -destination "generic/platform=iOS" \
  -sdk iphoneos \
  -allowProvisioningUpdates \
  DEVELOPMENT_TEAM=78E7V7QP35 \
  CODE_SIGN_STYLE=Manual \
  PROVISIONING_PROFILE_SPECIFIER="ExecuTorch Benchmark" \
  CODE_SIGN_IDENTITY="iPhone Distribution" \
  CODE_SIGNING_REQUIRED=No \
  CODE_SIGNING_ALLOWED=No

# Prepare the benchmark app.
pushd "${BUILD_DIR}/${MODE}-${PLATFORM}"

rm -rf Payload && mkdir Payload
APP_NAME=Benchmark

ls -lah
cp -r "${APP_NAME}.app" Payload && zip -vr "${APP_NAME}.ipa" Payload

popd

# Prepare the test suite
pushd "${BUILD_DIR}"

ls -lah
zip -vr "${APP_NAME}.xctestrun.zip" *.xctestrun

popd

if [[ -n "${ARTIFACTS_DIR_NAME}" ]]; then
  mkdir -p "${ARTIFACTS_DIR_NAME}"
  # Prepare all the artifacts to upload
  cp "${BUILD_DIR}/${MODE}-${PLATFORM}/${APP_NAME}.ipa" "${ARTIFACTS_DIR_NAME}/"
  cp "${BUILD_DIR}/${APP_NAME}.xctestrun.zip" "${ARTIFACTS_DIR_NAME}/"

  ls -lah "${ARTIFACTS_DIR_NAME}/"
fi
