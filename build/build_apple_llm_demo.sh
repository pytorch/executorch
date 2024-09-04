#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

TOKENIZER="${1:-bpe}"
ARTIFACTS_DIR_NAME="$2"

APP_PATH="examples/demo-apps/apple_ios/LLaMA/LLaMA"

if [[ "${TOKENIZER}" = "bpe" ]]; then
  xcodebuild build-for-testing \
    -project "${APP_PATH}.xcodeproj" \
    -scheme LLaMAPerfBenchmark \
    -destination platform="iOS" \
    -allowProvisioningUpdates \
    DEVELOPMENT_TEAM=78E7V7QP35 \
    CODE_SIGN_STYLE=Manual \
    PROVISIONING_PROFILE_SPECIFIER=iLLaMA \
    CODE_SIGN_IDENTITY="iPhone Distribution" \
    CODE_SIGNING_REQUIRED=No \
    CODE_SIGNING_ALLOWED=No \
    GCC_PREPROCESSOR_DEFINITIONS="DEBUG=1 ET_USE_TIKTOKEN=0"
else
  xcodebuild build-for-testing \
    -project "${APP_PATH}.xcodeproj" \
    -scheme LLaMAPerfBenchmark \
    -destination platform="iOS" \
    -allowProvisioningUpdates \
    DEVELOPMENT_TEAM=78E7V7QP35 \
    CODE_SIGN_STYLE=Manual \
    PROVISIONING_PROFILE_SPECIFIER=iLLaMA \
    CODE_SIGN_IDENTITY="iPhone Distribution" \
    CODE_SIGNING_REQUIRED=No \
    CODE_SIGNING_ALLOWED=No
fi

# The hack to figure out where the xctest package locates
BUILD_DIR=$(xcodebuild -showBuildSettings -project "$APP_PATH.xcodeproj" -json | jq -r ".[0].buildSettings.BUILD_DIR")

# Prepare the demo app, debug mode here is the default from xcodebuild and match
# with what we have in the test spec
# TODO (huydhn): See if we can switch to release mode here
MODE="Debug"
PLATFORM="iphoneos"
pushd "${BUILD_DIR}/${MODE}-${PLATFORM}"

rm -rf Payload && mkdir Payload
APP_NAME=LLaMAPerfBenchmark

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
