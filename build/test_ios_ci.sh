#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

APP_PATH="examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo"
MODEL_NAME="mv3"
SIMULATOR_NAME="executorch"

finish() {
  EXIT_STATUS=$?
  if xcrun simctl list | grep -q "$SIMULATOR_NAME"; then
    say "Deleting Simulator"
    xcrun simctl delete "$SIMULATOR_NAME"
  fi
  if [ $EXIT_STATUS -eq 0 ]; then
    say "SUCCEEDED"
  else
    say "FAILED"
  fi
  exit $EXIT_STATUS
}

trap finish EXIT

say() {
  echo -e "\033[1m\n\t** $1 **\n\033[0m"
}

say "Installing CoreML Backend Requirements"

./backends/apple/coreml/scripts/install_requirements.sh

say "Installing MPS Backend Requirements"

./backends/apple/mps/install_requirements.sh

say "Exporting Models"

python3 -m examples.portable.scripts.export --model_name="$MODEL_NAME" --segment_alignment=0x4000
python3 -m examples.apple.coreml.scripts.export --model_name="$MODEL_NAME"
python3 -m examples.apple.mps.scripts.mps_example --model_name="$MODEL_NAME"
python3 -m examples.xnnpack.aot_compiler --model_name="$MODEL_NAME" --delegate

mkdir -p "$APP_PATH/Resources/Models/MobileNet/"
mv $MODEL_NAME*.pte "$APP_PATH/Resources/Models/MobileNet/"

say "Downloading Labels"

curl https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt \
  -o "$APP_PATH/Resources/Models/MobileNet/imagenet_classes.txt"

say "Creating Simulator"

xcrun simctl create "$SIMULATOR_NAME" "iPhone 15"

say "Running Tests"

xcodebuild test \
  -project "$APP_PATH.xcodeproj" \
  -scheme MobileNetClassifierTest \
  -destination name="$SIMULATOR_NAME"

# NB: https://docs.aws.amazon.com/devicefarm/latest/developerguide/test-types-ios-xctest-ui.html
say "Package The Test Suite"

xcodebuild build-for-testing \
  -project "$APP_PATH.xcodeproj" \
  -scheme MobileNetClassifierTest \
  -destination platform="iOS" \
  -allowProvisioningUpdates \
  DEVELOPMENT_TEAM=78E7V7QP35 \
  CODE_SIGN_STYLE=Manual \
  PROVISIONING_PROFILE_SPECIFIER=ExecuTorchDemo \
  CODE_SIGN_IDENTITY="iPhone Distribution"

# The hack to figure out where the xctest package locates
BUILD_DIR=$(xcodebuild -showBuildSettings -project "$APP_PATH.xcodeproj" -json | jq -r ".[0].buildSettings.BUILD_DIR")

MODE="Debug"
PLATFORM="iphoneos"
pushd "${BUILD_DIR}/${MODE}-${PLATFORM}"

rm -rf Payload && mkdir Payload
MOCK_APP_NAME=DeviceFarm
cp -r "${MOCK_APP_NAME}.app" Payload && zip -vr "${MOCK_APP_NAME}.ipa" Payload

# DEBUG
ls -lah

popd
