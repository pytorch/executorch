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

say "Building Frameworks"

./build/build_apple_frameworks.sh --buck2="$(which buck2)" --flatc="$(which flatc)" --coreml --mps --portable --xnnpack
mv cmake-out "$APP_PATH/Frameworks"

say "Creating Simulator"

xcrun simctl create "$SIMULATOR_NAME" "iPhone 15"

say "Running Tests"

xcodebuild test \
  -project "$APP_PATH.xcodeproj" \
  -scheme MobileNetClassifierTest \
  -destination name="$SIMULATOR_NAME"

say "Installing Python Bindings"

# Ensure that the pybindings build works on macOS. TODO(shoumikhin): Make this a
# separate test to clarify that it's not required for the rest of the script.
# Until then, make this the last step so that nothing can depend on it.
EXECUTORCH_BUILD_PYBIND=ON \
BUCK="$(which buck2)" \
CMAKE_ARGS="-DEXECUTORCH_BUILD_COREML=ON -DEXECUTORCH_BUILD_MPS=ON -DEXECUTORCH_BUILD_XNNPACK=ON" \
CMAKE_BUILD_PARALLEL_LEVEL=9 \
pip install . --no-build-isolation -v
