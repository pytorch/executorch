#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage:
#   ./test_ios.sh [output]
# Arguments:
#   output - The directory where the repository will be cloned and built.
#            Default is 'executorch'.

set -e

OUTPUT="${1:-executorch}"
EXIT_STATUS=0
APP_PATH="examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo"
MODEL_NAME="mv3"
SIMULATOR_NAME="executorch"

finish() {
  EXIT_STATUS=$?
  if xcrun simctl list | grep -q "$SIMULATOR_NAME"; then
    say "Deleting Simulator"
    xcrun simctl delete "$SIMULATOR_NAME"
  fi
  if [ -d "$OUTPUT" ]; then
    popd > /dev/null
    say "Deleting Output Directory"
    rm -rf "$OUTPUT"
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

say "Cloning the Code"

pushd . > /dev/null
git clone https://github.com/pytorch/executorch.git "$OUTPUT"
cd "$OUTPUT"

say "Updating the Submodules"

git submodule update --init

say "Activating a Virtual Environment"

python3 -m venv .venv
source .venv/bin/activate

say "Installing Requirements"

pip install --upgrade cmake pip setuptools wheel zstd

./install_requirements.sh --pybind coreml mps xnnpack
export PATH="$(realpath third-party/flatbuffers/cmake-out):$PATH"
./build/install_flatc.sh

say "Installing CoreML Backend Requirements"

./backends/apple/coreml/scripts/install_requirements.sh

say "Installing MPS Backend Requirements"

./backends/apple/mps/install_requirements.sh

say "Exporting Models"

python3 -m examples.portable.scripts.export --model_name="$MODEL_NAME"
python3 -m examples.apple.coreml.scripts.export --model_name="$MODEL_NAME"
python3 -m examples.apple.mps.scripts.mps_example --model_name="$MODEL_NAME"
python3 -m examples.xnnpack.aot_compiler --model_name="$MODEL_NAME" --delegate

mkdir -p "$APP_PATH/Resources/Models/MobileNet/"
mv $MODEL_NAME*.pte "$APP_PATH/Resources/Models/MobileNet/"

say "Downloading Labels"

curl https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt \
  -o "$APP_PATH/Resources/Models/MobileNet/imagenet_classes.txt"

say "Building Frameworks"

./build/build_apple_frameworks.sh --coreml --custom --mps --optimized --portable --quantized --xnnpack
mv cmake-out "$APP_PATH/Frameworks"

say "Creating Simulator"

xcrun simctl create "$SIMULATOR_NAME" "iPhone 15"

say "Running Tests"

xcodebuild test \
  -project "$APP_PATH.xcodeproj" \
  -scheme MobileNetClassifierTest \
  -destination name="$SIMULATOR_NAME"
