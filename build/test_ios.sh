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

report() {
  if [ $EXIT_STATUS -eq 0 ]; then
    say "SUCCEEDED"
  else
    say "FAILED"
  fi
}

cleanup() {
  if xcrun simctl list | grep -q 'executorch'; then
    say "Deleting Simulator"
    xcrun simctl delete executorch
  fi

  if [ -d "$OUTPUT" ]; then
    popd > /dev/null
    say "Deleting Output Directory"
    rm -rf "$OUTPUT"
  fi
}

finish() {
  EXIT_STATUS=$?
  cleanup
  report
}

trap finish EXIT

say() {
  echo -e "\033[1m\n\t** $1 **\n\033[0m"
}

say "Cloning the Code"

pushd . > /dev/null
git clone --branch v0.1.0 https://github.com/pytorch/executorch.git "$OUTPUT"
cd "$OUTPUT"

say "Updating the Submodules"

git submodule sync
git submodule update --init

say "Activating a Virtual Environment"

python3 -m venv .venv
source .venv/bin/activate

say "Installing Requirements"

pip install cmake
./install_requirements.sh
export PATH="$(realpath third-party/flatbuffers/cmake-out):$PATH"
./build/install_flatc.sh

curl -LO "https://github.com/facebook/buck2/releases/download/2023-07-18/buck2-aarch64-apple-darwin.zst"
pip install zstd
zstd -cdq buck2-aarch64-apple-darwin.zst > .venv/bin/buck2 && chmod +x .venv/bin/buck2

say "Installing CoreML Backend Requirements"

./backends/apple/coreml/scripts/install_requirements.sh

say "Installing MPS Backend Requirements"

./backends/apple/mps/install_requirements.sh

say "Exporting Models"

python3 -m examples.portable.scripts.export --model_name="mv3"
python3 -m examples.apple.coreml.scripts.export_and_delegate --model_name="mv3"
python3 -m examples.apple.mps.scripts.mps_example --model_name="mv3"
python3 -m examples.xnnpack.aot_compiler --model_name="mv3" --delegate

mkdir -p examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo/Resources/Models/MobileNet/
mv mv3*.pte examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo/Resources/Models/MobileNet/

say "Downloading Labels"

curl https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt \
  -o examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo/Resources/Models/MobileNet/imagenet_classes.txt

say "Building Frameworks"

./build/build_apple_frameworks.sh --buck2="$(realpath .venv/bin/buck2)" --Release --coreml --mps --xnnpack
mv cmake-out examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo/Frameworks

say "Creating Simulator"

xcrun simctl create executorch "iPhone 15"

say "Running Tests"

xcodebuild clean test \
  -project examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo.xcodeproj \
  -scheme App \
  -destination name=executorch
