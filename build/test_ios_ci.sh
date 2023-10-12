#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

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
}

finish() {
  EXIT_STATUS=$?
  cleanup
  report
  exit $EXIT_STATUS
}

trap finish EXIT

say() {
  echo -e "\033[1m\n\t** $1 **\n\033[0m"
}

say "Installing Requirements"

./install_requirements.sh

say "Exporting Models"

python3 -m examples.portable.scripts.export --model_name="mv3"
python3 -m examples.xnnpack.aot_compiler --model_name="mv3" --delegate

mkdir -p examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo/Resources/Models/MobileNet/
mv mv3*.pte examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo/Resources/Models/MobileNet/

say "Downloading Labels"

curl https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt \
  -o examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo/Resources/Models/MobileNet/imagenet_classes.txt

say "Building Frameworks"

./build/build_apple_frameworks.sh --buck2="$(which buck2)" --flatc="$(which flatc)" --xnnpack
mv cmake-out examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo/Frameworks

say "Creating Simulator"

xcrun simctl create executorch "iPhone 14"

say "Running Tests"

xcodebuild clean test \
  -project examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo.xcodeproj \
  -scheme App \
  -destination name=executorch
