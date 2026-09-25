#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

APP_PATH="executorch-examples/mv3/apple/ExecuTorchDemo/ExecuTorchDemo"
MODEL_NAME="mv3"
SIMULATOR_NAME="executorch"

# If this is set, copy the build artifacts to this directory
ARTIFACTS_DIR_NAME="$1"

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

say "Cloning the Demo App"

git config --global http.postBuffer 524288000
git clone --depth 1 https://github.com/meta-pytorch/executorch-examples.git

say "Installing CoreML Backend Requirements"

./backends/apple/coreml/scripts/install_requirements.sh

say "Exporting Models"

python3 -m examples.portable.scripts.export --model_name="$MODEL_NAME" --segment_alignment=0x4000
python3 -m examples.apple.coreml.scripts.export --model_name="$MODEL_NAME"
python3 -m examples.xnnpack.aot_compiler --model_name="$MODEL_NAME" --delegate

# No generic examples/ entry point takes a model name and lowers it to MLX, so do
# it here. Lowering is ahead of time and imports no mlx runtime package. Assert MLX
# claimed the graph before writing: the partitioner only warns when it delegates
# nothing, so without this a program with no MLX delegate would be staged.
python3 - "$MODEL_NAME" <<'PY'
import sys

import torch
from executorch.backends.mlx import MLXPartitioner
from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

model_name = sys.argv[1]
model, example_inputs, _, _ = EagerModelFactory.create_model(
    *MODEL_NAME_TO_MODEL[model_name]
)
program = to_edge_transform_and_lower(
    torch.export.export(model.eval(), example_inputs),
    partitioner=[MLXPartitioner()],
    compile_config=EdgeCompileConfig(_skip_dim_order=True),
).to_executorch()

segments = sum(
    delegate.id == "MLXBackend"
    for plan in program.executorch_program.execution_plan
    for delegate in plan.delegates
)
if segments == 0:
    raise SystemExit(f"error: {model_name} produced no MLX delegate")

path = f"{model_name}_mlx.pte"
with open(path, "wb") as file:
    program.write_to_file(file)
print(f"{path}: {segments} MLX delegate segment(s)")
PY

mkdir -p "$APP_PATH/Resources/Models/MobileNet/"
mv $MODEL_NAME*.pte "$APP_PATH/Resources/Models/MobileNet/"

say "Downloading Labels"

curl --retry 3 --retry-all-errors https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt \
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
  CODE_SIGN_IDENTITY="iPhone Distribution" \
  CODE_SIGNING_REQUIRED=No \
  CODE_SIGNING_ALLOWED=No

# The hack to figure out where the xctest package locates
BUILD_DIR=$(xcodebuild -showBuildSettings -project "$APP_PATH.xcodeproj" -json | jq -r ".[0].buildSettings.BUILD_DIR")

# Prepare the demo app
MODE="Debug"
PLATFORM="iphoneos"
pushd "${BUILD_DIR}/${MODE}-${PLATFORM}"

rm -rf Payload && mkdir Payload
MOCK_APP_NAME=ExecuTorchDemo

ls -lah
cp -r "${MOCK_APP_NAME}.app" Payload && zip -vr "${MOCK_APP_NAME}.ipa" Payload

popd

# Prepare the test suite
pushd "${BUILD_DIR}"

ls -lah
zip -vr "${MOCK_APP_NAME}.xctestrun.zip" *.xctestrun

popd

if [[ -n "${ARTIFACTS_DIR_NAME}" ]]; then
  mkdir -p "${ARTIFACTS_DIR_NAME}"
  # Prepare all the artifacts to upload
  cp "${BUILD_DIR}/${MODE}-${PLATFORM}/${MOCK_APP_NAME}.ipa" "${ARTIFACTS_DIR_NAME}/"
  cp "${BUILD_DIR}/${MOCK_APP_NAME}.xctestrun.zip" "${ARTIFACTS_DIR_NAME}/"

  ls -lah "${ARTIFACTS_DIR_NAME}/"
fi
