#!/usr/bin/env bash
#
# Copyright © 2023 Apple Inc. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

SCRIPT_DIR_PATH="$(
    cd -- "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)"

EXECUTORCH_ROOT_PATH="$SCRIPT_DIR_PATH/../../../../"
COREML_DIR_PATH="$EXECUTORCH_ROOT_PATH/backends/apple/coreml"

#Generate models
echo "Executorch: Generating test models"
cd "$EXECUTORCH_ROOT_PATH"

MODELS=("add" "mul" "mv2" "mv3")
for MODEL in "${MODELS[@]}"
do
  python3 -m examples.apple.coreml.scripts.export_and_delegate --model_name "$MODEL" --save_processed_bytes
  mv -f "$MODEL""_coreml_all.pte" "$COREML_DIR_PATH/runtime/test/models"
  mv -f "$MODEL""_coreml_all.bin" "$COREML_DIR_PATH/runtime/test/models"
done
