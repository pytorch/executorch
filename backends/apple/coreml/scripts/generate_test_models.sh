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

EXECUTORCH_ROOT_PATH=$(realpath "$SCRIPT_DIR_PATH/../../../../")
COREML_DIR_PATH="$EXECUTORCH_ROOT_PATH/backends/apple/coreml"

cd "$EXECUTORCH_ROOT_PATH"

mkdir "$COREML_DIR_PATH/runtime/test/models/"
#Generate models
cd "$EXECUTORCH_ROOT_PATH"

MODELS=("add" "add_mul" "mul" "mv3")
for MODEL in "${MODELS[@]}"
do
  echo "Executorch: Generating $MODEL model" 
  # TODO: Don't use the script in examples directory.
  python3 -m examples.apple.coreml.scripts.export --model_name "$MODEL" --save_processed_bytes
  mv -f "$MODEL""_coreml_all.pte" "$COREML_DIR_PATH/runtime/test/models"
  mv -f "$MODEL""_coreml_all.bin" "$COREML_DIR_PATH/runtime/test/models"
done

echo "Executorch: Generating stateful model"
python3 "$SCRIPT_DIR_PATH/../runtime/test/export_stateful_model.py"
