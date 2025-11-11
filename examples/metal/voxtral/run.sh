#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if [ -z "$1" ]; then
  echo "Error: Input audio path must be provided as an argument"
  echo "Usage: $0 <input_audio_path> [dir_name]"
  echo ""
  echo "Arguments:"
  echo "  <input_audio_path>  Path to the input audio file"
  echo "  [dir_name]          Optional: Directory name containing model files (default: voxtral)"
  exit 1
fi

INPUT_AUDIO_PATH="$1"
DIR_NAME="${2:-voxtral}"

# Check if input audio file exists
if [ ! -f "$INPUT_AUDIO_PATH" ]; then
  echo "Error: Input audio file not found: $INPUT_AUDIO_PATH"
  exit 1
fi

# Check if directory exists
if [ ! -d "$DIR_NAME" ]; then
  echo "Error: Directory not found: $DIR_NAME"
  exit 1
fi

# Check for required files
REQUIRED_FILES=("model.pte" "aoti_metal_blob.ptd" "tekken.json" "voxtral_preprocessor.pte")
for file in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$DIR_NAME/$file" ]; then
    echo "Error: Required file not found: $DIR_NAME/$file"
    exit 1
  fi
done

/usr/bin/time -l ./cmake-out/examples/models/voxtral/voxtral_runner \
      --model_path "$DIR_NAME"/model.pte \
      --data_path "$DIR_NAME"/aoti_metal_blob.ptd \
      --tokenizer_path "$DIR_NAME"/tekken.json \
      --audio_path "$INPUT_AUDIO_PATH" \
      --processor_path "$DIR_NAME"/voxtral_preprocessor.pte \
      --temperature 0
