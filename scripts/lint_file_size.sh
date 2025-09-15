#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

status=0

green='\e[1;32m'; red='\e[1;31m'; cyan='\e[1;36m'; reset='\e[0m'

# Following is the rules for the file size linting:
# 1. For all files, the file size can't be larger than 1MB
# 2. For images/vidoes, the files size can't be larger than 7MB
# 3. There is an exception list defined in the script if it's really needed

# List of files to skip (relative paths)
EXCEPTIONS=(
  "examples/models/llama/params/demo_rand_params.pth"
  "examples/models/llama/tokenizer/test/resources/test_tiktoken_tokenizer.model"
  "examples/qualcomm/oss_scripts/llama/artifacts/stories260k_hybrid_llama_qnn.pte"
  # Following needs to be clean up
  "examples/mediatek/models/llm_models/weights/Llama-3.2-1B-Instruct/tokenizer.json"
  "examples/mediatek/models/llm_models/weights/Llama-3.2-3B-Instruct/tokenizer.json"
  "examples/mediatek/models/llm_models/weights/llama3-8B-instruct/tokenizer.json"
)

is_exception() {
  local f=$1
  for ex in "${EXCEPTIONS[@]}"; do
    if [[ "$f" == "$ex" ]]; then
      return 0
    fi
  done
  return 1
}

if [ $# -eq 2 ]; then
  base=$1
  head=$2
  echo "Checking changed files between $base...$head"
  files=$(git diff --name-only "$base...$head")
else
  echo "Checking all files in repository"
  files=$(git ls-files)
fi

for file in $files; do
  if is_exception "$file"; then
    echo -e "${cyan}SKIP${reset}  $file (in exception list)"
    continue
  fi
  if [ -f "$file" ]; then
    # Set size limit depending on extension
    if [[ "$file" =~ \.(png|jpg|jpeg|gif|svg|mp3|mp4)$ ]]; then
      MAX_SIZE=$((8 * 1024 * 1024))  # 5 MB for pictures
    else
      MAX_SIZE=$((1 * 1024 * 1024))  # 1 MB for others
    fi

    size=$(wc -c <"$file")
    if [ "$size" -gt "$MAX_SIZE" ]; then
      echo -e "${red}FAIL${reset} $file (${cyan}${size} bytes${reset}) exceeds ${MAX_SIZE} bytes"
      status=1
    else
      echo -e "${green}OK${reset}   $file (${size} bytes)"
    fi
  fi
done

exit $status
