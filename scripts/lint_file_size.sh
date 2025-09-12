#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

status=0

green='\e[1;32m'; red='\e[1;31m'; cyan='\e[1;36m'; reset='\e[0m'

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
