#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

# Check if dir1's files are also found in dir2 with the same
# contents. Exempt files named BUCK, CMakeLists.txt, TARGETS, or
# targets.bzl.

if [ $# -ne 2 ]; then
    echo "Usage: $0 dir1 dir2" >&2
    exit 1
fi
dir1="$1"
dir2="$2"

if [ ! -d "$dir1" ] || [ ! -d "$dir2" ]; then
    echo "Error: Both directories must exist" >&2
    exit 1
fi

exit_status=0
while IFS= read -r -d '' file; do
    base=$(basename "$file")
    case "$base" in
        "BUCK"|"CMakeLists.txt"|"TARGETS"|"targets.bzl")
            continue
            ;;
    esac
    # Construct the corresponding path in the second directory
    file2="$dir2/${file#$dir1/}"
    # Check if the corresponding file exists in the second directory
    if [ ! -f "$file2" ]; then
        echo "Error: File '$file' found in '$dir1' but not found in '$dir2'" >&2
        exit 1
    fi
    # Compare the contents of the two files using diff
    set +ex
    differences=$(diff -u -p "$file" "$file2")
    set -e # leave x off
    # If there are any differences, print an error message and exit with failure status
    if [ -n "$differences" ]; then
        echo "Error: Mismatch detected in file '$file':" >&2
        echo "$differences" >&2
        exit_status=1
    fi
    set -x
done < <(find "$dir1" -type f -print0)

exit $exit_status
