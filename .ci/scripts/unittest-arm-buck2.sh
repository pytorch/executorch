#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -euo pipefail

# This C++ package does not load the Arm Python tests' internal Buck helpers.
target=//backends/arm/test/vela:vela_external_blocks_test
buck2 test "${target}"
binary=$(buck2 build "${target}" --show-full-simple-output)

# Run each case in a fresh process, as internal CI does, to expose dependencies
# on initialization performed by an earlier case in the same test binary.
cases=$("${binary}" --gtest_list_tests | awk '
    /^[^ ]/ { suite = $1 }
    /^  [^ ]/ { print suite $1 }
')
if [[ -z "${cases}" ]]; then
    echo "No Vela parser tests were discovered" >&2
    exit 1
fi
while IFS= read -r test_case; do
    "${binary}" --gtest_filter="${test_case}"
done <<< "${cases}"
