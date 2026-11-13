#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -euo pipefail

# This C++ package does not load the Arm Python tests' internal Buck helpers.
target=//backends/arm/test/vela:vela_external_blocks_test
binary=$(buck2 build "${target}" --show-full-simple-output) || {
    status=$?
    echo "Failed to build Vela parser tests" >&2
    exit "${status}"
}
if [[ -z "${binary}" || ! -x "${binary}" ]]; then
    echo "Buck2 did not produce an executable Vela parser test binary: ${binary}" >&2
    exit 1
fi

# Run each case in a fresh process, as internal CI does, to expose dependencies
# on initialization performed by an earlier case in the same test binary.
listing=$("${binary}" --gtest_list_tests) || {
    status=$?
    echo "Failed to discover Vela parser tests" >&2
    exit "${status}"
}
cases=$(awk '
    /^[^ ]/ { suite = $1 }
    /^  [^ ]/ { print suite $1 }
' <<< "${listing}")
if [[ -z "${cases}" ]]; then
    echo "No Vela parser tests were discovered" >&2
    exit 1
fi

failed_cases=()
total=0
while IFS= read -r test_case; do
    total=$((total + 1))
    if ! "${binary}" --gtest_filter="${test_case}"; then
        failed_cases+=("${test_case}")
    fi
done <<< "${cases}"

failed=${#failed_cases[@]}
echo "Vela parser tests: $((total - failed)) passed, ${failed} failed (${total} total)."
if ((failed)); then
    printf 'Failed cases:\n' >&2
    printf '  - %s\n' "${failed_cases[@]}" >&2
    exit 1
fi
