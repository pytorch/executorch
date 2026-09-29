#!/bin/bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

source "$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)/setup_env.sh"

test_dir="${test_output_dir}/runner_size_optimization"
default_build_dir="${test_dir}/default"
optimized_build_dir="${test_dir}/optimized"

get_required_text_section_bytes() {
    local elf=$1
    local size_tool="arm-none-eabi-size"
    local value

    command -v "${size_tool}" >/dev/null \
        || { echo "Could not find ${size_tool} on PATH" >&2; exit 1; }

    value=$("${size_tool}" -A --radix=10 "${elf}" |
        awk '$1 == ".text" { print $2; found=1 } END { exit !found }')
    if [[ -z "${value}" ]]; then
        echo "Could not read .text size from ${elf}" >&2
        exit 1
    fi

    echo "${value}"
}

min_text_saving_bytes=$((50 * 1024))

echo "Compare runner text size with EXECUTORCH_OPTIMIZE_SIZE"
"${et_root_dir}/backends/arm/scripts/build_executor_runner.sh" \
    --pte=semihosting \
    --output="${default_build_dir}"

"${et_root_dir}/backends/arm/scripts/build_executor_runner.sh" \
    --pte=semihosting \
    --output="${optimized_build_dir}" \
    --extra_build_flags="-DEXECUTORCH_OPTIMIZE_SIZE=ON"

default_text=$(get_required_text_section_bytes "${default_build_dir}/arm_executor_runner")
optimized_text=$(get_required_text_section_bytes "${optimized_build_dir}/arm_executor_runner")
text_saving=$((default_text - optimized_text))

echo "default .text=${default_text} bytes"
echo "optimized .text=${optimized_text} bytes"
if (( text_saving < min_text_saving_bytes )); then
    echo "Expected EXECUTORCH_OPTIMIZE_SIZE to reduce .text by at least ${min_text_saving_bytes} bytes, got default=${default_text} optimized=${optimized_text} saving=${text_saving}" >&2
    exit 1
fi
