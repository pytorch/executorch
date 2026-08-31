#!/bin/bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

source "$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)/setup_env.sh"

target="ethos-u85-128"
test_dir="${test_output_dir}/memory_allocation_baseline"
runner_build_dir="${test_dir}/${target}_Release_arm-none-eabi-gcc"
log_file="${test_dir}/baseline.log"

build_baseline_runner() {
    echo "Build ethos-u baseline memory allocation runner"
    mkdir -p "${test_dir}"
    "${et_root_dir}/examples/arm/run.sh" \
        --et_build_root="${test_dir}" \
        --target="${target}" \
        --model_name="${et_root_dir}/examples/arm/example_modules/add.py" \
        --build_only
}

run_baseline_memory_allocation() {
    echo "Test ethos-u baseline memory allocation"
    "${et_root_dir}/examples/arm/run.sh" \
        --et_build_root="${test_dir}" \
        --build-dir="${runner_build_dir}" \
        --target="${target}" \
        --model_name="${et_root_dir}/examples/arm/example_modules/add.py" \
        2>&1 | tee "${log_file}"

    # method_allocator_input includes one 16-byte EValue input plus possible
    # alignment padding before that allocation.
    python3 "${et_root_dir}/backends/arm/test/test_memory_allocator_log.py" --log "${log_file}" \
        --require "model_pte_program_size" "<= 3200 B" \
        --require "method_allocator_planned" "<= 64 B" \
        --require "method_allocator_loaded" "<= 1024 B" \
        --require "method_allocator_input" "<= 24 B" \
        --require "Total DRAM used" "<= 0.06 KiB"
}

build_baseline_runner
run_baseline_memory_allocation
