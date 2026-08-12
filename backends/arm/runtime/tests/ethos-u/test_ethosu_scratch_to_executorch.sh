#!/bin/bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

source "$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)/setup_env.sh"

target="ethos-u55-128"
test_dir="${test_output_dir}/ethosu_scratch_to_executorch"
runner_build_dir="${test_dir}/runner"
mkdir -p "${test_dir}"
test_dir=$(realpath "${test_dir}")
truncate -s 4 "${test_dir}/dummy_input.bin"

for planning_mode in shared scratch legacy; do
    pte_file="${test_dir}/${planning_mode}.pte"
    bundle_file="${test_dir}/${planning_mode}.bpte"
    log_file="${test_dir}/${planning_mode}.log"
    temp_pool_size=0
    if [[ "${planning_mode}" == legacy ]]; then
        temp_pool_size=0x10000
    fi

    echo "Export and verify Ethos-U ${planning_mode} planning on FVP"
    python3 "${et_root_dir}/backends/arm/test/assets/export_ethosu_scratch_test_model.py" \
        --target="${target}" \
        --model=conv \
        --planning-mode="${planning_mode}" \
        --output="${pte_file}" \
        --bundle-output="${bundle_file}"

    "${et_root_dir}/backends/arm/scripts/build_executor_runner.sh" \
        --pte=semihosting \
        --bundleio \
        --target="${target}" \
        --output="${runner_build_dir}" \
        --select_ops_list="cortex_m::quantize_per_tensor.out,cortex_m::dequantize_per_tensor.out" \
        --extra_build_flags="-DFETCH_ETHOS_U_CONTENT=OFF -DET_NUM_INFERENCES=3 -DET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=${temp_pool_size} -DET_ARM_ETHOSU_PROFILE_IO_COPIES=ON -DET_LOG_DUMP_OUTPUT=OFF -DET_ATOL=0.000001 -DET_RTOL=0.000001"

    "${et_root_dir}/backends/arm/scripts/run_fvp.sh" \
        --elf="${runner_build_dir}/arm_executor_runner" \
        --target="${target}" \
        --semihosting-cwd="${test_dir}" \
        --semihosting-cmd-line="executor_runner -m ${planning_mode}.bpte -i dummy_input.bin -o ${planning_mode}" \
        2>&1 | tee "${log_file}"

    grep -q "3 inferences finished" "${log_file}"
    grep -q "Test_result: PASS" "${log_file}"
    if [[ "${planning_mode}" == shared ]]; then
        grep -q "Ethos-U IO copy calls: 0 " "${log_file}"
    else
        grep -Eq "Ethos-U input copy: [1-9][0-9]* calls" "${log_file}"
        grep -Eq "Ethos-U output copy: [1-9][0-9]* calls" "${log_file}"
    fi
    python3 "${et_root_dir}/backends/arm/test/test_memory_allocator_log.py" --log "${log_file}" \
        --require "method_allocator_planned" "> 0 B"
done
