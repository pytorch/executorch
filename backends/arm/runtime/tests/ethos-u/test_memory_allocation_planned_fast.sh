#!/bin/bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

source "$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)/setup_env.sh"

target="ethos-u55-128"
test_dir="${test_output_dir}/memory_allocation_planned_fast"
pte_file="${test_dir}/memory_planning_mem_id_3.pte"
runner_build_dir="${test_dir}/runner"
log_file="${test_dir}/planned.log"

echo "Build ethos-u planned slow and fast memory allocation runner"
mkdir -p "${test_dir}"

python3 "${et_root_dir}/backends/arm/test/assets/export_memory_planning_mem_id_3.py" \
    --output="${pte_file}"

"${et_root_dir}/backends/arm/scripts/build_executor_runner.sh" \
    --pte="${pte_file}" \
    --target="${target}" \
    --output="${runner_build_dir}" \
    --select_ops_list="aten::add.out,aten::mul.out" \
    --extra_build_flags="-DFETCH_ETHOS_U_CONTENT=OFF -DET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=0x180000 -DET_ARM_BAREMETAL_PLANNED_FAST_MEMORY_SIZE=0x1000"

echo "Test ethos-u planned slow and fast memory allocation"
"${et_root_dir}/backends/arm/scripts/run_fvp.sh" \
    --elf="${runner_build_dir}/arm_executor_runner" \
    --target="${target}" \
    2>&1 | tee "${log_file}"

python3 "${et_root_dir}/backends/arm/test/test_memory_allocator_log.py" --log "${log_file}" \
    --require "method_allocator_planned" "== 8 B" \
    --require "planned_fast_used" "== 8 B"
