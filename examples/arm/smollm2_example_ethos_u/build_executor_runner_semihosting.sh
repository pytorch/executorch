#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)

et_build_root="${repo_root}/arm_test"
output_dir="${repo_root}/cmake-out-smollm2-ethosu-semi"
toolchain="arm-none-eabi-gcc"
pte_file=""
target="ethos-u85-256"
system_config="Ethos_U85_SYS_DRAM_High"
memory_mode="Dedicated_Sram_512KB"
method_pool_size="0x00800000"
scratch_pool_size="0x00400000"
input_file_pool_size="0x00100000"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --pte=PATH             Embed this PTE in the semihosting runner ELF.
  --et_build_root=DIR    Build root. Default: ${et_build_root}
  --output=DIR           CMake output directory. Default: ${output_dir}
  --toolchain=NAME       Toolchain. Default: ${toolchain}
  --target=NAME          Ethos-U target. Default: ${target}
  --system_config=NAME   Vela system config. Default: ${system_config}
  --memory_mode=NAME     Vela memory mode. Default: ${memory_mode}
  --method_pool_size=HEX Method allocator pool size. Default: ${method_pool_size}
  --scratch_pool_size=HEX Scratch temp allocator pool size. Default: ${scratch_pool_size}
  --input_file_pool_size=HEX Input file allocator pool size. Default: ${input_file_pool_size}
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --pte=*) pte_file="${arg#*=}" ;;
    --et_build_root=*) et_build_root="${arg#*=}" ;;
    --output=*) output_dir="${arg#*=}" ;;
    --toolchain=*) toolchain="${arg#*=}" ;;
    --target=*) target="${arg#*=}" ;;
    --system_config=*) system_config="${arg#*=}" ;;
    --memory_mode=*) memory_mode="${arg#*=}" ;;
    --method_pool_size=*) method_pool_size="${arg#*=}" ;;
    --scratch_pool_size=*) scratch_pool_size="${arg#*=}" ;;
    --input_file_pool_size=*) input_file_pool_size="${arg#*=}" ;;
    *)
      echo "Unknown option: ${arg}" >&2
      usage
      exit 1
      ;;
  esac
done

cd "${repo_root}"
pte_arg="semihosting"
if [[ -n "${pte_file}" ]]; then
  pte_arg="${pte_file}"
fi

bash "${repo_root}/backends/arm/scripts/build_executor_runner.sh" \
  --et_build_root="${et_build_root}" \
  --output="${output_dir}" \
  --pte="${pte_arg}" \
  --build_type=Release \
  --target="${target}" \
  --system_config="${system_config}" \
  --memory_mode="${memory_mode}" \
  --extra_build_flags="-DSEMIHOSTING=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE=${method_pool_size} -DET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=${scratch_pool_size} -DET_ARM_BAREMETAL_SEMIHOSTING_FILE_ALLOCATOR_POOL_SIZE=${input_file_pool_size}" \
  --ethosu_tools_dir="${repo_root}/examples/arm/arm-scratch" \
  --toolchain="${toolchain}"
