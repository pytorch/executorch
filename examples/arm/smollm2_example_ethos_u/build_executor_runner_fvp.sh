#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Example build helper. This command-line interface is not a public API and may
# change without deprecation.

set -euo pipefail

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)

pte_file=""
et_build_root="${repo_root}/arm_test"
output_dir=""
toolchain="arm-none-eabi-gcc"
target="ethos-u85-256"
system_config="Ethos_U85_SYS_DRAM_High"
memory_mode="Dedicated_Sram_512KB"

usage() {
  cat <<EOF
Usage: $(basename "$0") --pte=PATH [options]
Note: this example build script is not a stable public API.

Options:
  --pte=PATH             PTE to include in the runner ELF.
  --et_build_root=DIR    Build root. Default: ${et_build_root}
  --output=DIR           CMake output directory override.
  --toolchain=NAME       Toolchain. Default: ${toolchain}
  --target=NAME          Ethos-U target. Default: ${target}
  --system_config=NAME   Vela system config. Default: ${system_config}
  --memory_mode=NAME     Vela memory mode. Default: ${memory_mode}
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
    *)
      echo "Unknown option: ${arg}" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${pte_file}" ]]; then
  echo "--pte is required" >&2
  exit 1
fi

cmd=(
  bash "${repo_root}/backends/arm/scripts/build_executor_runner.sh"
  --et_build_root=${et_build_root}
  --pte=${pte_file}
  --build_type=Release
  --target=${target}
  --system_config=${system_config}
  --memory_mode=${memory_mode}
  --extra_build_flags=-DET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=0x02000000
  --ethosu_tools_dir=${repo_root}/examples/arm/arm-scratch
  --toolchain=${toolchain}
)

if [[ -n "${output_dir}" ]]; then
  cmd+=(--output=${output_dir})
fi

cd "${repo_root}"
"${cmd[@]}"
