#!/usr/bin/env bash
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: More separation from the regular arm executor runner and testing.

set -eu

target="cortex-m55"
for arg in "$@"; do
    case $arg in
      --target=*) target="${arg#*=}";;
      *) ;;
    esac
done

# Forward to build_executorch.sh so the core libs share the runner's -mcpu.
if [[ ${target} =~ ^cortex-m([0-9]+(plus|p)?)(\+|$) ]]; then
    target_cpu="cortex-m${BASH_REMATCH[1]}"
else
    echo "Error: build_test_runner.sh only supports cortex-m<X> targets, got: ${target}"
    exit 1
fi

# Always rebuild executorch in case the cortex-m kernels has been updated.
script_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
et_root_dir=$(realpath "${script_dir}/../../..")
build_executorch="${et_root_dir}/backends/arm/scripts/build_executorch.sh"
${build_executorch} --devtools --target_cpu="${target_cpu}" --cmake-args="-DCORTEX_M_ENABLE_RUNTIME_CHECKS=ON"

# Build executor runner with selected aten ops and semi hosting
build_dir="${et_root_dir}/arm_test"
build_executor_runner="${et_root_dir}/backends/arm/scripts/build_executor_runner.sh"
build_root_test_dir="${et_root_dir}/arm_test/arm_semihosting_executor_runner_corstone-300"

select_ops_list="\
aten::add.out,\
aten::clamp.out,\
aten::mul.out,\
aten::convolution.out,\
dim_order_ops::_clone_dim_order.out,\
aten::cat.out,\
aten::full.out,\
aten::ge.Tensor_out,\
aten::unsqueeze_copy.out,\
aten::select_copy.int_out,\
aten::slice_copy.Tensor_out,\
aten::amax.out"

${build_executor_runner} --pte=semihosting --bundleio --target="${target}" --output="${build_root_test_dir}" --select_ops_list="${select_ops_list}" --extra_build_flags="-DET_ATOL=5.0 -DET_RTOL=1.0 -DET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=0"
