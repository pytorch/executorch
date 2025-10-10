#!/usr/bin/env bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: More separation from the regular arm executor runner and testing.

set -eu

# Always rebuild executorch in case the cortex-m kernels has been updated.
script_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
et_root_dir=$(realpath "${script_dir}/../../..")
build_executorch="${et_root_dir}/backends/arm/scripts/build_executorch.sh"
${build_executorch}

# Build executor runner with all portable ops selected and semi hosting
build_dir="${et_root_dir}/arm_test"
build_executor_runner="${et_root_dir}/backends/arm/scripts/build_executor_runner.sh"
build_root_test_dir="${et_root_dir}/arm_test/arm_semihosting_executor_runner_corstone-300"

${build_executor_runner} --pte=semihosting --target=ethos-u55-128 --output="${build_root_test_dir}"
