#!/usr/bin/env bash
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

script_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
et_root_dir=$(realpath "${script_dir}/../../..")
build_executor_runner=${et_root_dir}/backends/arm/scripts/build_executor_runner_vkml.sh
build_root_test_dir=${et_root_dir}/arm_test/arm_executor_runner_vkml

${build_executor_runner} --output="${build_root_test_dir}"
