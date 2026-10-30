#!/bin/bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

source "$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)/setup_env.sh"

test_dir="${test_output_dir}/runner_input_restoration"
mkdir -p "${test_dir}"

for target in ethos-u55-128 ethos-u85-128; do
    log_file="${test_dir}/${target}.log"

    # add.py reuses the input arena for its output. Without restoration,
    # x + x + 2 produces 4, 10, 22 instead of 4 on every iteration.
    "${et_root_dir}/examples/arm/run.sh" \
        --et_build_root="${test_dir}" \
        --output="${test_dir}/${target}" \
        --target="${target}" \
        --model_name="${et_root_dir}/examples/arm/example_modules/add.py" \
        --no_delegate \
        --no_quantize \
        --select_ops_list="aten::add.out" \
        --bundleio \
        --extra_build_flags="-DET_NUM_INFERENCES=3" \
        2>&1 | tee "${log_file}"

    grep -q "3 inferences finished" "${log_file}"
    grep -q "Test_result: PASS" "${log_file}"
done
