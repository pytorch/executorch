#!/bin/bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

et_root_dir=$(realpath "${1:?Usage: $(basename "$0") <EXECUTORCH_ROOT> <TEST_OUTPUT_DIR>}")
test_output_dir="${2:?Usage: $(basename "$0") <EXECUTORCH_ROOT> <TEST_OUTPUT_DIR>}"

scratch_dir=${et_root_dir}/examples/arm/arm-scratch
setup_path_script=${scratch_dir}/setup_path.sh
_setup_msg="please refer to ${et_root_dir}/examples/arm/setup.sh to properly install necessary tools."

[[ -f ${setup_path_script} ]] \
    || { echo "Missing ${setup_path_script}. ${_setup_msg}"; exit 1; }
source "${setup_path_script}"
