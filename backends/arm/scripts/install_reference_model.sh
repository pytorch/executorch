#!/usr/bin/env bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

# Installation script for TOSA reference model

tosa_reference_model_url="https://git.gitlab.arm.com/tosa/tosa-reference-model.git"
tosa_reference_model_1_0_rev="8aa2896be5b0625a7cde57abb2308da0d426198d" #2025.07.0

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

source ${script_dir}/utils.sh


function setup_tosa_reference_model() {
    local work_dir="$1"

    if [[ -z "$work_dir" ]]; then
        echo "Error: work_dir parameter is required."
        return 1
    fi

    mkdir -p "$work_dir"
    pushd "$work_dir" || exit 1

    # Install the 1.0 branch from upstream
    CMAKE_POLICY_VERSION_MINIMUM=3.5 BUILD_PYBIND=1 pip install "tosa-tools@git+${tosa_reference_model_url}@${tosa_reference_model_1_0_rev}" ml_dtypes==0.5.1 --no-dependencies flatbuffers
}

setup_tosa_reference_model $1
