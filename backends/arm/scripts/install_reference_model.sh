#!/usr/bin/env bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

# Installation script to manage transition to 1.0

# TOSA reference model
tosa_reference_model_url="https://git.gitlab.arm.com/tosa/tosa-reference-model.git"
tosa_reference_model_0_80_branch="v0.80"
tosa_reference_model_0_80_rev="70ed0b40fa831387e36abdb4f7fb9670a3464f5a"
tosa_serialization_lib_0_80_rev="v0.80.1"
tosa_reference_model_1_0_rev="d102f426dd2e3c1f25bbf23292ec8ee51aa9c677"

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

    # Install a patched version of TOSA reference model v0.80.1 to make it co-exist with 1.0 during the transition period
    if [[ ! -d "reference_model" ]]; then
        git clone --recurse-submodules --branch ${tosa_reference_model_0_80_branch} "$tosa_reference_model_url" reference_model
    fi

    patches_dir=${script_dir}/../third-party/reference_model/patches/v0.80
    patch_repo reference_model ${tosa_reference_model_0_80_rev} ${patches_dir}
    patch_repo reference_model/thirdparty/serialization_lib ${tosa_serialization_lib_0_80_rev} ${patches_dir}

    pushd reference_model
    rm -rf build
    # reference_model flatbuffers version clashes with Vela.
    # go with Vela's since it newer.
    # Vela's flatbuffer requirement is expected to loosen, then remove this. MLETORCH-565
    CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install . --no-dependencies flatbuffers
    popd

    # Install the 1.0 branch from upstream
    CMAKE_POLICY_VERSION_MINIMUM=3.5 BUILD_PYBIND=1 pip install "tosa-tools@git+${tosa_reference_model_url}@${tosa_reference_model_1_0_rev}" ml_dtypes==0.5.1 --no-dependencies flatbuffers
}

setup_tosa_reference_model $1
