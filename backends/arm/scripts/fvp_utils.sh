#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The purpose of this file is to isolate all the functions to download
# and setup FVP. The reasons for behind this are multiple.
# The FVP needs a EULA acceptance and since the software download
# differs for arch and os it becomes quite messy to try and handle
# inside setup.sh.

# Important to check for unset variables since this script is always sourced from setup.sh
set -u

# Check if the script is being sourced
(return 0 2>/dev/null)
if [[ $? -ne 0 ]]; then
    echo "Error: This script must be sourced."
    exit 1
fi

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "${script_dir}/utils.sh"
source "${script_dir}/../../../.ci/docker/common/arm_fvp_cache.sh"

function install_fvp() {
    if [[ "${OS}" == "Darwin" ]]; then
        log_step "fvp" "Skipping Linux FVP installation because host OS is Darwin"
        return 0
    fi

    # Download and install the Corstone FVP simulator platforms selected in setup.sh.
    local fvp
    for fvp in "${fvps[@]}"; do
        cd "${root_dir}"
        arm_fvp_cache_archive "${ARCH}" "${fvp}" "${root_dir}/FVP_${fvp}.tgz" || exit 1

        log_step "fvp" "Installing FVP ${fvp}"
        rm -rf FVP-${fvp}
        mkdir -p FVP-${fvp}
        cd FVP-${fvp}
        tar xf "${ARM_FVP_ARCHIVE}"

        # Install the FVP
        case ${fvp} in
            corstone300)
                ./FVP_Corstone_SSE-300.sh --i-agree-to-the-contained-eula --force --destination ./ --quiet --no-interactive
                ;;
            corstone320)
                ./FVP_Corstone_SSE-320.sh --i-agree-to-the-contained-eula --force --destination ./ --quiet --no-interactive
                ;;
            corstone1000)
                ./FVP_Corstone_1000-A320.sh --i-agree-to-the-contained-eula --force --destination ./ --quiet --no-interactive
                ;;
            *)
                log_step "fvp" "Error: Unknown FVP model ${fvp}. Exiting."
                exit 1
                ;;
        esac
    done
}

function check_fvp_eula () {
    if [[ "${OS}" == "Darwin" ]]; then
        return 0
    fi

    # Mandatory user arg --i-agree-to-the-contained-eula
    eula_acceptance_by_variable="${ARM_FVP_INSTALL_I_AGREE_TO_THE_CONTAINED_EULA:-False}"

    if [[ "${eula_acceptance}" -eq 0 ]]; then
        if [[ ${eula_acceptance_by_variable} != "True" ]]; then
            log_step "fvp" "Must pass '--i-agree-to-the-contained-eula' to download the FVP"
            log_step "fvp" "Alternatively set ARM_FVP_INSTALL_I_AGREE_TO_THE_CONTAINED_EULA=True"
            log_step "fvp" "Exiting due to missing EULA acceptance"
            exit 1
        else
            log_step "fvp" "Arm EULA accepted via ARM_FVP_INSTALL_I_AGREE_TO_THE_CONTAINED_EULA=True"
        fi
    fi
}

function setup_fvp() {
    if [[ "${OS}" != "Linux" ]]; then
        # Check if FVP is callable
        if command -v FVP_Corstone_SSE-300_Ethos-U55 &> /dev/null; then
            log_step "fvp" "Detected pre-installed MacOS FVP binaries; continuing"
            return 0  # If true exit gracefully and proceed with setup
        else
            log_step "fvp" "Warning: FVP setup only supported on Linux; Mac users should install via https://github.com/Arm-Examples/FVPs-on-Mac and ensure binaries are on PATH"
            return 1  # Throw error. User need to install FVP according to ^^^
        fi
    fi
}

function setup_path_fvp() {
    if [[ "${OS}" == "Darwin" ]]; then
        return 0
    fi

    local fvp
    local fvp_bin_path
    for fvp in "${fvps[@]}"; do
        arm_fvp_select_platform "${ARCH}" "${fvp}" || exit 1
        fvp_bin_path="${root_dir}/FVP-${fvp}/models/${ARM_FVP_MODEL_DIR}"
        append_env_in_setup_path PATH "${fvp_bin_path}"
        # The models link against the libpython shipped with each FVP
        append_env_in_setup_path LD_LIBRARY_PATH "${root_dir}/FVP-${fvp}/python/lib/"
    done

    local fvp_command
    for fvp_command in \
        FVP_Corstone_SSE-300_Ethos-U55 \
        FVP_Corstone_SSE-300_Ethos-U65 \
        FVP_Corstone_SSE-320 \
        FVP_Corstone-1000-A320; do
        echo "hash ${fvp_command} 2>/dev/null || true" >> "${setup_path_script}.sh"
    done
}
