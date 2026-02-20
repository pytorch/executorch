#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright 2025 Arm Limited and/or its affiliates.
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

if [[ "${ARCH}" == "x86_64" ]]; then
    # FVPs
    corstone300_url="https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.22_20_Linux64.tgz?rev=018659bd574f4e7b95fa647e7836ccf4&hash=22A79103C6FA5FFA7AFF3BE0447F3FF9"
    corstone300_model_dir="Linux64_GCC-9.3"
    corstone300_md5_checksum="98e93b949d0fbac977292d8668d34523"

    corstone320_url="https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-320/FVP_Corstone_SSE-320_11.27_25_Linux64.tgz?rev=a507bffc219a4d5792f1192ab7002d89&hash=D9A824AA8227D2E679C9B9787FF4E8B6FBE3D7C6"
    corstone320_model_dir="Linux64_GCC-9.3"
    corstone320_md5_checksum="3deb3c68f9b2d145833f15374203514d"
elif [[ "${ARCH}" == "aarch64" ]] || [[ "${ARCH}" == "arm64" ]]; then
    # FVPs
    corstone300_url="https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.22_20_Linux64_armv8l.tgz?rev=9cc6e9a32bb947ca9b21fa162144cb01&hash=7657A4CF27D42E892E3F08D452AAB073"
    corstone300_model_dir="Linux64_armv8l_GCC-9.3"
    corstone300_md5_checksum="cbbabbe39b07939cff7a3738e1492ef1"

    corstone320_url="https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-320/FVP_Corstone_SSE-320_11.27_25_Linux64_armv8l.tgz?rev=b6ebe0923cb84f739e017385fd3c333c&hash=8965C4B98E2FF7F792A099B08831FE3CB6120493"
    corstone320_model_dir="Linux64_armv8l_GCC-9.3"
    corstone320_md5_checksum="3889f1d80a6d9861ea4aa6f1c88dd0ae"
else
    log_step "fvp" "Error: only x86-64 & aarch64/arm64 architecture is supported for now!"
    exit 1
fi

function install_fvp() {
    # Download and install the Corstone 300 FVP simulator platform
    fvps=("corstone300" "corstone320")

    for fvp in "${fvps[@]}"; do
        cd "${root_dir}"
        if [[ ! -e "FVP_${fvp}.tgz" ]]; then
            log_step "fvp" "Downloading FVP ${fvp}"
            url_variable=${fvp}_url
            fvp_url=${!url_variable}
            curl --output "FVP_${fvp}.tgz" "${fvp_url}"
            md5_variable=${fvp}_md5_checksum
            fvp_md5_checksum=${!md5_variable}
            verify_md5 ${fvp_md5_checksum} FVP_${fvp}.tgz || exit 1
        fi

        log_step "fvp" "Installing FVP ${fvp}"
        rm -rf FVP-${fvp}
        mkdir -p FVP-${fvp}
        cd FVP-${fvp}
        tar xf ../FVP_${fvp}.tgz

        # Install the FVP
        case ${fvp} in
            corstone300)
                ./FVP_Corstone_SSE-300.sh --i-agree-to-the-contained-eula --force --destination ./ --quiet --no-interactive
                ;;
            corstone320)
                ./FVP_Corstone_SSE-320.sh --i-agree-to-the-contained-eula --force --destination ./ --quiet --no-interactive
                ;;
            *)
                log_step "fvp" "Error: Unknown FVP model ${fvp}. Exiting."
                exit 1
                ;;
        esac
    done
}

function check_fvp_eula () {
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
    fvps=("corstone300" "corstone320")
    for fvp in "${fvps[@]}"; do
        model_dir_variable=${fvp}_model_dir
        fvp_model_dir=${!model_dir_variable}
        fvp_bin_path="${root_dir}/FVP-${fvp}/models/${fvp_model_dir}"
        append_env_in_setup_path PATH ${fvp_bin_path}
    done

    # Fixup for Corstone-320 python dependency
    append_env_in_setup_path LD_LIBRARY_PATH "${root_dir}/FVP-corstone320/python/lib/"

    echo "hash FVP_Corstone_SSE-300_Ethos-U55" >> ${setup_path_script}.sh
    echo "hash FVP_Corstone_SSE-300_Ethos-U65" >> ${setup_path_script}.sh
    echo "hash FVP_Corstone_SSE-320" >> ${setup_path_script}.sh
}
