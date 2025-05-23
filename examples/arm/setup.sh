#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -u

########
### Hardcoded constants
########
script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
et_dir=$(realpath $script_dir/../..)
ARCH="$(uname -m)"
OS="$(uname -s)"

# Figure out if setup.sh was called or sourced and save it into "is_script_sourced"
(return 0 2>/dev/null) && is_script_sourced=1 || is_script_sourced=0

if [[ "${ARCH}" == "x86_64" ]]; then
    # FVPs
    corstone300_url="https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.22_20_Linux64.tgz?rev=018659bd574f4e7b95fa647e7836ccf4&hash=22A79103C6FA5FFA7AFF3BE0447F3FF9"
    corstone300_model_dir="Linux64_GCC-9.3"
    corstone300_md5_checksum="98e93b949d0fbac977292d8668d34523"

    corstone320_url="https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-320/FVP_Corstone_SSE-320_11.27_25_Linux64.tgz?rev=a507bffc219a4d5792f1192ab7002d89&hash=D9A824AA8227D2E679C9B9787FF4E8B6FBE3D7C6"
    corstone320_model_dir="Linux64_GCC-9.3"
    corstone320_md5_checksum="3deb3c68f9b2d145833f15374203514d"

    # toochain
    toolchain_url="https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu/13.3.rel1/binrel/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi.tar.xz"
    toolchain_dir="arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi"
    toolchain_md5_checksum="0601a9588bc5b9c99ad2b56133b7f118"
elif [[ "${ARCH}" == "aarch64" ]] || [[ "${ARCH}" == "arm64" ]]; then
    # FVPs
    corstone300_url="https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.22_20_Linux64_armv8l.tgz?rev=9cc6e9a32bb947ca9b21fa162144cb01&hash=7657A4CF27D42E892E3F08D452AAB073"
    corstone300_model_dir="Linux64_armv8l_GCC-9.3"
    corstone300_md5_checksum="cbbabbe39b07939cff7a3738e1492ef1"

    corstone320_url="https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-320/FVP_Corstone_SSE-320_11.27_25_Linux64_armv8l.tgz?rev=b6ebe0923cb84f739e017385fd3c333c&hash=8965C4B98E2FF7F792A099B08831FE3CB6120493"
    corstone320_model_dir="Linux64_armv8l_GCC-9.3"
    corstone320_md5_checksum="3889f1d80a6d9861ea4aa6f1c88dd0ae"

    # toochain
    if [[ "${OS}" == "Darwin" ]]; then
        toolchain_url="https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu/13.3.rel1/binrel/arm-gnu-toolchain-13.3.rel1-darwin-arm64-arm-none-eabi.tar.xz"
        toolchain_dir="arm-gnu-toolchain-13.3.rel1-darwin-arm64-arm-none-eabi"
        toolchain_md5_checksum="f1c18320bb3121fa89dca11399273f4e"
    elif [[ "${OS}" == "Linux" ]]; then
        toolchain_url="https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu/13.3.rel1/binrel/arm-gnu-toolchain-13.3.rel1-aarch64-arm-none-eabi.tar.xz"
        toolchain_dir="arm-gnu-toolchain-13.3.rel1-aarch64-arm-none-eabi"
        toolchain_md5_checksum="303102d97b877ebbeb36b3158994b218"
    fi
else
    echo "[main] Error: only x86-64 & aarch64/arm64 architecture is supported for now!"; exit 1;
fi

# vela
vela_repo_url="https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela"
vela_rev="8cac2b9a7204b57125a8718049519b091a98846c"

########
### Functions
########
function setup_root_dir() {
    # Handle a different root_dir set by the user as argument to the
    # script. This can only happen if the script is being executed and
    # not sourced.
    root_dir="${script_dir}/ethos-u-scratch"
    if [[ $is_script_sourced -eq 0 ]]; then
        root_dir=${2:-"${script_dir}/ethos-u-scratch"}
    fi
    mkdir -p ${root_dir}
    root_dir=$(realpath ${root_dir})
    setup_path_script="${root_dir}/setup_path.sh"
}

function check_fvp_eula () {
    # Mandatory user arg --i-agree-to-the-contained-eula
    eula_acceptance="${1:-'.'}"
    eula_acceptance_by_variable="${ARM_FVP_INSTALL_I_AGREE_TO_THE_CONTAINED_EULA:-False}"

    if [[ "${eula_acceptance}" != "--i-agree-to-the-contained-eula" ]]; then
        if [[ ${eula_acceptance_by_variable} != "True" ]]; then
            echo "Must pass first positional argument '--i-agree-to-the-contained-eula' to agree to EULA associated with downloading the FVP."
	    echo "Alternativly set environment variable ARM_FVP_INSTALL_I_AGREE_TO_THE_CONTAINED_EULA=True."
	    echo "Exiting!"
            exit 1
        else
            echo "Arm EULA for FVP agreed to with ARM_FVP_INSTALL_I_AGREE_TO_THE_CONTAINED_EULA=True environment variable"
        fi
    else
        shift; # drop this arg
    fi
}

function setup_fvp() {
    # check EULA, forward argument
    check_fvp_eula ${1:-'.'}

    if [[ "${OS}" != "Linux" ]]; then
        # Check if FVP is callable
        if command -v FVP_Corstone_SSE-300_Ethos-U55 &> /dev/null; then
            echo "[${FUNCNAME[0]}] Info: FVP for MacOS seem to be installed. Continuing..."
            return 0  # If true exit gracefully and proceed with setup
        else
            echo "[${FUNCNAME[0]}] Warning: FVP only supported with Linux OS, skipping FVP setup..."
            echo "[${FUNCNAME[0]}] Warning: For MacOS, using https://github.com/Arm-Examples/FVPs-on-Mac is recommended."
            echo "[${FUNCNAME[0]}] Warning:   Follow the instructions and make sure the path is set correctly."
            return 1  # Throw error. User need to install FVP according to ^^^
        fi
    fi

    # Download and install the Corstone 300 FVP simulator platform
    fvps=("corstone300" "corstone320")

    for fvp in "${fvps[@]}"; do
        cd "${root_dir}"
        if [[ ! -e "FVP_${fvp}.tgz" ]]; then
            echo "[${FUNCNAME[0]}] Downloading FVP ${fvp}..."
            url_variable=${fvp}_url
            fvp_url=${!url_variable}
            curl --output "FVP_${fvp}.tgz" "${fvp_url}"
            md5_variable=${fvp}_md5_checksum
            fvp_md5_checksum=${!md5_variable}
            verify_md5 ${fvp_md5_checksum} FVP_${fvp}.tgz || exit 1
        fi

        echo "[${FUNCNAME[0]}] Installing FVP ${fvp}..."
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
                echo "[${FUNCNAME[0]}] Error: Unknown FVP model ${fvp}. Exiting."
                exit 1
                ;;
        esac
    done
}

function setup_toolchain() {
    # Download and install the arm-none-eabi toolchain
    cd "${root_dir}"
    if [[ ! -e "${toolchain_dir}.tar.xz" ]]; then
        echo "[${FUNCNAME[0]}] Downloading toolchain ..."
        curl --output "${toolchain_dir}.tar.xz" "${toolchain_url}"
        verify_md5 ${toolchain_md5_checksum} "${toolchain_dir}.tar.xz" || exit 1
    fi

    echo "[${FUNCNAME[0]}] Installing toolchain ..."
    rm -rf "${toolchain_dir}"
    tar xf "${toolchain_dir}.tar.xz"
}

function setup_vela() {
    pip install ethos-u-vela@git+${vela_repo_url}@${vela_rev}
}

function setup_path() {
    echo $setup_path_script
}

function create_setup_path(){
    echo "" > "${setup_path_script}"
    fvps=("corstone300" "corstone320")
    for fvp in "${fvps[@]}"; do
        model_dir_variable=${fvp}_model_dir
        fvp_model_dir=${!model_dir_variable}
        fvp_bin_path="${root_dir}/FVP-${fvp}/models/${fvp_model_dir}"
        echo "export PATH=\${PATH}:${fvp_bin_path}" >> ${setup_path_script}
    done

    # Fixup for Corstone-320 python dependency
    echo "export LD_LIBRARY_PATH=${root_dir}/FVP-corstone320/python/lib/" >> ${setup_path_script}

    toolchain_bin_path="$(cd ${toolchain_dir}/bin && pwd)"
    echo "export PATH=\${PATH}:${toolchain_bin_path}" >> ${setup_path_script}

    echo "hash FVP_Corstone_SSE-300_Ethos-U55" >> ${setup_path_script}
    echo "hash FVP_Corstone_SSE-300_Ethos-U65" >> ${setup_path_script}
    echo "hash FVP_Corstone_SSE-320" >> ${setup_path_script}
}

function check_platform_support() {
    if [[ "${ARCH}" != "x86_64" ]] && [[ "${ARCH}" != "aarch64" ]] \
        && [[ "${ARCH}" != "arm64" ]]; then
        echo "[main] Error: only x86-64 & aarch64 architecture is supported for now!"
        exit 1
    fi

    # Make sure we are on a supported platform
    if [[ "${1:-'.'}" == "-h" || "${#}" -gt 2 ]]; then
        echo "Usage: $(basename $0) <--i-agree-to-the-contained-eula> [path-to-a-scratch-dir]"
        echo "Supplied args: $*"
        exit 1
    fi
}

########
### main
########

# script is not sourced! Lets run "main"
if [[ $is_script_sourced -eq 0 ]]
    then
    set -e

    check_platform_support

    cd "${script_dir}"

    # Setup the root dir
    setup_root_dir
    cd "${root_dir}"
    echo "[main] Using root dir ${root_dir}"

    # Import utils
    source $et_dir/backends/arm/scripts/utils.sh

    # Setup FVP
    setup_fvp ${1:-'.'}

    # Setup toolchain
    setup_toolchain

    # Create new setup_path script only if fvp and toolchain setup went well.
    create_setup_path

    # Setup the tosa_reference_model
    $et_dir/backends/arm/scripts/install_reference_model.sh ${root_dir}

    # Setup vela and patch in codegen fixes
    setup_vela

    echo "[main] update path by doing 'source ${setup_path_script}'"

    echo "[main] success!"
    exit 0
fi
