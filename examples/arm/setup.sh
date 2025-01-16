#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

if [[ "${1:-'.'}" == "-h" || "${#}" -gt 2 ]]; then
    echo "Usage: $(basename $0) <--i-agree-to-the-contained-eula> [path-to-a-scratch-dir]"
    echo "Supplied args: $*"
    exit 1
fi


########
### Helper functions
########
ARCH="$(uname -m)"
OS="$(uname -s)"

function verify_md5() {
    [[ $# -ne 2 ]]  \
        && { echo "[${FUNCNAME[0]}] Invalid number of args, expecting 2, but got $#"; exit 1; }
    local ref_checksum="${1}"
    local file="${2}"

    if [[ "${OS}" == "Darwin" ]]; then
        local file_checksum="$(md5 -q $file)"
    else
        local file_checksum="$(md5sum $file | awk '{print $1}')"
    fi
    if [[ ${ref_checksum} != ${file_checksum} ]]; then
        echo "Mismatched MD5 checksum for file: ${file}. Expecting ${ref_checksum} but got ${file_checksum}. Exiting."
        exit 1
    fi
}

########
### Hardcoded constants
########
script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

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

# ethos-u
ethos_u_repo_url="https://review.mlplatform.org/ml/ethos-u/ethos-u"
ethos_u_base_rev="24.08"

# tosa reference model
tosa_reference_model_url="https://review.mlplatform.org/tosa/reference_model"
tosa_reference_model_rev="v0.80.1"

# vela
vela_repo_url="https://review.mlplatform.org/ml/ethos-u/ethos-u-vela"
vela_rev="5427dc7e9c1a4c7d554163290faeea75f168772d"

########
### Mandatory user args
########
eula_acceptance="${1:-'.'}"
if [[ "${eula_acceptance}" != "--i-agree-to-the-contained-eula" ]]; then
    if [[ ${ARM_FVP_INSTALL_I_AGREE_TO_THE_CONTAINED_EULA} != "True" ]]; then
	echo "Must pass first positional argument '--i-agree-to-the-contained-eula' to agree to EULA associated with downloading the FVP. Exiting!"
	exit 1
    else
	echo "Arm EULA for FVP agreed to with ARM_FVP_INSTALL_I_AGREE_TO_THE_CONTAINED_EULA=True environment variable"
    fi
else
    shift; # drop this arg
fi

########
### Optional user args
########
root_dir=${1:-"${script_dir}/ethos-u-scratch"}
mkdir -p ${root_dir}
root_dir=$(realpath ${root_dir})

########
### Functions
########

function setup_fvp() {
    if [[ "${OS}" != "Linux" ]]; then
        echo "[${FUNCNAME[0]}] Warning: FVP only supported with Linux OS, skipping FVP setup..."
        echo "[${FUNCNAME[0]}] Warning: For MacOS, using https://github.com/Arm-Examples/FVPs-on-Mac is recommended."
        echo "[${FUNCNAME[0]}] Warning:   Follow the instructions and make sure the path is set correctly." 
        return 1
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
            verify_md5 ${fvp_md5_checksum} FVP_${fvp}.tgz
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

        model_dir_variable=${fvp}_model_dir
        fvp_model_dir=${!model_dir_variable}
        fvp_bin_path="$(cd models/${fvp_model_dir} && pwd)"
        export PATH=${PATH}:${fvp_bin_path}

        echo "export PATH=\${PATH}:${fvp_bin_path}" >> ${setup_path_script}
    done

    # Fixup for Corstone-320 python dependency
    echo "export LD_LIBRARY_PATH=${root_dir}/FVP-corstone320/python/lib/" >> ${setup_path_script}
}

function setup_toolchain() {
    # Download and install the arm-none-eabi toolchain
    cd "${root_dir}"
    if [[ ! -e "${toolchain_dir}.tar.xz" ]]; then
        echo "[${FUNCNAME[0]}] Downloading toolchain ..."
        curl --output "${toolchain_dir}.tar.xz" "${toolchain_url}"
        verify_md5 ${toolchain_md5_checksum} "${toolchain_dir}.tar.xz"
    fi

    echo "[${FUNCNAME[0]}] Installing toolchain ..."
    rm -rf "${toolchain_dir}"
    tar xf "${toolchain_dir}.tar.xz"
    toolchain_bin_path="$(cd ${toolchain_dir}/bin && pwd)"
    export PATH=${PATH}:${toolchain_bin_path}
    hash arm-none-eabi-gcc
    echo "export PATH=\${PATH}:${toolchain_bin_path}" >> ${setup_path_script}
}

function setup_ethos_u() {
    # This is the main dir which will pull more repos to do baremetal software dev for cs300
    echo "[${FUNCNAME[0]}] Setting up the repo"
    cd "${root_dir}"
    [[ ! -d ethos-u ]] && \
        git clone ${ethos_u_repo_url}
    cd ethos-u
    git reset --hard ${ethos_u_base_rev}
    python3 ./fetch_externals.py -c ${ethos_u_base_rev}.json fetch

    pip install pyelftools
    echo "[${FUNCNAME[0]}] Done @ $(git describe --all --long 3> /dev/null) in ${root_dir}/ethos-u dir."
}

function patch_repo() {
    # This is a temporary hack until it finds a better home in one for the ARM Ml repos
    name="$(basename $repo_dir)"
    echo -e "[${FUNCNAME[0]}] Preparing ${name}..."
    cd $repo_dir
    git fetch
    git reset --hard ${base_rev}

    patch_dir=${script_dir}/ethos-u-setup/${name}/patches/
    [[ -e ${patch_dir} && $(ls -A ${patch_dir}) ]] && \
        git am -3 ${patch_dir}/*.patch

    echo -e "[${FUNCNAME[0]}] Patched ${name} @ $(git describe --all --long 2> /dev/null) in ${repo_dir} dir.\n"
}

function setup_tosa_reference_model() {
    
    # reference_model flatbuffers version clashes with Vela.
    # go with Vela's since it newer.
    # Vela's flatbuffer requirement is expected to loosen, then remove this. MLETORCH-565
    pip install tosa-tools@git+${tosa_reference_model_url}@${tosa_reference_model_rev} --no-dependencies flatbuffers

}

function setup_vela() {
    #
    # Prepare the Vela compiler for AoT to Ethos-U compilation
    #
    pip install ethos-u-vela@git+${vela_repo_url}@${vela_rev}
}

########
### main
########
# do basic checks
# Make sure we are on a supported platform
if [[ "${ARCH}" != "x86_64" ]] && [[ "${ARCH}" != "aarch64" ]] \
    && [[ "${ARCH}" != "arm64" ]]; then
    echo "[main] Error: only x86-64 & aarch64 architecture is supported for now!"
    exit 1
fi

cd "${script_dir}"

# Setup the root dir
cd "${root_dir}"
echo "[main] Using root dir ${root_dir}"

setup_path_script="${root_dir}/setup_path.sh"
echo "" > "${setup_path_script}"

# Setup toolchain
setup_toolchain

# Setup the ethos-u dev environment
setup_ethos_u

# Patch the ethos-u dev environment to include executorch application
repo_dir="${root_dir}/ethos-u/core_platform"
base_rev=b728c774158248ba2cad8e78a515809e1eb9b77f
patch_repo

# Setup the tosa_reference_model
setup_tosa_reference_model

# Setup vela and patch in codegen fixes
setup_vela

# Setup FVP
setup_fvp

echo "[main] update path by doing 'source ${setup_path_script}'"

echo "[main] success!"
exit 0
