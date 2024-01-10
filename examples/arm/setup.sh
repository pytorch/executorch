#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

if [[ "${1:-'.'}" == "-h" || "${#}" -eq 0 || "${#}" -gt 2 ]]; then
    echo "Usage: $(basename $0) <--i-agree-to-the-contained-eula> [path-to-a-scratch-dir]"
    echo "Supplied args: $*"
    exit 1
fi

########
### Helper functions
########
function get_os_name() {
    # Returns the name of the system i.e. Linux or Darwin
    uname -s
}

function get_cpu_arch() {
    # Returns the cpu architecture like arm64 or x86-64
    uname -m
}

function verify_md5() {
    [[ $# -ne 2 ]]  \
        && { echo "[${FUNCNAME[0]}] Invalid number of args, expecting 2, but got $#"; exit 1; }
    local ref_checksum="${1}"
    local file="${2}"

    local file_checksum="$(md5sum $file | awk '{print $1}')"
    if [[ ${ref_checksum} != ${file_checksum} ]]; then
        echo "Mismatched MD5 checksum for file: ${file}. Expecting ${ref_checksum} but got ${file_checksum}. Exiting."
        exit 1
    fi
}

########
### Hardcoded constants
########
script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

if [[ $(get_cpu_arch) == "x86_64" ]]; then
    # FVP
    fvp_url="https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.22_20_Linux64.tgz?rev=018659bd574f4e7b95fa647e7836ccf4&hash=22A79103C6FA5FFA7AFF3BE0447F3FF9"
    fvp_model_dir="Linux64_GCC-9.3"
    fvp_md5_checksum="98e93b949d0fbac977292d8668d34523"

    # toochain
    toolchain_url="https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu/12.3.rel1/binrel/arm-gnu-toolchain-12.3.rel1-x86_64-arm-none-eabi.tar.xz"
    toolchain_dir="arm-gnu-toolchain-12.3.rel1-x86_64-arm-none-eabi"
    toolchain_md5_checksum="00ebb1b70b1f88906c61206457eacb61"
elif [[ $(get_cpu_arch) == "aarch64" ]]; then
    # FVP
    fvp_url="https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.22_20_Linux64_armv8l.tgz?rev=9cc6e9a32bb947ca9b21fa162144cb01&hash=7657A4CF27D42E892E3F08D452AAB073"
    fvp_model_dir="Linux64_armv8l_GCC-9.3"
    fvp_md5_checksum="cbbabbe39b07939cff7a3738e1492ef1"

    # toochain
    toolchain_url="https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu/12.3.rel1/binrel/arm-gnu-toolchain-12.3.rel1-aarch64-arm-none-eabi.tar.xz"
    toolchain_dir="arm-gnu-toolchain-12.3.rel1-aarch64-arm-none-eabi"
    toolchain_md5_checksum="02c9b0d3bb1110575877d8eee1f223f2"
else
    echo "[main] Error: only x86-64 & aarch64 architecture is supported for now!"; exit 1;
fi

# ethos-u
ethos_u_repo_url="https://review.mlplatform.org/ml/ethos-u/ethos-u"
ethos_u_base_rev="0995223100e3da8011700f58e491f1bf59511e3c"

########
### Mandatory user args
########
eula_acceptance="${1:-'.'}"; shift
if [[ "${eula_acceptance}" != "--i-agree-to-the-contained-eula" ]]; then
    echo "Must pass first positional argument '--i-agree-to-the-contained-eula' to agree to EULA associated with downloading the FVP. Exiting!"
    exit 1
fi

########
### Optional user args
########
root_dir=${1:-"${script_dir}/ethos-u-scratch"}
root_dir=$(realpath ${root_dir})

########
### Functions
########

function setup_fvp() {
    # Download and install the Corstone 300 FVP simulator platform
    cd "${root_dir}"
    if [[ ! -e FVP_cs300.tgz ]]; then
        echo "[${FUNCNAME[0]}] Downloading FVP ..."
        curl --output FVP_cs300.tgz "${fvp_url}"
        verify_md5 ${fvp_md5_checksum} FVP_cs300.tgz
    fi

    echo "[${FUNCNAME[0]}] Installing FVP ..."
    rm -rf FVP
    mkdir -p FVP
    cd FVP
    tar xf ../FVP_cs300.tgz
    ./FVP_Corstone_SSE-300.sh --i-agree-to-the-contained-eula --force --destination ./ --quiet --no-interactive

    fvp_bin_path="$(cd models/${fvp_model_dir} && pwd)"
    export PATH=${PATH}:${fvp_bin_path}

    hash FVP_Corstone_SSE-300_Ethos-U55
    echo "export PATH=\${PATH}:${fvp_bin_path}" >> ${setup_path_script}

}

function setup_toolchain() {
    # Download and install the arm-none-eabi toolchain
    cd "${root_dir}"
    if [[ ! -e gcc.tar.xz ]]; then
        echo "[${FUNCNAME[0]}] Downloading toolchain ..."
        curl --output gcc.tar.xz "${toolchain_url}"
        verify_md5 ${toolchain_md5_checksum} gcc.tar.xz
    fi

    echo "[${FUNCNAME[0]}] Installing toolchain ..."
    rm -rf "${toolchain_dir}"
    tar xf gcc.tar.xz
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
    python3 ./fetch_externals.py fetch
    pip install pyelftools
    echo "[${FUNCNAME[0]}] Done @ $(git describe --all --long 3> /dev/null) in ${root_dir}/ethos-u dir."
}

function patch_repo() {
    # This is a temporary hack until it finds a better home in one for the ARM Ml repos
    name="$(basename $repo_dir)"
    echo -e "[${FUNCNAME[0]}] Preparing ${name}..."
    cd $repo_dir

    git reset --hard ${base_rev}

    patch_dir=${script_dir}/ethos-u-setup/${name}/patches/
    [[ -e ${patch_dir} && $(ls -A ${patch_dir}) ]] && \
        git am -3 ${patch_dir}/*.patch

    echo -e "[${FUNCNAME[0]}] Patched ${name} @ $(git describe --all --long 2> /dev/null) in ${repo_dir} dir.\n"
}

function setup_tosa_reference_model() {
    # The debug flow on the host includes running on a reference implementation of TOSA
    # This is useful primarily for debug of quantization accuracy, but also for internal
    # errors for the early codebase
    cd "${root_dir}"
    if [[ ! -e reference_model ]]; then
        git clone https://review.mlplatform.org/tosa/reference_model -b main
        cd reference_model
        git submodule update --init --recursive
        cd ..
    fi
    cd reference_model
    mkdir -p build
    cd build
    cmake ..
    n=$(nproc)
    make -j"$((n - 5))"
    cd reference_model
    tosa_bin_path=`pwd`
    echo "export PATH=\${PATH}:${tosa_bin_path}" >> "${setup_path_script}"
}

function setup_vela() {
    #
    # Prepare the Vela compiler for AoT to Ethos-U compilation
    #
    cd "${root_dir}"
    if [[ ! -e ethos-u-vela ]]; then
        git clone https://review.mlplatform.org/ml/ethos-u/ethos-u-vela
        repo_dir="${root_dir}/ethos-u-vela"
        base_rev=00a15db3e1a188b25065d095152d701f4394cdc5
        patch_repo
    fi
    cd "${root_dir}/ethos-u-vela"
    pip install .
}

########
### main
########
# do basic checks
# Make sure we are on a supported platform
[[ $(get_cpu_arch) != "x86_64" ]] && [[ $(get_cpu_arch) != "aarch64" ]] \
    && { echo "[main] Error: only x86-64 & aarch64 architecture is supported for now!"; exit 1; }

# No OSx support for FVP
[[ "$(get_os_name)" != "Linux" ]] \
    && { echo "[main] Error: only Linux os is supported for now!"; exit 1; }

cd "${script_dir}"

# Setup the root dir
mkdir -p "${root_dir}"
cd "${root_dir}"
echo "[main] Using root dir ${root_dir}"

setup_path_script="${root_dir}/setup_path.sh"
echo "" > "${setup_path_script}"

# Setup FVP
setup_fvp

# Setup toolchain
setup_toolchain

# Setup the ethos-u dev environment
setup_ethos_u

# Patch the ethos-u dev environment to include executorch application
repo_dir="${root_dir}/ethos-u/core_platform"
base_rev=204210b1074071532627da9dc69950d058a809f4
patch_repo

# Setup the tosa_reference_model
setup_tosa_reference_model

# Setup vela and patch in codegen fixes
setup_vela

echo "[main] update path by doing 'source ${setup_path_script}'"

echo "[main] success!"
exit 0
