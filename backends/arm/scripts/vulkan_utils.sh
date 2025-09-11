#!/usr/bin/env bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Important to check for unset variables since this script is always sourced from setup.sh
set -u

# Check if the script is being sourced
(return 0 2>/dev/null)
if [[ $? -ne 0 ]]; then
    echo "Error: This script must be sourced."
    exit 1
fi

vulkan_sdk_version="1.4.321.1"
vulkan_sdk_base_dir="vulkan_sdk"

# MLSDK dependencies
mlsdk_manifest_dir="ml-sdk-for-vulkan-manifest"
vulkan_sdk_bin_dir="${vulkan_sdk_base_dir}/${vulkan_sdk_version}/${ARCH}/bin"


if [[ "${ARCH}" == "x86_64" ]]; then
    # Vulkan SDK
    vulkan_sdk_url="https://sdk.lunarg.com/sdk/download/${vulkan_sdk_version}/linux/vulkansdk-linux-x86_64-${vulkan_sdk_version}.tar.xz"
    vulkan_sdk_sha256="f22a3625bd4d7a32e7a0d926ace16d5278c149e938dac63cecc00537626cbf73"

elif [[ "${ARCH}" == "aarch64" ]] || [[ "${ARCH}" == "arm64" ]]; then
    # Vulkan SDK
    vulkan_sdk_url="https://github.com/jakoch/vulkan-sdk-arm/releases/download/1.4.321.1/vulkansdk-ubuntu-22.04-arm-1.4.321.1.tar.xz"
    vulkan_sdk_sha256="c57e318d0940394d3a304034bb7ddabda788b5b0b54638e80e90f7264efe9f84"
else
    echo "[main] Error: only x86-64 & aarch64/arm64 architecture is supported for now!"; exit 1;
fi

function setup_vulkan_sdk() {
    if command -v glslc > /dev/null 2>&1; then
        echo "[${FUNCNAME[0]}] GLSL already installed, no need to get Vulkan SDK..."
        enable_vulkan_sdk=0
        return
    fi

    cd "${root_dir}"

    vulkan_sdk_tar_file="${vulkan_sdk_url##*/}"
    if [[ ! -e "${vulkan_sdk_tar_file}" ]]; then
        echo "[${FUNCNAME[0]}] Downloading Vulkan SDK - ${vulkan_sdk_url}.."
        curl -L --output "${vulkan_sdk_tar_file}" "${vulkan_sdk_url}"
        echo "${vulkan_sdk_sha256} ${vulkan_sdk_tar_file}" | sha256sum -c -
        rm -fr ${vulkan_sdk_base_dir}
    fi

    mkdir -p ${vulkan_sdk_base_dir}
    tar -C ${vulkan_sdk_base_dir} -xJf "${vulkan_sdk_tar_file}"

    vulkan_sdk_bin_path="$(cd ${vulkan_sdk_bin_dir} && pwd)"
    if ${vulkan_sdk_bin_path}/glslc --version > /dev/null 2>&1; then
        echo "[${FUNCNAME[0]}] Vulkan SDK install (GLSL) OK"
    else
        echo "[${FUNCNAME[0]}] Vulkan SDK install NOK - glslc returned error"
        ${vulkan_sdk_bin_path}/glslc --version
        exit 1
    fi
}

function setup_path_vulkan() {
    cd "${root_dir}"
    vulkan_sdk_bin_path="$(cd ${vulkan_sdk_bin_dir} && pwd)"
    append_env_in_setup_path PATH ${vulkan_sdk_bin_path}
}
