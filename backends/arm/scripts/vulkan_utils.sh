#!/usr/bin/env bash
# Copyright 2025-2026 Arm Limited and/or its affiliates.
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

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "${script_dir}/utils.sh"

vulkan_sdk_version="1.4.321.1"
vulkan_sdk_base_dir="vulkan_sdk"

# MLSDK dependencies
mlsdk_manifest_dir="ml-sdk-for-vulkan-manifest"

os_name="${OS:-$(uname -s)}"
vulkan_sdk_arch="${ARCH}"

# Vulkan SDK selection differs between macOS and Linux; macOS has its own SDK version
if [[ "${os_name}" == "Darwin" ]]; then
    # Latest published macOS SDK is 1.4.321.0 (1.4.321.1 is not available for macOS)
    vulkan_sdk_version="1.4.321.0"
    vulkan_sdk_arch="macOS"
    vulkan_sdk_url="https://sdk.lunarg.com/sdk/download/${vulkan_sdk_version}/mac/vulkansdk-macos-${vulkan_sdk_version}.zip"
    vulkan_sdk_sha256="d873c43acacec1e3330fb530dafd541aa5d8a5726575a98a3f70ca505fc203db"
elif [[ "${os_name}" == "Linux" ]] && [[ "${ARCH}" == "x86_64" ]]; then
    vulkan_sdk_url="https://sdk.lunarg.com/sdk/download/${vulkan_sdk_version}/linux/vulkansdk-linux-x86_64-${vulkan_sdk_version}.tar.xz"
    vulkan_sdk_sha256="f22a3625bd4d7a32e7a0d926ace16d5278c149e938dac63cecc00537626cbf73"
elif [[ "${os_name}" == "Linux" ]] && ([[ "${ARCH}" == "aarch64" ]] || [[ "${ARCH}" == "arm64" ]]); then
    if [[ "${vulkan_sdk_arch}" == "arm64" ]]; then
        vulkan_sdk_arch="aarch64"
    fi
    vulkan_sdk_url="https://github.com/jakoch/vulkan-sdk-arm/releases/download/1.4.321.1/vulkansdk-ubuntu-22.04-arm-1.4.321.1.tar.xz"
    vulkan_sdk_sha256="c57e318d0940394d3a304034bb7ddabda788b5b0b54638e80e90f7264efe9f84"
else
    log_step "vulkan" "Error: only macOS and Linux are supported (detected ${os_name}); architecture must be x86-64 or aarch64/arm64"
    exit 1
fi

vulkan_sdk_bin_dir="${vulkan_sdk_base_dir}/${vulkan_sdk_version}/${vulkan_sdk_arch}/bin"

function download_and_extract_vulkan_sdk_linux() {
    local vulkan_sdk_tar_file="${vulkan_sdk_url##*/}"

    if [[ ! -e "${vulkan_sdk_tar_file}" ]]; then
        log_step "vulkan" "Downloading Vulkan SDK (${vulkan_sdk_version})"
        curl -L --output "${vulkan_sdk_tar_file}" "${vulkan_sdk_url}"
        echo "${vulkan_sdk_sha256} ${vulkan_sdk_tar_file}" | sha256sum -c - || exit 1
        rm -fr ${vulkan_sdk_base_dir}
    fi

    mkdir -p ${vulkan_sdk_base_dir}
    tar -C ${vulkan_sdk_base_dir} -xJf "${vulkan_sdk_tar_file}"
}

function install_vulkan_sdk_macos() {
    local vulkan_sdk_zip_file="${vulkan_sdk_url##*/}"

    if [[ ! -e "${vulkan_sdk_zip_file}" ]]; then
        log_step "vulkan" "Downloading Vulkan SDK (${vulkan_sdk_version}) for macOS"
        curl -L --output "${vulkan_sdk_zip_file}" "${vulkan_sdk_url}"
        echo "${vulkan_sdk_sha256}  ${vulkan_sdk_zip_file}" | shasum -a 256 -c - || exit 1
        rm -fr ${vulkan_sdk_base_dir}
    fi

    log_step "vulkan" "Extracting Vulkan SDK installer"
    unzip -q -o "${vulkan_sdk_zip_file}"

    local vulkan_sdk_app_path=""
    vulkan_sdk_app_path="$(find . -maxdepth 3 -type d -name "vulkansdk-macOS-${vulkan_sdk_version}.app" -print -quit)"
    if [[ -z "${vulkan_sdk_app_path}" ]]; then
        vulkan_sdk_app_path="$(find . -maxdepth 3 -type d -name "vulkansdk-macos-${vulkan_sdk_version}.app" -print -quit)"
    fi
    if [[ -z "${vulkan_sdk_app_path}" ]]; then
        log_step "vulkan" "Error: Vulkan SDK installer app not found after extracting ${vulkan_sdk_zip_file}"
        exit 1
    fi

    local vulkan_sdk_installer="${vulkan_sdk_app_path}/Contents/MacOS/$(basename "${vulkan_sdk_app_path}" .app)"
    if [[ ! -x "${vulkan_sdk_installer}" ]]; then
        log_step "vulkan" "Error: Vulkan SDK installer binary not found at ${vulkan_sdk_installer}"
        exit 1
    fi

    local install_root="$(cd "${root_dir}" && pwd)/${vulkan_sdk_base_dir}/${vulkan_sdk_version}"
    mkdir -p "${install_root}"
    local vulkan_sdk_root="${root_dir}/${vulkan_sdk_base_dir}"

    log_step "vulkan" "Installing Vulkan SDK (${vulkan_sdk_version}) to ${install_root}"
    ${vulkan_sdk_installer} --root "${install_root}" --accept-licenses --default-answer --confirm-command install
}

function setup_vulkan_sdk() {
    cd "${root_dir}"

    if [[ "${os_name}" == "Darwin" ]]; then
        install_vulkan_sdk_macos
    else
        download_and_extract_vulkan_sdk_linux
    fi

    if [[ ! -d "${root_dir}/${vulkan_sdk_bin_dir}" ]]; then
        log_step "vulkan" "Error: Vulkan SDK bin directory not found at ${root_dir}/${vulkan_sdk_bin_dir}"
        exit 1
    fi

    vulkan_sdk_bin_path="$(cd "${root_dir}/${vulkan_sdk_bin_dir}" && pwd)"
    if [[ ! -x "${vulkan_sdk_bin_path}/glslc" ]]; then
        log_step "vulkan" "Error: glslc not found at ${vulkan_sdk_bin_path}/glslc"
        exit 1
    fi

    if ${vulkan_sdk_bin_path}/glslc --version > /dev/null 2>&1; then
        log_step "vulkan" "Vulkan SDK validation (glslc) succeeded"
    else
        log_step "vulkan" "Error: Vulkan SDK validation failed"
        ${vulkan_sdk_bin_path}/glslc --version
        exit 1
    fi
}

function setup_path_vulkan() {
    cd "${root_dir}"
    if [[ ! -d "${root_dir}/${vulkan_sdk_bin_dir}" ]]; then
        log_step "vulkan" "Vulkan SDK not found; skipping PATH update"
        return
    fi

    local vulkan_sdk_arch_root="${vulkan_sdk_base_dir}/${vulkan_sdk_version}/${vulkan_sdk_arch}"

    if [[ ! -d "${vulkan_sdk_arch_root}" ]]; then
        log_step "vulkan" "Vulkan SDK arch path not found; skipping PATH update"
        return
    fi

    vulkan_sdk_arch_root="$(cd "${vulkan_sdk_arch_root}" && pwd)"
    vulkan_sdk_bin_path="$(cd "${vulkan_sdk_bin_dir}" && pwd)"

    append_env_in_setup_path PATH ${vulkan_sdk_bin_path}
    if [[ "${OS:-}" == "Darwin" ]]; then
        prepend_env_in_setup_path DYLD_LIBRARY_PATH "${vulkan_sdk_arch_root}/lib"
        local moltenvk_icd_path="${vulkan_sdk_arch_root}/share/vulkan/icd.d/MoltenVK_icd.json"
        if [[ -f "${moltenvk_icd_path}" ]]; then
            prepend_env_in_setup_path VK_DRIVER_FILES "${moltenvk_icd_path}"
            log_step "vulkan" "Configured VK_DRIVER_FILES to include ${moltenvk_icd_path}"
        else
            log_step "vulkan" "MoltenVK ICD manifest not found at ${moltenvk_icd_path}; skipping VK_DRIVER_FILES update"
        fi
    else
        prepend_env_in_setup_path LD_LIBRARY_PATH "${vulkan_sdk_arch_root}/lib"
    fi
    prepend_env_in_setup_path VULKAN_SDK "${vulkan_sdk_arch_root}"
}
