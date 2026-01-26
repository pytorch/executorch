#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
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

function gcc_select_toolchain() {
    if [[ "${ARCH}" == "x86_64" ]] ; then
        toolchain_url="https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu/13.3.rel1/binrel/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi.tar.xz"
        toolchain_dir="arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi"
        toolchain_md5_checksum="0601a9588bc5b9c99ad2b56133b7f118"
    elif [[ "${ARCH}" == "aarch64" ]] || [[ "${ARCH}" == "arm64" ]] ; then
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
        # This should never happen, it should be covered by setup.sh but catch it anyway
        log_step "toolchain" "Error: Unsupported architecture ${ARCH}"
        exit 1
    fi
}

function zephyr_select_toolchain() {
    if [[ "${OS}" != "Linux" ]] ; then
        log_step "toolchain" "Error: Linux is required for Zephyr toolchain support"
        exit 1
    fi

    if [[ "${ARCH}" == "x86_64" ]] ; then
        toolchain_url="https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.17.4/toolchain_linux-x86_64_arm-zephyr-eabi.tar.xz"
        toolchain_dir="arm-zephyr-eabi"
        toolchain_md5_checksum="68ae71edc0106c3093055b97aaa47017"
    elif [[ "${ARCH}" == "aarch64" ]] || [[ "${ARCH}" == "arm64" ]] ; then
        toolchain_url="https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.17.4/toolchain_linux-aarch64_arm-zephyr-eabi.tar.xz"
        toolchain_dir="arm-zephyr-eabi"
        toolchain_md5_checksum="d8a6dfd4314d55da713957d0b161d01f"
    else
        # This should never happen, it should be covered by setup.sh but catch it anyway
        log_step "toolchain" "Error: Unsupported architecture ${ARCH}"
        exit 1
    fi
}

function select_toolchain() {
    if [[ "${target_toolchain}" == "zephyr" ]]; then
        zephyr_select_toolchain
    else
        gcc_select_toolchain
    fi
    log_step "toolchain" "Selected ${toolchain_dir} for ${ARCH}/${OS}"
}

function setup_toolchain() {
    # Download and install the arm toolchain (default is arm-none-eabi)
    # setting --target-toolchain to zephyr sets this to arm-zephyr-eabi
    cd "${root_dir}"
    if [[ ! -e "${toolchain_dir}.tar.xz" ]]; then
        log_step "toolchain" "Downloading ${toolchain_dir} toolchain"
        curl --output "${toolchain_dir}.tar.xz" -L "${toolchain_url}"
        verify_md5 ${toolchain_md5_checksum} "${toolchain_dir}.tar.xz" || exit 1
    fi

    log_step "toolchain" "Installing ${toolchain_dir} toolchain"
    rm -rf "${toolchain_dir}"
    tar xf "${toolchain_dir}.tar.xz"
}

function setup_path_toolchain() {
    toolchain_bin_path="$(cd ${toolchain_dir}/bin && pwd)"
    append_env_in_setup_path PATH ${toolchain_bin_path}
}
