#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
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
        echo "[gcc_select_toolchain]: Unsupported arch!"; exit 1
    fi
}

function zephyr_select_toolchain() {
    if [[ "${OS}" != "Linux" ]] ; then
        echo "[zephyr_select_toolchain] Error: Linux is only supported for zephyr!"; exit 1;
    fi

    if [[ "${ARCH}" == "x86_64" ]] ; then
        toolchain_url="https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.17.2/toolchain_linux-x86_64_arm-zephyr-eabi.tar.xz"
        toolchain_dir="arm-zephyr-eabi"
        toolchain_md5_checksum="93128be0235cf5cf5f1ee561aa6eac5f"
    elif [[ "${ARCH}" == "aarch64" ]] || [[ "${ARCH}" == "arm64" ]] ; then
        toolchain_url="https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.17.2/toolchain_linux-aarch64_arm-zephyr-eabi.tar.xz"
        toolchain_dir="arm-zephyr-eabi"
        toolchain_md5_checksum="ef4ca56786204439a75270ba800cc64b"
    else
        # This should never happen, it should be covered by setup.sh but catch it anyway
        echo "[zephyr_select_toolchain]: Unsupported arch!"; exit 1
    fi
}

function select_toolchain() {
    if [[ "${target_toolchain}" == "zephyr" ]]; then
        zephyr_select_toolchain
    else
        gcc_select_toolchain
    fi
    echo "[main] Info selected ${toolchain_dir} for ${ARCH} - ${OS} platform"
}

function setup_toolchain() {
    # Download and install the arm toolchain (default is arm-none-eabi)
    # setting --target-toolchain to zephyr sets this to arm-zephyr-eabi
    cd "${root_dir}"
    if [[ ! -e "${toolchain_dir}.tar.xz" ]]; then
        echo "[${FUNCNAME[0]}] Downloading ${toolchain_dir} toolchain ..."
        curl --output "${toolchain_dir}.tar.xz" -L "${toolchain_url}"
        verify_md5 ${toolchain_md5_checksum} "${toolchain_dir}.tar.xz" || exit 1
    fi

    echo "[${FUNCNAME[0]}] Installing ${toolchain_dir} toolchain ..."
    rm -rf "${toolchain_dir}"
    tar xf "${toolchain_dir}.tar.xz"
}

function setup_path_toolchain() {
    toolchain_bin_path="$(cd ${toolchain_dir}/bin && pwd)"
    append_env_in_setup_path PATH ${toolchain_bin_path}
}
