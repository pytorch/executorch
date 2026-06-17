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
        toolchain_url="https://developer.arm.com/-/media/Files/downloads/gnu/15.2.rel1/binrel/arm-gnu-toolchain-15.2.rel1-x86_64-arm-none-eabi.tar.xz"
        toolchain_dir="arm-gnu-toolchain-15.2.rel1-x86_64-arm-none-eabi"
        toolchain_md5_checksum="da62bef8821e7fc2a9b5d023871036e0"
        toolchain_archive="${toolchain_dir}.tar.xz"
    elif [[ "${ARCH}" == "aarch64" ]] || [[ "${ARCH}" == "arm64" ]] ; then
        if [[ "${OS}" == "Darwin" ]]; then
            toolchain_url="https://developer.arm.com/-/media/Files/downloads/gnu/15.2.rel1/binrel/arm-gnu-toolchain-15.2.rel1-darwin-arm64-arm-none-eabi.tar.xz"
            toolchain_dir="arm-gnu-toolchain-15.2.rel1-darwin-arm64-arm-none-eabi"
            toolchain_md5_checksum="e91fd6348ba0f3e5ec35eeba1ad7e2b8"
            toolchain_archive="${toolchain_dir}.tar.xz"
        elif [[ "${OS}" == "Linux" ]]; then
            toolchain_url="https://developer.arm.com/-/media/Files/downloads/gnu/15.2.rel1/binrel/arm-gnu-toolchain-15.2.rel1-aarch64-arm-none-eabi.tar.xz"
            toolchain_dir="arm-gnu-toolchain-15.2.rel1-aarch64-arm-none-eabi"
            toolchain_md5_checksum="458c5d9b362726c9ac20c96f1894ae13"
            toolchain_archive="${toolchain_dir}.tar.xz"
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
        toolchain_url="https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v1.0.1/toolchain_gnu_linux-x86_64_arm-zephyr-eabi.tar.xz"
        toolchain_dir="arm-zephyr-eabi"
        toolchain_md5_checksum="da38a6b6a20bfae8249ab9c7eef72cdd"
        toolchain_archive="${toolchain_dir}.tar.xz"
    elif [[ "${ARCH}" == "aarch64" ]] || [[ "${ARCH}" == "arm64" ]] ; then
        toolchain_url="https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v1.0.1/toolchain_gnu_linux-aarch64_arm-zephyr-eabi.tar.xz"
        toolchain_dir="arm-zephyr-eabi"
        toolchain_md5_checksum="ebb47b3f29fa468c6c06fcaa43ebe108"
        toolchain_archive="${toolchain_dir}.tar.xz"
    else
        # This should never happen, it should be covered by setup.sh but catch it anyway
        log_step "toolchain" "Error: Unsupported architecture ${ARCH}"
        exit 1
    fi
}

function musl_select_toolchain() {
    if [[ "${OS}" != "Linux" ]] ; then
        log_step "toolchain" "Error: Linux is required for musl toolchain support"
        exit 1
    fi

    if [[ "${ARCH}" == "x86_64" ]] ; then
        toolchain_url="https://musl.cc/aarch64-linux-musl-cross.tgz"
        toolchain_dir="aarch64-linux-musl-cross"
        toolchain_md5_checksum="a6bb806af217a91cf575e15163e8b12b"
        toolchain_archive="${toolchain_dir}.tgz"
    else
        log_step "toolchain" "Error: Unsupported architecture ${ARCH} for musl toolchain"
        exit 1
    fi
}

function select_toolchain() {
    if [[ "${target_toolchain}" == "zephyr" ]]; then
        zephyr_select_toolchain
    elif [[ "${target_toolchain}" == "linux-musl" ]]; then
        musl_select_toolchain
    else
        gcc_select_toolchain
    fi
    log_step "toolchain" "Selected ${toolchain_dir} for ${ARCH}/${OS}"
}

function setup_toolchain() {
    # Download and install the arm toolchain (default is arm-none-eabi)
    # setting --target-toolchain to zephyr selects the arm-zephyr-eabi toolchain, and linux-musl selects the musl-based Linux toolchain
    cd "${root_dir}"
    if [[ -z "${toolchain_archive}" ]]; then
        log_step "toolchain" "Error: Toolchain archive not set"
        exit 1
    fi

    if [[ ! -e "${toolchain_archive}" ]]; then
        log_step "toolchain" "Downloading ${toolchain_dir} toolchain"
        download_with_retry "toolchain" "${toolchain_url}" "${toolchain_archive}" "${toolchain_md5_checksum}" || exit 1
    fi

    log_step "toolchain" "Installing ${toolchain_dir} toolchain"
    rm -rf "${toolchain_dir}"
    tar xf "${toolchain_archive}"
}

function setup_path_toolchain() {
    toolchain_bin_path="$(cd ${toolchain_dir}/bin && pwd)"
    append_env_in_setup_path PATH ${toolchain_bin_path}

    if [[ "${target_toolchain}" == "linux-musl" ]]; then
        local toolchain_root_path="$(cd ${toolchain_dir} && pwd)"
        set_env_in_setup_path MUSL_TOOLCHAIN_ROOT ${toolchain_root_path}
    fi
}
