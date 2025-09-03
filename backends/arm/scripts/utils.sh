#!/usr/bin/env bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

function verify_md5() {
    # Compare the md5 of a file with a provided expected value.

    # Arg 1: Expected checksum for file
    # Arg 2: Path to file
    # Exits with return code 1 if the number of arguments is incorrect.
    # Exits with return code 2 if the calculated mf5 does not match the given. 

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
        exit 2
    fi
}

function patch_repo() {
    # Patch git repo found in $repo_dir, starting from patch $base_rev and applying patches found in $patch_dir/$name.

    # Arg 1: Directory of repo to patch
    # Arg 2: Rev to start patching at
    # Arg 3: Directory 'setup-dir' containing patches in 'setup-dir/$name'
    # Exits with return code 1 if the number of arguments is incorrect.
    # Does not do any error handling if the base_rev or patch_dir is not found etc.

    [[ $# -ne 3 ]]  \
        && { echo "[${FUNCNAME[0]}] Invalid number of args, expecting 3, but got $#"; exit 1; }

    local repo_dir="${1}"
    local base_rev="${2}"
    local name="$(basename $repo_dir)"
    local patch_dir="${3}/$name"

    echo -e "[${FUNCNAME[0]}] Patching ${name} repo_dir:${repo_dir} base_rev:${base_rev} patch_dir:${patch_dir}"
    pushd $repo_dir
    git fetch
    git reset --hard ${base_rev}

    [[ -e ${patch_dir} && $(ls -A ${patch_dir}) ]] && \
        git am -3 ${patch_dir}/*.patch

    echo -e "[${FUNCNAME[0]}] Patched ${name} @ $(git describe --all --long 2> /dev/null) in ${repo_dir} dir.\n"
    popd
}

function check_platform_support() {
    # No args
    # Exits with return code 1 if the platform is unsupported

    # Make sure we are on a supported platform
    if [[ "${ARCH}" != "x86_64" ]] && [[ "${ARCH}" != "aarch64" ]] \
        && [[ "${ARCH}" != "arm64" ]]; then
        echo "[main] Error: only x86-64 & aarch64 architecture is supported for now!"
        exit 1
    fi
}

function check_os_support() {
    # No args
    # Exits with return code 1 if invalid combination of platform and os

    # Check valid combinations of OS and platform

    # Linux on x86_64
    if [[ "${ARCH}" == "x86_64" ]] && [[ "${OS}" != "Linux" ]]; then
        echo "Error: Only Linux is supported on x86_64"
        exit 1
    fi

    # Linux on arm64/aarch64
    # Darwin on arm64/aarch64
    if [[ "${ARCH}" == "aarch64" ]] || [[ "${ARCH}" == "arm64" ]]; then
        if [[ "${OS}" != "Darwin" ]] || [[ "${OS}" != "Linux" ]]; then
            echo "Error: Only Linux and Darwin are supported on arm64"
            exit 1
        fi
    fi
}
