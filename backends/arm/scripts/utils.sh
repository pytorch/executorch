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

# Usage:
#   log_step <context> <message>
# eg.
# log_step "step" "information message"
# outputs:
#   [setup/step] information message
function log_step() {
    local context="${1:-main}"
    shift || true
    local message="$*"
    printf "[Arm Setup/%s] %s\n" "${context}" "${message}"
}

function get_parallel_jobs() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1 && sysctl hw.logicalcpu >/dev/null 2>&1; then
        sysctl -n hw.logicalcpu
    elif command -v getconf >/dev/null 2>&1; then
        getconf _NPROCESSORS_ONLN
    elif [[ -n "${NUMBER_OF_PROCESSORS:-}" ]]; then
        echo "${NUMBER_OF_PROCESSORS}"
    else
        echo 1
    fi
}

function verify_md5() {
    # Compare the md5 of a file with a provided expected value.

    # Arg 1: Expected checksum for file
    # Arg 2: Path to file
    # Exits with return code 1 if the number of arguments is incorrect.
    # Returns 2 if the calculated md5 does not match the given. Returning
    # rather than exiting lets callers like download_with_retry treat a bad
    # checksum as a retryable failure (e.g. truncated download) instead of
    # tearing down the whole script.

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
        echo "Mismatched MD5 checksum for file: ${file}. Expecting ${ref_checksum} but got ${file_checksum}."
        return 2
    fi
}

function download_with_retry() {
    # Download a URL to a path and validate its MD5, retrying on transport
    # or checksum errors. developer.arm.com's CDN intermittently aborts the
    # download mid-stream with HTTP/2 INTERNAL_ERROR (curl exit 92), and
    # rare cases return a short error body that curl treats as success;
    # both are caught here. --fail rejects HTTP errors,
    # --retry-all-errors handles transport errors, and verify_md5 catches
    # truncation / wrong-content via the published archive checksum.

    # Arg 1: log context (passed to log_step)
    # Arg 2: URL to download
    # Arg 3: Output path
    # Arg 4: Expected MD5 checksum

    [[ $# -ne 4 ]] \
        && { echo "[${FUNCNAME[0]}] Invalid number of args, expecting 4, but got $#"; exit 1; }
    local context="${1}"
    local url="${2}"
    local output="${3}"
    local expected_md5="${4}"

    local max_attempts=5
    for attempt in $(seq 1 ${max_attempts}); do
        rm -f "${output}"
        if curl --fail --retry 3 --retry-delay 5 --retry-connrefused --retry-all-errors \
             -L --output "${output}" "${url}" \
           && verify_md5 "${expected_md5}" "${output}"; then
            return 0
        fi
        ls -l "${output}" 2>&1 || true
        if [[ "${attempt}" = "${max_attempts}" ]]; then
            log_step "${context}" "ERROR: download of ${url} failed after ${attempt} attempts"
            return 1
        fi
        log_step "${context}" "download attempt ${attempt} failed; retrying in $((attempt * 10))s..."
        sleep $((attempt * 10))
    done
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

    echo -e "[${FUNCNAME[0]}] Patching ${name}. repo_dir:${repo_dir}\t base_rev:${base_rev}\t patch_dir:${patch_dir}"
    pushd $repo_dir > /dev/null
    git fetch --quiet
    git reset --hard ${base_rev} --quiet

    [[ -e ${patch_dir} && $(ls -A ${patch_dir}) ]] && \
        git am -3 ${patch_dir}/*.patch

    echo -e "[${FUNCNAME[0]}] Patched ${name} @ $(git describe --all --long 2> /dev/null) in ${repo_dir} dir."
    popd > /dev/null
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
        if [[ "${OS}" != "Darwin" ]] && [[ "${OS}" != "Linux" ]]; then
            echo "Error: Only Linux and Darwin are supported on arm64"
            exit 1
        fi
    fi
}

function prepend_env_in_setup_path() {
    echo "export $1=$2:\${$1-}" >> ${setup_path_script}.sh
    echo "set --path -pgx $1 $2" >> ${setup_path_script}.fish
}

function append_env_in_setup_path() {
    echo "export $1=\${$1-}:$2" >> ${setup_path_script}.sh
    echo "set --path -agx $1 $2" >> ${setup_path_script}.fish
}

function set_env_in_setup_path() {
    echo "export $1=$2" >> ${setup_path_script}.sh
    echo "set -gx $1 $2" >> ${setup_path_script}.fish
}

function clear_setup_path() {
    # Clear setup_path_script
    echo "" > "${setup_path_script}.sh"
    echo "" > "${setup_path_script}.fish"
}
