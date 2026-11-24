#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../ci_commit_pins/arm-fvp.sh"

function arm_fvp_validate_archive() {
    local archive="${1}"
    local expected_md5="${2}"
    [[ -f "${archive}" ]] || return 1

    local actual_md5
    if command -v md5sum >/dev/null 2>&1; then
        actual_md5=$(md5sum "${archive}") || return 1
        actual_md5="${actual_md5%% *}"
    else
        actual_md5=$(md5 -q "${archive}") || return 1
    fi
    [[ "${actual_md5}" == "${expected_md5}" ]]
}

# Sets ARM_FVP_ARCHIVE to a validated archive without modifying an existing installation.
function arm_fvp_cache_archive() {
    arm_fvp_select_platform "${1}" "${2}" || return 1
    local legacy_archive="${3:-}"
    local cache_dir
    local cache_only="${ARM_FVP_CACHE_ONLY:-0}"
    if [[ -z "${ARM_FVP_CACHE_DIR:-}" && -d /opt/arm-fvp-cache ]]; then
        cache_dir="/opt/arm-fvp-cache"
        cache_only="${ARM_FVP_CACHE_ONLY:-1}"
    else
        cache_dir="${ARM_FVP_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME:?Set HOME or ARM_FVP_CACHE_DIR}/.cache}/executorch/fvps}"
    fi
    if [[ "${cache_dir}" != /* ]]; then
        cache_dir="${PWD}/${cache_dir}"
    fi
    ARM_FVP_ARCHIVE="${cache_dir}/${ARM_FVP_MD5}-${ARM_FVP_URL##*/}"

    if arm_fvp_validate_archive "${ARM_FVP_ARCHIVE}" "${ARM_FVP_MD5}"; then
        echo "[fvp] Using cached ${ARM_FVP_URL##*/}" >&2
        return 0
    fi

    local legacy_valid=0
    if [[ -n "${legacy_archive}" ]] && arm_fvp_validate_archive "${legacy_archive}" "${ARM_FVP_MD5}"; then
        legacy_valid=1
    elif [[ -n "${ARM_SETUP_DOWNLOAD_CACHE:-}" ]]; then
        local setup_archive="${ARM_SETUP_DOWNLOAD_CACHE}/${ARM_FVP_URL##*/}"
        if arm_fvp_validate_archive "${setup_archive}" "${ARM_FVP_MD5}"; then
            if [[ "${setup_archive}" != /* ]]; then
                setup_archive="${PWD}/${setup_archive}"
            fi
            legacy_archive="${setup_archive}"
            legacy_valid=1
        fi
    fi
    if [[ "${cache_only}" == "1" ]]; then
        if [[ "${legacy_valid}" == "1" ]]; then
            ARM_FVP_ARCHIVE="${legacy_archive}"
            return 0
        fi
        echo "[fvp] Cache miss for ${ARM_FVP_URL##*/}; ARM_FVP_CACHE_ONLY=1 forbids downloads. Rebuild or prewarm the FVP cache." >&2
        return 1
    fi

    if ! mkdir -p "${cache_dir}" || [[ ! -w "${cache_dir}" ]]; then
        if [[ "${legacy_valid}" == "1" ]]; then
            ARM_FVP_ARCHIVE="${legacy_archive}"
            return 0
        fi
        echo "[fvp] Cannot write FVP cache: ${cache_dir}" >&2
        return 1
    fi
    if ! command -v flock >/dev/null 2>&1; then
        if [[ "${legacy_valid}" == "1" ]]; then
            ARM_FVP_ARCHIVE="${legacy_archive}"
            return 0
        fi
        echo "[fvp] FVP cache population requires flock (util-linux)." >&2
        return 1
    fi

    (
        flock -w 600 9 || { echo "[fvp] Timed out waiting for ${ARM_FVP_ARCHIVE}" >&2; exit 1; }
        if arm_fvp_validate_archive "${ARM_FVP_ARCHIVE}" "${ARM_FVP_MD5}"; then
            exit 0
        fi

        temporary_archive=$(mktemp "${ARM_FVP_ARCHIVE}.tmp.XXXXXX") || exit 1
        trap 'rm -f "${temporary_archive}"' EXIT
        if [[ "${legacy_valid}" == "1" ]]; then
            cp "${legacy_archive}" "${temporary_archive}" || exit 1
            arm_fvp_validate_archive "${temporary_archive}" "${ARM_FVP_MD5}" || exit 1
            mv -f "${temporary_archive}" "${ARM_FVP_ARCHIVE}"
            exit $?
        fi

        local attempt
        for attempt in 1 2 3 4 5; do
            echo "[fvp] Downloading ${ARM_FVP_URL##*/} (attempt ${attempt}/5)" >&2
            if curl --fail --retry 3 --retry-delay 5 --retry-connrefused --retry-all-errors \
                    ${ARM_SETUP_CURL_PROGRESS_ARGS[@]+"${ARM_SETUP_CURL_PROGRESS_ARGS[@]}"} \
                    --user-agent "ExecuTorch-Arm-FVP-Setup/1" -L --output "${temporary_archive}" "${ARM_FVP_URL}" \
                    && arm_fvp_validate_archive "${temporary_archive}" "${ARM_FVP_MD5}"; then
                mv -f "${temporary_archive}" "${ARM_FVP_ARCHIVE}"
                exit $?
            fi
            if [[ "${attempt}" != "5" ]]; then
                sleep $((attempt * 10))
            fi
        done
        echo "[fvp] Download failed after 5 attempts: ${ARM_FVP_URL}" >&2
        exit 1
    ) 9>"${ARM_FVP_ARCHIVE}.lock"
}
