#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Single source of FVP download pins for developer setup and CI image builds.
ARM_FVP_PLATFORMS=(corstone300 corstone320 corstone1000)

function arm_fvp_select_platform() {
    local arch="${1}"
    local platform="${2}"
    local suffix
    case "${arch}" in
        x86_64) suffix="Linux64" ;;
        aarch64|arm64) suffix="Linux64_armv8l" ;;
        *)
            echo "[fvp] Unsupported host architecture: ${arch}" >&2
            return 1
            ;;
    esac

    local archive
    case "${platform}" in
        corstone300)
            archive="Corstone-300/FVP_Corstone_SSE-300_11.27_42_${suffix}.tgz"
            if [[ "${suffix}" == "Linux64" ]]; then
                ARM_FVP_MD5="0dc9538041296b8d479249e6fa5aab74"
            else
                ARM_FVP_MD5="74e7952817b338c8479bcf66874b23e0"
            fi
            ;;
        corstone320)
            archive="Corstone-320/FVP_Corstone_SSE-320_11.27_25_${suffix}.tgz"
            if [[ "${suffix}" == "Linux64" ]]; then
                ARM_FVP_MD5="3deb3c68f9b2d145833f15374203514d"
            else
                ARM_FVP_MD5="3889f1d80a6d9861ea4aa6f1c88dd0ae"
            fi
            ;;
        corstone1000)
            archive="Corstone-1000-with-Cortex-A320/FVP_Corstone_1000-A320_11.30_27_${suffix}.tgz"
            if [[ "${suffix}" == "Linux64" ]]; then
                ARM_FVP_MD5="80251ab4c7b9ca9d1b04ee7ff9f01ba0"
            else
                ARM_FVP_MD5="19585fa8b63a8f5084ef392a96c88e96"
            fi
            ;;
        *)
            echo "[fvp] Unknown FVP platform: ${platform}" >&2
            return 1
            ;;
    esac
    ARM_FVP_URL="https://developer.arm.com/-/cdn-downloads/permalink/FVPs-Corstone-IoT/${archive}"
    ARM_FVP_MODEL_DIR="${suffix}_GCC-9.3"
}
