#!/usr/bin/env bash
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

source ${script_dir}/utils.sh

function apply_emulation_layer_deploy_dir() {
    local deploy_dir="$1"

    if [[ -z "${deploy_dir}" ]] || [[ ! -d "${deploy_dir}" ]]; then
        return 1
    fi

    prepend_env_in_setup_path LD_LIBRARY_PATH "${deploy_dir}/lib"
    prepend_env_in_setup_path DYLD_LIBRARY_PATH "${deploy_dir}/lib"
    prepend_env_in_setup_path VK_LAYER_PATH "${deploy_dir}/share/vulkan/explicit_layer.d"
    prepend_env_in_setup_path VK_ADD_LAYER_PATH "${deploy_dir}/share/vulkan/explicit_layer.d"
    prepend_env_in_setup_path VK_INSTANCE_LAYERS VK_LAYER_ML_Tensor_Emulation
    prepend_env_in_setup_path VK_INSTANCE_LAYERS VK_LAYER_ML_Graph_Emulation
}

function find_vulkaninfo_binary() {
    if command -v vulkaninfo >/dev/null 2>&1; then
        command -v vulkaninfo
        return 0
    fi

    if [[ -n "${root_dir:-}" && \
          -n "${vulkan_sdk_bin_dir:-}" && \
          -x "${root_dir}/${vulkan_sdk_bin_dir}/vulkaninfo" ]]; then
        printf '%s\n' "${root_dir}/${vulkan_sdk_bin_dir}/vulkaninfo"
        return 0
    fi

    return 1
}

function detect_emulation_layer_float_as_double() {
    local vulkaninfo_bin=""
    if ! vulkaninfo_bin=$(find_vulkaninfo_binary); then
        log_step "mlsdk" \
            "vulkaninfo not found, can't detect shaderFloat64 support." >&2
        printf 'UNKNOWN\n'
        return 0
    fi

    if grep -qE 'shaderFloat64[[:space:]]*= false' < <("${vulkaninfo_bin}" 2>&1); then
        printf 'ON\n'
    else
        printf 'OFF\n'
    fi
}

function find_emulation_layer_pkg_dir() {
    local py="python3"
    if ! command -v "${py}" >/dev/null 2>&1; then
        py="python"
    fi

    if ! command -v "${py}" >/dev/null 2>&1; then
        return 1
    fi

    "${py}" - <<'PY'
import importlib.util
import os
import sys

spec = importlib.util.find_spec("emulation_layer")
if spec is None:
    print("")
    sys.exit(0)
if spec.submodule_search_locations:
    print(list(spec.submodule_search_locations)[0])
elif spec.origin:
    print(os.path.dirname(spec.origin))
else:
    print("")
PY
}

function setup_path_emulation_layer() {
    local pkg_dir=""
    if pkg_dir=$(find_emulation_layer_pkg_dir); then
        if [[ -n "${pkg_dir}" ]] && apply_emulation_layer_deploy_dir "${pkg_dir}/deploy"; then
            return
        fi
    fi

    echo "[mlsdk_utils] Failed to query emulation_layer environment; skipping"
}
