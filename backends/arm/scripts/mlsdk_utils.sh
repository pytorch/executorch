#!/usr/bin/env bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

# URL and tag of the MLSDK manifest repository. Can be overridden by environment variables.
# eg. export MLSDK_MANIFEST_URL=...; export MLSDK_MANIFEST_TAG=...
mlsdk_manifest_url="${MLSDK_MANIFEST_URL:-https://github.com/arm/ai-ml-sdk-manifest.git}"
mlsdk_manifest_tag="${MLSDK_MANIFEST_TAG:-refs/tags/v2025.12.0}"

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

source ${script_dir}/utils.sh

function mlsdk_sync_manifest() {
    local manifest_dir="$1"

    mkdir -p "${manifest_dir}"
    pushd "${manifest_dir}" || return 1
    local parallel_jobs="$(get_parallel_jobs)"

    if [[ ! -f repo ]]; then
        log_step "mlsdk" "Fetching repo tool"
        curl https://storage.googleapis.com/git-repo-downloads/repo > repo
        chmod u+x repo
    fi

    ./repo init \
           --depth=1 \
           --no-repo-verify \
           --manifest-url "${mlsdk_manifest_url}" \
           --manifest-branch "${mlsdk_manifest_tag}" \
           -g model-converter,emulation-layer,vgf-library

    local default_manifest=".repo/manifests/default.xml"

    ./repo sync --force-sync -j"${parallel_jobs}"

    popd
}

function download_ai_mlsdk_manifest() {
    local _manifest_dir="$1"

    if [[ -z "${_manifest_dir}" ]]; then
        log_step "mlsdk" "Error: _manifest_dir parameter missing"
        return 1
    fi

    if [[ -z "${mlsdk_manifest_url}" ]]; then
        log_step "mlsdk" "Error: mlsdk_manifest_url parameter missing"
        return 1
    fi

    if [[ ! -d "${_manifest_dir}/sw" ]] || [[ ! -d "${_manifest_dir}/dependencies" ]]; then
        log_step "mlsdk" "MLSDK checkout not found at ${_manifest_dir}; performing initial download"
        mlsdk_sync_manifest "${_manifest_dir}"
        return 0
    fi

    # If a checkout exists, get the URL and tag of the existing checkout.
    local cached_url=""
    local cached_tag=""
    local repo_config="${_manifest_dir}/.repo/manifests.git/config"
    if [[ -f "${repo_config}" ]]; then
        cached_url="$(git config --file "${repo_config}" remote.origin.url 2>/dev/null || echo "")"
        cached_tag="$(git config --file "${repo_config}" branch.default.merge 2>/dev/null || echo "")"
    fi

    # If the tag is main or refs/heads/main, always refresh the checkout.
    # This allows users to track the latest main branch without needing to manually
    # delete the checkout.
    local tag_tracks_main=0
    if [[ "${mlsdk_manifest_tag}" == "main" ]] || [[ "${mlsdk_manifest_tag}" == "refs/heads/main" ]]; then
        tag_tracks_main=1
    fi

    # If the URL and tag match, and the tag does not track main, reuse the existing checkout.
    # Skip fetching updates.
    if [[ "${cached_url}" == "${mlsdk_manifest_url}" ]] && [[ "${cached_tag}" == "${mlsdk_manifest_tag}" ]] && [[ "${tag_tracks_main}" -eq 0 ]]; then
        log_step "mlsdk" "Reusing cached MLSDK dependencies at ${_manifest_dir}"
        return 0
    fi

    # If we reach here, either the URL or tag changed, or the tag tracks main.
    # In all cases, refresh the checkout.
    if [[ "${tag_tracks_main}" -eq 1 ]]; then
        log_step "mlsdk" "Manifest tracks branch ${mlsdk_manifest_tag}; refreshing checkout"
    else
        log_step "mlsdk" "Manifest changed (url=${cached_url:-<unknown>} -> ${mlsdk_manifest_url}, tag=${cached_tag:-<unknown>} -> ${mlsdk_manifest_tag}); refreshing checkout"
    fi

    # Clean up any local manifest changes to avoid repo sync errors.
    # Since we patched in a local manifest for tosa_gitlab.xml,
    # remove any existing local manifests to avoid conflicts.
    # TODO: we should remove this at some point in the future, but its not hurting anything for now.
    if [[ -d "${_manifest_dir}/.repo/local_manifests" ]]; then
        rm -rf "${_manifest_dir}/.repo/local_manifests/"
    fi

    # Clean up any local changes in the manifests repository.
    if [[ -d "${_manifest_dir}/.repo/manifests.git" ]]; then
        git -C "${_manifest_dir}/.repo/manifests.git" reset --hard HEAD >/dev/null 2>&1 || true
        git -C "${_manifest_dir}/.repo/manifests.git" clean -fd >/dev/null 2>&1 || true
    fi

    # Clean up any local changes in the manifests working copy.
    if [[ -d "${_manifest_dir}/.repo/manifests" ]]; then
        git -C "${_manifest_dir}/.repo/manifests" reset --hard HEAD >/dev/null 2>&1 || true
        git -C "${_manifest_dir}/.repo/manifests" clean -fd >/dev/null 2>&1 || true
    fi

    # Going from v2025.10.0 to v2025.12.0 seems particular hard so just keep it simple.
    # TODO: Remove once this is history
    if [[ "${cached_tag}" == "refs/tags/v2025.10.0" ]] && [[ "${mlsdk_manifest_tag}" == "refs/tags/v2025.12.0" ]]; then
        pushd "${_manifest_dir}/.."
        log_step "mlsdk" "Deleting ${mlsdk_manifest_dir} and starting fresh"
        manifest_base_dir=$(basename "${_manifest_dir}")
        rm -fr $manifest_base_dir
        popd
    fi

    mlsdk_sync_manifest "${_manifest_dir}"
}

function setup_mlsdk() {
    local work_dir="$1"
    local manifest_dir="$2"
    local enable_model_converter="$3"
    local enable_vgf_lib="$4"
    local enable_emulation_layer="$5"

    if [[ -z "$work_dir" ]]; then
        log_step "mlsdk" "Error: work_dir parameter is required"
        return 1
    fi

    if [[ -z "$manifest_dir" ]]; then
        log_step "mlsdk" "Error: manifest_dir parameter is required"
        return 1
    fi

    mkdir -p "$work_dir"
    pushd "$work_dir" || exit 1

    log_step "mlsdk" "Syncing MLSDK manifest into ${manifest_dir}"
    download_ai_mlsdk_manifest "${manifest_dir}"

    pushd "$manifest_dir"
    local parallel_jobs="$(get_parallel_jobs)"

    # model-converter
    if [[ "${enable_model_converter}" -eq 1 ]]; then
        log_step "mlsdk" "Building MLSDK model-converter"
        python sw/model-converter/scripts/build.py -j"${parallel_jobs}"
        log_step "mlsdk" "MLSDK model-converter build complete"
    fi

    # libvgf
    if [[ "${enable_vgf_lib}" -eq 1 ]]; then
        log_step "mlsdk" "Building MLSDK VGF library"
        pushd sw/vgf-lib
        python scripts/build.py -j"${parallel_jobs}"
        cmake --install build --prefix deploy
        log_step "mlsdk" "MLSDK VGF library build complete"
        popd
    fi

    # emu layer
    if [[ "${enable_emulation_layer}" -eq 1 ]]; then
        log_step "mlsdk" "Building MLSDK Vulkan emulation layer"
        pushd sw/emulation-layer
        cmake -B build                                               \
            -DGLSLANG_PATH=../../dependencies/glslang                \
            -DSPIRV_CROSS_PATH=../../dependencies/SPIRV-Cross        \
            -DSPIRV_HEADERS_PATH=../../dependencies/SPIRV-Headers    \
            -DSPIRV_TOOLS_PATH=../../dependencies/SPIRV-Tools        \
            -DVULKAN_HEADERS_PATH=../../dependencies/Vulkan-Headers

        cmake --build build -j"${parallel_jobs}"
        cmake --install build --prefix deploy
        log_step "mlsdk" "MLSDK Vulkan emulation layer build complete"
        popd
    fi

    popd
}

function setup_path_model_converter() {
    cd "${root_dir}"
    model_converter_bin_path="$(cd "${mlsdk_manifest_dir}/sw/model-converter/build" && pwd)"
    append_env_in_setup_path PATH "${model_converter_bin_path}"
}

function setup_path_vgf_lib() {
    cd "${root_dir}"
    model_vgf_path="$(cd "${mlsdk_manifest_dir}/sw/vgf-lib/deploy" && pwd)"
    append_env_in_setup_path PATH "${model_vgf_path}/bin"
    append_env_in_setup_path LD_LIBRARY_PATH "${model_vgf_path}/lib"
    append_env_in_setup_path DYLD_LIBRARY_PATH "${model_vgf_path}/lib"
}

function setup_path_emulation_layer() {
    cd "${root_dir}"
    model_emulation_layer_path="$(cd "${mlsdk_manifest_dir}/sw/emulation-layer/" && pwd)"
    prepend_env_in_setup_path LD_LIBRARY_PATH "${model_emulation_layer_path}/deploy/lib"
    prepend_env_in_setup_path DYLD_LIBRARY_PATH "${model_emulation_layer_path}/deploy/lib"
    prepend_env_in_setup_path VK_LAYER_PATH "${model_emulation_layer_path}/deploy/share/vulkan/explicit_layer.d"
    prepend_env_in_setup_path VK_INSTANCE_LAYERS VK_LAYER_ML_Tensor_Emulation
    prepend_env_in_setup_path VK_INSTANCE_LAYERS VK_LAYER_ML_Graph_Emulation
}

function setup_path_emulation_layer_from_pip() {
    if ! command -v emulation_layer >/dev/null 2>&1; then
        echo "[mlsdk_utils] 'emulation_layer' command not found; skipping pip emulation layer path setup"
        return
    fi

    local output
    if ! output=$(emulation_layer 2>/dev/null); then
        echo "[mlsdk_utils] Failed to query emulation_layer environment; skipping"
        return
    fi

    local exports
    exports=$(echo "$output" | grep '^export ' || true)

    local ld_line
    ld_line=$(echo "$exports" | grep 'LD_LIBRARY_PATH=' || true)
    if [[ -n "${ld_line}" ]]; then
        local ld_value=${ld_line#export LD_LIBRARY_PATH=}
        ld_value=${ld_value%%:\$LD_LIBRARY_PATH*}
        if [[ -n "${ld_value}" ]]; then
            prepend_env_in_setup_path LD_LIBRARY_PATH "${ld_value}"
        fi
    fi

    local vk_add_line
    vk_add_line=$(echo "$exports" | grep 'VK_ADD_LAYER_PATH=' || true)
    if [[ -n "${vk_add_line}" ]]; then
        local vk_add_value=${vk_add_line#export VK_ADD_LAYER_PATH=}
        if [[ -n "${vk_add_value}" ]]; then
            prepend_env_in_setup_path VK_ADD_LAYER_PATH "${vk_add_value}"
        fi
    fi

    local vk_instance_line
    vk_instance_line=$(echo "$exports" | grep 'VK_INSTANCE_LAYERS=' || true)
    if [[ -n "${vk_instance_line}" ]]; then
        local vk_instance_value=${vk_instance_line#export VK_INSTANCE_LAYERS=}
        if [[ -n "${vk_instance_value}" ]]; then
            prepend_env_in_setup_path VK_INSTANCE_LAYERS "${vk_instance_value}"
        fi
    fi
}
