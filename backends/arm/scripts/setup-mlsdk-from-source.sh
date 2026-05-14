#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
et_dir=$(realpath "${script_dir}/../../..")
ARCH="$(uname -m)"
OS="$(uname -s)"

root_dir="${et_dir}/examples/arm/arm-scratch"
setup_path_script=""
mlsdk_manifest_dir="ml-sdk-for-vulkan-manifest"
mlsdk_manifest_url="${MLSDK_MANIFEST_URL:-https://github.com/arm/ai-ml-sdk-manifest.git}"
mlsdk_manifest_tag="${MLSDK_MANIFEST_TAG:-refs/tags/v2026.03.0}" # Keep this in sync with what is mentioned in requirements-arm-vgf.txt

enable_model_converter=0
enable_vgf_lib=0
enable_emulation_layer=0
enable_vulkan_sdk=0

source "${script_dir}/utils.sh"
source "${script_dir}/vulkan_utils.sh"
source "${script_dir}/mlsdk_utils.sh"

OPTION_LIST=(
  "--root-dir Path to scratch directory (default: examples/arm/arm-scratch)"
  "--manifest-dir Path to the MLSDK source checkout (default: <root-dir>/ml-sdk-for-vulkan-manifest)"
  "--mlsdk-manifest-url Override the MLSDK manifest repository URL"
  "--mlsdk-manifest-tag Override the MLSDK manifest tag or branch"
  "--enable-model-converter Build model-converter from source"
  "--enable-vgf-lib Build the VGF library from source"
  "--enable-emulation-layer Build the Vulkan emulation layer from source"
  "--enable-vulkan-sdk Download and export the Vulkan SDK"
  "--help Display help"
)

function print_usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo
    echo "Available options:"
    for entry in "${OPTION_LIST[@]}"; do
        opt="${entry%% *}"
        desc="${entry#* }"
        printf "  %-40s %s\n" "$opt" "$desc"
    done
    echo
    echo "When no component flags are provided, the script builds model-converter,"
    echo "vgf-lib, the emulation layer, and the Vulkan SDK."
}

function check_options() {
    while [[ "${#}" -gt 0 ]]; do
        case "$1" in
            --root-dir)
                if [[ $# -lt 2 ]]; then
                    print_usage
                    exit 1
                fi
                root_dir="$2"
                shift 2
                ;;
            --manifest-dir)
                if [[ $# -lt 2 ]]; then
                    print_usage
                    exit 1
                fi
                mlsdk_manifest_dir="$2"
                shift 2
                ;;
            --mlsdk-manifest-url)
                if [[ $# -lt 2 ]]; then
                    print_usage
                    exit 1
                fi
                mlsdk_manifest_url="$2"
                shift 2
                ;;
            --mlsdk-manifest-tag)
                if [[ $# -lt 2 ]]; then
                    print_usage
                    exit 1
                fi
                mlsdk_manifest_tag="$2"
                shift 2
                ;;
            --enable-model-converter)
                enable_model_converter=1
                shift
                ;;
            --enable-vgf-lib)
                enable_vgf_lib=1
                shift
                ;;
            --enable-emulation-layer)
                enable_emulation_layer=1
                shift
                ;;
            --enable-vulkan-sdk)
                enable_vulkan_sdk=1
                shift
                ;;
            --enable-mlsdk-deps)
                # Backwards-compatible alias for the default "build everything" behavior.
                enable_model_converter=1
                enable_vgf_lib=1
                enable_emulation_layer=1
                enable_vulkan_sdk=1
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                print_usage
                exit 1
                ;;
        esac
    done

    # If no component was selected explicitly, build the full MLSDK source stack.
    if [[ "${enable_model_converter}" -eq 0 && \
          "${enable_vgf_lib}" -eq 0 && \
          "${enable_emulation_layer}" -eq 0 && \
          "${enable_vulkan_sdk}" -eq 0 ]]; then
        enable_model_converter=1
        enable_vgf_lib=1
        enable_emulation_layer=1
        enable_vulkan_sdk=1
    fi
}

function setup_root_dir() {
    mkdir -p "${root_dir}"
    root_dir=$(realpath "${root_dir}")
    setup_path_script="${root_dir}/setup_path"
    log_step "main" "Prepared root dir at ${root_dir}"
}

function mlsdk_sync_manifest() {
    local manifest_dir="$1"

    mkdir -p "${manifest_dir}"
    pushd "${manifest_dir}" >/dev/null || return 1
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

    ./repo sync --force-sync -j"${parallel_jobs}"

    popd >/dev/null || return 1
}

function download_ai_mlsdk_manifest() {
    local manifest_dir="$1"

    if [[ -z "${manifest_dir}" ]]; then
        log_step "mlsdk" "Error: manifest_dir parameter is required"
        return 1
    fi

    if [[ -z "${mlsdk_manifest_url}" ]]; then
        log_step "mlsdk" "Error: mlsdk_manifest_url parameter is required"
        return 1
    fi

    if [[ ! -d "${manifest_dir}/sw" ]] || [[ ! -d "${manifest_dir}/dependencies" ]]; then
        log_step "mlsdk" "MLSDK checkout not found at ${manifest_dir}; performing initial download"
        mlsdk_sync_manifest "${manifest_dir}"
        return 0
    fi

    # A manifest checkout already exists. Compare its URL and branch/tag to the
    # requested manifest source and refresh it if they do not match.
    local cached_url=""
    local cached_tag=""
    local repo_config="${manifest_dir}/.repo/manifests.git/config"
    if [[ -f "${repo_config}" ]]; then
        cached_url="$(git config --file "${repo_config}" remote.origin.url 2>/dev/null || echo "")"
        cached_tag="$(git config --file "${repo_config}" branch.default.merge 2>/dev/null || echo "")"
    fi

    local tag_tracks_main=0
    if [[ "${mlsdk_manifest_tag}" == "main" ]] || [[ "${mlsdk_manifest_tag}" == "refs/heads/main" ]]; then
        tag_tracks_main=1
    fi

    if [[ "${cached_url}" == "${mlsdk_manifest_url}" && \
          "${cached_tag}" == "${mlsdk_manifest_tag}" && \
          "${tag_tracks_main}" -eq 0 ]]; then
        log_step "mlsdk" "Reusing cached MLSDK dependencies at ${manifest_dir}"
        return 0
    fi

    if [[ "${tag_tracks_main}" -eq 1 ]]; then
        log_step "mlsdk" "Manifest tracks branch ${mlsdk_manifest_tag}; refreshing checkout"
    else
        log_step "mlsdk" \
            "Manifest changed (url=${cached_url:-<unknown>} -> ${mlsdk_manifest_url}, tag=${cached_tag:-<unknown>} -> ${mlsdk_manifest_tag}); refreshing checkout"
    fi

    if [[ -d "${manifest_dir}/.repo/local_manifests" ]]; then
        rm -rf "${manifest_dir}/.repo/local_manifests/"
    fi

    if [[ -d "${manifest_dir}/.repo/manifests.git" ]]; then
        git -C "${manifest_dir}/.repo/manifests.git" reset --hard HEAD >/dev/null 2>&1 || true
        git -C "${manifest_dir}/.repo/manifests.git" clean -fd >/dev/null 2>&1 || true
    fi

    if [[ -d "${manifest_dir}/.repo/manifests" ]]; then
        git -C "${manifest_dir}/.repo/manifests" reset --hard HEAD >/dev/null 2>&1 || true
        git -C "${manifest_dir}/.repo/manifests" clean -fd >/dev/null 2>&1 || true
    fi

    # Keep this migration workaround from the old setup flow. Going directly
    # from v2025.10.0 to v2025.12.0 has required a clean checkout in practice.
    if [[ "${cached_tag}" == "refs/tags/v2025.10.0" && \
          "${mlsdk_manifest_tag}" == "refs/tags/v2025.12.0" ]]; then
        pushd "${manifest_dir}/.." >/dev/null || return 1
        log_step "mlsdk" "Deleting ${mlsdk_manifest_dir} and starting fresh"
        rm -rf "$(basename "${manifest_dir}")"
        popd >/dev/null || return 1
    fi

    mlsdk_sync_manifest "${manifest_dir}"
}

function setup_mlsdk_source() {
    local manifest_dir="$1"

    if [[ -z "${manifest_dir}" ]]; then
        log_step "mlsdk" "Error: manifest_dir parameter is required"
        return 1
    fi

    mkdir -p "${root_dir}"
    pushd "${root_dir}" >/dev/null || return 1

    log_step "mlsdk" "Syncing MLSDK manifest into ${manifest_dir}"
    download_ai_mlsdk_manifest "${manifest_dir}"

    pushd "${manifest_dir}" >/dev/null || return 1
    local parallel_jobs
    parallel_jobs="$(get_parallel_jobs)"

    if [[ "${enable_model_converter}" -eq 1 ]]; then
        log_step "mlsdk" "Building MLSDK model-converter"
        python sw/model-converter/scripts/build.py -j"${parallel_jobs}"
        log_step "mlsdk" "MLSDK model-converter build complete"
    fi

    if [[ "${enable_vgf_lib}" -eq 1 ]]; then
        log_step "mlsdk" "Building MLSDK VGF library"
        pushd sw/vgf-lib >/dev/null || return 1
        python scripts/build.py -j"${parallel_jobs}"
        cmake --install build --prefix deploy
        popd >/dev/null || return 1
        log_step "mlsdk" "MLSDK VGF library build complete"
    fi

    if [[ "${enable_emulation_layer}" -eq 1 ]]; then
        local float_as_double
        float_as_double="$(detect_emulation_layer_float_as_double)"
        if [[ "${float_as_double}" == "ON" ]]; then
            log_step "mlsdk" \
                "Detected missing shaderFloat64 support. Building Vulkan emulation layer with VMEL_USE_FLOAT_AS_DOUBLE=ON."
        elif [[ "${float_as_double}" == "UNKNOWN" ]]; then
            log_step "mlsdk" \
                "shaderFloat64 support could not be detected. Building Vulkan emulation layer with VMEL_USE_FLOAT_AS_DOUBLE=OFF."
        fi

        log_step "mlsdk" "Building MLSDK Vulkan emulation layer"
        pushd sw/emulation-layer >/dev/null || return 1
        cmake -B build \
            -DGLSLANG_PATH=../../dependencies/glslang \
            -DSPIRV_CROSS_PATH=../../dependencies/SPIRV-Cross \
            -DSPIRV_HEADERS_PATH=../../dependencies/SPIRV-Headers \
            -DSPIRV_TOOLS_PATH=../../dependencies/SPIRV-Tools \
            -DVULKAN_HEADERS_PATH=../../dependencies/Vulkan-Headers \
            -DVMEL_USE_FLOAT_AS_DOUBLE="${float_as_double/UNKNOWN/OFF}"
        cmake --build build -j"${parallel_jobs}"
        cmake --install build --prefix deploy
        popd >/dev/null || return 1
        log_step "mlsdk" "MLSDK Vulkan emulation layer build complete"
    fi

    popd >/dev/null || return 1
    popd >/dev/null || return 1
}

function setup_path_model_converter() {
    local model_converter_bin_path="${mlsdk_manifest_dir}/sw/model-converter/build"
    if [[ ! -d "${model_converter_bin_path}" ]]; then
        log_step "path" "model-converter build output not found; skipping PATH update"
        return
    fi

    model_converter_bin_path="$(cd "${model_converter_bin_path}" && pwd)"
    append_env_in_setup_path PATH "${model_converter_bin_path}"
}

function setup_path_vgf_lib() {
    local model_vgf_path="${mlsdk_manifest_dir}/sw/vgf-lib/deploy"
    if [[ ! -d "${model_vgf_path}" ]]; then
        log_step "path" "VGF deploy directory not found; skipping PATH update"
        return
    fi

    model_vgf_path="$(cd "${model_vgf_path}" && pwd)"
    append_env_in_setup_path PATH "${model_vgf_path}/bin"
    append_env_in_setup_path LD_LIBRARY_PATH "${model_vgf_path}/lib"
    append_env_in_setup_path DYLD_LIBRARY_PATH "${model_vgf_path}/lib"
}

function setup_path_source_emulation_layer() {
    local deploy_dir="${mlsdk_manifest_dir}/sw/emulation-layer/deploy"
    if [[ ! -d "${deploy_dir}" ]]; then
        log_step "path" "Emulation layer deploy directory not found; skipping Vulkan layer exports"
        return
    fi

    deploy_dir="$(cd "${deploy_dir}" && pwd)"
    apply_emulation_layer_deploy_dir "${deploy_dir}"
}

function create_setup_path() {
    cd "${root_dir}"

    clear_setup_path
    log_step "path" "Generating setup path scripts at ${setup_path_script}"

    if [[ -n "${VIRTUAL_ENV:-}" && -d "${VIRTUAL_ENV}/bin" ]]; then
        prepend_env_in_setup_path PATH "${VIRTUAL_ENV}/bin"
    elif [[ -d "${et_dir}/env/bin" ]]; then
        prepend_env_in_setup_path PATH "${et_dir}/env/bin"
    fi

    if [[ "${enable_vulkan_sdk}" -eq 1 ]]; then
        setup_path_vulkan
    fi

    if [[ "${enable_model_converter}" -eq 1 ]]; then
        setup_path_model_converter
    fi

    if [[ "${enable_vgf_lib}" -eq 1 ]]; then
        setup_path_vgf_lib
    fi

    if [[ "${enable_emulation_layer}" -eq 1 ]]; then
        setup_path_source_emulation_layer
    fi

    log_step "path" "Update PATH by sourcing ${setup_path_script}.{sh|fish}"
}

check_options "$@"
check_platform_support
check_os_support

setup_root_dir
if [[ "${mlsdk_manifest_dir}" != /* ]]; then
    mlsdk_manifest_dir="${root_dir}/${mlsdk_manifest_dir}"
fi

log_step "options" \
    "root=${root_dir}, manifest-dir=${mlsdk_manifest_dir}, manifest-url=${mlsdk_manifest_url}, manifest-tag=${mlsdk_manifest_tag}"
log_step "options" \
    "mlsdk: model-converter=${enable_model_converter}, vgf-lib=${enable_vgf_lib}, emu-layer=${enable_emulation_layer}, vulkan-sdk=${enable_vulkan_sdk}"

if [[ "${enable_vulkan_sdk}" -eq 1 ]]; then
    log_step "vulkan" "Configuring Vulkan SDK"
    setup_vulkan_sdk
fi

setup_mlsdk_source "${mlsdk_manifest_dir}"
create_setup_path
