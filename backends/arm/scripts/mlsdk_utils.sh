#!/usr/bin/env bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

mlsdk_manifest_url="https://github.com/arm/ai-ml-sdk-manifest.git"

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

source ${script_dir}/utils.sh

usage() { echo "Usage: $0 [-u <mlsdk-manifest-url>]" 1>&2; exit 1; }

while getopts ":u:" opt; do
    case "${opt}" in
        u)
            mlsdk_manifest_url=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done

function download_ai_mlsdk_manifest() {
    local _dada_dir="$1"

    if [[ -z "${_dada_dir}" ]]; then
        echo "Error: _dada_dir parameter missing?"
        return 1
    fi

    if [[ -z "${mlsdk_manifest_url}" ]]; then
        echo "Error: mlsdk_manifest_url parameter missing?"
        return 1
    fi

    if [[ ! -d "${_dada_dir}" ]]; then
        mkdir -p "$_dada_dir"
        pushd "$_dada_dir" || exit 1

        curl https://storage.googleapis.com/git-repo-downloads/repo > repo
        chmod u+x repo
        ./repo init  --no-repo-verify --depth=1  --manifest-url  ${mlsdk_manifest_url} -g model-converter,emulation-layer,vgf-library
        ./repo sync

        popd
    fi
}

function setup_model_converter() {
    local work_dir="$1"
    local manifest_dir="$2"
    local enable_model_converter="$3"
    local enable_vgf_lib="$4"
    local enable_emulation_layer="$5"

    if [[ -z "$work_dir" ]]; then
        echo "Error: work_dir parameter is required."
        return 1
    fi

    if [[ -z "$manifest_dir" ]]; then
        echo "Error: manifest_dir parameter is required."
        return 1
    fi

    mkdir -p "$work_dir"
    pushd "$work_dir" || exit 1

    download_ai_mlsdk_manifest ${manifest_dir}

    pushd "$manifest_dir"

    # model-converter
    if [[ "${enable_model_converter}" -eq 1 ]]; then
        # TODO: Remove this workaround once MLSDK has full Darwin support
        # Do not indent sed command, the whitespace is significant for the patch to work.
        if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' '/^ *print(f"Unsupported host platform/ i\
            if system == "Darwin":\
                return True\
\
' sw/model-converter/scripts/build.py
        fi
        python sw/model-converter/scripts/build.py -j$(nproc)
    fi

    # libvgf
    if [[ "${enable_vgf_lib}" -eq 1 ]]; then
        # TODO: Remove this workaround once MLSDK has full Darwin support
        # Do not indent sed command, the whitespace is significant for the patch to work.
        if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' '/^ *print(f"ERROR: Unsupported host platform/ i\
            if system == "Darwin":\
                return True\
\
' sw/vgf-lib/scripts/build.py
        fi
        pushd sw/vgf-lib
        python scripts/build.py -j$(nproc)
        cmake --install build --prefix deploy
        popd
    fi

    # emu layer
    if [[ "${enable_emulation_layer}" -eq 1 ]]; then
        pushd sw/emulation-layer
        cmake -B build                                               \
            -DGLSLANG_PATH=../../dependencies/glslang                \
            -DSPIRV_CROSS_PATH=../../dependencies/SPIRV-Cross        \
            -DSPIRV_HEADERS_PATH=../../dependencies/SPIRV-Headers    \
            -DSPIRV_TOOLS_PATH=../../dependencies/SPIRV-Tools        \
            -DVULKAN_HEADERS_PATH=../../dependencies/Vulkan-Headers

        cmake --build build
        cmake --install build --prefix deploy
        popd
    fi

    popd
}

#setup_model_converter() $1
# `"$manifest_dir"'
