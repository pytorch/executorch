#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file builds the `flatc` commandline tool from the
# `third-party/flatbuffers` directory and help users install it correctly.

set -o errexit
set -o nounset
set -o pipefail

EXECUTORCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly EXECUTORCH_ROOT

readonly FLATBUFFERS_PATH="${EXECUTORCH_ROOT}/third-party/flatbuffers"
readonly BUILD_DIR="${FLATBUFFERS_PATH}/cmake-out"
readonly BUILT_FLATC="${BUILD_DIR}/flatc"

# Must use "echo -e" to expand these escape sequences.
readonly GREEN="\033[0;32m" # GREEN Color
readonly RED="\033[0;31m" # Red Color
readonly NC="\033[0m" # No Color

# Prints the flatbuffers version of the git submodule.
print_flatbuffers_version(){
    local version_file="${FLATBUFFERS_PATH}/package.json"
    local version
    # Extract the version from the first line like `"version": "23.5.26",`
    # First remove the final double quote, then remove everything
    # before the now-final double quote.
    version="$(
        grep '"version"\s*:' "${version_file}" \
        | head -1 \
        | sed -e 's/"[^"]*$//' \
        | sed -e 's/.*"//'
        )"
    if [[ ${version} =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "${version}"
    else
        echo "ERROR: Bad version '${version}'; could not find version in ${version_file}" >&2
        exit 1
    fi
}

main() {
    local flatbuffers_version
    flatbuffers_version="$(print_flatbuffers_version)"
    echo "Version of ${FLATBUFFERS_PATH} is ${flatbuffers_version}"

    local flatc_path
    flatc_path="$(which flatc 2>/dev/null || echo '')"
    if [[ -f "${flatc_path}" ]]; then
        # A flatc is already on the PATH.
        if { "${flatc_path}" --version | grep -q "${flatbuffers_version}"; }; then
            echo -e "${GREEN}A compatible version of flatc is on the PATH" \
                "and ready to use.${NC}"
            return 0
        else
            echo -e "${RED}WARNING: An incompatible version of flatc" \
                "is on the PATH at ${flatc_path}."
            echo -e "  Required version: flatc version ${flatbuffers_version}"
            echo -e "  Actual version: $("${flatc_path}" --version)${NC}"

            if [[ "${flatc_path}" == *miniconda* ]]; then
                echo -e "${RED}ERROR: ${flatc_path} appears to be installed" \
                    "with conda, which can cause consistency problems."
                echo -e "Please run the following command to remove it: "
                echo -e "  conda uninstall flatbuffers${NC}"
                return 1
            fi

            # Continue to build a compatible version.
        fi
    fi

    if [[ -f "${BUILT_FLATC}" ]]; then
        echo -e "${BUILT_FLATC} is already built."
    else
        # Build the tool if not already built.
        echo "Building flatc under ${FLATBUFFERS_PATH}..."
        # Generate cache.
        (rm -rf "${BUILD_DIR}" && mkdir "${BUILD_DIR}" && cd "${BUILD_DIR}" && cmake -DCMAKE_BUILD_TYPE=Release ..)
        # Build.
        (cd "${FLATBUFFERS_PATH}" && cmake --build "${BUILD_DIR}" --target flatc -j9)

        echo -e "Finished building ${BUILT_FLATC}."
    fi

    echo -e ""
    echo -e "***** Run the following commands to add a compatible flatc"\
        "to the PATH and re-run this script:"
    echo -e "  ${RED}export PATH=\"${BUILD_DIR}:\${PATH}\""
    echo -e "  bash ${EXECUTORCH_ROOT}/build/install_flatc.sh${NC}"
}

main "$@"
