#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -u

########################
### Hardcoded constants
########################
script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
et_dir=$(realpath $script_dir/../..)
ARCH="$(uname -m)"
OS="$(uname -s)"
root_dir="${script_dir}/ethos-u-scratch"  # TODO: rename
eula_acceptance=0
enable_baremetal_toolchain=1
target_toolchain=""
enable_fvps=1
enable_vela=1
enable_model_converter=0   # model-converter tool for VGF output
enable_vgf_lib=0  # vgf reader - runtime backend dependency
enable_emulation_layer=0  # Vulkan layer driver - emulates Vulkan ML extensions
enable_vulkan_sdk=0  # Download and export Vulkan SDK required by emulation layer
mlsdk_manifest_url="https://github.com/arm/ai-ml-sdk-manifest.git"

# Figure out if setup.sh was called or sourced and save it into "is_script_sourced"
(return 0 2>/dev/null) && is_script_sourced=1 || is_script_sourced=0

# Global scope these so they can be set later
toolchain_url=""
toolchain_dir=""
toolchain_md5_checksum=""


# List of supported options and their descriptions
OPTION_LIST=(
  "--i-agree-to-the-contained-eula (required) Agree to the EULA"
  "--root-dir Path to scratch directory"
  "--enable-baremetal-toolchain Enable baremetal toolchain setup"
  "--enable-fvps Enable FVP setup"
  "--enable-vela Enable VELA setup"
  "--enable-model-converter Enable MLSDK model converter setup"
  "--enable-vgf-lib Enable MLSDK vgf library setup"
  "--enable-emulation-layer Enable MLSDK Vulkan emulation layer"
  "--disable-ethos-u-deps Do not setup what is needed for Ethos-U"
  "--enable-mlsdk-deps Setup what is needed for MLSDK"
  "--mlsdk-manifest-url URL to the MLSDK manifest for vulkan."
  "--help Display help"
)


########
### Functions
########

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
    echo "Supplied args: $*"
}

function check_options() {
    while [[ "${#}" -gt 0 ]]; do
        case "$1" in
            --i-agree-to-the-contained-eula)
                eula_acceptance=1
                shift
                ;;
            --root-dir)
                # Only change default root dir if the script is being executed and not sourced.
                if [[ $is_script_sourced -eq 0 ]]; then
                    root_dir=${2:-"${root_dir}"}
                fi

                if [[ $# -ge 2 ]]; then
                    shift 2
                else
                    print_usage "$@"
                    exit 1
                fi
                ;;
            --enable-baremetal-toolchain)
                enable_baremetal_toolchain=1
                shift
                ;;
            --target-toolchain)
                # Only change default root dir if the script is being executed and not sourced.
                if [[ $is_script_sourced -eq 0 ]]; then
                    target_toolchain=${2:-"${target_toolchain}"}
                fi

                if [[ $# -ge 2 ]]; then
                    shift 2
                else
                    print_usage "$@"
                    exit 1
                fi
                ;;
            --enable-fvps)
                enable_fvps=1
                shift
                ;;
            --enable-vela)
                enable_vela=1
                shift
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
            --disable-ethos-u-deps)
                enable_baremetal_toolchain=0
                enable_fvps=0
                enable_vela=0
                shift
                ;;
            --enable-mlsdk-deps)
                enable_model_converter=1
                enable_vgf_lib=1
                enable_emulation_layer=1
                enable_vulkan_sdk=1
                shift
                ;;
            --mlsdk-manifest-url)
                # Ensure that there is a url provided.
                if [[ -n "$2" && "${2:0:1}" != "-" ]]; then
                    mlsdk_manifest_url="$2"
                    shift 2
                else
                    echo "Error: --mlsdk-manifest-url requires a URL argument."
                    print_usage "$@"
                    exit 1
                fi
                ;;
            --setup-test-dependency)
                echo "Installing test dependency..."
                source $et_dir/backends/arm/scripts/install_models_for_test.sh
                exit 0
                ;;
            --help)
                print_usage "$@"
                exit 0
                ;;
            *)
                print_usage "$@"
                exit 1
                ;;
        esac
    done
}

function setup_root_dir() {
    mkdir -p ${root_dir}
    root_dir=$(realpath ${root_dir})
    setup_path_script="${root_dir}/setup_path"
}

function setup_ethos_u_tools() {
    CMAKE_POLICY_VERSION_MINIMUM=3.5 BUILD_PYBIND=1 pip install --no-dependencies -r $et_dir/backends/arm/requirements-arm-ethos-u.txt
}

function create_setup_path(){
    cd "${root_dir}"

    clear_setup_path

    if [[ "${enable_fvps}" -eq 1 ]]; then
        setup_path_fvp
    fi

    if [[ "${enable_baremetal_toolchain}" -eq 1 ]]; then
        setup_path_toolchain
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
        setup_path_emulation_layer
    fi

    echo "[main] Update path by running 'source ${setup_path_script}.sh'"
    echo "[main] Or for fish shell use 'source ${setup_path_script}.fish'"
}


########
### main
########

# script is not sourced! Lets run "main"
if [[ $is_script_sourced -eq 0 ]]; then
    set -e

    check_options "$@"

    # Import utils
    source $et_dir/backends/arm/scripts/utils.sh
    source $et_dir/backends/arm/scripts/fvp_utils.sh
    source $et_dir/backends/arm/scripts/toolchain_utils.sh
    source $et_dir/backends/arm/scripts/vulkan_utils.sh

    echo "[main]: Checking platform and os"
    check_platform_support
    check_os_support

    cd "${script_dir}"

    # Setup the root dir
    setup_root_dir
    cd "${root_dir}"
    echo "[main] Using root dir ${root_dir} and options:"
    echo "enable-fvps=${enable_fvps}"
    echo "target-toolchain=${target_toolchain}"
    echo "enable-baremetal-toolchain=${enable_baremetal_toolchain}"
    echo "enable-model-converter=${enable_model_converter}"
    echo "enable-vgf-lib=${enable_vgf_lib}"
    echo "enable-emulation-layer=${enable_emulation_layer}"
    echo "enable-vulkan-sdk=${enable_vulkan_sdk}"
    echo "enable-vela=${enable_vela}"
    echo "mlsdk-manifest-url=${mlsdk_manifest_url}"


    # Setup toolchain
    if [[ "${enable_baremetal_toolchain}" -eq 1 ]]; then
        # Select appropriate toolchain
        select_toolchain
        setup_toolchain
    fi

    # Setup FVP
    if [[ "${enable_fvps}" -eq 1 ]]; then
        check_fvp_eula
        setup_fvp
        install_fvp
    fi

    # Setup Vulkan SDK
    if [[ "${enable_vulkan_sdk}" -eq 1 ]]; then
        setup_vulkan_sdk
    fi

    if [[ "${enable_model_converter}" -eq 1 || \
          "${enable_vgf_lib}" -eq 1 || \
          "${enable_emulation_layer}" -eq 1 ]]; then
        source $et_dir/backends/arm/scripts/mlsdk_utils.sh -u "${mlsdk_manifest_url}"
        setup_model_converter ${root_dir} ${mlsdk_manifest_dir} ${enable_model_converter} ${enable_vgf_lib} ${enable_emulation_layer}
    fi

    # Create the setup_path.sh used to create the PATH variable for shell
    create_setup_path

    # Setup the tosa_reference_model and dependencies
    CMAKE_POLICY_VERSION_MINIMUM=3.5 BUILD_PYBIND=1 pip install --no-dependencies -r $et_dir/backends/arm/requirements-arm-tosa.txt

    if [[ "${enable_vela}" -eq 1 ]]; then
        setup_ethos_u_tools
    fi

    echo "[main] success!"
    exit 0
fi
