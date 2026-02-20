#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright 2023-2026 Arm Limited and/or its affiliates.
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
root_dir="${script_dir}/arm-scratch"
eula_acceptance=0
enable_baremetal_toolchain=1
target_toolchain=""
enable_fvps=1
enable_vela=1
enable_model_converter=0   # model-converter tool for VGF output
enable_vgf_lib=0  # vgf reader - runtime backend dependency
enable_emulation_layer=0  # Vulkan layer driver - emulates Vulkan ML extensions
enable_vulkan_sdk=0  # Download and export Vulkan SDK required by emulation layer
enable_mlsdk_pip_install=1

# Figure out if setup.sh was called or sourced and save it into "is_script_sourced"
(return 0 2>/dev/null) && is_script_sourced=1 || is_script_sourced=0

# Global scope these so they can be set later
toolchain_url=""
toolchain_dir=""
toolchain_md5_checksum=""

# Load logging helpers early so option parsing can emit status messages.
source "$et_dir/backends/arm/scripts/utils.sh"


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
  "--install-mlsdk-deps-with-pip (default) Use MLSDK PyPi package. This flag will be removed."
  "--install-mlsdk-deps-from-src Build from source instead of using MLSDK PyPi packages"
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
            --install-mlsdk-deps-with-pip)
                enable_mlsdk_pip_install=1
                shift
                ;;
            --install-mlsdk-deps-from-src)
                enable_mlsdk_pip_install=0
                shift
                ;;
            --enable-mlsdk-deps)
                enable_model_converter=1
                enable_vgf_lib=1
                enable_emulation_layer=1
                enable_vulkan_sdk=1
                shift
                ;;
            --setup-test-dependency)
                log_step "deps" "Installing test dependency..."
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
    mkdir -p "${root_dir}"
    root_dir=$(realpath "${root_dir}")
    log_step "main" "Prepared root dir at ${root_dir}"
    setup_path_script="${root_dir}/setup_path"
}

function setup_ethos_u_tools() {
    log_step "ethos-u-tools" "Installing Ethos-U Python tooling"
    CMAKE_POLICY_VERSION_MINIMUM=3.5 BUILD_PYBIND=1 pip install --no-dependencies -r $et_dir/backends/arm/requirements-arm-ethos-u.txt
}

function setup_mlsdk_dependencies() {
    log_step "mlsdk" "Installing MLSDK dependencies from pip"
    pip install -r $et_dir/backends/arm/requirements-arm-vgf.txt
}

function create_setup_path(){
    cd "${root_dir}"

    clear_setup_path
    log_step "path" "Generating setup path scripts at ${setup_path_script}"

    local use_mlsdk_pip=0
    if use_mlsdk_pip_package; then
        use_mlsdk_pip=1
    fi

    if [[ "${enable_fvps}" -eq 1 ]]; then
        setup_path_fvp
    fi

    if [[ "${enable_baremetal_toolchain}" -eq 1 ]]; then
        setup_path_toolchain
    fi

    if [[ "${enable_vulkan_sdk}" -eq 1 ]]; then
        setup_path_vulkan
    fi

    if [[ "${enable_model_converter}" -eq 1 && "${use_mlsdk_pip}" -eq 0 ]]; then
        setup_path_model_converter
    fi

    if [[ "${enable_vgf_lib}" -eq 1 && "${use_mlsdk_pip}" -eq 0 ]]; then
        setup_path_vgf_lib
    fi

    if [[ "${enable_emulation_layer}" -eq 1 ]]; then
        if [[ "${use_mlsdk_pip}" -eq 0 ]]; then
            setup_path_emulation_layer
        else
            setup_path_emulation_layer_from_pip
        fi
    fi

   log_step "path" "Update PATH by sourcing ${setup_path_script}.{sh|fish}"
}

function use_mlsdk_pip_package() {
    if [[ "${enable_mlsdk_pip_install}" -eq 0 ]]; then
        return 1
    fi

    return 0
}


########
### main
########

# script is not sourced! Lets run "main"
if [[ $is_script_sourced -eq 0 ]]; then
    set -e

    check_options "$@"

    # Import utils
    source $et_dir/backends/arm/scripts/fvp_utils.sh
    source $et_dir/backends/arm/scripts/toolchain_utils.sh
    source $et_dir/backends/arm/scripts/vulkan_utils.sh
    source $et_dir/backends/arm/scripts/mlsdk_utils.sh

    log_step "main" "Checking platform and OS"
    check_platform_support
    check_os_support

    cd "${script_dir}"

    # Setup the root dir
    setup_root_dir
    cd "${root_dir}"

    if [[ "${mlsdk_manifest_dir}" != /* ]]; then
        mlsdk_manifest_dir="${root_dir}/${mlsdk_manifest_dir}"
    fi

    log_step "options" \
             "root=${root_dir}, target-toolchain=${target_toolchain:-<default>}, mlsdk-dir=${mlsdk_manifest_dir}"
    log_step "options" \
             "ethos-u: fvps=${enable_fvps}, toolchain=${enable_baremetal_toolchain}, vela=${enable_vela} | " \
             "mlsdk: model-converter=${enable_model_converter}, vgf-lib=${enable_vgf_lib}, " \
                    "emu-layer=${enable_emulation_layer}, vulkan-sdk=${enable_vulkan_sdk}"

    # Setup toolchain
    if [[ "${enable_baremetal_toolchain}" -eq 1 ]]; then
        log_step "toolchain" "Configuring baremetal toolchain (${target_toolchain:-gnu})"
        # Select appropriate toolchain
        select_toolchain
        setup_toolchain
    fi

    # Setup FVP
    if [[ "${enable_fvps}" -eq 1 ]]; then
        log_step "fvp" "Setting up Arm Fixed Virtual Platforms"
        check_fvp_eula
        setup_fvp
        install_fvp
    fi

    # Setup Vulkan SDK
    if [[ "${enable_vulkan_sdk}" -eq 1 ]]; then
        log_step "vulkan" "Setting up Vulkan SDK"
        setup_vulkan_sdk
    fi

    if [[ "${enable_model_converter}" -eq 1 || \
          "${enable_vgf_lib}" -eq 1 || \
          "${enable_emulation_layer}" -eq 1 ]]; then
        log_step "mlsdk" "Configuring MLSDK components (model-converter=${enable_model_converter}, " \
                         "vgf-lib=${enable_vgf_lib}, emu-layer=${enable_emulation_layer})"
        if use_mlsdk_pip_package; then
            setup_mlsdk_dependencies
        else
            log_step "mlsdk" "Installing MLSDK dependencies from source"
            setup_mlsdk ${root_dir} \
                        ${mlsdk_manifest_dir} \
                        ${enable_model_converter} \
                        ${enable_vgf_lib} \
                        ${enable_emulation_layer}
        fi
    fi

    # Create the setup_path.sh used to create the PATH variable for shell
    create_setup_path

    # Setup the TOSA reference model and serialization dependencies
    log_step "deps" "Installing TOSA reference model dependencies"
    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
        pip install --no-dependencies -r "$et_dir/backends/arm/requirements-arm-tosa.txt"

    pushd "$root_dir"
    if [[ ! -d "tosa-tools" ]]; then
        git clone https://git.gitlab.arm.com/tosa/tosa-tools.git
    fi

    pushd tosa-tools
    git checkout v2025.11.0

    if [[ ! -d "reference_model" ]]; then
        log_step "main" "[error] Missing reference_model directory in tosa-tools repo."
        exit 1
    fi
    if [[ ! -d "serialization" ]]; then
        log_step "main" "[error] Missing serialization directory in tosa-tools repo."
        exit 1
    fi


    export CMAKE_BUILD_PARALLEL_LEVEL="$(get_parallel_jobs)"

    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
        BUILD_PYBIND=1 \
        pip install --no-dependencies ./reference_model

    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
        BUILD_PYBIND=1 \
        pip install --no-dependencies ./serialization
    popd
    popd

    if [[ "${enable_vela}" -eq 1 ]]; then
        log_step "deps" "Installing Ethos-U Vela compiler"
        setup_ethos_u_tools
    fi

    log_step "main" "Setup complete"
    exit 0
fi
