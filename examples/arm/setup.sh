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
vulkan_sdk_version="1.4.321.1"
vulkan_sdk_base_dir="vulkan_sdk"

# Figure out if setup.sh was called or sourced and save it into "is_script_sourced"
(return 0 2>/dev/null) && is_script_sourced=1 || is_script_sourced=0

# Global scope these so they can be set later
toolchain_url=""
toolchain_dir=""
toolchain_md5_checksum=""

if [[ "${ARCH}" == "x86_64" ]]; then
    # Vulkan SDK
    vulkan_sdk_url="https://sdk.lunarg.com/sdk/download/${vulkan_sdk_version}/linux/vulkansdk-linux-x86_64-${vulkan_sdk_version}.tar.xz"
    vulkan_sdk_sha256="f22a3625bd4d7a32e7a0d926ace16d5278c149e938dac63cecc00537626cbf73"

elif [[ "${ARCH}" == "aarch64" ]] || [[ "${ARCH}" == "arm64" ]]; then
    # Vulkan SDK
    vulkan_sdk_url="https://github.com/jakoch/vulkan-sdk-arm/releases/download/1.4.321.1/vulkansdk-ubuntu-22.04-arm-1.4.321.1.tar.xz"
    vulkan_sdk_sha256="c57e318d0940394d3a304034bb7ddabda788b5b0b54638e80e90f7264efe9f84"

else
    echo "[main] Error: only x86-64 & aarch64/arm64 architecture is supported for now!"; exit 1;
fi

# MLSDK dependencies
mlsdk_manifest_dir="ml-sdk-for-vulkan-manifest"
vulkan_sdk_bin_dir="${vulkan_sdk_base_dir}/${vulkan_sdk_version}/${ARCH}/bin"

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

function setup_vulkan_sdk() {

    if command -v vulkaninfo > /dev/null 2>&1; then
        echo "[${FUNCNAME[0]}] Vulkan SDK already installed..."
        enable_vulkan_sdk=0
        return
    fi

    cd "${root_dir}"

    vulkan_sdk_tar_file="${vulkan_sdk_url##*/}"
    if [[ ! -e "${vulkan_sdk_tar_file}" ]]; then
        echo "[${FUNCNAME[0]}] Downloading Vulkan SDK - ${vulkan_sdk_url}.."
        curl -L --output "${vulkan_sdk_tar_file}" "${vulkan_sdk_url}"
        echo "${vulkan_sdk_sha256} ${vulkan_sdk_tar_file}" | sha256sum -c -
        rm -fr ${vulkan_sdk_base_dir}
    fi

    mkdir -p ${vulkan_sdk_base_dir}
    tar -C ${vulkan_sdk_base_dir} -xJf "${vulkan_sdk_tar_file}"

    vulkan_sdk_bin_path="$(cd ${vulkan_sdk_bin_dir} && pwd)"
    if ${vulkan_sdk_bin_path}/vulkaninfo > /dev/null 2>&1; then
        echo "[${FUNCNAME[0]}] Vulkan SDK OK"
    else
        echo "[${FUNCNAME[0]}] Vulkan SDK NOK - perhaps need manual install of swifthshader or mesa-vulkan driver?"
        exit 1
    fi
}

function select_toolchain() {
    if [[ "${ARCH}" == "x86_64" ]]; then
        if [[ "${OS}" == "Linux" ]]; then
	    if [[ "${target_toolchain}" == "zephyr" ]]; then
	        # TODO can include support for zephyr toolchain for other host platforms later
                toolchain_url="https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.17.2/toolchain_linux-x86_64_arm-zephyr-eabi.tar.xz"
                toolchain_dir="arm-zephyr-eabi"
                toolchain_md5_checksum="93128be0235cf5cf5f1ee561aa6eac5f"
            else
                toolchain_url="https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu/13.3.rel1/binrel/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi.tar.xz"
                toolchain_dir="arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi"
                toolchain_md5_checksum="0601a9588bc5b9c99ad2b56133b7f118"
	    fi
        else
            echo "[main] Error: only Linux is currently supported for x86-64 architecture now!"; exit 1;
	fi
   elif [[ "${ARCH}" == "aarch64" ]] || [[ "${ARCH}" == "arm64" ]]; then
        if [[ "${OS}" == "Darwin" ]]; then
	    if [[ "${target_toolchain}" == "zephyr" ]]; then
                echo "[main] Error: only Linux OS is currently supported for aarch64 architecture targeting Zephyr now!"; exit 1;
	    else
                toolchain_url="https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu/13.3.rel1/binrel/arm-gnu-toolchain-13.3.rel1-darwin-arm64-arm-none-eabi.tar.xz"
                toolchain_dir="arm-gnu-toolchain-13.3.rel1-darwin-arm64-arm-none-eabi"
                toolchain_md5_checksum="f1c18320bb3121fa89dca11399273f4e"
	    fi
        elif [[ "${OS}" == "Linux" ]]; then
	    if [[ "${target_toolchain}" == "zephyr" ]]; then
                toolchain_url="https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.17.2/toolchain_linux-aarch64_arm-zephyr-eabi.tar.xz"
                toolchain_dir="arm-zephyr-eabi"
		toolchain_md5_checksum="ef4ca56786204439a75270ba800cc64b"
	    else
                toolchain_url="https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu/13.3.rel1/binrel/arm-gnu-toolchain-13.3.rel1-aarch64-arm-none-eabi.tar.xz"
                toolchain_dir="arm-gnu-toolchain-13.3.rel1-aarch64-arm-none-eabi"
                toolchain_md5_checksum="303102d97b877ebbeb36b3158994b218"
	    fi
        fi
    else
        echo "[main] Error: only x86-64 & aarch64/arm64 architecture is supported for now!"; exit 1;
    fi
    echo "[main] Info selected ${toolchain_dir} for ${ARCH} - ${OS} platform"
}

function setup_toolchain() {
    # Download and install the arm toolchain (default is arm-none-eabi)
    # setting --target-toolchain to zephyr sets this to arm-zephyr-eabi
    cd "${root_dir}"
    if [[ ! -e "${toolchain_dir}.tar.xz" ]]; then
        echo "[${FUNCNAME[0]}] Downloading ${toolchain_dir} toolchain ..."
        curl --output "${toolchain_dir}.tar.xz" -L "${toolchain_url}"
        verify_md5 ${toolchain_md5_checksum} "${toolchain_dir}.tar.xz" || exit 1
    fi

    echo "[${FUNCNAME[0]}] Installing ${toolchain_dir} toolchain ..."
    rm -rf "${toolchain_dir}"
    tar xf "${toolchain_dir}.tar.xz"
}

function setup_ethos_u_tools() {
    CMAKE_POLICY_VERSION_MINIMUM=3.5 BUILD_PYBIND=1 pip install --no-dependencies -r $et_dir/backends/arm/requirements-arm-ethos-u.txt
}

function prepend_env_in_setup_path() {
    echo "export $1=$2:\${$1-}" >> ${setup_path_script}.sh
    echo "set --path -pgx $1 $2" >> ${setup_path_script}.fish
}

function append_env_in_setup_path() {
    echo "export $1=\${$1-}:$2" >> ${setup_path_script}.sh
    echo "set --path -agx $1 $2" >> ${setup_path_script}.fish
}

function create_setup_path(){
    cd "${root_dir}"

    # Clear setup_path_script
    echo "" > "${setup_path_script}.sh"
    echo "" > "${setup_path_script}.fish"

    if [[ "${enable_fvps}" -eq 1 ]]; then
        setup_path_fvp
    fi

    if [[ "${enable_baremetal_toolchain}" -eq 1 ]]; then
        toolchain_bin_path="$(cd ${toolchain_dir}/bin && pwd)"
        append_env_in_setup_path PATH ${toolchain_bin_path}
    fi

    if [[ "${enable_vulkan_sdk}" -eq 1 ]]; then
        cd "${root_dir}"
        vulkan_sdk_bin_path="$(cd ${vulkan_sdk_bin_dir} && pwd)"
        append_env_in_setup_path PATH ${vulkan_sdk_bin_path}
    fi

    if [[ "${enable_model_converter}" -eq 1 ]]; then
        cd "${root_dir}"
        model_converter_bin_path="$(cd ${mlsdk_manifest_dir}/sw/model-converter/build && pwd)"
        append_env_in_setup_path PATH ${model_converter_bin_path}
    fi

    # Add Path for vgf-lib and emulation-layer
    if [[ "${enable_vgf_lib}" -eq 1 ]]; then
        cd "${root_dir}"
        model_vgf_path="$(cd ${mlsdk_manifest_dir}/sw/vgf-lib/deploy && pwd)"
        append_env_in_setup_path PATH ${model_vgf_path}/bin
        append_env_in_setup_path LD_LIBRARY_PATH "${model_vgf_path}/lib"
        append_env_in_setup_path DYLD_LIBRARY_PATH "${model_vgf_path}/lib"
    fi

    if [[ "${enable_emulation_layer}" -eq 1 ]]; then
        cd "${root_dir}"
        model_emulation_layer_path="$(cd ${mlsdk_manifest_dir}/sw/emulation-layer/ && pwd)"
        prepend_env_in_setup_path LD_LIBRARY_PATH "${model_emulation_layer_path}/deploy/lib"
        prepend_env_in_setup_path DYLD_LIBRARY_PATH "${model_emulation_layer_path}/deploy/lib"
        prepend_env_in_setup_path VK_INSTANCE_LAYERS VK_LAYER_ML_Tensor_Emulation
        prepend_env_in_setup_path VK_INSTANCE_LAYERS VK_LAYER_ML_Graph_Emulation
        prepend_env_in_setup_path VK_ADD_LAYER_PATH "${model_emulation_layer_path}/deploy/share/vulkan/explicit_layer.d"
    fi
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

    # Select appropriate toolchain
    select_toolchain

    # Setup toolchain
    if [[ "${enable_baremetal_toolchain}" -eq 1 ]]; then
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

    # Create new setup_path script
    if [[ "${enable_baremetal_toolchain}" -eq 1 || \
           "${enable_fvps}" -eq 1 || \
           "${enable_vulkan_sdk}" -eq 1 || \
          "${enable_model_converter}" -eq 1 ]]; then
        create_setup_path
    fi

    # Setup the tosa_reference_model and dependencies
    CMAKE_POLICY_VERSION_MINIMUM=3.5 BUILD_PYBIND=1 pip install --no-dependencies -r $et_dir/backends/arm/requirements-arm-tosa.txt

    if [[ "${enable_vela}" -eq 1 ]]; then
        setup_ethos_u_tools
    fi

    echo "[main] Update path by running 'source ${setup_path_script}.sh'"
    hash fish 2>/dev/null && echo >&2 "[main] Or for fish shell use 'source ${setup_path_script}.fish'"
    echo "[main] success!"
    exit 0
fi
