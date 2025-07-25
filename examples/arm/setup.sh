#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -u

########
### Hardcoded constants
########
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
mlsdk_manifest_url=""


# Figure out if setup.sh was called or sourced and save it into "is_script_sourced"
(return 0 2>/dev/null) && is_script_sourced=1 || is_script_sourced=0

# Global scope these so they can be set later
toolchain_url=""
toolchain_dir=""
toolchain_md5_checksum=""

if [[ "${ARCH}" == "x86_64" ]]; then
    # FVPs
    corstone300_url="https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.22_20_Linux64.tgz?rev=018659bd574f4e7b95fa647e7836ccf4&hash=22A79103C6FA5FFA7AFF3BE0447F3FF9"
    corstone300_model_dir="Linux64_GCC-9.3"
    corstone300_md5_checksum="98e93b949d0fbac977292d8668d34523"

    corstone320_url="https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-320/FVP_Corstone_SSE-320_11.27_25_Linux64.tgz?rev=a507bffc219a4d5792f1192ab7002d89&hash=D9A824AA8227D2E679C9B9787FF4E8B6FBE3D7C6"
    corstone320_model_dir="Linux64_GCC-9.3"
    corstone320_md5_checksum="3deb3c68f9b2d145833f15374203514d"
elif [[ "${ARCH}" == "aarch64" ]] || [[ "${ARCH}" == "arm64" ]]; then
    # FVPs
    corstone300_url="https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.22_20_Linux64_armv8l.tgz?rev=9cc6e9a32bb947ca9b21fa162144cb01&hash=7657A4CF27D42E892E3F08D452AAB073"
    corstone300_model_dir="Linux64_armv8l_GCC-9.3"
    corstone300_md5_checksum="cbbabbe39b07939cff7a3738e1492ef1"

    corstone320_url="https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-320/FVP_Corstone_SSE-320_11.27_25_Linux64_armv8l.tgz?rev=b6ebe0923cb84f739e017385fd3c333c&hash=8965C4B98E2FF7F792A099B08831FE3CB6120493"
    corstone320_model_dir="Linux64_armv8l_GCC-9.3"
    corstone320_md5_checksum="3889f1d80a6d9861ea4aa6f1c88dd0ae"
else
    echo "[main] Error: only x86-64 & aarch64/arm64 architecture is supported for now!"; exit 1;
fi

# Vela
vela_repo_url="https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela"
vela_rev="d37febc1715edf0d236c2ff555739a8a9aadcf9a"

# MLSDK dependencies
mlsdk_manifest_dir="ml-sdk-for-vulkan-manifest"

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
    setup_path_script="${root_dir}/setup_path.sh"
}

function check_fvp_eula () {
    # Mandatory user arg --i-agree-to-the-contained-eula
    eula_acceptance_by_variable="${ARM_FVP_INSTALL_I_AGREE_TO_THE_CONTAINED_EULA:-False}"

    if [[ "${eula_acceptance}" -eq 0 ]]; then
        if [[ ${eula_acceptance_by_variable} != "True" ]]; then
            echo "Must pass argument '--i-agree-to-the-contained-eula' to agree to EULA associated with downloading the FVP."
            echo "Alternativly set environment variable ARM_FVP_INSTALL_I_AGREE_TO_THE_CONTAINED_EULA=True."
            echo "Exiting!"
            exit 1
        else
            echo "Arm EULA for FVP agreed to with ARM_FVP_INSTALL_I_AGREE_TO_THE_CONTAINED_EULA=True environment variable"
        fi
    fi
}

function setup_fvp() {
    # check EULA, forward argument
    check_fvp_eula

    if [[ "${OS}" != "Linux" ]]; then
        # Check if FVP is callable
        if command -v FVP_Corstone_SSE-300_Ethos-U55 &> /dev/null; then
            echo "[${FUNCNAME[0]}] Info: FVP for MacOS seem to be installed. Continuing..."
            return 0  # If true exit gracefully and proceed with setup
        else
            echo "[${FUNCNAME[0]}] Warning: FVP only supported with Linux OS, skipping FVP setup..."
            echo "[${FUNCNAME[0]}] Warning: For MacOS, using https://github.com/Arm-Examples/FVPs-on-Mac is recommended."
            echo "[${FUNCNAME[0]}] Warning:   Follow the instructions and make sure the path is set correctly."
            return 1  # Throw error. User need to install FVP according to ^^^
        fi
    fi

    # Download and install the Corstone 300 FVP simulator platform
    fvps=("corstone300" "corstone320")

    for fvp in "${fvps[@]}"; do
        cd "${root_dir}"
        if [[ ! -e "FVP_${fvp}.tgz" ]]; then
            echo "[${FUNCNAME[0]}] Downloading FVP ${fvp}..."
            url_variable=${fvp}_url
            fvp_url=${!url_variable}
            curl --output "FVP_${fvp}.tgz" "${fvp_url}"
            md5_variable=${fvp}_md5_checksum
            fvp_md5_checksum=${!md5_variable}
            verify_md5 ${fvp_md5_checksum} FVP_${fvp}.tgz || exit 1
        fi

        echo "[${FUNCNAME[0]}] Installing FVP ${fvp}..."
        rm -rf FVP-${fvp}
        mkdir -p FVP-${fvp}
        cd FVP-${fvp}
        tar xf ../FVP_${fvp}.tgz

        # Install the FVP
        case ${fvp} in
            corstone300)
                ./FVP_Corstone_SSE-300.sh --i-agree-to-the-contained-eula --force --destination ./ --quiet --no-interactive
                ;;
            corstone320)
                ./FVP_Corstone_SSE-320.sh --i-agree-to-the-contained-eula --force --destination ./ --quiet --no-interactive
                ;;
            *)
                echo "[${FUNCNAME[0]}] Error: Unknown FVP model ${fvp}. Exiting."
                exit 1
                ;;
        esac
    done
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
	        # eventually, this can be support by downloading the the toolchain from 
		# "https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.17.2/toolchain_linux-aarch64_arm-zephyr-eabi.tar.xz"
		# but for now, we error if user tries to specify this
                echo "[main] Error: currently target_toolchain zephyr is only support for x86-64 Linux host systems!"; exit 1;
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

function setup_vela() {
    pip install ethos-u-vela@git+${vela_repo_url}@${vela_rev}
}

function create_setup_path(){
    cd "${root_dir}"

    echo "" > "${setup_path_script}"

    if [[ "${enable_fvps}" -eq 1 ]]; then
        fvps=("corstone300" "corstone320")
        for fvp in "${fvps[@]}"; do
            model_dir_variable=${fvp}_model_dir
            fvp_model_dir=${!model_dir_variable}
            fvp_bin_path="${root_dir}/FVP-${fvp}/models/${fvp_model_dir}"
            echo "export PATH=\${PATH}:${fvp_bin_path}" >> ${setup_path_script}
        done

        # Fixup for Corstone-320 python dependency
        echo "export LD_LIBRARY_PATH=${root_dir}/FVP-corstone320/python/lib/" >> ${setup_path_script}

        echo "hash FVP_Corstone_SSE-300_Ethos-U55" >> ${setup_path_script}
        echo "hash FVP_Corstone_SSE-300_Ethos-U65" >> ${setup_path_script}
        echo "hash FVP_Corstone_SSE-320" >> ${setup_path_script}
    fi

    if [[ "${enable_baremetal_toolchain}" -eq 1 ]]; then
        toolchain_bin_path="$(cd ${toolchain_dir}/bin && pwd)"
        echo "export PATH=\${PATH}:${toolchain_bin_path}" >> ${setup_path_script}
    fi

    if [[ "${enable_model_converter}" -eq 1 ]]; then
        cd "${root_dir}"
        model_converter_bin_path="$(cd ${mlsdk_manifest_dir}/sw/model-converter/build && pwd)"
        echo "export PATH=\${PATH}:${model_converter_bin_path}" >> ${setup_path_script}
    fi

    # Add Path for vgf-lib and emulation-layer
    if [[ "${enable_vgf_lib}" -eq 1 ]]; then
        cd "${root_dir}"
        model_vgf_lib_bin_path="$(cd ${mlsdk_manifest_dir}/sw/vgf-lib/build && pwd)"
        echo "export PATH=\${PATH}:${model_vgf_lib_bin_path}" >> ${setup_path_script}
    fi

    if [[ "${enable_emulation_layer}" -eq 1 ]]; then
        cd "${root_dir}"
        model_emulation_layer_bin_path="$(cd ${mlsdk_manifest_dir}/sw/vgf-lib/build && pwd)"
        echo "export PATH=\${PATH}:${model_emulation_layer_bin_path}" >> ${setup_path_script}
    fi
}

function check_platform_support() {
    # Make sure we are on a supported platform
    if [[ "${ARCH}" != "x86_64" ]] && [[ "${ARCH}" != "aarch64" ]] \
        && [[ "${ARCH}" != "arm64" ]]; then
        echo "[main] Error: only x86-64 & aarch64 architecture is supported for now!"
        exit 1
    fi
}


########
### main
########

# script is not sourced! Lets run "main"
if [[ $is_script_sourced -eq 0 ]]; then
    set -e

    check_options "$@"

    check_platform_support

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
    echo "enable-vela=${enable_vela}"
    echo "mlsdk-manifest-url=${mlsdk_manifest_url}"

    # Import utils
    source $et_dir/backends/arm/scripts/utils.sh

    # Select appropriate toolchain
    select_toolchain

    # Setup toolchain
    if [[ "${enable_baremetal_toolchain}" -eq 1 ]]; then
        setup_toolchain
    fi

    # Setup FVP
    if [[ "${enable_fvps}" -eq 1 ]]; then
        setup_fvp
    fi


    if [[ -z "$mlsdk_manifest_url" && "${enable_model_converter}" -eq 1 ]]; then
        echo "Warning: mlsdk-manifest-url is not set, but model converter setup is not skipped."
        echo "         Please set the --mlsdk-manifest-url option to the correct URL."
        echo "         Skipping MLSDK model converter setup."
        enable_model_converter=0  # Q: Can we assume if we enable mlsdk, we will always enable model converter
        enable_vgf_lib=0
        enable_emulation_layer=0
    fi

    if [[ "${enable_model_converter}" -eq 1 ]]; then
        source $et_dir/backends/arm/scripts/mlsdk_utils.sh -u "${mlsdk_manifest_url}"
        setup_model_converter ${root_dir} ${mlsdk_manifest_dir} ${enable_vgf_lib} ${enable_emulation_layer}
    fi

    # Create new setup_path script
    if [[ "${enable_baremetal_toolchain}" -eq 1 || \
          "${enable_fvps}" -eq 1 || \
          "${enable_model_converter}" -eq 1 ]]; then
        create_setup_path
    fi

    # Setup the tosa_reference_model
    $et_dir/backends/arm/scripts/install_reference_model.sh ${root_dir}

    # Setup vela and patch in codegen fixes
    if [[ "${enable_vela}" -eq 1 ]]; then
        setup_vela
    fi

    echo "[main] update path by doing 'source ${setup_path_script}'"

    echo "[main] success!"
    exit 0
fi
