#!/usr/bin/env bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Optional parameter:
# --build_type= "Release" | "Debug" | "RelWithDebInfo"
# --etdump      build with devtools-etdump support

set -eu
set -o pipefail

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
et_root_dir=$(cd ${script_dir}/../../.. && pwd)
et_root_dir=$(realpath ${et_root_dir})
setup_path_script=${et_root_dir}/examples/arm/ethos-u-scratch/setup_path.sh
_setup_msg="please refer to ${et_root_dir}/examples/arm/setup.sh to properly install necessary tools."


model=""
opt_flags=""
build_path="cmake-out-vkml"
converter="model-converter"

help() {
    echo "Usage: $(basename $0) [options]"
    echo "Options:"
    echo "  --model=<MODEL_FILE>    .pte model file to run"
    echo "  --build_path=<BUILD_PATH>  Path to executor_runner build. for Default: ${build_path}"
    exit 0
}

for arg in "$@"; do
    case $arg in
      -h|--help) help ;;
      --optional_flags=*) opt_flags="${arg#*=}";;
      --model=*) model="${arg#*=}";;
      --build_path=*) build_path="${arg#*=}";;
      *)
      ;;
    esac
done

if [[ -z ${model} ]]; then echo "Model name needs to be provided"; exit 1; fi


# Source the tools
# This should be prepared by the setup.sh
[[ -f ${setup_path_script} ]] \
    || { echo "Missing ${setup_path_script}. ${_setup_msg}"; exit 1; }

source ${setup_path_script}

if ! command -v "${converter}" >/dev/null 2>&1; then
    if command -v model_converter >/dev/null 2>&1; then
        converter="model_converter"
    fi
fi

command -v "${converter}" >/dev/null 2>&1 \
    || { echo "Could not find a model converter executable (tried model-converter, model_converter). ${_setup_msg}"; exit 1; }


runner=$(find ${build_path} -name executor_runner -type f)


echo "--------------------------------------------------------------------------------"
echo "Running ${model} with ${runner} ${opt_flags}"
echo "WARNING: The VK_ML layer driver will not provide accurate performance information"
echo "--------------------------------------------------------------------------------"

# Check if stdbuf is intalled and use stdbuf -oL together with tee below to make the output
# go all the way to the console more directly and not be buffered

if hash stdbuf 2>/dev/null; then
    nobuf="stdbuf -oL"
else
    nobuf=""
fi

log_file=$(mktemp)


${nobuf} ${runner} -model_path ${model} ${opt_flags} | tee ${log_file}
echo "[${BASH_SOURCE[0]}] execution complete, $?"

# Most of these can happen for bare metal or linx executor_runner runs.
echo "Checking for problems in log:"
! grep -E "^(F|E|\\[critical\\]|Hard fault.|Info: Simulation is stopping. Reason: CPU time has been exceeded.).*$" ${log_file}
if [ $? != 0 ]; then
    echo "Found ERROR"
    rm "${log_file}"
    exit 1
fi
echo "No problems found!"
rm "${log_file}"
