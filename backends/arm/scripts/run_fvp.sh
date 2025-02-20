#!/usr/bin/env bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Optional parameter:
# --build_type= "Release" | "Debug" | "RelWithDebInfo"
# --etdump      build with devtools-etdump support

set -eu

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
et_root_dir=$(cd ${script_dir}/../../.. && pwd)
et_root_dir=$(realpath ${et_root_dir})
setup_path_script=${et_root_dir}/examples/arm/ethos-u-scratch/setup_path.sh
_setup_msg="please refer to ${et_root_dir}/examples/arm/setup.sh to properly install necessary tools."


elf_file=""
target="ethos-u55-128"

help() {
    echo "Usage: $(basename $0) [options]"
    echo "Options:"
    echo "  --elf=<ELF_FILE>         elf file to run"
    echo "  --target=<TARGET>        Target to build and run for Default: ${target}"
    exit 0
}

for arg in "$@"; do
    case $arg in
      -h|--help) help ;;
      --elf=*) elf_file="${arg#*=}";;
      --target=*) target="${arg#*=}";;
      *)
      ;;
    esac
done

elf_file=$(realpath ${elf_file})

if [[ ${target} == *"ethos-u55"*  ]]; then
    fvp_model=FVP_Corstone_SSE-300_Ethos-U55
else
    fvp_model=FVP_Corstone_SSE-320
fi

# Source the tools
# This should be prepared by the setup.sh
[[ -f ${setup_path_script} ]] \
    || { echo "Missing ${setup_path_script}. ${_setup_msg}"; exit 1; }

source ${setup_path_script}

# basic checks before we get started
hash ${fvp_model} \
    || { echo "Could not find ${fvp_model} on PATH, ${_setup_msg}"; exit 1; }


[[ ! -f $elf_file ]] && { echo "[${BASH_SOURCE[0]}]: Unable to find executor_runner elf: ${elf_file}"; exit 1; }
num_macs=$(echo ${target} | cut -d - -f 3)

echo "--------------------------------------------------------------------------------"
echo "Running ${elf_file} for ${target} run with FVP:${fvp_model} num_macs:${num_macs}"
echo "--------------------------------------------------------------------------------"

log_file=$(mktemp)

if [[ ${target} == *"ethos-u55"*  ]]; then
    ${fvp_model}                                            \
        -C ethosu.num_macs=${num_macs}                      \
        -C mps3_board.visualisation.disable-visualisation=1 \
        -C mps3_board.telnetterminal0.start_telnet=0        \
        -C mps3_board.uart0.out_file='-'                    \
        -C mps3_board.uart0.shutdown_on_eot=1               \
        -a "${elf_file}"                                         \
        --timelimit 220 2>&1 | tee ${log_file} || true # seconds
    echo "[${BASH_SOURCE[0]}] Simulation complete, $?"
elif [[ ${target} == *"ethos-u85"*  ]]; then
    ${fvp_model}                                            \
        -C mps4_board.subsystem.ethosu.num_macs=${num_macs} \
        -C mps4_board.visualisation.disable-visualisation=1 \
        -C vis_hdlcd.disable_visualisation=1                \
        -C mps4_board.telnetterminal0.start_telnet=0        \
        -C mps4_board.uart0.out_file='-'                    \
        -C mps4_board.uart0.shutdown_on_eot=1               \
        -a "${elf_file}"                                         \
        --timelimit 220 2>&1 | tee ${log_file} || true # seconds
    echo "[${BASH_SOURCE[0]}] Simulation complete, $?"
else
    echo "Running ${elf_file} for ${target} is not supported"
    exit 1
fi

echo "Checking for problems in log:"
! grep -E "^(F|E|\\[critical\\]|Hard fault.|Info: Simulation is stopping. Reason: CPU time has been exceeded.).*$" ${log_file}
if [ $? != 0 ]; then
    echo "Found ERROR"
    rm "${log_file}"
    exit 1
fi
echo "No problems found!"
rm "${log_file}"
