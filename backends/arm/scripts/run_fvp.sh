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
data_file=""
target="ethos-u55-128"
timeout="600"
etrecord_file=""
trace_file=""

help() {
    echo "Usage: $(basename $0) [options]"
    echo "Options:"
    echo "  --elf=<ELF_FILE>         elf file to run"
    echo "  --data=<FILE>@<ADDRESS>  Place a file in memory at this address, useful to emulate a PTE flashed into memory instead as part of the code."
    echo "  --target=<TARGET>        Target to build and run for Default: ${target}"
    echo "  --timeout=<TIME_IN_SEC>  Maximum target runtime, used to detect hanging, might need to be higer on large models Default: ${timeout}"
    echo "  --etrecord=<FILE>        If ETDump is used you can supply a ETRecord file matching the PTE"
    echo "  --trace_file=<FILE>      File to write PMU trace output to"
    exit 0
}

for arg in "$@"; do
    case $arg in
      -h|--help) help ;;
      --elf=*) elf_file="${arg#*=}";;
      --data=*) data_file="--data ${arg#*=}";;
      --target=*) target="${arg#*=}";;
      --timeout=*) timeout="${arg#*=}";;
      --etrecord=*) etrecord_file="${arg#*=}";;
      --trace_file=*) trace_file="${arg#*=}";;
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
echo "Running ${elf_file} for ${target} run with FVP:${fvp_model} num_macs:${num_macs} timeout:${timeout}"
echo "WARNING: Corstone FVP is not cycle accurate and should NOT be used to determine valid runtime"
echo "--------------------------------------------------------------------------------"

# Check if stdbuf is intalled and use stdbuf -oL together with tee below to make the output
# go all the way to the console more directly and not be buffered

if hash stdbuf 2>/dev/null; then
    nobuf="stdbuf -oL"
else
    nobuf=""
fi

log_file=$(mktemp)

extra_args_u55=()
extra_args_u85=()

if [[ -n "${trace_file}" ]]; then
    extra_args_u55+=(-C "ethosu.extra_args=--pmu-trace ${trace_file}")
    extra_args_u85+=(-C "mps4_board.subsystem.ethosu.extra_args=--pmu-trace ${trace_file}")
fi

if [[ ${target} == *"ethos-u55"*  ]]; then
    ${nobuf} ${fvp_model}                                   \
        -C ethosu.num_macs=${num_macs}                      \
        -C mps3_board.visualisation.disable-visualisation=1 \
        -C mps3_board.telnetterminal0.start_telnet=0        \
        -C mps3_board.uart0.out_file='-'                    \
        -C mps3_board.uart0.shutdown_on_eot=1               \
        "${extra_args_u55[@]}"                              \
        -a "${elf_file}"                                    \
        ${data_file}                                        \
        --timelimit ${timeout} 2>&1 | sed 's/\r$//' | tee ${log_file} || true # seconds
    echo "[${BASH_SOURCE[0]}] Simulation complete, $?"
elif [[ ${target} == *"ethos-u85"*  ]]; then
    ${nobuf} ${fvp_model}                                   \
        -C mps4_board.subsystem.ethosu.num_macs=${num_macs} \
        -C mps4_board.visualisation.disable-visualisation=1 \
        -C vis_hdlcd.disable_visualisation=1                \
        -C mps4_board.telnetterminal0.start_telnet=0        \
        -C mps4_board.uart0.out_file='-'                    \
        -C mps4_board.uart0.shutdown_on_eot=1               \
        "${extra_args_u85[@]}"                              \
        -a "${elf_file}"                                    \
        ${data_file}                                        \
        --timelimit ${timeout} 2>&1 | sed 's/\r$//' | tee ${log_file} || true # seconds
    echo "[${BASH_SOURCE[0]}] Simulation complete, $?"
else
    echo "Running ${elf_file} for ${target} is not supported"
    exit 1
fi

echo "Checking for a etdump in log"
! grep "#\[RUN THIS\]" ${log_file} >/dev/null
if [ $? != 0 ]; then
    echo "Found ETDump in log!"
    devtools_extra_args=""
    echo "#!/bin/sh" > etdump_script.sh
    sed -n '/^#\[RUN THIS\]$/,/^#\[END\]$/p' ${log_file} >> etdump_script.sh
    # You can run etdump_script.sh if you do
    # $ chmod a+x etdump_script.sh
    # $ ./etdump_script.sh
    # But lets not trust the script as a bad patch would run bad code on your machine
    grep ">etdump.bin" etdump_script.sh | cut -d\" -f2- | cut -d\" -f1 | base64 -d >etdump.bin
    ! grep ">debug_buffer.bin" etdump_script.sh >/dev/null
    if [ $? != 0 ]; then
        grep ">debug_buffer.bin" etdump_script.sh | cut -d\" -f2- | cut -d\" -f1 | base64 -d >debug_buffer.bin
        devtools_extra_args="${devtools_extra_args} --debug_buffer_path debug_buffer.bin"
    fi
    if [[ ${etrecord_file} != "" ]]; then
        devtools_extra_args="${devtools_extra_args} --etrecord_path ${etrecord_file}"
    fi
    python3 -m devtools.inspector.inspector_cli --etdump_path etdump.bin ${devtools_extra_args} --source_time_scale cycles --target_time_scale cycles
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
