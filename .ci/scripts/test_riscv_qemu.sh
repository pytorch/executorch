#!/usr/bin/env bash
# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CI wrapper: install RISC-V cross-compile + qemu-user tooling, then run the
# RISC-V smoke test (export, cross-compile, qemu-user execution) via
# examples/riscv/run.sh. The bundled-IO comparison and Test_result: PASS
# check are done by run.sh.

set -eu

script_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
et_root_dir=$(realpath "${script_dir}/../..")

model="add"
xnnpack=false
quantize=false
verbose_xnnpack=false
debug_xnnpack=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]
Options:
  --model=<NAME>     Which model to export and run (default: add)
  --xnnpack          Enable the XNNPACK backend (AOT partitioner + runtime)
  --quantize         Produce an 8-bit quantized model
  --verbose-xnnpack  Build XNNPACK with XNN_LOG_LEVEL=4 to log microkernel dispatch
  --debug-xnnpack    Enable XNNPACK partitioner DEBUG logging and dump the lowered graph
  -h, --help         Show this help
EOF
}

for arg in "$@"; do
    case $arg in
        --model=*) model="${arg#*=}" ;;
        --xnnpack) xnnpack=true ;;
        --quantize) quantize=true ;;
        --debug-xnnpack) debug_xnnpack=true ;;
        --verbose-xnnpack) verbose_xnnpack=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; usage; exit 1 ;;
    esac
done

run_extra_args=()
if ${xnnpack}; then
    run_extra_args+=(--xnnpack)
fi
if ${quantize}; then
    run_extra_args+=(--quantize)
fi
if ${debug_xnnpack}; then
    run_extra_args+=(--debug-xnnpack)
fi
if ${verbose_xnnpack}; then
    run_extra_args+=(--verbose-xnnpack)
fi

bash "${et_root_dir}/examples/riscv/setup.sh"
bash "${et_root_dir}/examples/riscv/run.sh" --model="${model}" "${run_extra_args[@]}"
