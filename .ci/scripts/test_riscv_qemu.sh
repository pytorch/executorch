#!/usr/bin/env bash
# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CI wrapper: install RISC-V cross-compile + qemu-user tooling, then run the
# RISC-V Phase 1 smoke test (export, cross-compile, qemu-user execution) via
# examples/riscv/run.sh. The bundled-IO comparison and Test_result: PASS
# check are done by run.sh.

set -eu

script_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
et_root_dir=$(realpath "${script_dir}/../..")

xnnpack=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]
Options:
  --xnnpack       Enable the XNNPACK backend (AOT partitioner + runtime)
  -h, --help      Show this help
EOF
}

for arg in "$@"; do
    case $arg in
        --xnnpack) xnnpack=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; usage; exit 1 ;;
    esac
done

run_extra_args=()
if ${xnnpack}; then
    run_extra_args+=(--xnnpack)
fi

bash "${et_root_dir}/examples/riscv/setup.sh"
bash "${et_root_dir}/examples/riscv/run.sh" "${run_extra_args[@]}"
