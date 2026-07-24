#!/usr/bin/env bash
# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CI wrapper: install riscv32/64 cross-compile + qemu tooling, then drive
# examples/riscv/run.sh which does the export, cross-compile, qemu run, and
# bundled-IO PASS check.

set -eu

script_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
et_root_dir=$(realpath "${script_dir}/../..")

model="add"
backend="portable"
quantize=false
os="linux"
arch="rv64"
qemu_cpu_ext=""
verbose_xnnpack=false
debug_xnnpack=false
build_dir=

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]
Options:
  --model=<NAME>          Which model to export and run (default: ${model})
  --quantize              Produce an 8-bit quantized model
  --backend=<NAME>        AOT backend (portable|xnnpack) (default: ${backend})
  --os=<NAME>             Target OS (linux|baremetal) (default: ${os})
  --arch=<NAME>           Target arch (rv32|rv64) (default: ${arch})
  --qemu-cpu-ext=<EXT>    QEMU -cpu extensions (no rv32/rv64 prefix, default: none)
  --build-dir=<DIR>       Build/output directory for this configuration (required)
  --verbose-xnnpack       Build XNNPACK with XNN_LOG_LEVEL=4 to log microkernel dispatch
  --debug-xnnpack         Enable XNNPACK partitioner DEBUG logging and dump the lowered graph
  -h, --help              Show this help
EOF
}

for arg in "$@"; do
    case $arg in
        --model=*) model="${arg#*=}" ;;
        --quantize) quantize=true ;;
        --backend=*) backend="${arg#*=}" ;;
        --os=*) os="${arg#*=}" ;;
        --arch=*) arch="${arg#*=}" ;;
        --qemu-cpu-ext=*) qemu_cpu_ext="${arg#*=}" ;;
        --build-dir=*) build_dir="${arg#*=}" ;;
        --debug-xnnpack) debug_xnnpack=true ;;
        --verbose-xnnpack) verbose_xnnpack=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "${build_dir}" ]]; then
    echo "[test_riscv_qemu.sh] --build-dir is required" >&2; usage; exit 1
fi

run_extra_args=()
if [ -n "${qemu_cpu_ext}" ]; then
    run_extra_args+=(--qemu-cpu-ext="${qemu_cpu_ext}")
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

bash "${et_root_dir}/examples/riscv/setup-${os}.sh"
bash "${et_root_dir}/examples/riscv/run.sh" \
    --model="${model}" --backend="${backend}" --os="${os}" --arch="${arch}" \
    --build-dir="${build_dir}" \
    "${run_extra_args[@]}"
