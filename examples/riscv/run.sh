#!/usr/bin/env bash
# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# RISC-V Phase 1 smoke test driver (pytorch/executorch#18991):
#   1. Export a tiny model to a BundledProgram (.bpte) on the x86_64 host.
#   2. Cross-compile executor_runner for riscv64 Linux glibc.
#   3. Invoke the runner under qemu-user-static and grep its stdout for the
#      Test_result: PASS marker emitted by the bundled-IO comparison path.

set -eu

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
et_root_dir=$(realpath "${script_dir}/../..")

build_only=false
build_dir="${et_root_dir}/cmake-out-riscv"
output_dir="${et_root_dir}/riscv_test"
qemu="qemu-riscv64-static"
qemu_timeout="600"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]
Options:
  --build_only            Only export and cross-compile; do not invoke QEMU
  --build_dir=<DIR>       CMake build directory (default: ${build_dir})
  --output_dir=<DIR>      Directory for the exported .bpte (default: ${output_dir})
  --qemu=<BIN>            qemu-user binary (default: ${qemu})
  --timeout=<SECONDS>     Maximum QEMU runtime; matches run_fvp.sh --timelimit (default: ${qemu_timeout})
  -h, --help              Show this help
EOF
}

for arg in "$@"; do
    case $arg in
        --build_only) build_only=true ;;
        --build_dir=*) build_dir="${arg#*=}" ;;
        --output_dir=*) output_dir="${arg#*=}" ;;
        --qemu=*) qemu="${arg#*=}" ;;
        --timeout=*) qemu_timeout="${arg#*=}" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; usage; exit 1 ;;
    esac
done

mkdir -p "${output_dir}"
bpte_path="${output_dir}/add_riscv.bpte"

echo "[run.sh] Step 1/3: AOT export on host"
python "${script_dir}/aot_riscv.py" --output "${bpte_path}"

echo "[run.sh] Step 2/3: cross-compile executor_runner for riscv64-linux"
cmake -S "${et_root_dir}" -B "${build_dir}" \
    --preset riscv64-linux \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" -j"$(nproc)" --target executor_runner

runner="${build_dir}/executor_runner"
[[ -x "${runner}" ]] || { echo "[run.sh] runner not found at ${runner}" >&2; exit 1; }

if file "${runner}" | grep -q "RISC-V"; then
    echo "[run.sh] runner is a RISC-V ELF: $(file -b "${runner}")"
else
    echo "[run.sh] ERROR: ${runner} does not look like a RISC-V ELF"
    file "${runner}"
    exit 1
fi

if ${build_only}; then
    echo "[run.sh] --build_only set, skipping QEMU invocation"
    exit 0
fi

echo "[run.sh] Step 3/3: run under ${qemu}"
hash "${qemu}" 2>/dev/null || {
    echo "[run.sh] ERROR: ${qemu} not found on PATH; install with examples/riscv/setup.sh" >&2
    exit 1
}

# QEMU_LD_PREFIX points qemu-user at the riscv64 sysroot so the dynamic
# linker (ld-linux-riscv64-lp64d.so.1) referenced in the ELF resolves.
export QEMU_LD_PREFIX="${QEMU_LD_PREFIX:-/usr/riscv64-linux-gnu}"

log_file=$(mktemp)
trap 'rm -f "${log_file}"' EXIT

set +e
timeout --signal=KILL "${qemu_timeout}" "${qemu}" "${runner}" \
    --model_path="${bpte_path}" \
    2>&1 | tee "${log_file}"
qemu_status=${PIPESTATUS[0]}
set -e

echo "[run.sh] qemu exit status: ${qemu_status}"

if grep -q "Test_result: PASS" "${log_file}"; then
    echo "[run.sh] Bundled I/O check PASSED"
    exit 0
elif grep -q "Test_result: FAIL" "${log_file}"; then
    echo "[run.sh] ERROR: Bundled I/O check FAILED"
    exit 1
else
    echo "[run.sh] ERROR: No Test_result line found in QEMU output"
    exit 1
fi
