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
model="add"
xnnpack=false
quantize=false
debug_xnnpack=false
verbose_xnnpack=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]
Options:
  --model=<NAME>          Which model to export and run (default: ${model})
  --xnnpack               Enable the XNNPACK backend (AOT partitioner + runtime)
  --quantize              Produce an 8-bit quantized model
  --verbose-xnnpack       Build XNNPACK with XNN_LOG_LEVEL=4 to log microkernel dispatch at runtime
  --debug-xnnpack         Enable XNNPACK partitioner DEBUG logging and dump the lowered graph
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
        --model=*) model="${arg#*=}" ;;
        --xnnpack) xnnpack=true ;;
        --quantize) quantize=true ;;
        --debug-xnnpack) debug_xnnpack=true ;;
        --verbose-xnnpack) verbose_xnnpack=true ;;
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
bpte_path="${output_dir}/${model}_riscv.bpte"

echo "[run.sh] Step 1/3: AOT export on host"
aot_extra_args=()
if ${xnnpack}; then
    aot_extra_args+=(--xnnpack)
fi
if ${quantize}; then
    aot_extra_args+=(--quantize)
fi
if ${debug_xnnpack}; then
    aot_extra_args+=(--debug-xnnpack)
fi
python "${script_dir}/aot_riscv.py" --model "${model}" "${aot_extra_args[@]}" --output "${bpte_path}"

echo "[run.sh] Step 2/3: cross-compile executor_runner for riscv64-linux"
cmake_extra_args=()
if ${xnnpack}; then
    cmake_extra_args+=(-DEXECUTORCH_BUILD_XNNPACK=ON)
fi
if ${verbose_xnnpack}; then
    cmake_extra_args+=(-DEXECUTORCH_XNNPACK_LOG_LEVEL=4 -DEXECUTORCH_BUILD_RISCV_ETDUMP=ON)
fi
cmake -S "${et_root_dir}" -B "${build_dir}" \
    --preset riscv64-linux \
    "${cmake_extra_args[@]}" \
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

if [[ -n "${QEMU_CPU+x}" ]]; then
    echo "[run.sh] QEMU_CPU=${QEMU_CPU}"
fi

runner_extra_args=()
if ${quantize}; then
    runner_extra_args+=(--bundleio_rtol=0.1 --bundleio_atol=0.25)
fi
etdump_path=""
if ${verbose_xnnpack}; then
    etdump_path="${output_dir}/${model}_riscv.etdump"
    rm -f "${etdump_path}"
    runner_extra_args+=(--etdump_path="${etdump_path}")
fi

# etdump_summary.py reads the XNN_LOG_LEVEL=4 registrations.
log_file="${output_dir}/${model}_riscv.run.log"
rm -f "${log_file}"

set +e
timeout --signal=KILL "${qemu_timeout}" "${qemu}" "${runner}" \
    --model_path="${bpte_path}" \
    "${runner_extra_args[@]}" \
    2>&1 | tee "${log_file}"
qemu_status=${PIPESTATUS[0]}
set -e

echo "[run.sh] qemu exit status: ${qemu_status}"

if [[ -n "${etdump_path}" && -f "${etdump_path}" ]]; then
    python "${script_dir}/etdump_summary.py" "${etdump_path}" \
        --run-log "${log_file}" \
        --json "${etdump_path}.json" || true
fi

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
