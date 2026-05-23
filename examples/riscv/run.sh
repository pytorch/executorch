#!/usr/bin/env bash
# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# RISC-V smoke test driver:
#   1. Export a small model to a BundledProgram (.bpte) on the host.
#   2. Cross-compile a riscv32/64 runner (linux glibc or baremetal).
#   3. Invoke under qemu and grep stdout for the Test_result: PASS marker.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
et_root_dir=$(realpath "${script_dir}/../..")

build_only=false
build_dir=
qemu_timeout="1800"
model="add"
backend="portable"
os="linux"
arch="rv64"
qemu_cpu_ext=""
quantize=false
debug_xnnpack=false
verbose_xnnpack=false
qemu_override=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]
Options:
  --model=<NAME>          Which model to export and run (default: ${model})
  --quantize              Produce an 8-bit quantized model
  --backend=<NAME>        AOT backend (default: ${backend}):
                           - 'portable': portable kernels only
                           - 'xnnpack':  XNNPACK delegate (linux only)
  --os=<NAME>             Target OS (default: ${os}):
                           - 'linux':    glibc, qemu-user
                           - 'baremetal': no OS, qemu-system + semihosting
  --arch=<NAME>           Target arch (default: ${arch}):
                           - 'rv64': riscv64
                           - 'rv32': riscv32
  --qemu-cpu-ext=<EXT>    QEMU -cpu extensions appended after the arch base
                          (e.g. 'v=true,vlen=128'); no rv32/rv64 prefix.
  --verbose-xnnpack       Build XNNPACK with XNN_LOG_LEVEL=4 to log microkernel dispatch
  --debug-xnnpack         Enable XNNPACK partitioner DEBUG logging and dump the lowered graph
  --build_only            Only export and cross-compile; do not invoke QEMU
  --build-dir=<DIR>       Build/output directory for this configuration (required)
  --qemu=<BIN>            Override qemu binary
  --timeout=<SECONDS>     Maximum QEMU runtime (default: ${qemu_timeout})
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
        --debug-xnnpack) debug_xnnpack=true ;;
        --verbose-xnnpack) verbose_xnnpack=true ;;
        --build_only) build_only=true ;;
        --build-dir=*) build_dir="${arg#*=}" ;;
        --qemu=*) qemu_override="${arg#*=}" ;;
        --timeout=*) qemu_timeout="${arg#*=}" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; usage; exit 1 ;;
    esac
done

case "${backend}" in
    portable|xnnpack) ;;
    *) echo "Unknown backend: ${backend}" >&2; usage; exit 1 ;;
esac
case "${os}" in
    linux|baremetal) ;;
    *) echo "Unknown os: ${os}" >&2; usage; exit 1 ;;
esac
case "${arch}" in
    rv32|rv64) ;;
    *) echo "Unknown arch: ${arch}" >&2; usage; exit 1 ;;
esac

# xnnpack needs pthreads + dynamic loading: baremetal has neither, and the
# Ubuntu xnnpack microkernels don't ship an rv32 build.
if [[ "${backend}" == "xnnpack" && "${os}" == "baremetal" ]]; then
    echo "[run.sh] --backend=xnnpack requires --os=linux" >&2
    exit 1
fi
if [[ "${backend}" == "xnnpack" && "${arch}" == "rv32" ]]; then
    echo "[run.sh] --backend=xnnpack requires --arch=rv64" >&2
    exit 1
fi
# Ubuntu doesn't package a riscv32-linux-gnu cross (riscv64-linux-gnu has no
# rv32 multilib either), so rv32 linux is blocked on a custom toolchain build.
if [[ "${arch}" == "rv32" && "${os}" == "linux" ]]; then
    echo "[run.sh] --arch=rv32 --os=linux not supported: no riscv32-linux-gnu toolchain on Ubuntu" >&2
    exit 1
fi

if ${debug_xnnpack} && [[ "${backend}" != "xnnpack" ]]; then
    echo "[run.sh] --debug-xnnpack requires --backend=xnnpack" >&2
    exit 1
fi
if ${verbose_xnnpack} && [[ "${backend}" != "xnnpack" ]]; then
    echo "[run.sh] --verbose-xnnpack requires --backend=xnnpack" >&2
    exit 1
fi

if [[ -z "${build_dir}" ]]; then
    echo "[run.sh] --build-dir is required" >&2; usage; exit 1
fi
mkdir -p "${build_dir}"

bpte_path="${build_dir}/model.bpte"

echo "[run.sh] Step 1/3: AOT export on host (backend=${backend} os=${os} arch=${arch})"
aot_extra_args=()
if ${quantize}; then
    aot_extra_args+=(--quantize)
fi
if ${debug_xnnpack}; then
    aot_extra_args+=(--debug-xnnpack)
fi
python "${script_dir}/aot_riscv.py" --model "${model}" --backend "${backend}" --os "${os}" "${aot_extra_args[@]}" --output "${bpte_path}"

echo "[run.sh] Step 2/3: cross-compile executor_runner for ${arch}-${os}"
cmake_extra_args=()
if [[ "${backend}" == "xnnpack" ]]; then
    cmake_extra_args+=(-DEXECUTORCH_BUILD_XNNPACK=ON)
fi
if ${verbose_xnnpack}; then
    cmake_extra_args+=(-DEXECUTORCH_XNNPACK_LOG_LEVEL=4 -DEXECUTORCH_BUILD_RISCV_ETDUMP=ON)
fi

# Map our short arch (rv32/rv64) to the canonical riscv32/riscv64 prefix used
# by the cross toolchain and qemu binary names.
case "${arch}" in
    rv32) arch_long="riscv32" ;;
    rv64) arch_long="riscv64" ;;
esac

if [[ "${os}" == "linux" ]]; then
    build_target="executor_runner"
    qemu_default="qemu-${arch_long}-static"
    cmake -S "${et_root_dir}" -B "${build_dir}" --fresh \
        --preset "${arch_long}-linux" \
        "${cmake_extra_args[@]}" \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "${build_dir}" -j"$(nproc)" --target "${build_target}"
    runner="${build_dir}/${build_target}"

elif [[ "${os}" == "baremetal" ]]; then
    build_target="executor_runner_baremetal"
    qemu_default="qemu-system-${arch_long}"
    # Standalone build (mirrors examples/arm/executor_runner/standalone)
    cmake -S "${et_root_dir}/examples/riscv/baremetal" -B "${build_dir}" --fresh \
        -DCMAKE_TOOLCHAIN_FILE=${et_root_dir}/examples/riscv/${arch_long}-unknown-elf-toolchain.cmake \
        -DEXECUTORCH_BUILD_PRESET_FILE=${et_root_dir}/tools/cmake/preset/riscv_baremetal.cmake \
        -DEXECUTORCH_ROOT="${et_root_dir}" \
        -DRISCV_BAREMETAL_PTE="${bpte_path}" \
        "${cmake_extra_args[@]}" \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "${build_dir}" -j"$(nproc)" --target "${build_target}"
    runner="${build_dir}/${build_target}.elf"

else
    echo "Unknown os: ${os}" >&2
    usage
    exit 1
fi

qemu="${qemu_override:-${qemu_default}}"
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

log_file="${build_dir}/run.log"
rm -f "${log_file}"

# Compose the QEMU -cpu value once: ${arch} alone, or ${arch},${ext} when an
# extension list was supplied. qemu-user reads $QEMU_CPU; qemu-system takes
# -cpu on the command line.
qemu_cpu="${arch}"
if [[ -n "${qemu_cpu_ext}" ]]; then
    qemu_cpu="${arch},${qemu_cpu_ext}"
fi
echo "[run.sh] qemu -cpu = ${qemu_cpu}"

if [[ "${os}" == "linux" ]]; then
    # QEMU_LD_PREFIX points qemu-user at the cross sysroot so the dynamic
    # linker (ld-linux-riscv*) referenced in the ELF resolves.
    if [[ "${arch}" == "rv64" ]]; then
        export QEMU_LD_PREFIX="${QEMU_LD_PREFIX:-/usr/riscv64-linux-gnu}"
    else
        export QEMU_LD_PREFIX="${QEMU_LD_PREFIX:-/usr/riscv32-linux-gnu}"
    fi
    export QEMU_CPU="${qemu_cpu}"

    runner_extra_args=()
    if ${quantize}; then
        runner_extra_args+=(--bundleio_rtol=0.1 --bundleio_atol=0.25)
    fi
    etdump_path=""
    if ${verbose_xnnpack}; then
        etdump_path="${build_dir}/run.etdump"
        rm -f "${etdump_path}"
        runner_extra_args+=(--etdump_path="${etdump_path}")
    fi

    set +e
    timeout --signal=KILL "${qemu_timeout}" "${qemu}" "${runner}" \
        --model_path="${bpte_path}" \
        "${runner_extra_args[@]}" \
      |& tee "${log_file}"
    qemu_status=${PIPESTATUS[0]}
    set -e

    if [[ -n "${etdump_path}" && -f "${etdump_path}" ]]; then
        python "${script_dir}/etdump_summary.py" "${etdump_path}" \
            --run-log "${log_file}" \
            --json "${etdump_path}.json" || true
    fi

elif [[ "${os}" == "baremetal" ]]; then
    # qemu-system -machine virt boots at 0x80000000; -bios none skips OpenSBI;
    # semihosting target=native routes SYS_WRITE0/SYS_EXIT to host stdio.
    # For deeper debugging, add: -accel tcg,one-insn-per-tb=on -d in_asm,nochain
    #                            -D <trace.log>
    set +e
    timeout --signal=KILL "${qemu_timeout}" "${qemu}" \
        -machine virt -cpu "${qemu_cpu}" -m 512M -nographic -bios none \
        -semihosting-config enable=on,target=native \
        -kernel "${runner}" \
      |& tee "${log_file}"
    qemu_status=${PIPESTATUS[0]}
    set -e

else
    echo "Unknown os: ${os}" >&2
    usage
    exit 1
fi

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
