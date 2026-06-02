#!/usr/bin/env bash
# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Local mirror of riscv64.yml's matrix using two docker containers:
#
#   - executorch-riscv-linux (ubuntu:24.04 + gcc-14).
#   - executorch-riscv-baremetal (ubuntu:26.04 + gcc-15).
#     26.04 is the only release shipping libstdc++-riscv64-unknown-elf-picolibc.
#
# Usage:
#   examples/riscv/test-matrix.sh                    # full sweep
#   examples/riscv/test-matrix.sh --model=mv2        # one model, all configs
#   examples/riscv/test-matrix.sh --os=baremetal     # one OS
#   examples/riscv/test-matrix.sh --quantize-only    # skip the no-q half
#   examples/riscv/test-matrix.sh --setup-only       # bootstrap containers, don't run
#
# Re-runs are cheap when the per-cell build dirs survive (set --keep-build).

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
et_root_dir=$(realpath "${script_dir}/../..")

model_filter=""
os_filter=""
arch_filter=""
variant_filter=""
backend_filter=""
quantize_filter=""
setup_only=false
keep_build=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]
Options:
  --model=<NAME>     Only run cells for this model
  --os=<linux|baremetal>
  --arch=<rv64|rv32>
  --backend=<portable|xnnpack>
  --variant=<scalar|rvv128|rvv256|rvv512>
  --quantize=<yes,no>
  --setup-only       Make sure both containers are ready, then exit
  --keep-build       Reuse riscv_test/<cell> dirs instead of starting fresh
  -h, --help
EOF
}

for arg in "$@"; do
    case $arg in
        --model=*)     model_filter="${arg#*=}"    ;;
        --os=*)        os_filter="${arg#*=}"       ;;
        --arch=*)      arch_filter="${arg#*=}"     ;;
        --backend=*)   backend_filter="${arg#*=}"  ;;
        --variant=*)   variant_filter="${arg#*=}"  ;;
        --quantize=*)  quantize_filter="${arg#*=}" ;;
        --setup-only)  setup_only=true             ;;
        --keep-build)  keep_build=true             ;;
        -h|--help)     usage; exit 0               ;;
        *)             echo "Unknown: $arg" >&2; usage; exit 1 ;;
    esac
done

# Container names + image tags match what the CI workflow consumes.
LINUX_CTR=executorch-riscv-linux
BAREMETAL_CTR=executorch-riscv-baremetal

MODELS="add mv2 resnet18 mobilebert llama2 yolo26"
BACKENDS="portable xnnpack"

# qemu-cpu-ext sweeps; keep parity with the JSON arrays in riscv64.yml.
SCALAR_EXT="v=false"
RVV128_EXT="v=true,vext_spec=v1.0,vlen=128"
RVV256_EXT="v=true,vext_spec=v1.0,vlen=256"
RVV512_EXT="v=true,vext_spec=v1.0,vlen=512"

# Check if a cell combination should be excluded (matching riscv64.yml excludes)
should_exclude() {
    local os=$1 arch=$2 backend=$3 variant=$4 model=$5 quantize=$6

    # Disable quantization testing with Portable Kernels
    if [[ "${backend}" == "portable" && "${quantize}" == "true" ]]; then
        return 0
    fi
    # XNNPACK needs pthreads + dynamic loading (no baremetal)
    if [[ "${backend}" == "xnnpack" && "${os}" == "baremetal" ]]; then
        return 0
    fi
    # XNNPACK needs RVV
    if [[ "${backend}" == "xnnpack" && "${variant}" == "scalar" ]]; then
        return 0
    fi
    # No quantization recipe for Yolo26
    if [[ "${model}" == "yolo26" && "${quantize}" == "true" ]]; then
        return 0
    fi
    # No riscv32-linux-gnu cross is packaged on Ubuntu
    if [[ "${os}" == "linux" && "${arch}" == "rv32" ]]; then
        return 0
    fi

    return 1
}

# ---- container bootstrap (idempotent) -------------------------------------

ensure_linux() {
    if ! docker ps -a --format '{{.Names}}' | grep -qx "${LINUX_CTR}"; then
        echo "[matrix] starting ${LINUX_CTR} (ubuntu:24.04)"
        docker run -d --name "${LINUX_CTR}" \
            -e DEBIAN_FRONTEND=noninteractive \
            -v "${et_root_dir}":/executorch -w /executorch \
            ubuntu:24.04 sleep infinity >/dev/null
    fi
    docker start "${LINUX_CTR}" >/dev/null
    if ! docker exec "${LINUX_CTR}" test -d /executorch/.venv-docker-linux; then
        echo "[matrix] bootstrapping ${LINUX_CTR} (this takes a few minutes)"
        docker exec "${LINUX_CTR}" bash -eu -c '
            set -e
            apt-get update -qq && apt-get install -y -qq --no-install-recommends \
                python3 python3-pip ca-certificates sudo
            python3 -m pip install --break-system-packages --quiet uv
            uv python install 3.10
            cd /executorch
            uv venv --python 3.10 --seed .venv-docker-linux
        '
    fi
    docker exec "${LINUX_CTR}" bash -eu -c '
        set -e
        cd /executorch
        source .venv-docker-linux/bin/activate
        pip install --upgrade pip
        pip install executorch
        bash examples/riscv/setup-linux.sh
    '
}

ensure_baremetal() {
    if ! docker ps -a --format '{{.Names}}' | grep -qx "${BAREMETAL_CTR}"; then
        echo "[matrix] starting ${BAREMETAL_CTR} (ubuntu:26.04)"
        docker run -d --name "${BAREMETAL_CTR}" \
            -e DEBIAN_FRONTEND=noninteractive \
            -v "${et_root_dir}":/executorch -w /executorch \
            ubuntu:26.04 sleep infinity >/dev/null
    fi
    docker start "${BAREMETAL_CTR}" >/dev/null
    if ! docker exec "${BAREMETAL_CTR}" test -d /executorch/.venv-docker-baremetal; then
        echo "[matrix] bootstrapping ${BAREMETAL_CTR} (this takes a few minutes)"
        docker exec "${BAREMETAL_CTR}" bash -eu -c '
            set -e
            apt-get update -qq && apt-get install -y -qq --no-install-recommends \
                python3 python3-pip ca-certificates sudo
            python3 -m pip install --break-system-packages --quiet uv
            uv python install 3.10
            cd /executorch
            uv venv --python 3.10 --seed .venv-docker-baremetal
        '
    fi
    docker exec "${BAREMETAL_CTR}" bash -eu -c '
        set -e
        cd /executorch
        source .venv-docker-baremetal/bin/activate
        pip install --upgrade pip
        pip install executorch
        bash examples/riscv/setup-baremetal.sh
    '
}

ensure_linux
ensure_baremetal
if ${setup_only}; then exit 0; fi

# ---- one cell --------------------------------------------------------------

# Args: ctr venv os arch backend variant ext model quantize_flag
run_cell() {
    local ctr=$1 venv=$2 os=$3 arch=$4 backend=$5 variant=$6 ext=$7 model=$8 q=$9
    local cell="${model}${q:++q}-${backend}/${os}-${arch}"
    local model_q="${model}${q:+-q}"
    local variant_slug="${ext//,/_}"; variant_slug="${variant_slug//=/_}"; variant_slug="${variant_slug:-base}"
    local build_dir="/executorch/riscv_test/${model_q}/${backend}/${os}-${arch}-${variant_slug}"
    if ! ${keep_build}; then
        docker exec "${ctr}" rm -rf "${build_dir}"
    fi
    if docker exec "${ctr}" bash -lc "
            cd /executorch && source ${venv}/bin/activate &&
            timeout 1800 bash -eu examples/riscv/run.sh \
              --model=${model} ${q} --backend=${backend} \
              --os=${os} --arch=${arch} \
              --qemu-cpu-ext='${ext}' \
              --build-dir=${build_dir} --timeout=900
        "; then
        echo "  PASS  ${cell}"
        return 0
    else
        echo "  FAIL  ${cell}"
        return 1
    fi
}

# ---- iterate ---------------------------------------------------------------

passed=0; total=0
for m in ${MODELS}; do
for backend in ${BACKENDS}; do
for os_arch in "linux:rv64" "baremetal:rv64" "baremetal:rv32"; do
for variant_lbl in "scalar:${SCALAR_EXT}" "rvv128:${RVV128_EXT}" "rvv256:${RVV256_EXT}" "rvv512:${RVV512_EXT}"; do
    os="${os_arch%%:*}"; arch="${os_arch##*:}"; variant="${variant_lbl%%:*}"; ext="${variant_lbl#*:}"

    if [[ -n "${model_filter}" && "${m}" != "${model_filter}" ]]; then continue; fi
    if [[ -n "${backend_filter}" && "${backend}" != "${backend_filter}" ]]; then continue; fi
    if [[ -n "${os_filter}" && "${os}" != "${os_filter}" ]]; then continue; fi
    if [[ -n "${arch_filter}" && "${arch}" != "${arch_filter}" ]]; then continue; fi
    if [[ -n "${variant_filter}" && "${variant}" != "${variant_filter}" ]]; then continue; fi

    if [[ "${os}" == "linux" ]]; then ctr="${LINUX_CTR}"; venv=/executorch/.venv-docker-linux;
    else                              ctr="${BAREMETAL_CTR}"; venv=/executorch/.venv-docker-baremetal; fi

    if [[ -z "${quantize_filter}" || "${quantize_filter}" = "no" ]]; then
        if should_exclude "${os}" "${arch}" "${backend}" "${variant}" "${m}" "false"; then continue; fi
        total=$((total+1))
        run_cell "${ctr}" "${venv}" "${os}" "${arch}" "${backend}" "${variant}" "${ext}" "${m}" "" \
            && passed=$((passed+1)) || exit 1
    fi
    if [[ -z "${quantize_filter}" || "${quantize_filter}" = "yes" ]]; then
        if should_exclude "${os}" "${arch}" "${backend}" "${variant}" "${m}" "true"; then continue; fi
        total=$((total+1))
        run_cell "${ctr}" "${venv}" "${os}" "${arch}" "${backend}" "${variant}" "${ext}" "${m}" "--quantize" \
            && passed=$((passed+1)) || exit 1
    fi
done
done
done
done

echo ""
echo "===== ${passed}/${total} cells passed ====="
test "${passed}" -eq "${total}"
