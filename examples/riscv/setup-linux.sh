#!/usr/bin/env bash
# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install host tooling needed for the RISC-V Phase 1.0 smoke test:
# - gcc/g++/binutils for riscv64-linux-gnu (cross-compiler + sysroot)
# - qemu-user-static (qemu-riscv64 user-mode emulator)

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

if ! command -v apt-get >/dev/null 2>&1; then
    echo "[$(basename "$0")] this setup script targets Debian/Ubuntu (apt-get not found)" >&2
    exit 1
fi

SUDO=""
if [[ $EUID -ne 0 ]]; then
    SUDO="sudo"
fi

source /etc/os-release

GCC_VERSION=""
if [[ "${VERSION_ID:-}" == "24.04" ]]; then
    GCC_VERSION="14"
fi

${SUDO} apt-get update
${SUDO} apt-get install -y --no-install-recommends \
    build-essential \
    gcc${GCC_VERSION:+-${GCC_VERSION}}-riscv64-linux-gnu \
    g++${GCC_VERSION:+-${GCC_VERSION}}-riscv64-linux-gnu \
    binutils-riscv64-linux-gnu \
    libc6-riscv64-cross \
    libc6-dev-riscv64-cross \
    cmake \
    file \
    ca-certificates \
    qemu-user-static \
    libglib2.0-0t64 \
    libxcb1 \
    libgl1

if [[ -n "${GCC_VERSION+x}" ]]; then
    ${SUDO} update-alternatives --install /usr/bin/riscv64-linux-gnu-gcc riscv64-linux-gnu-gcc /usr/bin/riscv64-linux-gnu-gcc${GCC_VERSION:+-${GCC_VERSION}} 100
    ${SUDO} update-alternatives --install /usr/bin/riscv64-linux-gnu-g++ riscv64-linux-gnu-g++ /usr/bin/riscv64-linux-gnu-g++${GCC_VERSION:+-${GCC_VERSION}} 100
fi

riscv64-linux-gnu-gcc --version | head -n1
qemu-riscv64 --version | head -n1

# Some python packages also need to be installed
pip install -r "${script_dir}/requirements.txt"
