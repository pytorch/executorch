#!/usr/bin/env bash
# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Host tooling for the RISC-V smoke tests. Targets Ubuntu 26.04: that's where
# libstdc++-riscv64-unknown-elf-picolibc was first packaged, and the baremetal
# build chain needs C++ stdlib headers paired with picolibc.

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

${SUDO} apt-get update
${SUDO} apt-get install -y --no-install-recommends \
    build-essential \
    gcc-riscv64-linux-gnu \
    g++-riscv64-linux-gnu \
    binutils-riscv64-linux-gnu \
    libc6-riscv64-cross \
    libc6-dev-riscv64-cross \
    gcc-riscv64-unknown-elf \
    picolibc-riscv64-unknown-elf \
    libstdc++-riscv64-unknown-elf-picolibc \
    cmake \
    file \
    ca-certificates \
    qemu-user \
    qemu-system-riscv \
    libglib2.0-0t64 \
    libxcb1 \
    libgl1

riscv64-linux-gnu-gcc --version | head -n1
qemu-riscv64 --version | head -n1

# Some python packages also need to be installed
pip install -r "${script_dir}/requirements.txt"
