#!/usr/bin/env bash
# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install host tooling needed for the RISC-V Phase 1.0 smoke test:
# - gcc/g++/binutils for riscv64-linux-gnu (cross-compiler + sysroot)
# - qemu-user-static (qemu-riscv64 user-mode emulator)

set -eu

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
    cmake \
    file \
    qemu-user-static

riscv64-linux-gnu-gcc --version | head -n1
qemu-riscv64-static --version | head -n1
