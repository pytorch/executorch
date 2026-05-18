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

bash "${et_root_dir}/examples/riscv/setup.sh"
bash "${et_root_dir}/examples/riscv/run.sh"
