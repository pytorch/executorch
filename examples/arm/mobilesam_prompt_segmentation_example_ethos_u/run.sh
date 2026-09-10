#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

if (($#)); then
    echo "This example has one supported configuration; run it without arguments."
    exit 2
fi

example_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${example_dir}/../../.." && pwd)
work_dir="${repo_root}/arm_test/mobilesam"
export_dir="${work_dir}/export"
io_dir="${work_dir}/io"
runner_dir="${work_dir}/runner"

cd "${repo_root}"
[[ -f examples/arm/arm-scratch/setup_path.sh ]] || {
    echo "Arm tools are missing. Run ./examples/arm/setup.sh first."
    exit 1
}
source examples/arm/arm-scratch/setup_path.sh
mkdir -p "${io_dir}"

if [[ "$(uname -s)" == "Darwin" ]]; then
    export FVP_MOUNT_DIR="${FVP_MOUNT_DIR:-${repo_root}}"
    export FVP_WORKDIR="${FVP_WORKDIR:-${repo_root}}"
fi

echo "[1/4] Prepare MobileSAM"
python3 "${example_dir}/model_export/prepare_mobilesam.py"

echo "[2/4] Export, quantize, and lower to Ethos-U"
python3 "${example_dir}/model_export/export_mobilesam.py"

echo "[3/4] Build the standard Arm executor runner"
backends/arm/scripts/build_executor_runner.sh \
    --pte="${export_dir}/mobilesam.pte" \
    --target=ethos-u85-256 \
    --output="${runner_dir}" \
    '--extra_build_flags=-DSEMIHOSTING=ON -DET_COMPILED_PTE=ON'

cp "${export_dir}/input.bin" "${io_dir}/input.bin"
rm -f "${io_dir}/output-0.bin"

echo "[4/4] Run on FVP and validate the output"
backends/arm/scripts/run_fvp.sh \
    --elf="${runner_dir}/arm_executor_runner" \
    --target=ethos-u85-256 \
    --timeout=300 \
    --semihosting-cwd="${io_dir}" \
    '--semihosting-cmd-line=executor_runner -i input.bin -o output' \
    --fast | tee "${work_dir}/fvp.log"
python3 "${example_dir}/runtime/visualize_fvp_output.py"

echo "MobileSAM example: PASS"
