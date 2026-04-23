#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2026 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# End-to-end test for Cortex-M backend: export a model via aot_arm_compiler
# with cortex-m55+int8 target, then run the .bpte on Corstone-300 FVP.
#
# Usage: bash .ci/scripts/test_cortex_m_e2e.sh <model_name>
# Example: bash .ci/scripts/test_cortex_m_e2e.sh mv2

set -eux

MODEL=$1
mkdir -p "./cortex_m_e2e/${MODEL}"
WORK_DIR=$(realpath "./cortex_m_e2e/${MODEL}")

echo "=== Exporting ${MODEL} with cortex-m55+int8 ==="
python -m backends.arm.scripts.aot_arm_compiler \
    -m "${MODEL}" \
    --target=cortex-m55+int8 \
    --quantize \
    --bundleio \
    --intermediates="${WORK_DIR}/intermediates" \
    --output="${WORK_DIR}/${MODEL}.bpte"

BPTE="${WORK_DIR}/${MODEL}.bpte"
test -f "${BPTE}" || { echo "FAIL: ${BPTE} not produced"; exit 1; }
echo "=== Exported ${BPTE} ($(stat --printf='%s' "${BPTE}") bytes) ==="

ELF="arm_test/arm_semihosting_executor_runner_corstone-300/arm_executor_runner"
test -f "${ELF}" || { echo "FAIL: executor runner not found at ${ELF}"; exit 1; }

LOG_FILE=$(mktemp)

# Create a tiny dummy input file — the runner requires -i but BundleIO
# ignores it and uses the embedded test inputs instead.
dd if=/dev/zero of="${WORK_DIR}/dummy.bin" bs=4 count=1 2>/dev/null

echo "=== Running ${MODEL} on Corstone-300 FVP ==="
FVP_Corstone_SSE-300_Ethos-U55 \
    -C ethosu.num_macs=128 \
    -C mps3_board.visualisation.disable-visualisation=1 \
    -C mps3_board.telnetterminal0.start_telnet=0 \
    -C mps3_board.uart0.out_file='-' \
    -C mps3_board.uart0.shutdown_on_eot=1 \
    -C cpu0.semihosting-enable=1 \
    -C cpu0.semihosting-stack_base=0 \
    -C cpu0.semihosting-heap_limit=0 \
    -C "cpu0.semihosting-cwd=${WORK_DIR}" \
    -C "ethosu.extra_args='--fast'" \
    -C "cpu0.semihosting-cmd_line='executor_runner -m ${MODEL}.bpte -i dummy.bin -o out'" \
    -a "${ELF}" \
    --timelimit 300 2>&1 | tee "${LOG_FILE}" || true

echo "=== Checking FVP output ==="

if grep -q "Test_result: PASS" "${LOG_FILE}"; then
    echo "=== SUCCESS: ${MODEL} e2e BundleIO test PASSED on FVP ==="
    rm "${LOG_FILE}"
    exit 0
fi

if grep -q "Test_result: FAIL" "${LOG_FILE}"; then
    echo "FAIL: ${MODEL} BundleIO output mismatch"
    rm "${LOG_FILE}"
    exit 1
fi

if grep -qE "(^[EF][: ].*$)|(^.*Hard fault.*$)|(^.*Assertion.*$)" "${LOG_FILE}"; then
    echo "FAIL: ${MODEL} FVP run hit a fatal error"
    rm "${LOG_FILE}"
    exit 1
fi

echo "FAIL: ${MODEL} no BundleIO test result found in FVP output"
rm "${LOG_FILE}"
exit 1
