#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build the CMSIS Pack and exercise it with a consumer project that enables
# EVERY operator component, so a single cbuild links the whole operator surface
# (build/link coverage). Phase 2 will add per-op .pte execution on the FVP.
#
# Mirrors smoke/run.sh: csolution + cbuild against an in-tree project, run
# inside the AVH-MLOps Docker image. The only addition is generating the
# all-ops cproject (every operator component) before the build.
#
# Prerequisites / environment overrides: see smoke/run.sh.

set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ET_ROOT="$(cd "${TEST_DIR}/../../../../.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${ET_ROOT}/arm_test/cmake-out}"
OUTPUT_DIR="${OUTPUT_DIR:-${ET_ROOT}/arm_test/cmsis-pack-output}"
BASE_VER="$(sed 's/a0$//' "${ET_ROOT}/version.txt")"
PACK_VERSION="${PACK_VERSION:-${BASE_VER}-stage}"
DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/arm-software/avh-mlops/arm-mlops-docker-licensed-community:latest-arm64}"

echo "=== Build pack ${PACK_VERSION} ==="
"${ET_ROOT}/backends/arm/cmsis_pack/scripts/build_pack.sh" \
    --executorch-root "${ET_ROOT}" \
    --build-dir       "${BUILD_DIR}" \
    --version         "${PACK_VERSION}" \
    --output-dir      "${OUTPUT_DIR}"

PACK_BASENAME="PyTorch.ExecuTorch.${PACK_VERSION}.pack"
PACK_FILE="${OUTPUT_DIR}/${PACK_BASENAME}"
[[ -f "${PACK_FILE}" ]] || { echo "Pack file not found: ${PACK_FILE}"; exit 1; }

echo
echo "=== Validate pack structure ==="
python3 "${TEST_DIR}/../validate_pack.py" "${PACK_FILE}"

# Target selection: default is Corstone-315 (SSE-315, Cortex-M85 host, no NPU),
# which runs every non-delegated op on the CPU. ETHOS_U=1 selects the
# Corstone-320 (SSE-320) + Ethos-U85 variant: every op is exported through Vela;
# whatever partitions runs on the NPU, the rest falls back to the host core.
if [[ "${ETHOS_U:-0}" == "1" ]]; then
    TARGET_TYPE="SSE-320"
    GEN_CPROJECT_ARGS=(--ethos-u)
    FVP="${FVP_BIN:-FVP_Corstone_SSE-320}"
else
    TARGET_TYPE="SSE-315"
    GEN_CPROJECT_ARGS=()
    FVP="${FVP_BIN:-FVP_Corstone_SSE-315}"
fi
CONTEXT="all_ops.Debug+${TARGET_TYPE}"

echo
echo "=== Generate all-ops project (every operator component) [${TARGET_TYPE}] ==="
python3 "${TEST_DIR}/gen_cproject.py" \
    --source-dir "${ET_ROOT}" \
    --output     "${TEST_DIR}/all_ops.cproject.yml" \
    "${GEN_CPROJECT_ARGS[@]}"

# -------------------------------------------------------------------------
# Embed the per-op test models into the image (.incbin in .rodata) so the
# firmware self-tests every op in one boot -- no semihosting filesystem access
# -- and reports a per-op DWT cycle count. Models must exist before the build.
#
# RUN_FVP=1 exports a fresh self-contained .pte for every op (needs the torch
# export env: examples/arm/setup.sh, incl. cmsis_nn for the Cortex-M models).
# Each .pte embeds its own test data as constant methods -- inputs, expected
# outputs, tolerances -- so there are no side-band .bin files; the manifest
# only maps ops to .pte names.
# Without it, gen_embedded_models.py emits an empty stub, so the image still
# links and boots (build/link coverage only).
# -------------------------------------------------------------------------
# ETHOS_U=1 lowers every op through Vela (delegated to the NPU where it
# partitions, host-core fallback otherwise); the default exports the CPU recipes.
EXPORT_ARGS=()
[[ "${ETHOS_U:-0}" == "1" ]] && EXPORT_ARGS=(--ethos-u)

MODELS_DIR="${TEST_DIR}/models"
if [[ "${RUN_FVP:-0}" == "1" ]]; then
    echo
    echo "=== Export per-op self-contained .pte files [${TARGET_TYPE}] ==="
    python3 "${TEST_DIR}/generate_test_models.py" \
        --source-dir "${ET_ROOT}" \
        --output-dir "${MODELS_DIR}" \
        --continue-on-error \
        "${EXPORT_ARGS[@]}"
fi
echo
echo "=== Embed models into firmware ==="
python3 "${TEST_DIR}/gen_embedded_models.py" \
    --models-dir "${MODELS_DIR}" \
    --output-dir "${TEST_DIR}"

echo
echo "=== Consumer build (${DOCKER_IMAGE}) ==="
docker run --rm \
    -e PACK_BASENAME="${PACK_BASENAME}" \
    -e CONTEXT="${CONTEXT}" \
    -v "${TEST_DIR}:/workspace" \
    -v "${OUTPUT_DIR}:/pack-output:ro" \
    "${DOCKER_IMAGE}" \
    bash -lc '
set -euo pipefail

# Docker Desktop for Mac bind mounts fail the csolution --update-rte device-file
# copy (gRPC-FUSE / std::filesystem). Build on a container-local path, then copy
# the outputs back to the bind-mounted project so the host FVP step finds the
# ELF. Stale cbuild locks/outputs are dropped so packs re-resolve cleanly.
rm -rf /tmp/build
cp -r /workspace /tmp/build
cd /tmp/build
rm -rf out tmp RTE *.cbuild-idx.yml *.cbuild-pack.yml *.cbuild-set.yml

# Acquire the toolchain set declared in vcpkg-configuration.json. The pinned
# AVH-MLOps image ships a stale vcpkg registry snapshot (predates cmsis-toolbox
# 2.14.1); refresh it so the pinned toolbox version resolves.
vcpkg x-update-registry --all >/dev/null 2>&1 || true
export Z_VCPKG_POSTSCRIPT="$(mktemp /tmp/vcpkg.XXXXXX.sh)"
vcpkg activate
source "${Z_VCPKG_POSTSCRIPT}"

# Always reinstall the freshly built pack into a clean container-local pack
# root, isolated from any host pack store. --packs pulls ARM::CMSIS-NN (needed
# by the Cortex-M CMSIS-NN ops) from the public index.
export CMSIS_PACK_ROOT=/tmp/cmsis-pack-root
cpackget init https://www.keil.com/pack/index.pidx
cpackget add --agree-embedded-license -F "/pack-output/${PACK_BASENAME}"

cbuild all_ops.csolution.yml --packs --update-rte --context "${CONTEXT}"

# Publish build outputs back to the bind-mounted project for the host FVP step.
rm -rf /workspace/out
cp -r /tmp/build/out /workspace/out
'
echo
echo "=== Build/link coverage PASS (every operator component linked) ==="

# -------------------------------------------------------------------------
# Execution + per-op cycles (RUN_FVP=1): the image embeds every model (above),
# so a SINGLE FVP boot self-tests all ops and prints a per-op DWT cycle table
# plus a final "Test_result: SUMMARY <pass>/<total> PASS" line.
#
# Requires the Corstone FVP on PATH (examples/arm/setup.sh --enable-fvps).
# -------------------------------------------------------------------------
if [[ "${RUN_FVP:-0}" != "1" ]]; then
    echo "(set RUN_FVP=1 to export + embed models and run on the FVP)"
    exit 0
fi

# FVP already selected above (SSE-315 default, SSE-320 for ETHOS_U=1).
ELF="$(find "${TEST_DIR}/out" -name '*.elf' | head -1)"
[[ -n "${ELF}" ]] || { echo "No firmware ELF found under out/"; exit 1; }

# The all-ops image is linked into DDR @0x90000000 (SSE315_large.ld) because it
# is far too large for the on-chip ITCM/SRAM. Point the CPU's reset vector table
# at DDR so reset fetches SP/PC from there. Both SSE-315 and SSE-320 are MPS4
# subsystems, so the config namespace is mps4_board.subsystem.* for both.
BOOT_BASE=0x90000000

# Ethos-U variant: the FVP NPU MAC count must match the AOT compile target
# (ethos-u85-256 -> num_macs=256), or the Vela command stream is rejected at
# runtime. STDIO is semihosting on both (the BSP's UART support is limited).
ETHOS_U_ARGS=()
if [[ "${ETHOS_U:-0}" == "1" ]]; then
    ETHOS_U_ARGS=(
        -C mps4_board.subsystem.ethosu.num_macs=256
        -C "mps4_board.subsystem.ethosu.extra_args=--fast"
    )
fi

echo
echo "=== Run embedded all-ops image on ${FVP} ==="
log="$("${FVP}" \
    -C mps4_board.subsystem.cpu0.semihosting-enable=1 \
    -C "mps4_board.subsystem.cpu0.INITSVTOR=${BOOT_BASE}" \
    -C "mps4_board.subsystem.iotss3_systemcontrol.INITSVTOR_RST=${BOOT_BASE}" \
    -C mps4_board.visualisation.disable-visualisation=1 \
    -C mps4_board.telnetterminal0.start_telnet=0 \
    -C mps4_board.uart0.out_file=- \
    -C mps4_board.uart0.shutdown_on_eot=1 \
    "${ETHOS_U_ARGS[@]}" \
    -a "${ELF}" --timelimit 600 2>&1 || true)"
echo "${log}"

# The per-op cycle table + SUMMARY are the artifact. The FVP's own exit status
# reports the simulator stop reason (the benign CPU timelimit) rather than the
# test outcome, hence the `|| true` above; the SUMMARY line is what decides.
# Two distinct failures: the image never reached SUMMARY (crash / hang / boot
# failure), or it ran to completion with one or more operators failing.
summary="$(grep -oE "Test_result: SUMMARY [0-9]+/[0-9]+" <<<"${log}" | tail -1)"
if [[ -z "${summary}" ]]; then
    echo "=== Execution FAILED: image did not reach SUMMARY ==="
    exit 1
fi
counts="${summary##* }"
passed="${counts%/*}"
total="${counts#*/}"
if (( total == 0 )); then
    echo "=== Execution FAILED: no operators were tested ==="
    exit 1
fi
if [[ "${passed}" != "${total}" ]]; then
    echo "=== Execution FAILED: ${passed}/${total} operators passed ==="
    exit 1
fi
echo
echo "=== PASS (build/link + execution): ${passed}/${total} ==="
