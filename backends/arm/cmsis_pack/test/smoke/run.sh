#!/usr/bin/env bash
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build the CMSIS Pack and exercise it as a real consumer would:
# csolution + cbuild against the in-tree smoke project, run inside the
# AVH-MLOps Docker image. All flags come from the PDSC + the
# cmsis-toolbox cdefault — none are hand-curated here.
#
# Prerequisites:
#   - docker
#   - $BUILD_DIR (default arm_test/cmake-out) populated by
#     backends/arm/scripts/build_executorch.sh; build_pack.sh consumes
#     the FlatBuffers / schema headers from there.
#
# Environment overrides:
#   PACK_VERSION   default: <version.txt without trailing a0>-stage
#   BUILD_DIR      default: <repo>/arm_test/cmake-out
#   OUTPUT_DIR     default: <repo>/arm_test/cmsis-pack-output
#   DOCKER_IMAGE   default: ghcr.io/arm-software/avh-mlops/arm-mlops-docker-licensed-community:latest-arm64

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

echo
echo "=== Consumer build (${DOCKER_IMAGE}) ==="
docker run --rm \
    -v "${TEST_DIR}:/workspace" \
    -v "${OUTPUT_DIR}:/pack-output:ro" \
    "${DOCKER_IMAGE}" \
    bash -lc '
set -euo pipefail
cd /workspace

# Acquire the toolchain set declared in vcpkg-configuration.json
# (cmsis-toolbox + arm-none-eabi-gcc + cmake + ninja).
export Z_VCPKG_POSTSCRIPT="$(mktemp /tmp/vcpkg.XXXXXX.sh)"
vcpkg activate
source "${Z_VCPKG_POSTSCRIPT}"

# Always reinstall the freshly built pack into a clean container-local
# pack root, isolated from any host pack store.
export CMSIS_PACK_ROOT=/tmp/cmsis-pack-root
cpackget init https://www.keil.com/pack/index.pidx
cpackget add --agree-embedded-license -F "/pack-output/'"${PACK_BASENAME}"'"

cbuild smoke.csolution.yml --packs --update-rte --context smoke.Debug+ARMCM55
'
echo
echo "=== PASS ==="
