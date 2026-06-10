#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and install the licensed Cadence Xtensa toolchain + core config for
# a given backend, then export the environment that
# backends/cadence/cadence.cmake and xt-run need.
#
# The artifacts (host tools, the core tarball, and the bundled license) cannot
# be hosted publicly, so they are fetched at runtime from an auth-gated object
# store. The store location is provided by the caller via XTENSA_S3_BUCKET (set
# from a CI variable); credentials are obtained out of band before this runs.
#
# Usage:
#   XTENSA_S3_BUCKET=<bucket> .ci/scripts/setup-xtensa-tools.sh <backend>
#     backend = hifi4 | vision | fusion_g3
#
# In GitHub Actions this appends the toolchain env to $GITHUB_ENV so later
# steps inherit it. Run locally to populate a workspace for manual builds.
#
# Modeled on .ci/scripts/setup-arm-baremetal-tools.sh.

set -euo pipefail

BACKEND="${1:-}"
if [[ -z "${BACKEND}" ]]; then
  echo "ERROR: usage: XTENSA_S3_BUCKET=<bucket> $0 <hifi4|vision|fusion_g3>" >&2
  exit 1
fi

S3_BUCKET="${XTENSA_S3_BUCKET:-}"
if [[ -z "${S3_BUCKET}" ]]; then
  echo "ERROR: XTENSA_S3_BUCKET is not set (provide it from a CI variable)." >&2
  exit 1
fi
S3_TOOLCHAIN_PREFIX="${XTENSA_S3_TOOLCHAIN_PREFIX:-toolchains}"
S3_CORE_PREFIX="${XTENSA_S3_CORE_PREFIX:-cores}"

# Per-backend mapping: core tarball, toolchain tarball, core name, OPT flag.
# The toolchain's clang major must match the core's codegen plugin:
#   hifi4 / fusion_g3 cores (RI-2022.10, clang 10) -> RI-2022.9 host tools
#   vision core           (RJ-2025.5,  clang 15)   -> RJ-2025.5 host tools
case "${BACKEND}" in
  hifi4)
    CORE_NAME="hifi4_ss_spfpu_7_et_ci2"
    CORE_TARBALL="hifi4_ss_spfpu_7_et_ci2_linux.tgz"
    TOOLCHAIN_TARBALL="XtensaTools_RI_2022_9_linux.tgz"
    TOOLCHAIN_VER="RI-2022.9-linux"
    OPT_FLAG="EXECUTORCH_NNLIB_OPT"
    ;;
  fusion_g3)
    CORE_NAME="XRC_FuG3_TYP_SPVFPU_et_c2"
    CORE_TARBALL="XRC_FuG3_TYP_SPVFPU_et_c2_linux.tgz"
    TOOLCHAIN_TARBALL="XtensaTools_RI_2022_9_linux.tgz"
    TOOLCHAIN_VER="RI-2022.9-linux"
    OPT_FLAG="EXECUTORCH_FUSION_G3_OPT"
    ;;
  vision)
    CORE_NAME="XRC_Vision_110_AO_et_ci2"
    CORE_TARBALL="XRC_Vision_110_AO_et_ci2_linux.tgz"
    TOOLCHAIN_TARBALL="XtensaTools_RJ_2025_5_linux.tgz"
    TOOLCHAIN_VER="RJ-2025.5-linux"
    OPT_FLAG="EXECUTORCH_VISION_OPT"
    ;;
  *)
    echo "ERROR: unknown backend '${BACKEND}' (expected hifi4|vision|fusion_g3)" >&2
    exit 1
    ;;
esac

XTENSA_ROOT="${XTENSA_ROOT:-/tmp/xtensa}"
TOOLS_ROOT="${XTENSA_ROOT}/tools"     # contains <ver>-linux/XtensaTools
CORES_ROOT="${XTENSA_ROOT}/cores"     # contains <corever>-linux/<core>
REGISTRY_ROOT="${XTENSA_ROOT}/registry/${CORE_NAME}"
DL_DIR="${XTENSA_ROOT}/download"
mkdir -p "${TOOLS_ROOT}" "${CORES_ROOT}" "${REGISTRY_ROOT}" "${DL_DIR}"

s3_get() {
  # $1 = s3 key, $2 = local dest
  local key="$1" dest="$2"
  echo "Downloading s3://${S3_BUCKET}/${key} ..."
  aws s3 cp "s3://${S3_BUCKET}/${key}" "${dest}" --only-show-errors
}

# 1. Toolchain (host xt-clang/xt-run). Skip re-extract if already present.
if [[ ! -d "${TOOLS_ROOT}/${TOOLCHAIN_VER}/XtensaTools" ]]; then
  s3_get "${S3_TOOLCHAIN_PREFIX}/${TOOLCHAIN_TARBALL}" "${DL_DIR}/${TOOLCHAIN_TARBALL}"
  tar xzf "${DL_DIR}/${TOOLCHAIN_TARBALL}" -C "${TOOLS_ROOT}"
fi
TOOLCHAIN_HOME="${TOOLS_ROOT}/${TOOLCHAIN_VER}/XtensaTools"
if [[ ! -x "${TOOLCHAIN_HOME}/bin/xt-clang" ]]; then
  echo "ERROR: xt-clang not found at ${TOOLCHAIN_HOME}/bin after extract" >&2
  exit 1
fi

# 2. Core config (ISA libs, params, examples, bundled magic-key license).
s3_get "${S3_CORE_PREFIX}/${CORE_TARBALL}" "${DL_DIR}/${CORE_TARBALL}"
tar xzf "${DL_DIR}/${CORE_TARBALL}" -C "${CORES_ROOT}"
CORE_DIR=$(echo "${CORES_ROOT}"/*/"${CORE_NAME}")
if [[ ! -d "${CORE_DIR}" ]]; then
  echo "ERROR: core dir for ${CORE_NAME} not found under ${CORES_ROOT}" >&2
  exit 1
fi

# 3. Build a local Xtensa core registry with the XPG-internal build paths in
#    the params file rewritten to our extracted toolchain + core locations.
#    The vendor ships params referencing /././home/xpgcust/... build paths.
PARAMS_SRC="${CORE_DIR}/config/${CORE_NAME}-params"
TOOLS_PFX=$(sed -n 's/^install-prefix = //p' "${PARAMS_SRC}" | head -1)
TOOLSUB_PFX=$(sed -n 's/^xtensa-tools = //p' "${PARAMS_SRC}" | head -1)
CFG_PFX=$(sed -n 's/^config-prefix = //p' "${PARAMS_SRC}" | head -1)
sed \
  -e "s|${TOOLS_PFX}|${TOOLCHAIN_HOME}|g" \
  -e "s|${TOOLSUB_PFX}|${TOOLCHAIN_HOME}/Tools|g" \
  -e "s|${CFG_PFX}|${CORE_DIR}|g" \
  "${PARAMS_SRC}" > "${REGISTRY_ROOT}/${CORE_NAME}-params"
ln -sf "${CORE_NAME}-params" "${REGISTRY_ROOT}/default-params"

LICENSE_FILE="${CORE_DIR}/misc/license.dat"

# 4. Export environment. cadence.cmake reads XTENSA_TOOLCHAIN/TOOLCHAIN_VER;
#    xt-clang/xt-run read XTENSA_SYSTEM/XTENSA_CORE; xtensad reads
#    XTENSAD_LICENSE_FILE (the bundled uncounted magic key, no server needed).
emit() {
  # Export into the current shell (so callers that `source` this script get the
  # vars) and append to $GITHUB_ENV (so later workflow steps inherit them too).
  echo "$1"
  export "${1?}"
  if [[ -n "${GITHUB_ENV:-}" ]]; then echo "$1" >> "${GITHUB_ENV}"; fi
}
echo "=== Xtensa env for backend '${BACKEND}' (core ${CORE_NAME}) ==="
emit "XTENSA_TOOLCHAIN=${TOOLS_ROOT}"
emit "TOOLCHAIN_VER=${TOOLCHAIN_VER}"
emit "XTENSA_SYSTEM=${REGISTRY_ROOT}"
emit "XTENSA_CORE=${CORE_NAME}"
emit "XTENSAD_LICENSE_FILE=${LICENSE_FILE}"
emit "CADENCE_OPT_FLAG=${OPT_FLAG}"
if [[ -n "${GITHUB_PATH:-}" ]]; then
  echo "${TOOLCHAIN_HOME}/bin" >> "${GITHUB_PATH}"
fi
export PATH="${TOOLCHAIN_HOME}/bin:${PATH}"

echo "=== sanity ==="
xt-clang --version 2>&1 | head -1
xt-run --show-config=cores 2>&1 | sed -n '/available/,/registry/p' | head -6
echo "Xtensa toolchain ready for ${BACKEND}."
