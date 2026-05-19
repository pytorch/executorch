#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Consume prebuilt ExecuTorch wheels produced by build_macos_wheels.sh and
# downloaded into ${RUNNER_ARTIFACT_DIR} via the macos_job.yml
# `download-artifact:` input, then run install_executorch.sh against them.
#
# This script:
#   1. Moves the downloaded *.whl files out of ${RUNNER_ARTIFACT_DIR} so a
#      subsequent `upload-artifact:` from the same job does not re-upload
#      them as part of an unrelated artifact (e.g. an exported .pte).
#   2. Invokes install_executorch.sh --prebuilt-wheel-dir <moved-dir>,
#      forwarding any additional flags after the optional --.
#
# Usage:
#   install_executorch_from_wheels.sh [-- <extra install_executorch.sh flags>]
#
# Required environment:
#   RUNNER_ARTIFACT_DIR  (set by pytorch/test-infra macos_job.yml)
#   CONDA_RUN            (optional; used as conda env wrapper if present)
#
# Notes:
#   - Honors EXECUTORCH_BUILD_KERNELS_TORCHAO / TORCHAO_BUILD_EXPERIMENTAL_MPS
#     etc., but those should match the values used at wheel-build time.
#   - The repo root must be the current working directory when invoked.

set -euxo pipefail

if [[ -z "${RUNNER_ARTIFACT_DIR:-}" ]]; then
  echo "ERROR: RUNNER_ARTIFACT_DIR is not set." >&2
  exit 1
fi

WHEEL_DIR="${RUNNER_TEMP:-/tmp}/prebuilt_executorch_wheels"
mkdir -p "${WHEEL_DIR}"

# Move every wheel out of the artifact dir so it isn't re-uploaded.
shopt -s nullglob
WHEELS=( "${RUNNER_ARTIFACT_DIR}"/*.whl )
shopt -u nullglob
if [[ ${#WHEELS[@]} -eq 0 ]]; then
  echo "ERROR: no *.whl files found in ${RUNNER_ARTIFACT_DIR}." >&2
  echo "Did the consumer job set download-artifact correctly?" >&2
  exit 1
fi
mv -v "${WHEELS[@]}" "${WHEEL_DIR}/"

EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--" ]]; then
    shift
  fi
  EXTRA_ARGS=( "$@" )
fi

# Forward to install_executorch.sh. Honor ${CONDA_RUN} if set (matches the
# convention used in metal.yml and friends).
if [[ -n "${CONDA_RUN:-}" ]]; then
  ${CONDA_RUN} ./install_executorch.sh \
    --prebuilt-wheel-dir "${WHEEL_DIR}" \
    "${EXTRA_ARGS[@]}"
else
  ./install_executorch.sh \
    --prebuilt-wheel-dir "${WHEEL_DIR}" \
    "${EXTRA_ARGS[@]}"
fi
