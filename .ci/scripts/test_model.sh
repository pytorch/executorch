#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/test_model_utils.sh"

MODEL_NAME=$1
if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "Missing model name, exiting..."
  exit 1
fi

BUILD_TOOL=$2
if [[ -z "${BUILD_TOOL:-}" ]]; then
  echo "Missing build tool (require buck2 or cmake), exiting..."
  exit 1
fi

BACKEND=$3
if [[ -z "${BACKEND:-}" ]]; then
  echo "Missing backend (require portable or xnnpack), exiting..."
  exit 1
fi

UPLOAD_DIR=${4:-}

if [[ "${BACKEND}" == "portable" ]]; then
  echo "Testing ${MODEL_NAME} with portable kernels..."
  test_model $MODEL_NAME
elif [[ "${BACKEND}" == "qnn" ]]; then
  echo "Testing ${MODEL_NAME} with qnn..."
  test_model_with_qnn $MODEL_NAME
  if [[ $? -eq 0 ]]; then
    prepare_artifacts_upload $UPLOAD_DIR
  fi
elif [[ "${BACKEND}" == "coreml" ]]; then
  echo "Testing ${MODEL_NAME} with coreml..."
  test_model_with_coreml $MODEL_NAME $BUILD_TOOL
  if [[ $? -eq 0 ]]; then
    prepare_artifacts_upload $UPLOAD_DIR
  fi
elif [[ "${BACKEND}" == "xnnpack" ]]; then
  echo "Testing ${MODEL_NAME} with xnnpack..."
  WITH_QUANTIZATION=true
  WITH_DELEGATION=true
  if [[ "$MODEL_NAME" == "mobilebert" ]]; then
    # TODO(T197452682)
    WITH_QUANTIZATION=false
  fi
  test_model_with_xnnpack "${MODEL_NAME}" "${WITH_QUANTIZATION}" "${WITH_DELEGATION}" "${BUILD_TOOL}"
  if [[ $? -eq 0 ]]; then
    prepare_artifacts_upload $UPLOAD_DIR
  fi
else
  set +e
  if [[ "${BACKEND}" == *"quantization"* ]]; then
    echo "::group::Testing ${MODEL_NAME} with XNNPACK quantization only..."
    test_model_with_xnnpack "${MODEL_NAME}" true false "${BUILD_TOOL}" || Q_ERROR="error"
    echo "::endgroup::"
  fi
  if [[ "${BACKEND}" == *"delegation"* ]]; then
    echo "::group::Testing ${MODEL_NAME} with XNNPACK delegation only..."
    test_model_with_xnnpack "${MODEL_NAME}" false true "${BUILD_TOOL}" || D_ERROR="error"
    echo "::endgroup::"
  fi
  if [[ "${BACKEND}" == *"quantization"* ]] && [[ "${BACKEND}" == *"delegation"* ]]; then
    echo "::group::Testing ${MODEL_NAME} with XNNPACK quantization and delegation..."
    test_model_with_xnnpack "${MODEL_NAME}" true true "${BUILD_TOOL}" || Q_D_ERROR="error"
    echo "::endgroup::"
  fi
  set -e
  if [[ -n "${Q_ERROR:-}" ]] || [[ -n "${D_ERROR:-}" ]] || [[ -n "${Q_D_ERROR:-}" ]]; then
    echo "Portable q8 ${Q_ERROR:-ok}," "Delegation fp32 ${D_ERROR:-ok}," "Delegation q8 ${Q_D_ERROR:-ok}"
    exit 1
  else
    prepare_artifacts_upload $UPLOAD_DIR
  fi
fi
