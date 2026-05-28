#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

# Keep AOTInductor precompiled headers scoped to this job. The default cache
# location can persist across macOS self-hosted runner jobs and produce stale
# PCH failures after PyTorch is reinstalled.
export TORCHINDUCTOR_CACHE_DIR="$(mktemp -d "${RUNNER_TEMP:-/tmp}/torchinductor_cache_XXXXXX")"
trap 'rm -rf "${TORCHINDUCTOR_CACHE_DIR}"' EXIT

# Run pytest with coverage. --timeout surfaces hung tests with a thread dump
# and faulthandler_timeout periodically dumps every worker's threads while
# tests are still running, so we can see which test is dragging before it
# trips the hard timeout.
${CONDA_RUN} pytest -n auto --cov=./ --cov-report=xml \
  --timeout=600 --timeout-method=thread \
  -o faulthandler_timeout=180
# Run gtest
LLVM_PROFDATA="xcrun llvm-profdata" LLVM_COV="xcrun llvm-cov" \
${CONDA_RUN} test/run_oss_cpp_tests.sh
