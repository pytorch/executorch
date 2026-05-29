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

# EXPERIMENT: run without xdist entirely so output is unbuffered and each test
# name prints immediately (with -n 1, xdist still buffers all output in a
# worker process, hiding which test is hanging). -v prints test names as they
# start; faulthandler_timeout dumps threads if a single test stalls.
${CONDA_RUN} pytest -p no:xdist -v --cov=./ --cov-report=xml \
  --timeout=1500 --timeout-method=thread \
  -o faulthandler_timeout=180
# Run gtest
LLVM_PROFDATA="xcrun llvm-profdata" LLVM_COV="xcrun llvm-cov" \
${CONDA_RUN} test/run_oss_cpp_tests.sh
