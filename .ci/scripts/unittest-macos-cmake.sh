#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

# Run pytest with coverage
${CONDA_RUN} pytest -n auto --cov=./ --cov-report=xml
# Run gtest
LLVM_PROFDATA="xcrun llvm-profdata" LLVM_COV="xcrun llvm-cov" \
${CONDA_RUN} test/run_oss_cpp_tests.sh
${CONDA_RUN} test/check_for_installed_private_headers_in_cmake_out.sh
