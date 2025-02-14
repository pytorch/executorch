#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

# Run pytest with coverage
pytest -n auto --cov=./ --cov-report=xml
# Run gtest
LLVM_PROFDATA=llvm-profdata-12 LLVM_COV=llvm-cov-12 \
test/run_oss_cpp_tests.sh
