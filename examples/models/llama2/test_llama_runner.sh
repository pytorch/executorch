#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test Llama runner in examples/models/llama2/main.cpp
# 1. Export a llama-like model
# 2. Build llama runner binary
# 3. Run model with the llama runner binary with prompt
set -e
bash "$(dirname "${BASH_SOURCE[0]}")/../../../.ci/scripts/test_llama.sh" "$@"
