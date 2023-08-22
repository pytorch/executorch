#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test the end-to-end quantization flow.

set -e

# TODO(larryliu0820): Add CMake build
test_buck2_quantization() {
  echo "Building quantized ops shared library"
  SO_LIB=$(buck2 build //kernels/quantized:aot_lib --show-output | grep "buck-out" | cut -d" " -f2)

  echo "Run example.py"
  python -m "examples.quantization.example" --so_library="$SO_LIB" --model_name="$1"
}

test_buck2_quantization "$1"
