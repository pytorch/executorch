#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

# TODO: expand this to //...
# TODO: can't query cadence & vulkan backends
# TODO: can't query //kernels/prim_ops because of a cpp_unittest and
# broken code in shim to read oss.folly_cxx_tests. Sending fix but it
# needs to propagate and we need a submodule update.
buck2 query "//backends/apple/... + //backends/example/... + \
//backends/mediatek/... + //backends/test/... + //backends/transforms/... + \
//backends/xnnpack/... + //configurations/... + //kernels/aten/... + \
//kernels/optimized/... + //kernels/portable/... + //kernels/quantized/... + \
//kernels/test/... + //runtime/... + //schema/... + //test/... + //util/..."

UNBUILDABLE_OPTIMIZED_OPS_REGEX="gelu|fft_r2c|log_softmax"
BUILDABLE_OPTIMIZED_OPS=$(buck2 query //kernels/optimized/cpu/... | grep -E -v $UNBUILDABLE_OPTIMIZED_OPS_REGEX)

BUILDABLE_KERNELS_PRIM_OPS_TARGETS=$(buck2 query //kernels/prim_ops/... | grep -v prim_ops_test_py)
# TODO: expand the covered scope of Buck targets.
# //runtime/kernel/... is failing because //third-party:torchgen_files's shell script can't find python on PATH.
# //runtime/test/... requires Python torch, which we don't have in our OSS buck setup.
buck2 test $BUILDABLE_OPTIMIZED_OPS //kernels/portable/... \
      $BUILDABLE_KERNELS_PRIM_OPS_TARGETS //runtime/backend/... //runtime/core/... \
      //runtime/executor: //runtime/kernel/... //runtime/platform/...
