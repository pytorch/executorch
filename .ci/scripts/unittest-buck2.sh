#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

# TODO: expand this to //...
# TODO: can't query cadence & vulkan backends
# TODO: can't query //kernels/prim_ops because of non-buckified stuff in OSS.
# TODO: Make //backends/arm tests use runtime wrapper so we can just query //backends/arm/...
buck2 query "//backends/apple/... + //backends/arm: + //backends/arm/debug/... + \
//backends/arm/operator_support/... + //backends/arm/operators/... + \
//backends/arm/_passes/... + //backends/arm/runtime/... + //backends/arm/tosa/... \
+ //backends/example/... + \
//backends/mediatek/... + //backends/transforms/... + \
//backends/xnnpack/... + //codegen/tools/... + \
//configurations/... + //extension/flat_tensor: + \
//extension/llm/runner: + //kernels/aten/... + //kernels/optimized/... + \
//kernels/portable/... + //kernels/quantized/... + //kernels/test/... + \
//runtime/... + //schema/... + //test/... + //util/..."

# TODO: optimized ops are unbuildable because they now use ATen; put
# them back after we can use PyTorch in OSS buck.
UNBUILDABLE_OPTIMIZED_OPS_REGEX="_elu|gelu|fft|log_softmax"
BUILDABLE_OPTIMIZED_OPS= #$(buck2 query //kernels/optimized/cpu/... | grep -E -v $UNBUILDABLE_OPTIMIZED_OPS_REGEX)

# TODO: build prim_ops_test_cpp again once supported_features works in
# OSS buck.
BUILDABLE_KERNELS_PRIM_OPS_TARGETS=$(buck2 query //kernels/prim_ops/... | grep -v prim_ops_test)
# TODO: expand the covered scope of Buck targets.
# //runtime/kernel/... is failing because //third-party:torchgen_files's shell script can't find python on PATH.
# //runtime/test/... requires Python torch, which we don't have in our OSS buck setup.
for op in "build" "test"; do
    buck2 $op $BUILDABLE_OPTIMIZED_OPS \
          //examples/selective_build:select_all_dtype_selective_lib_portable_lib \
          //kernels/portable/... \
          $BUILDABLE_KERNELS_PRIM_OPS_TARGETS //runtime/backend/... //runtime/core/... \
          //runtime/executor: //runtime/kernel/... //runtime/platform/...
done

# Build only without testing
buck2 build //codegen/tools/... # Needs torch for testing which we don't have in our OSS buck setup.
