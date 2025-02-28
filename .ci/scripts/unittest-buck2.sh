#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

# TODO: expand this to //...
# TODO: can't query cadence & vulkan backends
buck2 query "//backends/apple/... + //backends/example/... + \
//backends/mediatek/... + //backends/test/... + //backends/transforms/... + \
//backends/xnnpack/... + //configurations/... + //kernels/portable/cpu/... + \
//runtime/... + //schema/... + //test/... + //util/..."

# TODO: expand the covered scope of Buck targets.
# //runtime/kernel/... is failing because //third-party:torchgen_files's shell script can't find python on PATH.
# //runtime/test/... requires Python torch, which we don't have in our OSS buck setup.
buck2 build //runtime/backend/... //runtime/core/... //runtime/executor: //runtime/kernel/... //runtime/platform/...
buck2 test //runtime/backend/... //runtime/core/... //runtime/executor: //runtime/kernel/... //runtime/platform/...
