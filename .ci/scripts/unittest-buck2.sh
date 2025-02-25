#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

# TODO: expand this to //...
# TODO: can't query cadence & vulkan backends
buck2 query "//backends/apple/... + //backends/arm/... + \
//backends/example/... + //backends/mediatek/... + //backends/test/... + \
//backends/transforms/... + //backends/xnnpack/... + //configurations/... + \
//kernels/portable/cpu/... + //runtime/... + //schema/... + //test/... + \
//util/..."

# TODO: expand the covered scope of Buck targets.
buck2 build //runtime/core/portable_type/...
buck2 test //runtime/core/portable_type/...
