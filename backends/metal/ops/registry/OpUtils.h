/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// OpUtils.h — convenience header that pulls in all op-side helpers.
//
// Bundled headers:
//   Elementwise.h     — ElementwiseVariant + classifyBinary +
//                       isRowContiguous / sameShape / variantPrefix
//   StrideUtils.h     — broadcastStrides + collapseContiguousDims +
//                       makeContiguousStrides + isColContiguous
//   GridDims.h        — getBlockDims + get2DGridDims + workPerThread
//   MetalDeviceInfo.h — DeviceTier + getDeviceTierFromName
//
// Prefer including the focused header(s) you actually need; this file
// is a single-include catch-all when several are required at once.

#include <executorch/backends/metal/ops/registry/Elementwise.h>
#include <executorch/backends/metal/ops/registry/GridDims.h>
#include <executorch/backends/metal/core/MetalDeviceInfo.h>
#include <executorch/backends/metal/ops/registry/StrideUtils.h>
