/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// OpUtils.h — back-compat re-export header.
//
// Pre-2026, this file held a 400-LOC kitchen sink of host-side helpers
// for ops. It was split into focused headers in 2026:
//
//   Elementwise.h     — ElementwiseVariant + classifyBinary +
//                       isRowContiguous / sameShape / variantPrefix
//   StrideUtils.h     — broadcastStrides + collapseContiguousDims +
//                       makeContiguousStrides + isColContiguous
//   GridDims.h        — getBlockDims + get2DGridDims + workPerThread
//   MetalDeviceInfo.h — DeviceTier + getDeviceTierFromName
//
// New code should include the focused header(s) it actually needs. This
// file remains as a single-include catch-all for back-compat with op
// files that haven't been migrated yet.

#include <executorch/backends/portable/runtime/metal_v2/Elementwise.h>
#include <executorch/backends/portable/runtime/metal_v2/GridDims.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalDeviceInfo.h>
#include <executorch/backends/portable/runtime/metal_v2/StrideUtils.h>
