/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>

#include <cstdint>

namespace executorch::backends::webgpu {

// Shared builder for the elementwise NumPy-broadcasting binary ops
// (minimum/pow/floor_divide): validates operands, builds the 3 per-tensor
// broadcast TensorMeta UBOs, a 2D dispatch, and dynamic-shape resize hooks.
// Callers resolve their own arg layout first (e.g. floor_divide's
// rounding-mode check) and pass the resolved tensor value ids plus the op's
// WGSL source and default workgroup size. `op_name` is used in error messages
// and dispatch labels.
void add_binary_broadcast_op(
    WebGPUGraph& graph,
    int in1_id,
    int in2_id,
    int out_id,
    const char* wgsl_code,
    uint32_t wg_size_default,
    const char* op_name);

} // namespace executorch::backends::webgpu
