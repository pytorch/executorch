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

// Dummy min/max for no-param activations; mirrors Vulkan kDummyFloat.
inline constexpr float kUnaryDummyFloat = -1.0f;

// Generic elementwise unary op; mirrors Vulkan add_unary_op_node.
// Exact integer clamp bounds for the i32 shader variants.
struct UnaryIntBounds {
  int32_t min;
  int32_t max;
};

void add_unary_op(
    WebGPUGraph& graph,
    int in_id,
    int out_id,
    const char* wgsl_source,
    uint32_t wg_size_x,
    const char* op_name,
    float min = kUnaryDummyFloat,
    float max = kUnaryDummyFloat,
    // Non-null selects an op's int32 shader variant and supplies the bounds
    // exactly. Defaults null so every other unary op keeps rejecting int
    // operands it would otherwise read as f32. Bounds are passed as i32
    // rather than reusing min/max because float cannot represent every
    // int32 exactly (16777217 would round to 16777216).
    const UnaryIntBounds* int_bounds = nullptr);

} // namespace executorch::backends::webgpu
