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
void add_unary_op(
    WebGPUGraph& graph,
    int in_id,
    int out_id,
    const char* wgsl_source,
    uint32_t wg_size_x,
    const char* op_name,
    float min = kUnaryDummyFloat,
    float max = kUnaryDummyFloat,
    // Opt-in for ops with an int32 shader variant (min/max are then
    // reinterpreted as i32). Defaults off so every other unary op keeps
    // rejecting int operands it would otherwise read as f32.
    bool int_variant = false);

} // namespace executorch::backends::webgpu
